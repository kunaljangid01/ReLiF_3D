import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils


from dataloaders.la import (LA, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)

from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps

from val_3D_la import test_all_case

from semisam_plus import semisam_branch

from losses import *


import math
from typing import List


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training_Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/ReLiF_3D', help='experiment_name')
parser.add_argument('--prompt', type=str,
                    default='unc')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=2,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument('--output_dir', type=str,
                    default='output')

parser.add_argument('--contrast', type=float, default=0.08, help='max weight for contrastive loss')
parser.add_argument('--contrast_T', type=float, default=0.2, help='temperature for NT-Xent')
parser.add_argument('--contrast_warmup', type=int, default=2000, help='iters to ramp contrast weight 0->max')
parser.add_argument('--contrast_on', type=int, default=1, help='enable (1) or disable (0) contrastive term')

parser.add_argument('--soct_on', type=int, default=1)
parser.add_argument('--soct_t0', type=float, default=0.90)  # start conf thr (strict)
parser.add_argument('--soct_t1', type=float, default=0.70)  # end conf thr (looser)
parser.add_argument('--soct_w',  type=float, default=1.0)   # global weight for gated terms



args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



# Global buffer for hooked features
_feature_buf: List[torch.Tensor] = []

def _hook_store_features(_, __, output):
    # output: [B, C, D, H, W]
    _feature_buf.append(output.detach())

def register_last_conv3d_hook(model: nn.Module):
    # find last Conv3d module and register a forward hook
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    if last is None:
        raise RuntimeError("No Conv3d layer found for feature hook.")
    last.register_forward_hook(_hook_store_features)

class ProjectionHead(nn.Module):
    """
    Lazy 2-layer MLP: (C)->256->128 with ReLU, L2 normalize the output.
    Uses LazyLinear so we don't need to know C.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, z):      # z: [B, C]
        z = self.net(z)        # [B, out_dim]
        z = F.normalize(z, p=2, dim=1)
        return z

def nt_xent(z1, z2, T=0.2):
    """
    SimCLR NT-Xent. z1,z2: [N, D], positives are (i in z1) with (i in z2).
    Uses all other samples in the batch as negatives.
    """
    N, D = z1.size()
    z = torch.cat([z1, z2], dim=0)                     # [2N, D]
    sim = torch.mm(z, z.t())                           # cosine sims (already L2 normed)
    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim / T
    sim.masked_fill_(mask, -1e9)

    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    logits = sim
    labels = pos
    loss = F.cross_entropy(logits, labels)
    return loss




def masked_gap3d(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    feat: [B,C,D,H,W], mask: [B,1,D,H,W] in [0,1]
    returns [B,C] pooled only over masked voxels
    """
    w = mask
    num = (feat * w).sum(dim=(2,3,4))
    den = w.sum(dim=(2,3,4)).clamp_min(eps)
    return num / den



def _gauss_blur3d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return F.avg_pool3d(x, kernel_size=k, stride=1, padding=k//2)

@torch.no_grad()
def _per_sample_q(x: torch.Tensor, q: float) -> torch.Tensor:
    # x: [B,1,D,H,W] -> [B,1,1,1,1]
    v = x.flatten(2)
    qv = torch.quantile(v, q, dim=2, keepdim=True)
    return qv.view(x.size(0), 1, 1, 1, 1)

@torch.no_grad()
def sobf_augment(x: torch.Tensor,
                 iter_num: int, max_iter: int,
                 k_mul: int = 41, k_add: int = 21,
                 a_mul_max: float = 0.25, a_add_max: float = 0.12):
    """
    Spectrally-Orthogonal Bias Fields (SOBF)
    x: [B,1,D,H,W] (float, MRI volume in [~])
    Returns x2: same shape, augmented
    """
    B, C, D, H, W = x.shape
    device = x.device
    assert C == 1, "SOBF assumes single-channel input."

    # schedule: ramp up augmentation strength smoothly
    s = min(1.0, float(iter_num) / max(1, max_iter))
    a_mul = a_mul_max * (0.5 + 0.5 * s)  # multiplicative strength
    a_add = a_add_max * (0.5 + 0.5 * s)  # additive strength


    n1 = torch.randn_like(x)
    m_raw = _gauss_blur3d(n1, k=k_mul)
    m_raw = m_raw - m_raw.mean(dim=(2,3,4), keepdim=True)
    m_raw = m_raw / (m_raw.std(dim=(2,3,4), keepdim=True) + 1e-6)

    n2 = torch.randn_like(x)
    a_raw = _gauss_blur3d(n2, k=k_add)
    a_raw = a_raw - a_raw.mean(dim=(2,3,4), keepdim=True)
    a_raw = a_raw / (a_raw.std(dim=(2,3,4), keepdim=True) + 1e-6)


    dot = (a_raw.flatten(2) * m_raw.flatten(2)).sum(dim=2, keepdim=True)  # [B,1,1]
    mm  = (m_raw.flatten(2)**2).sum(dim=2, keepdim=True) + 1e-6
    a_orth = a_raw - (dot / mm).view(B,1,1,1,1) * m_raw

    M = 1.0 + a_mul * m_raw.tanh()           
    A = a_add * a_orth.tanh()             
    y = x * M + A

    lo = _per_sample_q(x, 0.01)
    hi = _per_sample_q(x, 0.99)
    y = y.clamp_(lo, hi)
    return y



def _lin_sched(it, T, a, b):
    s = min(1.0, max(0.0, it / max(1, T)))
    return a*(1 - s) + b*s



def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    
    model_label_convent = create_model()
    model_label_convent.train()
    

    proj_head = ProjectionHead(out_dim=128).cuda()
    register_last_conv3d_hook(model_label_convent)

    

    db_train = LA(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 75))    
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    
    optimizer_label_convent = optim.SGD(model_label_convent.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)    


    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    log_buffer = []
    

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            
            progress = iter_num / args.max_iterations
            lambda_overlap = 1.0 - progress
            lambda_pixel = progress


            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            
            labeled_label_batch = label_batch[:args.labeled_bs]
            

            volume_batch_fft = sobf_augment(
                volume_batch,
                iter_num=iter_num,
                max_iter=args.max_iterations,
                k_mul=41,       
                k_add=21,       
                a_mul_max=0.25, 
                a_add_max=0.12 
            )

            
            outputs_convent = model_label_convent(volume_batch_fft)
            outputs_convent_soft = torch.softmax(outputs_convent, dim=1)
            
            outputs_convent_clean = model_label_convent(volume_batch)
            outputs_convent_clean_soft = torch.softmax(outputs_convent_clean, dim=1)
            
            

            soct_mask = None
            if args.soct_on:
                # per-voxel confidence of each teacher
                conf_clean, idx_clean = outputs_convent_clean_soft.max(dim=1, keepdim=True)  # [B,1,D,H,W]
                conf_freq,  idx_freq  = outputs_convent_soft.max(dim=1, keepdim=True)       # [B,1,D,H,W]

                agree = (idx_clean == idx_freq).float()                                      # [B,1,D,H,W]

                thr = _lin_sched(iter_num, args.max_iterations, args.soct_t0, args.soct_t1)  # scalar
                confident = (conf_clean >= thr).float() * (conf_freq >= thr).float()         # [B,1,D,H,W]

                soct_mask = (agree * confident).detach()                                     # [B,1,D,H,W]

            
            noise = torch.clamp(torch.randn_like(
                volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise
            
            lo=labeled_label_batch.unsqueeze(1)
            lp = lo[:,0:1,:,:,:] + outputs_convent_soft[:,1:2,:,:,:]


            with torch.no_grad():
                samseg_mask_label, _ = semisam_branch(ema_inputs, lp, generalist='SAM-Med3D',prompt='mask')
                samseg_mask_label_soft = torch.softmax(samseg_mask_label, dim=1)
                
    
            outputs_label = outputs_convent +  samseg_mask_label
            outputs_label_soft = torch.softmax(outputs_label, dim=1)
    
            
            
            loss_boundary = 0
            
            loss_ce = ce_loss(  outputs_convent[:args.labeled_bs],
                              labeled_label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss( outputs_convent_soft[:args.labeled_bs], labeled_label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = lambda_overlap * loss_dice + lambda_pixel * (loss_ce + loss_boundary)
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            
            tau = 2.0
            P_t = F.softmax(samseg_mask_label / tau, dim=1)
            Q_t = F.softmax(outputs_convent / tau,   dim=1)
            consistency_loss = F.kl_div(Q_t.log(), P_t, reduction='mean')

            
            if args.soct_on and soct_mask is not None and (volume_batch.size(0) - args.labeled_bs) > 0:
                U_m = soct_mask[args.labeled_bs:]  # [U,1,D,H,W]
                diff = (outputs_convent_clean_soft[args.labeled_bs:] - outputs_convent_soft[args.labeled_bs:])**2  # [U,2,D,H,W]
                diff = diff.sum(dim=1, keepdim=True)  # per-voxel scalar [U,1,D,H,W]
                unsup_l = (diff * U_m).sum() / (U_m.sum() + 1e-6)
            
            
            # === LAPOC START: lesion-aware positive-only contrast ===
            lapoc_loss = torch.tensor(0.0, device=volume_batch.device)

            if args.contrast_on:
                assert len(_feature_buf) >= 2, "Feature hook buffer underflow; ensure both forwards ran."
                feat_freq  = _feature_buf[-2]      # [B,C,D,H,W]
                feat_clean = _feature_buf[-1]
                del _feature_buf[:]               
                u0 = args.labeled_bs
                feat_f_u = feat_freq[u0:]
                feat_c_u = feat_clean[u0:]
                if feat_f_u.size(0) > 0:
                    with torch.no_grad():
                        sam_u = samseg_mask_label_soft[u0:]           # [U,2,D,H,W]
                        fg_u  = sam_u[:, 1:2]                         # [U,1,D,H,W]
                        fg_conf = (fg_u ** 1.5)


                        valid = (fg_conf.sum(dim=(2,3,4)) > 50).squeeze(1)  # at least 50 voxels

                    if valid.any():
                        feat_f_u = feat_f_u[valid]
                        feat_c_u = feat_c_u[valid]
                        fg_conf  = fg_conf[valid]

                        emb_f = masked_gap3d(feat_f_u, fg_conf)   # [U', C]
                        emb_c = masked_gap3d(feat_c_u, fg_conf)   # [U', C]

                        zf = proj_head(emb_f)                     # [U', d]
                        zc = proj_head(emb_c)                     # [U', d]

                        cos_sim = F.cosine_similarity(zf, zc.detach(), dim=1)  # [U']
                        lapoc_loss = (1.0 - cos_sim).mean()
                    else:
                        lapoc_loss = torch.tensor(0.0, device=volume_batch.device)

            
            if iter_num < 1000: lapoc_loss = 0
            
            # weight & add to total loss
            contrast_w = args.contrast * min(1.0, float(iter_num) / max(1, args.contrast_warmup))

            loss = (supervised_loss
                    + consistency_weight * (args.soct_w * consistency_loss if args.soct_on else consistency_loss)
                    + (args.soct_w * unsup_l if args.soct_on else unsup_l)
                    + contrast_w * lapoc_loss)

            
            # === LAPOC END ===


            optimizer_label_convent.zero_grad()
            loss.backward()
            optimizer_label_convent.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_label_convent.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/unsup_l',
                              unsup_l, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            # ===   ===
            writer.add_scalar('contrast/lapoc_loss', float(lapoc_loss), iter_num)
            writer.add_scalar('contrast/weight', float(contrast_w), iter_num)
            # ===   ===


            log_buffer.append("\niteration {} : loss : {}, loss_ce: {}, loss_dice: {}, consistency_loss: {}, unsup_l: {}\n".format(iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss.item() ,unsup_l.item()))  
            
            writer.add_scalar('loss/loss', loss, iter_num)
            
            
            if len(log_buffer) > 10:
                logging.info("\n".join(log_buffer))
                log_buffer = []
                
                
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_label_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model_label_convent.eval()
                avg_metric = test_all_case(
                    model_label_convent, args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model_label_convent.state_dict(), save_mode_path)
                    torch.save(model_label_convent.state_dict(), save_best)

                    
                writer.add_scalar('info/test_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/test_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model_label_convent.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model_label_convent.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../output/{}/{}_{}/{}".format( 
        args.output_dir, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')


    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    torch.cuda.empty_cache()  # Free up GPU memory before starting training
    
    train(args, snapshot_path)
