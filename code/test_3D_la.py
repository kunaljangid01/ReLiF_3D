import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from test_3D_util_la import test_all_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training_Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='output/ReLiF_3D/LA/ReLiF_3D_2', help='experiment_name')  
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')


def Inference(FLAGS):
    snapshot_path = "{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.list", num_classes=num_classes,
                               patch_size=(128, 128, 128), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    
    # Save metrics to a log file
    log_file = os.path.join("{}/{}/log_test.txt".format(FLAGS.exp, FLAGS.model))
    with open(log_file, "w") as f:
        f.write("Arguments:\n")
        f.write(str(FLAGS) + "\n")
        
        f.write("Test Metrics:\n")
        f.write(str(metric) + "\n")
