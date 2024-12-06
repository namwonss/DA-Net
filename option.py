import argparse
import os
import warnings
import torch
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# RSID_100
# RICE1_100
# RICE2_100
# SateHaze1K/Distributed_haze1k
parser.add_argument('--dataset_dir', type=str, default='./dataset/RSID_100')
parser.add_argument('--train', type=str, default='train', help='train dir')
# test
# test_thin
# test_moderate
# test_thick
parser.add_argument('--test', type=str, default='test', help='test dir')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--eval_step', type=int, default=1)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir', type=str, default='./trained_models/')
parser.add_argument('--net', type=str, default='DA-Net')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=240, help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--clip', action='store_true', help='use grad clip')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')


opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(opt)
print('model_dir:', opt.model_dir)

if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
