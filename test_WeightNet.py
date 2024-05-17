import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio
import time
import cv2

from model.WeightNet_VGG_models import WeightNet_VGG
from model.WeightNet_Res_models import WeightNet_Res
from data import test_dataset

import warnings
warnings.filterwarnings("ignore")

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = './dataset/test_dataset/'

if opt.is_ResNet:
    model = WeightNet_Res()
    model.load_state_dict(torch.load('/data/dlm/WeightNet/models/WeightNet_Res/WeightNet_ResNet.pth.41'))
else:
    model = WeightNet_VGG()
    model.load_state_dict(torch.load('/data/dlm/WeightNet/models/WeightNet_VGG/WeightNet_VGG.pth.59'))

model.cuda()
model.eval()

#test_datasets = ['EORSSD']
#test_datasets = ['ORSSD']
test_datasets = ['ORSI-4199']
#test_datasets = ['SSO']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = './results/VGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
#    image_root = '/data/dlm/ORSSD/Image-test/'
#    image_root = '/data/dlm/EORSSD/Image-test/'
    image_root = '/data/dlm/ors-4199/testset/images/'
#    image_root = '/data/dlm/ors-4199/challenge/SSO/images/'
    print(dataset)
#    gt_root = '/data/dlm/ORSSD/GT-test/'
#    gt_root = '/data/dlm/EORSSD/GT-test/'
    gt_root = '/data/dlm/ors-4199/testset/gt/'
#    gt_root = '/data/dlm/ors-4199/challenge/SSO/gt/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image,  gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s2, s3, s4, s5, s1_1, s2_1, s3_1, s4_1, s5_1 = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*256)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))