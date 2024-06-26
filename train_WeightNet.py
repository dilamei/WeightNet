import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.WeightNet_VGG_models import WeightNet_VGG
from model.WeightNet_Res_models import WeightNet_Res
from data import get_loader
from utils import clip_gradient, adjust_lr

import warnings
warnings.filterwarnings("ignore")

import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# For vgg, batchsize is 6; for ResNet, batchsize is 8.
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = WeightNet_Res()
else:
    model = WeightNet_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

#image_root = '/data/dlm/ORSSD/train-images/'
#gt_root = '/data/dlm/ORSSD/train-labels/'

image_root = '/data/dlm/EORSSD/train-images/'
gt_root = '/data/dlm/EORSSD/train-labels/'

#image_root = '/data/dlm/ors-4199/trainset/images/'
#gt_root = '/data/dlm/ors-4199/trainset/gt/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
floss = pytorch_fm.FLoss()

sigmoid = torch.nn.Sigmoid()

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)

        images = images.cuda()
        gts = gts.cuda()


        s1, s2, s3, s4, s5, s1_1, s2_1, s3_1, s4_1 = model(images)

        loss1 = CE(s1, gts) + IOU(sigmoid(s1), gts) + floss(sigmoid(s1), gts)
        loss2 = CE(s2, gts) + IOU(sigmoid(s2), gts) + floss(sigmoid(s2), gts)
        loss3 = CE(s3, gts) + IOU(sigmoid(s3), gts) + floss(sigmoid(s3), gts)
        loss4 = CE(s4, gts) + IOU(sigmoid(s4), gts) + floss(sigmoid(s4), gts)
        loss5 = CE(s5, gts) + IOU(sigmoid(s5), gts) + floss(sigmoid(s5), gts)
        
        loss1_1 = CE(s1_1, gts) + IOU(sigmoid(s1_1), gts) + floss(sigmoid(s1_1), gts)
        loss2_1 = CE(s2_1, gts) + IOU(sigmoid(s2_1), gts) + floss(sigmoid(s2_1), gts)
        loss3_1 = CE(s3_1, gts) + IOU(sigmoid(s3_1), gts) + floss(sigmoid(s3_1), gts)
        loss4_1 = CE(s4_1, gts) + IOU(sigmoid(s4_1), gts) + floss(sigmoid(s4_1), gts)
        loss5_1 = CE(s5_1, gts) + IOU(sigmoid(s5_1), gts) + floss(sigmoid(s5_1), gts)
        

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss1_1 + loss2_1 + loss3_1 + loss4_1 + loss5_1
#        + loss2 + loss3 + loss4 + loss5

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                 '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1_1: {:.4f} , Loss1_2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data, loss1_1.data))

    if opt.is_ResNet:
        save_path = '/data/dlm/WeightNet/models/WeightNet_Res/'
    else:
        save_path = '/data/dlm/WeightNet/models/WeightNet_hy/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) > 30:
        if opt.is_ResNet:
            torch.save(model.state_dict(), save_path + 'WeightNet_ResNet.pth' + '.%d' % epoch)
        else:
            torch.save(model.state_dict(), save_path + 'WeightNet_VGG.pth' + '.%d' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
