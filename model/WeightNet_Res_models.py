import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ResNet50 import Backbone_ResNet50_in3


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# for conv5
class FLFA_5(nn.Module):
    def __init__(self, channel=512):
        super(FLFA_5, self).__init__()
        self.relu = nn.ReLU(True)

        self.downsample2 = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.downsample4 = nn.MaxPool2d(4, stride=4,ceil_mode=True)
        self.downsample8 = nn.MaxPool2d(8, stride=8,ceil_mode=True)
        self.downsample16 = nn.MaxPool2d(16, stride=16,ceil_mode=True)
        
        self.conv_c1 = BasicConv2d(channel*5, channel,1, padding=0)
        self.conv_c2 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x_cur, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp):
        # full-level feature Aggregation
        
        #-------------Copy SOD to different channels--------------#
        s5_lbp = self.downsample2(s5_lbp)
        SOD5 = torch.cat((s5_lbp,s5_lbp),1) #2 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #4 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #8 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #16 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #32 Channels
        SOD5_64 = torch.cat((SOD5,SOD5),1) #64 Channels
        SOD5_128 = torch.cat((SOD5_64,SOD5_64),1) #128 Channels
        SOD5_256 = torch.cat((SOD5_128,SOD5_128),1) #256 Channels
        SOD5_512 = torch.cat((SOD5_256,SOD5_256),1) #512 Channels
        
        s4_lbp = self.downsample4(s4_lbp)
        SOD4 = torch.cat((s4_lbp,s4_lbp),1) #2 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #4 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #8 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #16 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #32 Channels
        SOD4_64 = torch.cat((SOD4,SOD4),1) #64 Channels
        SOD4_128 = torch.cat((SOD4_64,SOD4_64),1) #128 Channels
        SOD4_256 = torch.cat((SOD4_128,SOD4_128),1) #256 Channels
        SOD4_512 = torch.cat((SOD4_256,SOD4_256),1) #512 Channels
        
        s3_lbp = self.downsample8(s3_lbp)
        SOD3 = torch.cat((s3_lbp,s3_lbp),1) #2 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #4 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #8 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #16 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #32 Channels
        SOD3_64 = torch.cat((SOD3,SOD3),1) #64 Channels
        SOD3_128 = torch.cat((SOD3_64,SOD3_64),1) #128 Channels
        SOD3_256 = torch.cat((SOD3_128,SOD3_128),1) #256 Channels
        SOD3_512 = torch.cat((SOD3_256,SOD3_256),1) #512 Channels
        
        s2_lbp = self.downsample16(s2_lbp)
        SOD2 = torch.cat((s2_lbp,s2_lbp),1) #2 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #4 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #8 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #16 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #32 Channels
        SOD2_64 = torch.cat((SOD2,SOD2),1) #64 Channels
        SOD2_128 = torch.cat((SOD2_64,SOD2_64),1) #128 Channels
        SOD2_256 = torch.cat((SOD2_128,SOD2_128),1) #256 Channels
        SOD2_512 = torch.cat((SOD2_256,SOD2_256),1) #512 Channels
        
        s1_lbp = self.downsample16(s1_lbp)
        SOD1 = torch.cat((s1_lbp,s1_lbp),1) #2 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #4 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #8 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #16 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #32 Channels
        SOD1_64 = torch.cat((SOD1,SOD1),1) #64 Channels
        SOD1_128 = torch.cat((SOD1_64,SOD1_64),1) #128 Channels
        SOD1_256 = torch.cat((SOD1_128,SOD1_128),1) #256 Channels
        SOD1_512 = torch.cat((SOD1_256,SOD1_256),1) #512 Channels       

        x = torch.cat((torch.mul(x_cur, SOD5_512)+x_cur,torch.mul(x_cur, SOD4_512)+x_cur,torch.mul(x_cur, SOD3_512)+x_cur,torch.mul(x_cur, SOD2_512)+x_cur,torch.mul(x_cur, SOD1_512)+x_cur),1)
        x = self.conv_c1(x)
        x_1 = self.conv_c2(x)
        x_1 = self.conv_c2(x_1)
        x_LocAndGlo = x + x_1  
        x_LocAndGlo = x_LocAndGlo + x_cur

        return x_LocAndGlo

# for conv4
class FLFA_4(nn.Module):
    def __init__(self, channel=512):
        super(FLFA_4, self).__init__()
        self.relu = nn.ReLU(True)

        self.downsample2 = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.downsample4 = nn.MaxPool2d(4, stride=4,ceil_mode=True)
        self.downsample8 = nn.MaxPool2d(8, stride=8,ceil_mode=True)
        
        self.conv_c1 = BasicConv2d(channel*5, channel, 1, padding=0)
        self.conv_c2 = BasicConv2d(channel, channel, 3, padding=1)
        

    def forward(self, x_cur, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp):
        # full-level feature Aggregation
        #-------------Copy SOD to different channels--------------#
        
        SOD5 = torch.cat((s5_lbp,s5_lbp),1) #2 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #4 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #8 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #16 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #32 Channels
        SOD5_64 = torch.cat((SOD5,SOD5),1) #64 Channels
        SOD5_128 = torch.cat((SOD5_64,SOD5_64),1) #128 Channels
        SOD5_256 = torch.cat((SOD5_128,SOD5_128),1) #256 Channels
        SOD5_512 = torch.cat((SOD5_256,SOD5_256),1) #512 Channels
        
        s4_lbp = self.downsample2(s4_lbp)
        SOD4 = torch.cat((s4_lbp,s4_lbp),1) #2 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #4 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #8 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #16 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #32 Channels
        SOD4_64 = torch.cat((SOD4,SOD4),1) #64 Channels
        SOD4_128 = torch.cat((SOD4_64,SOD4_64),1) #128 Channels
        SOD4_256 = torch.cat((SOD4_128,SOD4_128),1) #256 Channels
        SOD4_512 = torch.cat((SOD4_256,SOD4_256),1) #512 Channels
        
        s3_lbp = self.downsample4(s3_lbp)
        SOD3 = torch.cat((s3_lbp,s3_lbp),1) #2 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #4 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #8 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #16 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #32 Channels
        SOD3_64 = torch.cat((SOD3,SOD3),1) #64 Channels
        SOD3_128 = torch.cat((SOD3_64,SOD3_64),1) #128 Channels
        SOD3_256 = torch.cat((SOD3_128,SOD3_128),1) #256 Channels
        SOD3_512 = torch.cat((SOD3_256,SOD3_256),1) #512 Channels
        
        s2_lbp = self.downsample8(s2_lbp)
        SOD2 = torch.cat((s2_lbp,s2_lbp),1) #2 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #4 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #8 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #16 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #32 Channels
        SOD2_64 = torch.cat((SOD2,SOD2),1) #64 Channels
        SOD2_128 = torch.cat((SOD2_64,SOD2_64),1) #128 Channels
        SOD2_256 = torch.cat((SOD2_128,SOD2_128),1) #256 Channels
        SOD2_512 = torch.cat((SOD2_256,SOD2_256),1) #512 Channels
        
        s1_lbp = self.downsample8(s1_lbp)
        SOD1 = torch.cat((s1_lbp,s1_lbp),1) #2 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #4 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #8 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #16 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #32 Channels
        SOD1_64 = torch.cat((SOD1,SOD1),1) #64 Channels
        SOD1_128 = torch.cat((SOD1_64,SOD1_64),1) #128 Channels
        SOD1_256 = torch.cat((SOD1_128,SOD1_128),1) #256 Channels
        SOD1_512 = torch.cat((SOD1_256,SOD1_256),1) #512 Channels       
        
        x = torch.cat((torch.mul(x_cur, SOD5_512)+x_cur,torch.mul(x_cur, SOD4_512)+x_cur,torch.mul(x_cur, SOD3_512)+x_cur,torch.mul(x_cur, SOD2_512)+x_cur,torch.mul(x_cur, SOD1_512)+x_cur),1)
        x = self.conv_c1(x)
        x_1 = self.conv_c2(x)
        x_1 = self.conv_c2(x_1)
        x_LocAndGlo = x + x_1  
        x_LocAndGlo = x_LocAndGlo + x_cur

        return x_LocAndGlo

# for conv3
class FLFA_3(nn.Module):
    def __init__(self, channel=256):
        super(FLFA_3, self).__init__()
        self.relu = nn.ReLU(True)

        self.downsample2 = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.downsample4 = nn.MaxPool2d(4, stride=4,ceil_mode=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_c1 = BasicConv2d(channel*5, channel, 1, padding=0)
        self.conv_c2 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x_cur, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp):
        # full-level feature Aggregation
        #-------------Copy SOD to different channels--------------#
        s5_lbp = self.upsample2(s5_lbp)
        SOD5 = torch.cat((s5_lbp,s5_lbp),1) #2 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #4 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #8 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #16 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #32 Channels
        SOD5_64 = torch.cat((SOD5,SOD5),1) #64 Channels
        SOD5_128 = torch.cat((SOD5_64,SOD5_64),1) #128 Channels
        SOD5_256 = torch.cat((SOD5_128,SOD5_128),1) #256 Channels
        
#        s4_lbp = self.upsample2(s4_lbp)
        SOD4 = torch.cat((s4_lbp,s4_lbp),1) #2 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #4 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #8 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #16 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #32 Channels
        SOD4_64 = torch.cat((SOD4,SOD4),1) #64 Channels
        SOD4_128 = torch.cat((SOD4_64,SOD4_64),1) #128 Channels
        SOD4_256 = torch.cat((SOD4_128,SOD4_128),1) #256 Channels
        
        s3_lbp = self.downsample2(s3_lbp)
        SOD3 = torch.cat((s3_lbp,s3_lbp),1) #2 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #4 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #8 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #16 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #32 Channels
        SOD3_64 = torch.cat((SOD3,SOD3),1) #64 Channels
        SOD3_128 = torch.cat((SOD3_64,SOD3_64),1) #128 Channels
        SOD3_256 = torch.cat((SOD3_128,SOD3_128),1) #256 Channels
        
        s2_lbp = self.downsample4(s2_lbp)
        SOD2 = torch.cat((s2_lbp,s2_lbp),1) #2 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #4 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #8 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #16 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #32 Channels
        SOD2_64 = torch.cat((SOD2,SOD2),1) #64 Channels
        SOD2_128 = torch.cat((SOD2_64,SOD2_64),1) #128 Channels
        SOD2_256 = torch.cat((SOD2_128,SOD2_128),1) #256 Channels
        
        s1_lbp = self.downsample4(s1_lbp)
        SOD1 = torch.cat((s1_lbp,s1_lbp),1) #2 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #4 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #8 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #16 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #32 Channels
        SOD1_64 = torch.cat((SOD1,SOD1),1) #64 Channels
        SOD1_128 = torch.cat((SOD1_64,SOD1_64),1) #128 Channels
        SOD1_256 = torch.cat((SOD1_128,SOD1_128),1) #256 Channels
        
        x = torch.cat((torch.mul(x_cur, SOD5_256)+x_cur,torch.mul(x_cur, SOD4_256)+x_cur,torch.mul(x_cur, SOD3_256)+x_cur,torch.mul(x_cur, SOD2_256)+x_cur,torch.mul(x_cur, SOD1_256)+x_cur),1)
        x = self.conv_c1(x)
        x_1 = self.conv_c2(x)
        x_1 = self.conv_c2(x_1)
        x_LocAndGlo = x + x_1  
        x_LocAndGlo = x_LocAndGlo + x_cur

        return x_LocAndGlo
        
# for conv2
class FLFA_2(nn.Module):
    def __init__(self, channel=128):
        super(FLFA_2, self).__init__()
        self.relu = nn.ReLU(True)

        self.downsample2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.downsample4 = nn.MaxPool2d(4, stride=4, ceil_mode=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_c1 = BasicConv2d(channel*5, channel, 1, padding=0)
        self.conv_c2 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x_cur, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp):
        # full-level feature Aggregation
        #-------------Copy SOD to different channels--------------#
        s5_lbp = self.upsample4(s5_lbp)
        SOD5 = torch.cat((s5_lbp,s5_lbp),1) #2 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #4 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #8 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #16 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #32 Channels
        SOD5_64 = torch.cat((SOD5,SOD5),1) #64 Channels
        SOD5_128 = torch.cat((SOD5_64,SOD5_64),1) #128 Channels
        
        s4_lbp = self.upsample2(s4_lbp)
        SOD4 = torch.cat((s4_lbp,s4_lbp),1) #2 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #4 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #8 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #16 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #32 Channels
        SOD4_64 = torch.cat((SOD4,SOD4),1) #64 Channels
        SOD4_128 = torch.cat((SOD4_64,SOD4_64),1) #128 Channels
        
#        s3_lbp = self.upsample4(s3_lbp)
        SOD3 = torch.cat((s3_lbp,s3_lbp),1) #2 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #4 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #8 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #16 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #32 Channels
        SOD3_64 = torch.cat((SOD3,SOD3),1) #64 Channels
        SOD3_128 = torch.cat((SOD3_64,SOD3_64),1) #128 Channels
        
        s2_lbp = self.downsample2(s2_lbp)
        SOD2 = torch.cat((s2_lbp,s2_lbp),1) #2 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #4 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #8 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #16 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #32 Channels
        SOD2_64 = torch.cat((SOD2,SOD2),1) #64 Channels
        SOD2_128 = torch.cat((SOD2_64,SOD2_64),1) #128 Channels
        
        s1_lbp = self.downsample2(s1_lbp)
        SOD1 = torch.cat((s1_lbp,s1_lbp),1) #2 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #4 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #8 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #16 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #32 Channels
        SOD1_64 = torch.cat((SOD1,SOD1),1) #64 Channels
        SOD1_128 = torch.cat((SOD1_64,SOD1_64),1) #128 Channels

        x = torch.cat((torch.mul(x_cur, SOD5_128)+x_cur,torch.mul(x_cur, SOD4_128)+x_cur,torch.mul(x_cur, SOD3_128)+x_cur,torch.mul(x_cur, SOD2_128)+x_cur,torch.mul(x_cur, SOD1_128)+x_cur),1)
        x = self.conv_c1(x)
        x_1 = self.conv_c2(x)
        x_1 = self.conv_c2(x_1)
        x_LocAndGlo = x + x_1  
        x_LocAndGlo = x_LocAndGlo + x_cur 

        return x_LocAndGlo

# for conv1
class FLFA_1(nn.Module):
    def __init__(self, channel=64):
        super(FLFA_1, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_c1 = BasicConv2d(channel*5, channel, 1, padding=0)
        self.conv_c2 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x_cur, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp):
        # full-level feature Aggregation
        #-------------Copy SOD to different channels--------------#
        s5_lbp = self.upsample8(s5_lbp)
        SOD5 = torch.cat((s5_lbp,s5_lbp),1) #2 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #4 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #8 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #16 Channels
        SOD5 = torch.cat((SOD5,SOD5),1) #32 Channels
        SOD5_64 = torch.cat((SOD5,SOD5),1) #64 Channels
        
        s4_lbp = self.upsample4(s4_lbp)
        SOD4 = torch.cat((s4_lbp,s4_lbp),1) #2 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #4 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #8 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #16 Channels
        SOD4 = torch.cat((SOD4,SOD4),1) #32 Channels
        SOD4_64 = torch.cat((SOD4,SOD4),1) #64 Channels
        
        s3_lbp = self.upsample2(s3_lbp)
        SOD3 = torch.cat((s3_lbp,s3_lbp),1) #2 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #4 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #8 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #16 Channels
        SOD3 = torch.cat((SOD3,SOD3),1) #32 Channels
        SOD3_64 = torch.cat((SOD3,SOD3),1) #64 Channels
        
#        s2_lbp = self.downsample2(s2_lbp)
        SOD2 = torch.cat((s2_lbp,s2_lbp),1) #2 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #4 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #8 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #16 Channels
        SOD2 = torch.cat((SOD2,SOD2),1) #32 Channels
        SOD2_64 = torch.cat((SOD2,SOD2),1) #64 Channels
        
#        s1_lbp = self.downsample2(s1_lbp)
        SOD1 = torch.cat((s1_lbp,s1_lbp),1) #2 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #4 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #8 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #16 Channels
        SOD1 = torch.cat((SOD1,SOD1),1) #32 Channels
        SOD1_64 = torch.cat((SOD1,SOD1),1) #64 Channels

        x = torch.cat((torch.mul(x_cur, SOD5_64)+x_cur,torch.mul(x_cur, SOD4_64)+x_cur,torch.mul(x_cur, SOD3_64)+x_cur,torch.mul(x_cur, SOD2_64)+x_cur,torch.mul(x_cur, SOD1_64)+x_cur),1)
        x = self.conv_c1(x)
        x_1 = self.conv_c2(x)
        x_1 = self.conv_c2(x_1)
        x_LocAndGlo = x + x_1  
        x_LocAndGlo = x_LocAndGlo + x_cur
        

        return x_LocAndGlo
        

class Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=2, dilation_2=2, dilation_3=2):
        super(Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_1, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_3, 3, padding=1)
        self.conv3_Dila = BasicConv2d(channel_2, channel_3, 3, padding=dilation_3, dilation=dilation_3)


    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x)        
        x1_all = x1+x1_dila

        x2 = self.conv2(x1_all)
        x2_dila = self.conv2_Dila(x1_all)        
        x2_all = x2 + x2_dila

        x3 = self.conv3(x2_all)
        x3_dila = self.conv3_Dila(x2_all)
        x3_all = x3 + x3_dila

        return x3_all


class decoder_1(nn.Module):
    def __init__(self, channel=512):
        super(decoder_1, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            Decoder(512, 512, 512, 3, 2, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            Decoder(1024, 512, 256, 3, 2, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            Decoder(512, 256, 128, 5, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            Decoder(256, 128, 64, 5, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            Decoder(128, 64, 32, 5, 3, 2)
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)
        x5_sig = self.sigmoid(s5)

        x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
        s4 = self.S4(x4_up)
        x4_sig = self.sigmoid(s4)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)
        x3_sig = self.sigmoid(s3)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)
        x2_sig = self.sigmoid(s2)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        s1 = self.S1(x1_up)
        x1_sig = self.sigmoid(s1)

        return s1, s2, s3, s4, s5, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig
        
class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            Decoder(512, 512, 512, 3, 2, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            Decoder(1024+1, 512, 256, 3, 2, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            Decoder(512+1, 256, 128, 5, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            Decoder(256+1, 128, 64, 5, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            Decoder(128+1, 64, 32, 5, 3, 2)
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        
        self.downsample2 = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.downsample4 = nn.MaxPool2d(4, stride=4,ceil_mode=True)
        self.downsample8 = nn.MaxPool2d(8, stride=8,ceil_mode=True)
        self.downsample16 = nn.MaxPool2d(16, stride=16,ceil_mode=True)   
        
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
        
        self.conv1 = BasicConv2d(128, 128, 3, padding=1)
        self.conv2 = BasicConv2d(128, 1, 3, padding=1)
        
        self.conv3 = BasicConv2d(512, 128, 3, padding=1)
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x5, x4, x3, x2, x1, xx1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        
        SOD = torch.cat((xx1,xx1),1) #2 Channels
        SOD = torch.cat((SOD,SOD),1) #4 Channels
        SOD = torch.cat((SOD,SOD),1) #8 Channels
        SOD = torch.cat((SOD,SOD),1) #16 Channels
        SOD = torch.cat((SOD,SOD),1) #32 Channels
        SOD_64 = torch.cat((SOD,SOD),1) #64 Channels
        SOD_128 = torch.cat((SOD_64,SOD_64),1) #128 Channels
        
        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)
        
        x5 = self.conv3(x5)

        x4_add = self.upsample2(x5)+torch.mul(self.upsample2(x5), self.downsample8(SOD_128))
        x4_add = self.conv1(x4_add)
        x4_add = self.sigmoid(self.conv2(x4_add))
        x4_up = self.decoder4(torch.cat((x4, x5_up, x4_add), 1))
        s4 = self.S4(x4_up)

        x3_add = self.upsample4(x5)+torch.mul(self.upsample4(x5), self.downsample4(SOD_128))
        x3_add = self.conv1(x3_add)
        x3_add = self.sigmoid(self.conv2(x3_add))
        x3_up = self.decoder3(torch.cat((x3, x4_up,x3_add), 1))
        s3 = self.S3(x3_up)

        x2_add = self.upsample8(x5)+torch.mul(self.upsample8(x5), self.downsample2(SOD_128))
        x2_add = self.conv1(x2_add)
        x2_add = self.sigmoid(self.conv2(x2_add))
        x2_up = self.decoder2(torch.cat((x2, x3_up, x2_add), 1))
        s2 = self.S2(x2_up)

        x1_add = self.upsample16(x5)+torch.mul(self.upsample16(x5), SOD_128)
        x1_add = self.conv1(x1_add)
        x1_add = self.sigmoid(self.conv2(x1_add))
        x1_up = self.decoder1(torch.cat((x1, x2_up, x1_add), 1))
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4, s5 
            
class WeightNet_Res(nn.Module):
    def __init__(self, channel=32):
        super(WeightNet_Res, self).__init__()
        #Backbone model
        # ---- ResNet50 Backbone ----
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_ResNet50_in3()

        # Lateral layers
        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(1024, 512, 3, stride=1, padding=1)
        self.lateral_conv4 = BasicConv2d(2048, 512, 3, stride=1, padding=1)

        self.FLFA5 = FLFA_5(512)
        self.FLFA4 = FLFA_4(512)
        self.FLFA3 = FLFA_3(256)
        self.FLFA2 = FLFA_2(128)
        self.FLFA1 = FLFA_1(64)

        # self.agg2_rgbd = aggregation(channel)
        self.decoder_rgb = decoder(512)
        
        self.decoder_1 = decoder_1(512)
        

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

    def forward(self, rgb):
        x0_1 = self.encoder1(rgb)
        x1_1 = self.encoder2(x0_1)
        x2_1 = self.encoder4(x1_1)
        x3_1 = self.encoder8(x2_1)
        x4_1 = self.encoder16(x3_1)

        x1_e1 = self.lateral_conv0(x0_1)
        x2_e1 = self.lateral_conv1(x1_1)
        x3_e1 = self.lateral_conv2(x2_1)
        x4_e1 = self.lateral_conv3(x3_1)
        x5_e1 = self.lateral_conv4(x4_1) 
              
        s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig = self.decoder_1(x5_e1, x4_e1, x3_e1, x2_e1, x1_e1) 
        
        x0 = self.encoder1(rgb)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        x1_rgb = self.lateral_conv0(x0)
        x2_rgb = self.lateral_conv1(x1)
        x3_rgb = self.lateral_conv2(x2)
        x4_rgb = self.lateral_conv3(x3)
        x5_rgb = self.lateral_conv4(x4)       

        # up means update
        x5_FLFA = self.FLFA5(x5_rgb, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig)
        x4_FLFA = self.FLFA4(x4_rgb, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig)
        x3_FLFA = self.FLFA3(x3_rgb, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig)
        x2_FLFA = self.FLFA2(x2_rgb, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig)
        x1_FLFA = self.FLFA1(x1_rgb, x1_sig, x2_sig, x3_sig, x4_sig, x5_sig)
        

        s1, s2, s3, s4, s5 = self.decoder_rgb(x5_FLFA, x4_FLFA, x3_FLFA, x2_FLFA, x1_FLFA, x1_sig)
        
        # At test phase, we can use the HA to post-processing our saliency map
        s1_lbp = self.upsample2(s1_lbp)
        s2_lbp = self.upsample2(s2_lbp)
        s3_lbp = self.upsample4(s3_lbp)
        s4_lbp = self.upsample8(s4_lbp)
        s5_lbp = self.upsample16(s5_lbp)
        
        s1 = self.upsample2(s1)
        s2 = self.upsample2(s2)
        s3 = self.upsample4(s3)
        s4 = self.upsample8(s4)
        s5 = self.upsample16(s5)
        
        return s1, s2, s3, s4, s5, s1_lbp, s2_lbp, s3_lbp, s4_lbp, s5_lbp
