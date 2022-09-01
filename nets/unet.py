""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np

from torch.autograd import Variable
from torchkeras import summary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn1(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,
                 activation=F.relu, space_dropout=True):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.bn1(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

class UNetPytorch(nn.Module):
    def __init__(self):
        super(UNetPytorch, self).__init__()

        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_32 = UNetConvBlock(1, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)

        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)

        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)

        self.last = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        block1 = self.conv_block1_32(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block128_256(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block256_512(pool4)

        up1 = self.up_block512_256(block5, block4)
        up2 = self.up_block256_128(up1, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, block1)
        out = self.last(up4)
        out = torch.sigmoid(out)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,init_feature_num=32, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        feature_num = init_feature_num
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, feature_num)
        self.down1 = Down(feature_num, feature_num*2)
        self.down2 = Down(feature_num*2, feature_num*4)
        self.down3 = Down(feature_num*4, feature_num*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(feature_num*8, feature_num*16 // factor)
        self.up1 = Up(feature_num*16, feature_num*8, bilinear)
        self.up2 = Up(feature_num*8, feature_num*4, bilinear)
        self.up3 = Up(feature_num*4, feature_num*2, bilinear)
        self.up4 = Up(feature_num*2, feature_num * factor, bilinear)
        self.outc_out = OutConv(feature_num, out_channels)
    # @autocast()
    def forward(self, x):
        #with autocast():
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc_out(x)
        out = torch.sigmoid(out)
        return out
if __name__ == '__main__':
    device = torch.device('cpu')  #cuda:0
    inputs = torch.rand(1,144,144).unsqueeze(0).to(device)
    net1 = UNet(in_channels=1, out_channels=1,init_feature_num=64)
    res1 = net1(inputs)
    print(summary(net1, (1,144,144)))
    print('res1 shape:', res1.shape)

    net2 = UNetPytorch()
    res2 = net2(inputs)
    print(summary(net2, (1,144,144)))
    print('res2 shape:', res2.shape)



