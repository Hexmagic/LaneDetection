from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block_nested(nn.Module):
    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 relu=True,
                 stride=1,
                 padding=1,
                 dilation=1):
        super(conv_block_nested, self).__init__()
        layers = [
            nn.Conv2d(in_channel,
                      mid_channel,
                      kernel_size=3,
                      padding=dilation,
                      groups=1,
                      stride=stride,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(True),
            nn.Conv2d(mid_channel,
                      out_channel,
                      kernel_size=3,
                      padding=dilation,
                      groups=1,
                      stride=stride,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(True),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


#Nested Unet


class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, n_class=8, n1=16):
        super(NestedUNet, self).__init__()
        in_ch = 3
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # [16,32,64,128,256]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3],
                                         filters[4],
                                         filters[4],
                                         dilation=2)

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0],
                                         filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1],
                                         filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2],
                                         filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3],
                                         filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1],
                                         filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2],
                                         filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3],
                                         filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1],
                                         filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2],
                                         filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1],
                                         filters[0], filters[0])
        self.super1 = nn.Conv2d(filters[0], n_class, kernel_size=1)
        self.super2 = nn.Conv2d(filters[0], n_class, kernel_size=1)
        self.super3 = nn.Conv2d(filters[0], n_class, kernel_size=1)
        self.super4 = nn.Conv2d(filters[0], n_class, kernel_size=1)
        self.final = nn.Conv2d(n_class * 4, n_class, kernel_size=1)
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight,
                                              mode='fan_out')

    def up(self, ipt, dst):
        h, w = dst.size()[2:]
        return F.interpolate(ipt, (h, w), mode='bilinear', align_corners=True)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3,
                       self.up(x1_3, x0_0)], 1))
        deep = torch.cat([
            self.super1(x0_1),
            self.super2(x0_2),
            self.super3(x0_3),
            self.super4(x0_4)
        ],
                         dim=1)
        output = self.final(deep)
        return output


#Dictioary Unet
#if required for getting the filters and model parameters for each step

if __name__ == "__main__":
    data = torch.rand((1, 3, 255, 243))
    net = NestedUNet()
    from thop import profile
    rnt = profile(net, (data, ))
    #Ernt = net(data)
    print(rnt)
