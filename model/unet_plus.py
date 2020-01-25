##########################################
# @subject : Unet++ implementation       #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
## pytorch implementation of unet++ , just use its main idea, the model is not the same as the origin unet++ mentioned in paper
## paper : UNet++: A Nested U-Net Architecture for Medical Image Segmentation
## https://arxiv.org/abs/1807.10165
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):  # x1--up , x2 ---down
        x1 = nn.functional.interpolate(x1,
                                       x2.size()[2:],
                                       mode='bilinear',
                                       align_corners=True)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (
            diffY // 2,
            diffY - diffY // 2,
            diffX // 2,
            diffX - diffX // 2,
        ))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        # print(x1.shape)
        x1 = nn.functional.interpolate(x1,
                                       x2.size()[2:],
                                       mode='bilinear',
                                       align_corners=True)
        #x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = nn.functional.interpolate(x1,
                                       x2.size()[2:],
                                       mode='bilinear',
                                       align_corners=True)
        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up5, self).__init__()
        self.conv = double_conv2(in_ch, out_ch)
    def forward(self, x1, x2, x3, x4, x5):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = nn.functional.interpolate(x1,
                                       x2.size()[2:],
                                       mode='bilinear',
                                       align_corners=True)
        #x1 = self.up(x1)
        x = torch.cat([x5, x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=4)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, dst):
        x = nn.functional.interpolate(x,
                                      dst.size()[2:],
                                      mode='bilinear',
                                      align_corners=True)
        x = self.conv(x)
        #x = F.sigmoid(x)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 5, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


cc = 16  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing


class Unet(nn.Module):
    def __init__(self, n_classes):
        super(Unet, self).__init__()
        self.inconv = inconv(3, cc)  # 3 16
        self.down1 = down(cc, 2 * cc)  # 16 32
        self.down2 = down(2 * cc, 4 * cc)  # 32 64
        self.down3 = down(4 * cc, 8 * cc)  # 64 128
        self.down4 = down(8 * cc, 16 * cc)
        self.up1 = up(24 * cc, 8 * cc)
        self.up21 = up(12 * cc, 4 * cc)
        self.up2 = up3(16 * cc, 4 * cc)  #  144 32
        self.up11 = up(6 * cc, 2 * cc)
        self.up12 = up3(8 * cc, 2 * cc)
        self.up3 = up4(10 * cc, 2 * cc)
        self.up01 = up(3 * cc, cc)  # 48 16
        self.up02 = up3(4 * cc, cc)  # 64 16
        self.up03 = up4(5 * cc, cc)  # 80 16
        self.up04 = up5(6 * cc, cc)
        self.outconv0 = outconv(cc, n_classes)
        self.outconv1 = outconv(cc, n_classes)
        self.outconv2 = outconv(cc, n_classes)
        self.outconv3 = outconv(cc, n_classes)
        self.outconv = nn.Conv2d(n_classes * 4, n_classes,kernel_size=1)

    def forward(self, x):
        x00 = self.inconv(x)  #cc
        x10 = self.down1(x00)  #2
        x20 = self.down2(x10)  #4
        x30 = self.down3(x20)  #8
        x40 = self.down4(x30)
        x31 = self.up1(x40, x30)  #8
        x21 = self.up21(x30, x20)  #4
        x22 = self.up2(x31, x21, x20)
        x11 = self.up11(x20, x10)  #2
        x12 = self.up12(x21, x11, x10)  #2cc
        x13 = self.up3(x22, x12, x11, x10)  # cc+2cc+2cc+2cc
        x01 = self.up01(x10, x00)  #cc
        x02 = self.up02(x11, x01, x00)
        x03 = self.up03(x12, x02, x01, x00)  #2cc,cc,cc,2cc
        x04 = self.up04(x13, x03, x02, x01, x00)
        y0 = self.outconv0(x01, x)
        y1 = self.outconv0(x02, x)
        y2 = self.outconv0(x03, x)
        y3 = self.outconv0(x04, x)
        y = torch.cat([y0, y1, y2, y3], dim=1)
        return self.outconv(y)


if __name__ == '__main__':
    import time
    import thop
    x = torch.rand((1, 3, 846, 255))
    lnet = Unet(3)
    rst = thop.profile(lnet, (x, ))
    print(rst)
    rtn = lnet(x)
    print(rtn.shape)
