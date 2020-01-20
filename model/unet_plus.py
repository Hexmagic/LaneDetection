from torch.nn import *
import torch
import torch.nn.functional as F


class ConvBlock(Module):
    def __init__(self, in_channel, out_channel, k=3, p=1, g=1, d=1, s=1):
        super(ConvBlock, self).__init__()
        self.net = Sequential(
            BatchNorm2d(in_channel), ReLU(True),
            Conv2d(in_channel,
                   out_channel,
                   kernel_size=k,
                   padding=p,
                   groups=g,
                   dilation=d,
                   stride=s))

    def forward(self, x):
        return self.net(x)


class DownBlock(Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(DownBlock, self).__init__()
        self.conv = Sequential(ConvBlock(in_channel, out_channel),
                               ConvBlock(out_channel, out_channel))
        if stride == 2:
            self.down = MaxPool2d(kernel_size=3, padding=1, stride=2)
        else:
            self.down = Sequential()

    def forward(self, x):
        conv = self.conv(x)
        pool = self.down(conv)
        return conv, pool


class Encode(Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.down1 = DownBlock(3, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 1024, stride=1)

    def forward(self, x):
        a, x = self.down1(x)
        b, x = self.down2(x)
        c, x = self.down3(x)
        d, x = self.down4(x)
        _, x = self.down5(x)
        return [a, b, c, d], x


class UpBlock(Module):
    def __init__(self, in_channel, out_channel, last=False):
        super(UpBlock, self).__init__()
        self.conv = Sequential(ConvBlock(in_channel, out_channel),
                               ConvBlock(out_channel, out_channel))
        self.proj = ConvBlock(in_channel, out_channel)

    def forward(self, x, short):
        x = self.proj(x)
        cat = torch.cat([short, x], dim=1)
        return self.conv(cat)


class UnetPlus(Module):
    def __init__(self, n_class):
        super(UnetPlus, self).__init__()
        self.encode = Encode()
        self.u31 = UpBlock(1024, 512)

        self.c20_21 = ConvBlock(256, 256)
        self.u21 = UpBlock(512, 256)
        self.c21_22 = ConvBlock(256, 256)
        self.c20_22 = ConvBlock(256, 256)

        self.u22 = UpBlock(512, 256)

        self.c10_11 = ConvBlock(128, 128)
        self.u11 = UpBlock(256, 128)
        self.c10_12 = ConvBlock(128, 128)
        self.c10_13 = ConvBlock(128, 128)
        self.c11_12 = ConvBlock(128, 128)
        self.u12 = UpBlock(256, 128)
        self.c11_13 = ConvBlock(128, 128)
        self.c12_13 = ConvBlock(128, 128)
        self.u13 = UpBlock(256, 128)

        self.c00_01 = ConvBlock(64, 64)
        self.u01 = UpBlock(128, 64)
        self.c00_02 = ConvBlock(64, 64)
        self.c00_03 = ConvBlock(64, 64)
        self.c00_04 = ConvBlock(64, 64)
        self.c01_02 = ConvBlock(64, 64)
        self.c01_03 = ConvBlock(64, 64)
        self.c01_04 = ConvBlock(64, 64)
        self.u02 = UpBlock(128, 64)
        self.c02_03 = ConvBlock(64, 64)
        self.c02_04 = ConvBlock(64, 64)
        self.u03 = UpBlock(128, 64)
        self.c03_04 = ConvBlock(64, 64)
        self.u04 = UpBlock(128, 64, last=True)
        self.deepsuper1 = ConvBlock(64, n_class, k=1, p=0)
        self.deepsuper2 = ConvBlock(64, n_class, k=1, p=0)
        self.deepsuper3 = ConvBlock(64, n_class, k=1, p=0)
        self.deepsuper4 = ConvBlock(64, n_class, k=1, p=0)
        self.classifer = Conv2d(32, n_class, 1)

    def forward(self, x):
        h, w = x.size()[2:]
        [x00, x10, x20, x30], x40 = self.encode(x)
        x40 = F.interpolate(x40,
                            x30.size()[2:],
                            mode='bilinear',
                            align_corners=True)

        x31 = self.u31(x40, x30)
        x30 = F.interpolate(x30,
                            x20.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x21 = self.u21(x30, self.c20_21(x20))
        x31 = F.interpolate(x31,
                            x30.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x22 = self.u22(x31, self.c21_22(x21) + self.c20_22(x20))

        x20 = F.interpolate(x20,
                            x10.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x11 = self.u11(x20, self.c10_11(x10))
        x21 = F.interpolate(x21,
                            x10.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x12 = self.u12(x21, self.c11_12(x11)) + self.c10_12(x10)
        x22 = F.interpolate(x22,
                            x20.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x13 = self.u13(x22,
                       self.c12_13(x12) + self.c11_13(x11) + self.c10_13(x10))
        x10 = F.interpolate(x10, (h, w), mode='bilinear', align_corners=True)
        x01 = self.u01(x10, self.c00_01(x00))
        x11 = F.interpolate(x11, (h, w), mode='bilinear', align_corners=True)
        x02 = self.u02(x11, self.c01_02(x01) + self.c00_02(x00))
        x12 = F.interpolate(x12, (h, w), mode='bilinear', align_corners=True)
        x03 = self.u03(x12,
                       self.c02_03(x02) + self.c01_03(x01) + self.c00_03(x00))
        x13 = F.interpolate(x13, (h, w), align_corners=True, mode='bilinear')
        x04 = self.u04(
            x13,
            self.c03_04(x03) + self.c02_04(x02) + self.c01_04(x01) +
            self.c00_04(x00))
        deepsuper = torch.cat([
            self.deepsuper1(x01),
            self.deepsuper2(x02),
            self.deepsuper3(x03),
            self.deepsuper4(x04)
        ],
                              dim=1)
        return self.classifer(deepsuper)


if __name__ == "__main__":
    import torch
    SIZE1 = [[846, 255], 2, 15]
    SIZE2 = [[1128, 340], 2, 10]
    SIZE3 = [[1692, 510], 1, 15]
    data = torch.rand((1, 3, 846, 255))
    net = UnetPlus(8)
    rtn = net(data)
    print(rtn.shape)