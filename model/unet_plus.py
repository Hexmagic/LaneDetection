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
    def __init__(self, stride=16):
        super(Encode, self).__init__()
        self.down1 = DownBlock(3, stride)
        self.down2 = DownBlock(stride, stride * 2)
        self.down3 = DownBlock(stride * 2, stride * 4)
        self.down4 = DownBlock(stride * 4, stride * 8)
        self.down5 = DownBlock(stride * 8, stride * 16, stride=1)

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
    def __init__(self, n_class, stride=16):
        super(UnetPlus, self).__init__()
        self.encode = Encode(stride=stride)
        self.u31 = UpBlock(stride * 16, stride * 8)

        self.u21 = UpBlock(stride * 8, stride * 4)
        self.u22 = UpBlock(stride * 8, stride * 4)

        self.u11 = UpBlock(stride * 4, stride * 2)
        self.u12 = UpBlock(stride * 4, stride * 2)
        self.u13 = UpBlock(stride * 4, stride * 2)

        self.u01 = UpBlock(stride * 2, stride)
        self.u02 = UpBlock(stride * 2, stride)
        self.u03 = UpBlock(stride * 2, stride)
        self.u04 = UpBlock(stride * 2, stride, last=True)

        self.deepsuper1 = ConvBlock(stride, n_class, k=1, p=0)
        self.deepsuper2 = ConvBlock(stride, n_class, k=1, p=0)
        self.deepsuper3 = ConvBlock(stride, n_class, k=1, p=0)
        self.deepsuper4 = ConvBlock(stride, n_class, k=1, p=0)
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
        x21 = self.u21(x30, x20)
        x31 = F.interpolate(x31,
                            x30.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x22 = self.u22(x31, x21 + x20)

        x20 = F.interpolate(x20,
                            x10.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x11 = self.u11(x20, x10)
        x21 = F.interpolate(x21,
                            x10.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x12 = self.u12(x21, x11 + x10)
        x22 = F.interpolate(x22,
                            x20.size()[2:],
                            mode='bilinear',
                            align_corners=True)
        x13 = self.u13(x22, x12 + x11 + x10)
        x10 = F.interpolate(x10, (h, w), mode='bilinear', align_corners=True)
        x01 = self.u01(x10, x00)
        x11 = F.interpolate(x11, (h, w), mode='bilinear', align_corners=True)
        x02 = self.u02(x11, x01 + x00)
        x12 = F.interpolate(x12, (h, w), mode='bilinear', align_corners=True)
        x03 = self.u03(x12, x02 + x01 + x00)
        x13 = F.interpolate(x13, (h, w), align_corners=True, mode='bilinear')
        x04 = self.u04(x13, x03 + x02 + x01 + x00)
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
    from thop import profile
    net = UnetPlus(8,stride=32)
    data = torch.rand((1, 3, 255, 288))
    rtn = profile(net, inputs=(data, ))
    print(rtn)
    rtn = net(data)
    print(rtn.shape)