from torch.nn import *
import torch.nn.functional as F
import torch
from model.resnet import ResNet


class SparableConv(Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 relu=True,
                 stride=1,
                 padding=1,
                 dilation=1):
        super(SparableConv, self).__init__()
        layers = [
            Conv2d(in_channel,
                   in_channel,
                   kernel_size=3,
                   padding=padding,
                   groups=in_channel,
                   stride=stride,
                   dilation=dilation,
                   bias=False)
        ]
        layers.append(Conv2d(in_channel, out_channel, kernel_size=1))
        if relu:
            layers += [BatchNorm2d(out_channel), ReLU(True)]
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvBlock(Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.net = Sequential(
            BatchNorm2d(in_channel), ReLU(True),
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1))

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, channels, depth=4):
        super(Decoder, self).__init__()
        self.dconv1 = self.make_layer(channels[0])
        self.proj1 = ConvBlock(channels[0], channels[1])

        self.dconv2 = self.make_layer(channels[1])
        self.proj2 = ConvBlock(channels[1], channels[2])

        self.dconv3 = self.make_layer(channels[2])
        self.proj3 = ConvBlock(channels[2], channels[3])

        self.dconv4 = self.make_layer(channels[3]//2)
        self.proj4 = ConvBlock(channels[3],channels[4])

    def make_layer(self, in_channel):
        return Sequential(SparableConv(in_channel, in_channel // 2),
                          SparableConv(in_channel // 2, in_channel // 2))

    def forward(self, x, shorts):
        a, b, c, d = shorts
        x = self.proj1(x)
        up = F.interpolate(x,
                           d.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        cat = torch.cat([d, up], dim=1)
        x = self.dconv1(cat)

        x = self.proj2(x)
        up = F.interpolate(x,
                           c.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        cat = torch.cat([c, up], dim=1)
        x = self.dconv2(cat)

        x = self.proj3(x)
        up = F.interpolate(x,
                           b.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        cat = torch.cat([b, up], dim=1)
        x = self.dconv3(cat)

        x = self.proj4(x)
        up = F.interpolate(x,
                           a.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        cat = torch.cat([a, up], dim=1)
        x = self.dconv4(cat)
        return x


class Encoder(Module):
    def __init__(self, in_channel=3, depth=4):
        super(Encoder, self).__init__()
        self.layers = []
        self.downs = []
        for i in range(4):
            layer = self.make_layer(in_channel)
            setattr(self, f'econv{i}', layer)
            self.layers.append(layer)
            down_layer = MaxPool2d(kernel_size=3, padding=1, stride=2)
            setattr(self, f'epool{i}', down_layer)
            self.downs.append(down_layer)
            if in_channel == 3:
                in_channel = 64
            else:
                in_channel *= 2

    def make_layer(self, in_channel):
        out_channel = 64 if in_channel == 3 else in_channel * 2
        return Sequential(ConvBlock(in_channel, out_channel),
                          ConvBlock(out_channel, out_channel))

    def forward(self, x):
        rtn = []
        for layer, down in zip(self.layers, self.downs):
            x = layer(x)
            rtn.append(x)
            x = down(x)
        return rtn


class Unet(Module):
    def __init__(self, n_class=8):
        super(Unet, self).__init__()
        self.classifer = Sequential(BatchNorm2d(64), ReLU(True),
                                    Conv2d(64, n_class, 1))
        self.encoder = ResNet(depth=50,
                              base_width=4,
                              group=32,
                              n_class=n_class,
                              dilations=[1, 1, 2, 2])
        #self.mid = Sequential(ConvBlock(in_channel=2048, out_channel=1024))

        self.decoder = Decoder(channels=[2048,1024, 512, 256, 64])

    def forward(self, x):
        short, x = self.encoder(x)
        #x = self.mid(x)
        x = self.decoder(x, short)
        return self.classifer(x)


if __name__ == "__main__":
    data = torch.rand((1, 3, 245, 234))
    data = torch.rand((1, 3, 255, 243))
    net = Unet()
    from thop import profile
    rtn = profile(net, (data, ))
    print(rtn)
