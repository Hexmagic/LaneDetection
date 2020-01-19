from torch.nn import *
import torch.nn.functional as F
import torch


class ConvBlock(Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.net = Sequential(
            BatchNorm2d(in_channel), ReLU(True),
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1))

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, in_channel=1024, depth=4):
        super(Decoder, self).__init__()
        self.dconv1 = self.make_layer(in_channel)
        self.proj1 = ConvBlock(in_channel, in_channel // 2)
        in_channel //= 2
        self.dconv2 = self.make_layer(in_channel)
        self.proj2 = ConvBlock(in_channel, in_channel // 2)
        in_channel //= 2
        self.dconv3 = self.make_layer(in_channel)
        self.proj3 = ConvBlock(in_channel, in_channel // 2)
        in_channel //= 2
        self.dconv4 = self.make_layer(in_channel)
        self.proj4 = ConvBlock(in_channel, in_channel // 2)

    def make_layer(self, in_channel):
        return Sequential(ConvBlock(in_channel, in_channel // 2),
                          ConvBlock(in_channel // 2, in_channel // 2))

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
    def __init__(self, n_class=20):
        super(Unet, self).__init__()
        self.classifer = Sequential(BatchNorm2d(64), ReLU(True),
                                    Conv2d(64, n_class, 1))
        self.encoder = Encoder()
        self.mid = Sequential(ConvBlock(in_channel=512, out_channel=1024),
                              ConvBlock(in_channel=1024, out_channel=1024))

        self.decoder = Decoder()

    def forward(self, x):
        short = self.encoder(x)
        x = self.mid(short[-1])
        x = self.decoder(x, short)
        return self.classifer(x)


if __name__ == "__main__":
    data = torch.rand((1, 3, 245, 234))
    net = Unet()
    print(net(data).shape)