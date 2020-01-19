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
        self.layers = []
        self.projections = []
        for i in range(depth):
            layer = self.make_layer(in_channel)
            setattr(self, f'dconv{i}', layer)
            self.layers.append(layer)
            pro = ConvBlock(in_channel, in_channel // 2)
            setattr(self, 'dproj{i}', pro)
            self.projections.append(pro)
            in_channel //= 2

    def make_layer(self, in_channel):
        return Sequential(ConvBlock(in_channel, in_channel // 2),
                          ConvBlock(in_channel // 2, in_channel // 2))

    def forward(self, x, shorts):
        shorts.reverse()
        for layer, short, projection in zip(self.layers, shorts,
                                            self.projections):
            h, w = short.size()[2:]
            x = projection(x)
            up = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
            cat = torch.cat([short, up], dim=1)
            x = layer(cat)
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
