import torch
from torch.nn import *


class SparableConv(Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 relu=False,
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
                   dilation=dilation)
        ]
        layers.append(Conv2d(in_channel, out_channel, kernel_size=1))
        if relu:
            layers += [BatchNorm2d(out_channel), ReLU(True)]
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AligenResidualBlock(Module):
    def __init__(self, in_channel, out_channel, stride=2, relu_first=True):
        super(AligenResidualBlock, self).__init__()
        net = [
            SparableConv(in_channel, out_channel),
            BatchNorm2d(out_channel),
            ReLU(True),
            SparableConv(out_channel, out_channel),
            BatchNorm2d(out_channel),
            ReLU(True),
            SparableConv(out_channel, out_channel, stride=2, padding=1)
        ]
        if stride == 1:
            # middle flow 需要三个可分离卷积
            net = [SparableConv(in_channel, out_channel), ReLU(True)]
        if relu_first:
            # 第一个residual block开始不需要relu
            net = [ReLU(True)] + net

        if stride == 1:
            # middle flow 不需要maxpooling
            net[-1] = SparableConv(out_channel, out_channel)
            self.net = Sequential(*net)
        else:
            self.net = Sequential(*net)
        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            self.downsample = Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        short = x
        x = self.net(x)
        if self.downsample:
            short = self.downsample(short)
        return x + short


class Xception(Module):
    def __init__(self, n_class=20, aligen=False):
        super(Xception, self).__init__()
        # input 299,299,3
        self.residual_cls = AligenResidualBlock
        self.entry_0 = Sequential(*[
            Sequential(Conv2d(3, 32, 3, stride=2, padding=1), ReLU(True),
                       Conv2d(32, 64, 3, padding=1)),
            self.residual_cls(64, 128, stride=2, relu_first=False),
        ])

        # 分开方便deeplabv3+ 取出low feature
        self.entry_1 = Sequential(*[
            self.residual_cls(128, 256, stride=2),
            self.residual_cls(256, 728, stride=2),
        ])

        self.middle = Sequential(*[
            self.residual_cls(728, 728, stride=1)
            for _ in range(16 if aligen else 8)
        ])

        dilation = 2 if aligen else 1  # Deeplab 需要用到三层的conv和atros conv
        self.exit = Sequential(
            self.residual_cls(728, 1024, stride=1 if aligen else 2),
            SparableConv(1024, 1536, padding=dilation, dilation=dilation),
            SparableConv(1536, 1536, padding=dilation, dilation=dilation)
            if aligen else Sequential(),
            SparableConv(1536, 2048, padding=dilation, dilation=dilation),
        )
        self.avgpool = AdaptiveAvgPool2d(1)
        self.classifer = Linear(2048, n_class)
        self.layers = [
            self.entry_0, self.entry_1, self.middle, self.exit, self.avgpool
        ]

    def forward(self, x):
        self.entry_0_out = self.entry_0(x)
        entry_1_out = self.entry_1(self.entry_0_out)
        middle_out = self.middle(entry_1_out)
        self.exit_out = self.exit(middle_out)
        #avg_out = self.avgpool(self.exit_out)
        #flatten = torch.flatten(avg_out)
        return self.exit_out


class SepDilationConv(Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(SepDilationConv, self).__init__()
        self.net = Sequential(
            SparableConv(in_channel,
                         out_channel,
                         3,
                         padding=dilation,
                         dilation=dilation), BatchNorm2d(out_channel),
            ReLU(True))

    def forward(self, x):
        return self.net(x)


class SepAspPooling(Module):
    def __init__(self, in_channel, out_channel):
        super(SepAspPooling, self).__init__()
        self.layers_1 = SepDilationConv(in_channel, out_channel, 1)
        self.layers_6 = SepDilationConv(in_channel, out_channel, 6)
        self.layers_12 = SepDilationConv(in_channel, out_channel, 12)
        self.layers_18 = SepDilationConv(in_channel, out_channel, 18)

        self.pooling_layers = Sequential(
            AdaptiveAvgPool2d((1, 1)),
            SparableConv(in_channel, out_channel),
            # BatchNorm2d(out_channel),
            ReLU(True))

        self.projection = Sequential(Conv2d(256 * 5, out_channel, 1),
                                     BatchNorm2d(out_channel),
                                     ReLU(True))  # 映射回原来的深度

    def forward(self, x):
        conv_rsts = [
            self.layers_1(x),
            self.layers_6(x),
            self.layers_12(x),
            self.layers_18(x),
        ]
        mean = self.pooling_layers(x)
        pooling = Upsample(x.size()[2:], mode='bilinear',
                           align_corners=True)(mean)
        conv_rsts.append(pooling)
        return self.projection(torch.cat(conv_rsts, dim=1))


class DeeplabV3Plus(Module):
    def __init__(self, n_class=20):
        super(DeeplabV3Plus, self).__init__()
        self.n_class = n_class
        self.backbone = Xception(aligen=True)
        self.aspp = SepAspPooling(512 * 4, 256)
        self.low_projection = Sequential(Conv2d(128, 48, kernel_size=1))
        self.projection = Sequential(BatchNorm2d(256 + 48), ReLU(True),
                                     Conv2d(256 + 48, 256, 1),
                                     BatchNorm2d(256), ReLU(True),
                                     Conv2d(256, n_class, 1))

    def forward(self, x):
        self.backbone(x)
        feature_map = self.backbone.exit_out
        low_feature = self.backbone.entry_0_out
        feature_map = self.aspp(feature_map)
        low_feature = self.low_projection(low_feature)

        h, w = low_feature.size()[2:]
        feature_map = Upsample((h, w), mode='bilinear',
                               align_corners=True)(feature_map)
        feature_map = torch.cat([low_feature, feature_map], dim=1)

        h, w = x.size()[2:]
        feature_map = Upsample((h, w), mode='bilinear',
                               align_corners=True)(feature_map)
        return self.projection(feature_map)


if __name__ == "__main__":
    data = torch.rand((1, 3, 399, 399))
    model = DeeplabV3Plus(n_class=20)
    rst = model(data)
    print(rst.shape)
