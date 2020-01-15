import torch
from torch.nn import *


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


class AligenResidualBlock(Module):
    def __init__(self, in_channel, out_channel, stride=2, relu_first=True):
        super(AligenResidualBlock, self).__init__()
        net = [
            SparableConv(in_channel, out_channel),
            SparableConv(out_channel, out_channel),
            SparableConv(out_channel, out_channel, stride=stride, padding=1)
        ]
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

    def forward(self, x):
        entry_0_out = self.entry_0(x)
        entry_1_out = self.entry_1(entry_0_out)
        middle_out = self.middle(entry_1_out)
        exit_out = self.exit(middle_out)
        #avg_out = self.avgpool(self.exit_out)
        #flatten = torch.flatten(avg_out)
        return entry_0_out, exit_out


class SepAspPooling(Module):
    def __init__(self, in_channel, out_channel):
        super(SepAspPooling, self).__init__()
        self.layers_1 = SparableConv(in_channel,
                                     out_channel,
                                     dilation=1,
                                     padding=1)
        self.layers_6 = SparableConv(in_channel,
                                     out_channel,
                                     dilation=6,
                                     padding=6)
        self.layers_12 = SparableConv(in_channel,
                                      out_channel,
                                      dilation=12,
                                      padding=12)
        self.layers_18 = SparableConv(in_channel,
                                      out_channel,
                                      dilation=18,
                                      padding=18)

        self.pooling_layers = Sequential(AdaptiveAvgPool2d((1, 1)),
                                         Conv2d(in_channel, out_channel, 1))

        self.projection = Sequential(SparableConv(256 * 5, out_channel, 1))

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
        self.d1 = Dropout(0.4)
        self.d2 = Dropout(0.4)
        self.low_projection = Sequential(Conv2d(128, 48, kernel_size=1),
                                         BatchNorm2d(48), ReLU())
        self.projection = Sequential(SparableConv(256 + 48, 256), Dropout(0.4),
                                     SparableConv(256, 256))
        self.classifer = Sequential(SparableConv(256, 256), Dropout(0.4),
                                    BatchNorm2d(256), ReLU(True),
                                    Conv2d(256, n_class, 1, bias=True))

        for n in self.modules():
            if isinstance(n, Conv2d):
                init.kaiming_normal_(n.weight.data, mode='fan_out')

    def forward(self, x):

        low_feature, feature_map = self.backbone(x)
        feature_map = self.aspp(feature_map)
        low_feature = self.low_projection(low_feature)

        feature_map = self.d1(feature_map)
        feature_map = self.d1(feature_map)
        h, w = low_feature.size()[2:]

        feature_map = UpsamplingBilinear2d((h, w))(feature_map)
        feature_map = torch.cat([low_feature, feature_map], dim=1)

        feature_map = self.d2(feature_map)
        feature_map = self.projection(feature_map)
        h, w = x.size()[2:]
        feature_map = UpsamplingBilinear2d((h, w))(feature_map)

        return self.classifer(feature_map)
