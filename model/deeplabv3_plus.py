import torch
from torch.nn import *
import torch.nn.functional as F


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
            layers += [BatchNorm2d(out_channel), LeakyReLU()]
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AligenResidualBlock(Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=2,
                 relu_first=True,
                 dilation=1):
        super(AligenResidualBlock, self).__init__()
        net = [
            SparableConv(in_channel,
                         out_channel,
                         padding=dilation,
                         dilation=dilation),
            SparableConv(out_channel,
                         out_channel,
                         padding=dilation,
                         dilation=dilation),
            SparableConv(out_channel,
                         out_channel,
                         stride=stride,                         
                         dilation=dilation,
                         padding=dilation)
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
            Conv2d(3, 32, 3, stride=2, padding=1),
            BatchNorm2d(32),
            LeakyReLU(),
            Conv2d(32, 64, 3, padding=1),
            BatchNorm2d(64),
            self.residual_cls(64, 128, stride=2, relu_first=False),
        ])

        # 分开方便deeplabv3+ 取出low feature
        self.entry_1 = self.residual_cls(128, 256, stride=2)
        self.entry_2 = self.residual_cls(256, 728, stride=2)

        self.middle = Sequential(*[
            self.residual_cls(728, 728, stride=1, dilation=2)
            for _ in range(16)
        ])

        dilation = 4
        self.exit = Sequential(
            self.residual_cls(728, 1024, stride=1 if aligen else 2),
            SparableConv(1024, 1536, padding=dilation, dilation=dilation),
            SparableConv(1536, 1536, padding=dilation, dilation=dilation),
            SparableConv(1536, 2048, padding=dilation, dilation=dilation),
        )

    def forward(self, x):
        entry_0_out = self.entry_0(x)
        entry_1_out = self.entry_1(entry_0_out)
        entry_2_out = self.entry_2(entry_1_out)
        middle_out = self.middle(entry_2_out)
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
        self.layers_4 = SparableConv(in_channel,
                                     out_channel,
                                     dilation=4,
                                     padding=4)
        self.layers_8 = SparableConv(in_channel,
                                      out_channel,
                                      dilation=8,
                                      padding=8)
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

        self.projection = Sequential(SparableConv(256 * 6, out_channel, 1))

    def forward(self, x):
        conv_rsts = [
            self.layers_1(x),
            self.layers_4(x),
            self.layers_8(x),
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
        self.low_projection = Sequential(
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2d(128, 48, kernel_size=1),
        )

        self.projection = Sequential(SparableConv(256 + 48, 256), Dropout(0.2),
                                     SparableConv(256, 256))

        self.classifer = Sequential(BatchNorm2d(256), LeakyReLU(),
                                    Conv2d(256, n_class, 1, bias=True))

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight,
                                              mode='fan_out')
        #         if layer.bias is not None:
        #             torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x):

        low_feature, feature_map = self.backbone(x)
        feature_map = self.aspp(feature_map)
        low_feature = self.low_projection(low_feature)
        #low_feature = self.d1(low_feature)
        #middle_feature = self.mid_projection(middle_feature)
        #middle_feature = self.d2(middle_feature)

        h, w = low_feature.size()[2:]

        feature_map = F.interpolate(feature_map, (h, w),
                                    mode='bilinear',
                                    align_corners=True)
        feature_map = torch.cat([low_feature, feature_map], dim=1)

        feature_map = self.projection(feature_map)

        h, w = x.size()[2:]
        feature_map = F.interpolate(feature_map, [h, w],
                                    mode='bilinear',
                                    align_corners=True)
        return self.classifer(feature_map)


if __name__ == "__main__":
    net = DeeplabV3Plus(8)
    import thop
    data = torch.rand((1, 3, 288, 288))
    rtn = thop.profile(net, inputs=(data, ))
    print(rtn)