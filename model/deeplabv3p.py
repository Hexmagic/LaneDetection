#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from torch import nn
from torch.nn import functional as F
from model.backbone_xception import Xception
# from backbone_mobilenetv2 import MobileNetv2
# from torchsummary import summary

# In[10]:

# class DeepLabHead(nn.Sequential):
#     def __init__(slef, in_channels, num_classes):
#         super(DeepLabHead, self).__init__(
#             ASPP(in_channels, [12, 24, 36]),
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, num_classes, 1)
#         )

# In[11]:


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# In[12]:


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x,
                             size=size,
                             mode='bilinear',
                             align_corners=False)


# In[13]:


class Encoder(nn.Module):
    def __init__(self, in_channels, output_stride=16):
        super(Encoder, self).__init__()
        out_channels = 256
        modules = []

        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise Exception("deeplab only support stride 8 or 16")

        # Image Pooling, average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        # 1x1 Conv
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU()))

        # ASPP
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))

        # ModelList没有实现forward方法，需要在forward中实现；
        self.convs = nn.ModuleList(modules)

        # 连接之后的1x1 Conv
        # Sequential实现了forward方法，可以直接调用
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# In[14]:


class Decoder(nn.Module):
    def __init__(self, shortcut_in_channels, separable=True):
        super(Decoder, self).__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(shortcut_in_channels, 48, 1),
                                      nn.BatchNorm2d(48), nn.ReLU())
        self.up = nn.Upsample(mode='bilinear', scale_factor=4)

        if separable:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(304, 304, 3, padding=1, groups=304), nn.Dropout(0.3),
                nn.Conv2d(304, 256, 1), nn.BatchNorm2d(256))
        else:
            self.conv3x3 = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1),
                                         nn.BatchNorm2d(out_channels))

    def forward(self, x, y):
        x = self.shortcut(x)
        # up = self.up(y)
        up = F.interpolate(y,
                           size=x.shape[2:],
                           mode="bilinear",
                           align_corners=False)
        out = torch.cat((up, x), 1)
        out = self.conv3x3(out)

        return out


# In[15]:


class DeepLabV3P(nn.Module):
    def __init__(self, n_classes=10, backbone='Xception'):
        super(DeepLabV3P, self).__init__()
        if backbone == 'Xception':
            data_channels = 728
            shortcut_channels = 128
            self.backbone = Xception()
        elif backbone == 'MobileNet':
            data_channels = 64
            shortcut_channels = 24
            self.backbone = MobileNetv2()
        else:
            raise Exception(
                "deeplab only support backbone Xception or MobileNet")

        self.encoder = Encoder(data_channels)
        self.decoder = Decoder(shortcut_channels)

        self.last = Sequential(nn.Dropout(0.3), nn.Conv2d(256, n_classes, 1))
        self.up = nn.Upsample(mode='bilinear', scale_factor=4)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            # else isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        data, shortcut = self.backbone(x)
        data = self.encoder(data)
        out = self.decoder(shortcut, data)
        #out = self.last(out)
        # # out = self.up(out)
        # out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = F.interpolate(out,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=False)
        out = self.last(out)

        return out


# In[16]:

# x = torch.randn((1, 3, 299, 299))
# x.size()
# model = DeepLabV3P()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# summary(model, (3,299,299))
# model.eval()
# y = model(x)
# y.size()

# In[17]:

# x2 = torch.randn((1, 3, 224, 224))
# x2.size()
# model2 = DeepLabV3P(10, 'MobileNet')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# summary(model2, (3,224,224))
# model2.eval()
# y2 = model2(x2)
# y2.size()

# In[ ]:
