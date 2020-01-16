import torch
from torch.nn import *
from model.deeplabv3_plus import SparableConv


class Projection(Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.prev = Sequential(SparableConv(1, 16))
        self.deconv = Sequential(
            SparableConv(16, 32),
            ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1))
        self.short = UpsamplingBilinear2d(scale_factor=2)
        self.suff = Sequential(SparableConv(48, 16), Conv2d(16, 3, 1))

    def forward(self, x):
        x = self.prev(x)
        short = self.short(x)
        x = self.deconv(x)
        x = torch.cat([x, short], dim=1)
        return self.suff(x)


if __name__ == '__main__':
    data = torch.rand((1, 1, 510, 1692))
    pro = Projection()
    rst = pro(data)
    print(rst.shape)