import torch
from torch.nn import *
from model.deeplabv3_plus import SparableConv


class Projection(Module):
    def __init__(self):
        super(Projection, self).__init__()

        self.short = UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):

        x = self.short(x)
        return x


if __name__ == '__main__':
    data = torch.rand((1, 1, 510, 1692))
    pro = Projection()
    rst = pro(data)
    print(rst.shape)
