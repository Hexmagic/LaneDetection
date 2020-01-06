from torch.nn import *
import torch


class Unet(Module):
	def __init__(self, n_class=2):
		super(Unet, self).__init__()
		self.base = 64
		self.c1 = self.contract(3, 64)
		self.c2 = self.contract(64, 128,stride=2)
		self.c3 = self.contract(128, 256,stride=2)
		self.c4 = self.contract(256, 512,stride=2)
		self.c5 = self.contract(512, 1024,stride=2)
		self.exp1 = self.expansive(1024, 512)
		self.exp_contract1 = self.contract(1024, 512)
		self.exp2 = self.expansive(512, 256)
		self.exp_contract2 = self.contract(512, 256)
		self.exp3 = self.expansive(256, 128)
		self.exp_contract3 = self.contract(256, 128)
		self.exp4 = self.expansive(128, 64)
		self.exp_contract4 = self.contract(128, 64)
		self.classifier = Conv2d(64, n_class, kernel_size=1)

	def contract(self, in_channel, out_channel, stride=1):
		return Sequential(
			MaxPool2d(2, ceil_mode=True) if stride == 2 else Sequential(),
			Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
			BatchNorm2d(out_channel),
			ReLU(True),
			Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
			BatchNorm2d(out_channel),
			ReLU(True),
		)

	def expansive(self, in_channel, out_channel):
		return ConvTranspose2d(in_channel,
							   out_channel,
							   kernel_size=4,
							   stride=2,
							   padding=2)

	def crop(self, c, e):
		w, h = e.size()[2:]
		dw, dh = c.size()[2:]
		offx = (dw - w) // 2
		offy = (dh - h) // 2
		return c[:, :, offx:offx + w, offy:offy + h]

	def forward(self, x):
		w,h = x.size()[2:]
		c1 = self.c1(x)
		c2 = self.c2(c1)
		c3 = self.c3(c2)
		c4 = self.c4(c3)
		c5 = self.c5(c4)
		e1 = self.exp1(c5)

		c4 = self.crop(c4,e1)
		e1 = torch.cat([e1, c4], dim=1)
		e1 = self.exp_contract1(e1)
		e2 = self.exp2(e1)

		c3 = self.crop(c3,e2)
		e2 = torch.cat([e2, c3], dim=1)
		e2 = self.exp_contract2(e2)
		e3 = self.exp3(e2)

		c2 = self.crop(c2,e3)
		e3 = torch.cat([e3, c2], dim=1)
		e3 = self.exp_contract3(e3)
		e4 = self.exp4(e3)

		c1 = self.crop(c1,e4)
		e4 = torch.cat([e4, c1], dim=1)
		e4 = Upsample((w,h),mode='bilinear',align_corners=True)(e4)
		e4 = self.exp_contract4(e4)
		return self.classifier(e4)


if __name__ == "__main__":
	import torch
	model = Unet(n_class=2)
	data = torch.rand((1, 3, 572, 572))
	rst = model(data)
	print(rst.shape)
