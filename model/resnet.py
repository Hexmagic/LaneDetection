import torch
from torch.nn import *


class ConvBlock(Module):
	def __init__(self,
				 in_channel,
				 out_channel,
				 kernel_size,
				 stride=1,
				 padding=1,
				 dilation=1,
				 norm=True):
		'''
		参数:
			in_channel ： 输入通道
			out_channel:  输出通道数
			kernel_size:  卷积核大小
			stride: 步长默认1

		'''
		super(ConvBlock, self).__init__()
		layer = [
			Conv2d(in_channel,
				   out_channel,
				   kernel_size,
				   padding=padding,
				   stride=stride,
				   dilation=dilation)
		]
		if norm:
			layer.append(BatchNorm2d(out_channel))
		self.net = Sequential(*layer)

	def forward(self, X):
		return self.net(X)


class BottleBlock(Module):
	expansion = 4  # 最终输出是输入的几倍

	def __init__(self, channels, norm=True, group=32, stride=1, dilation=1):
		super(BottleBlock, self).__init__()
		'''
		参数:
			channel ： 存储用到的输入和输出通道
			norm:  是否使用norm
		'''
		[first, second, third] = channels
		layers = [
			ConvBlock(first, second, 1, norm=norm, padding=0),
			LeakyReLU(),
			ConvBlock(second,
					  second,
					  3,
					  padding=dilation,
					  stride=stride,
					  norm=norm,
					  dilation=dilation),
			LeakyReLU(),
			ConvBlock(second, third, 1, norm=norm, padding=0),
		]
		self.net = Sequential(*layers)
		self.downsample = Sequential()
		if stride != 1 or first != third:
			self.downsample = ConvBlock(first,
										third,
										kernel_size=1,
										padding=0,
										stride=stride,
										norm=norm)

	def forward(self, X):
		I = X
		if self.downsample:
			I = self.downsample(X)
		X = LeakyReLU(True)(self.net(X) + I)
		return X


class BasicBlock(Module):
	expansion = 1  # 最终输出是输入的几倍

	def __init__(self, channels, norm=True, group=32, stride=1, dilation=1):
		'''
		'''
		super(BasicBlock, self).__init__()
		[first, second, third] = channels
		layers = [
			ConvBlock(first, second, 3, padding=1, norm=norm),
			LeakyReLU(),
			ConvBlock(second,
					  third,
					  3,
					  padding=1,
					  stride=stride,
					  norm=norm,
					  dilation=dilation),
		]

		self.net = Sequential(*layers)
		self.transform = Sequential()
		if first != third or stride != 1:
			# 兼容resnet34
			self.transform = Conv2d(first, third, kernel_size=1, stride=stride)

	def forward(self, X):
		I = X
		if self.transform:
			I = self.transform(X)
		return LeakyReLU(True)(I + self.net(X))


class ResNet(Module):
	layerMap = {
		34: [3, 4, 6, 3],
		50: [3, 4, 6, 3],
		101: [3, 4, 23, 3],
		152: [3, 8, 36, 3]
	}

	def __init__(self,
				 norm=True,
				 n_class=1000,
				 depth=None,
				 group=1,
				 base_width=64,
				 dilations=[1, 1, 1, 1]):
		'''
		param:
			norm: 是否使用batchnorm
			n_class:分类数目
			group: resnet默认为1，resnext需要自己自动自定义group
			base_width: bottlenet的宽度，resnext-4d的宽度为4
			depth: 网络的深度
			dilation: 是为了支持deeplab系列
		'''
		super(ResNet, self).__init__()
		self.depth = depth
		self.layers = self.layerMap[depth]
		self.norm = norm
		self.blk_class = BasicBlock if depth == 34 else BottleBlock  # 34层以上使用bottleblock
		self.width = base_width
		self.group = group
		# begin layers
		self.pre = Sequential(Conv2d(3, 64, 7, stride=2, padding=3), LeakyReLU(),
							  BatchNorm2d(64),
							  MaxPool2d(3, 2, 1))  # 输出大小56x56

		self.channel = 64

		self.conv_2 = self._maker_layers(in_channel=64,
										 layers=self.layers[0],
										 dilation=dilations[0])
		self.conv_3 = self._maker_layers(in_channel=128,
										 layers=self.layers[1],
										 stride=2,
										 dilation=dilations[1])
		self.conv_4 = self._maker_layers(in_channel=256,
										 layers=self.layers[2],
										 stride=2,
										 dilation=dilations[2])
		self.conv_5 = self._maker_layers(in_channel=512,
										 layers=self.layers[3],
										 stride=1,
										 dilation=dilations[3])

		# self.avgpool = AvgPool2d(7)
		# self.classifier = Linear(self.blk_class.expansion * 512, n_class)
		# self.layers = [
		#     self.pre, self.conv_2, self.conv_3, self.conv_4, self.conv_5,
		#     self.avgpool, self.classifier
		# ]

	def _maker_layers(self, in_channel, layers, stride=1, dilation=1):
		lst = []
		first = self.channel
		second = int(in_channel * (self.width / 64)) * self.group  # 每个阶段的输入
		third = in_channel * self.blk_class.expansion  # 输出
		channels = [first, second, third]

		# bottle net [64,64,256]

		# 2层以后的层需要downsample下采样，conv2的stride=1,所以不会进行下采样
		lst.append(
			self.blk_class(channels,
						   norm=self.norm,
						   group=self.group,
						   stride=stride,
						   dilation=dilation))

		channels[0] = in_channel * self.blk_class.expansion
		for _ in range(1, layers):
			lst.append(
				self.blk_class(channels,
							   norm=self.norm,
							   group=self.group,
							   dilation=dilation))

			# bottle net [256,64,256]

		self.channel = in_channel * self.blk_class.expansion

		# lst.append(MaxPool2d(2))
		return Sequential(*lst)

	def forward(self, X):
		l2 = X = self.pre(X)
		l4 = X = self.conv_2(X)
		l8 = X = self.conv_3(X)
		l16 = X = self.conv_4(X)
		X = self.conv_5(X)

		return [l2,l4,l8,l16],X


def makeRest(depth, n_class=1000):
	return ResNet(depth=depth, n_class=n_class)


def makeResNext_4d(depth, n_class=1000):
	return ResNet(depth=depth, base_width=4, group=32, n_class=n_class)


if __name__ == "__main__":
	data = torch.rand((1, 3, 224, 224))
	resnet = ResNet(depth=50, dilations=[1, 1, 2, 4])
	rst = resnet(data)
	print(rst.shape)
