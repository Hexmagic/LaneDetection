import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import setup_logger
from torch.autograd import Variable
from torch.nn import BCELoss, BCEWithLogitsLoss, MultiLabelMarginLoss, NLLLoss
from torch.optim import Adam, AdamW, RMSprop, ASGD
from tqdm import tqdm
from visdom import Visdom

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import LanDataSet, get_train_loader, get_valid_loader
from util.label_util import label_to_color_mask, mask_to_label

#torch.cuda.set_device(6)

if not os.path.exists("log"):
	os.mkdir("log")


class FocalLoss(nn.Module):
	r"""
		This criterion is a implemenation of Focal Loss, which is proposed in 
		Focal Loss for Dense Object Detection.
			Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
		The losses are averaged across observations for each minibatch.
		Args:
			alpha(1D Tensor, Variable) : the scalar factor for this criterion
			gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
								   putting more focus on hard, misclassiﬁed examples
			size_average(bool): By default, the losses are averaged over observations for each minibatch.
								However, if the field size_average is set to False, the losses are
								instead summed for each minibatch.
	"""
	def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
		super(FocalLoss, self).__init__()
		if alpha is None:
			self.alpha = Variable(torch.ones(class_num, 1))
		else:
			if isinstance(alpha, Variable):
				self.alpha = alpha
			else:
				self.alpha = Variable(alpha)
		self.gamma = gamma
		self.class_num = class_num
		self.size_average = size_average

	def forward(self, inputs, targets):
		N = inputs.size(0)
		C = inputs.size(1)
		P = F.softmax(inputs)

		class_mask = inputs.data.new(N, C).fill_(0)
		class_mask = Variable(class_mask)
		ids = targets.view(-1, 1)
		class_mask.scatter_(1, ids.data, 1.)
		#print(class_mask)

		if inputs.is_cuda and not self.alpha.is_cuda:
			self.alpha = self.alpha.cuda()
		alpha = self.alpha[ids.data.view(-1)]

		probs = (P * class_mask).sum(1).view(-1, 1)

		log_p = probs.log()
		#print('probs size= {}'.format(probs.size()))
		#print(probs)

		batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
		#print('-----bacth_loss------')
		#print(batch_loss)

		if self.size_average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()
		return loss


logger = setup_logger(name="",
					  logfile="log/loss.log",
					  level=logging.INFO,
					  maxBytes=1e6,
					  backupCount=3)


def encode(labels):
	label = labels[0]
	msk = label_to_color_mask(label)
	return msk


def train():
	vis = Visdom()
	model = DeeplabV3Plus(n_class=8).cuda()
	loss_func = BCEWithLogitsLoss().cuda()
	opt = AdamW(params=model.parameters())
	loader = get_train_loader()
	loss_list = []
	def adjust_learning_rate(optimizer):
		for param_group in optimizer.param_groups:
			param_group['lr'] = 0.001
	with open('loss.log', 'w') as f:
		for epoch in range(10):
			i = 0
			adjust_learning_rate(opt)
			for batch in tqdm(loader, desc=f"Epoch {epoch} process"):
				i += 1
				x, y = batch
				xv, yv = Variable(x).cuda(), Variable(y).cuda()
				yout = model(xv)
				#yout = torch.sigmoid(yhat) 如果用BCELoss需要取消注释
				opt.zero_grad()
				loss = loss_func(yout, yv)
				if i % 1000 == 0:
					torch.save(model.state_dict(), '\parameter.pkl')
				if i % 10 == 0:
					print(
						f"Epoch {epoch} batch {i} loss {np.mean(loss_list)}"
					)
					#continue
					yout = torch.sigmoid(yout)
					output_np = yout.cpu().detach().numpy().copy(
					)  # output_np.shape = (4, 2, 160, 160)
					output_np = np.argmax(output_np, axis=1)
					bag_msk_np = yv.cpu().detach().numpy().copy(
					)  # bag_msk_np.shape = (4, 2, 160, 160)
					msk = encode(output_np)
					mask = np.argmax(bag_msk_np, axis=1)
					label = encode(mask)

					msk = np.array([msk.transpose((2, 0, 1))])
					bag_msk_np = np.array([label.transpose((2, 0, 1))])
					vis.images(msk,
							   win='train_pred',
							   opts=dict(title='train prediction'))
					vis.images(bag_msk_np,
							   win='train_label',
							   opts=dict(title='train prediction'))
					vis.line(loss_list,
							 win='train_iter_loss',
							 opts=dict(title='train iter loss'))

				loss_list.append(loss.item())
				f.write(str(loss.item()) + '\n')
				#logger.info(f"Loss Value {loss.item()}")
				loss.backward()
				opt.step()


train()
