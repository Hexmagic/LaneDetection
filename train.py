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

torch.cuda.set_device(7)

if not os.path.exists("log"):
    os.mkdir("log")


def encode(labels):
    label = labels[0]
    msk = label_to_color_mask(label)
    return msk


def train():
    #vis = Visdom()
    model = DeeplabV3Plus(n_class=8).cuda()
    loss_func = BCEWithLogitsLoss().cuda()
    opt = AdamW(params=model.parameters())
    loader = get_train_loader()
    loss_list = []

    def adjust_learning_rate(optimizer, epoch):
        if epoch<3:
            lr = 0.003/epoch
        else:
            lr = 0.0005/(epoch//2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    with open('loss.log', 'w') as f:
        for epoch in range(10):
            i = 0
            #adjust_learning_rate(opt, epoch + 1)
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
                    print(f"Epoch {epoch} batch {i} loss {np.mean(loss_list)}")
                    #continue
                    # yout = torch.sigmoid(yout)
                    # output_np = yout.cpu().detach().numpy().copy(
                    # )  # output_np.shape = (4, 2, 160, 160)
                    # output_np = np.argmax(output_np, axis=1)
                    # bag_msk_np = yv.cpu().detach().numpy().copy(
                    # )  # bag_msk_np.shape = (4, 2, 160, 160)
                    # msk = encode(output_np)
                    # mask = np.argmax(bag_msk_np, axis=1)
                    # label = encode(mask)

                    # msk = np.array([msk.transpose((2, 0, 1))])
                    # bag_msk_np = np.array([label.transpose((2, 0, 1))])
                    # vis.images(msk,
                    # 		   win='train_pred',
                    # 		   opts=dict(title='train prediction'))
                    # vis.images(bag_msk_np,
                    # 		   win='train_label',
                    # 		   opts=dict(title='train prediction'))
                    # vis.line(loss_list,
                    # 		 win='train_iter_loss',
                    # 		 opts=dict(title='train iter loss'))

                loss_list.append(loss.item())
                f.write(str(loss.item()) + '\n')
                #logger.info(f"Loss Value {loss.item()}")
                loss.backward()
                opt.step()


train()
