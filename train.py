import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import BCELoss, CrossEntropyLoss, NLLLoss
from torch.optim import Adam, RMSprop

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import LanDataSet, get_train_loader
from util.label_util import label_to_color_mask, mask_to_label




def train():
    model = DeeplabV3Plus(n_class=8).cuda()
    loss_func =BCELoss().cuda()
    opt = Adam(params=model.parameters())
    loader = get_train_loader()
    loss_list = []
    for epoch in range(5):
        for i, batch in enumerate(loader):
            x, y = batch
            xv, yv = Variable(x).cuda(), Variable(y).cuda()
            yhat = model(xv)
            opt.zero_grad()
            yhat = torch.sigmoid(yhat)
            yv = torch.sigmoid(yv)
            loss = loss_func(yhat, yv)
            loss_list.append(loss.item())
            if i % 10 == 0:
                print(f'Epoch {epoch} loss {sum(loss_list)/len(loss_list)}')
            loss.backward()
            opt.step()


testModel()
