import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from logzero import setup_logger
from torch.autograd import Variable
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import LanDataSet, get_train_loader
from util.label_util import label_to_color_mask, mask_to_label

torch.cuda.set_device(6)

if not os.path.exists("log"):
    os.mkdir("log")

logger = setup_logger(
    name="", logfile="log/loss.log", level=logging.ERROR, maxBytes=1e6, backupCount=3
)


def train():
    model = DeeplabV3Plus(n_class=8).cuda()
    loss_func = BCEWithLogitsLoss().cuda()
    opt = Adam(params=model.parameters())
    loader = get_train_loader()
    for epoch in range(5):
        i = 0
        loss_list = []
        for batch in tqdm(loader, desc=f"Epoch {epoch} process"):
            i += 1
            x, y = batch
            xv, yv = Variable(x).cuda(), Variable(y).cuda()
            yhat = model(xv)
            yhat = torch.sigmoid(yhat)
            yv = torch.sigmoid(yv)
            opt.zero_grad()
            loss = loss_func(yhat, yv)
            if i % 20 == 0:
                print(f"Epoch {epoch} batch {i} loss {sum(loss_list)/len(loss_list)}")
            loss_list.append(loss.item())
            logger.info(f"Loss Value {loss.item()}")
            loss.backward()
            opt.step()


train()
