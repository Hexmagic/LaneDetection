from util.label_util import label_to_color_mask, mask_to_label
import cv2
import matplotlib.pyplot as plt
import visdom
import torch

def testMask():
    img = cv2.imread('test2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = mask_to_label(img)
    mask = label_to_color_mask(label)
    plt.imshow(mask)
    plt.show()


from util.datagener import LanDataSet, get_train_loader


def testLoader():
    d = LanDataSet(root='data_list/train.csv')


from model.deeplabv3_plus import DeeplabV3Plus
from torch.autograd import Variable
from torch.nn import BCELoss, NLLLoss,CrossEntropyLoss
from torch.optim import Adam,RMSprop
import numpy as np


def encode(label):
    h,w = label.shape[1:]
    img = np.zeros((h,w))
    for i in range(7,0,-1):
        img[label[i]>= i] = i
    return img


def testModel():
    vis = visdom.Visdom()
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
                img =yhat.cpu().data.numpy()[0]
                img = encode(img)
                img = label_to_color_mask(img)

            loss.backward()
            opt.step()


testModel()