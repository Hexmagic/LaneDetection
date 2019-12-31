from util.label_util import label_to_color_mask, mask_to_label
import cv2
import matplotlib.pyplot as plt


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
from torch.nn import BCELoss,NLLLoss
from torch.optim import Adam


def testModel():
    model = DeeplabV3Plus(n_class=8).cuda()
    loss_func = NLLLoss()
    opt = Adam(params=model.parameters())
    loader = get_train_loader()
    loss_list =[]
    for epoch in range(5):
        for i, batch in enumerate(loader):
            x, y = batch
            xv, yv = Variable(x).cuda(), Variable(y).cuda()
            yhat = model(xv)
            opt.zero_grad()
            loss = loss_func(yhat, yv.long())
            loss_list.append(loss.item(0))
            if i%15==0:
                print(f'Epoch {epoch} loss {sum(loss_list)/len(loss_list)}')
            loss.backward()
            opt.step()

testModel()