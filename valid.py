from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader
import torch
from torch.autograd import Variable
from torch.nn import BCELoss
loss_fuc = BCELoss()
net = DeeplabV3Plus()
net.load_state_dict(torch.load('parameter.pkl'))
test_loader = get_test_loader()
loss_list  =[]
for batch in  test_loader:
	x,y = batch
	xv,yv = Variable(x),Variable(y)
	yout = model(xv)
	yout = torch.sigmoid(yout)
	loss = loss_fuc(yout,yv)
	loss_list.append(loss.item())

print(f'Valid Losss {sum(loss_list)/len(loss_list)}')
