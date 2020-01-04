import torch
from torch.autograd import Variable
from torch.nn import BCELoss
from tqdm import tqdm

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader

loss_fuc = BCELoss().cuda()
model = DeeplabV3Plus(n_class=8).cuda()
model.load_state_dict(torch.load('parameter.pkl'))
test_loader = get_test_loader()
loss_list = []
i = 0
for batch in tqdm(test_loader):
	x, y = batch
	i+=1
	if i>100:
		break
	xv, yv = Variable(x).cuda(), Variable(y).cuda()
	yout = model(xv)
	yout = torch.sigmoid(yout)
	loss = loss_fuc(yout, yv)
	loss_list.append(loss.item())

print(f'Valid Losss {sum(loss_list)/len(loss_list)}')
