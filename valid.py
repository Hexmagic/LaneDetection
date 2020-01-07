import torch
from torch.autograd import Variable
from torch.nn import BCELoss
from tqdm import tqdm

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader
from model.unet import Unet
from collections import defaultdict

def compute_iou(pred, gt, result):
	"""
	pred : [N, H, W]
	gt: [N, H, W]
	"""
	pred = pred.detach().cpu().numpy()
	gt = gt.detach().cpu().numpy()
	for i in range(8):
		single_gt = gt == i
		single_pred = pred == i
		temp_tp = np.sum(single_gt * single_pred)
		temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
		result["TP"][i] += temp_tp
		result["TA"][i] += temp_ta
	return result


def validLoss():
	loss_fuc = BCELoss().cuda()
	model = Unet(n_class=8).cuda()
	model.load_state_dict(torch.load('parameter.pkl'))
	test_loader = get_test_loader()
	loss_list = []
	i = 0
	result = {"TP": defaultdict(int), "TA": defaultdict(int)}
	for batch in tqdm(test_loader):
		x, y = batch
		i +=1
		if i>100:
			break
		xv, yv = Variable(x).cuda(), Variable(y).cuda()
		yout = model(xv)
		yout = torch.sigmoid(yout)
		loss = loss_fuc(yout, yv)
		loss_list.append(loss.item())
		result = compute_iou(pred, gt, result)
	print(f'Valid Losss {sum(loss_list)/len(loss_list)}')
	for i in range(8):
		print(f"Class {i} IOU {result['TP'][i]/result['TA'][i]}")


validLoss()