import torch
from util.datagener import get_train_loader,get_test_loader,get_valid_loader

mean = torch.zeros(3)
std = torch.zeros(3)
loader = get_train_loader(batch_size=1)
for ipt,t in loader:
	for i in range(3):
		mean[i]+=ipt[:,i,:,:].mean()
		std[i]+=ipt[:,i,:,:].std()

mean.div_(len(loader))
std.div_(len(loader))
print(f'mean {mean}')
print(f'std: {std}')