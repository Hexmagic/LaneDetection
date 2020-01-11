from util.datagener import get_test_loader, get_train_loader, get_valid_loader
from collections import defaultdict
from tqdm import tqdm
import sys

batch = int(sys.argv[1])
print(f"batch {batch}")
a = get_train_loader(batch_size=batch)
b = get_valid_loader(batch_size=batch)
c = get_test_loader(batch_size=batch)


def analyze(gen):
	cnt = defaultdict(int)
	for batch in tqdm(gen):
		_, y = batch
		y = y[0]
		import pdb; pdb.set_trace()
		for i in range(8):
			ele = y[i]
			s = sum(ele)
			cnt[i]+=s
	print(cnt)


def main():
	print("valid")
	analyze(b)
	print("test")
	analyze(c)
	print('train ')
	analyze(a)
	
main()