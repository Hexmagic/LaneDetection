from util.datagener import get_test_loader, get_train_loader, get_valid_loader
from collections import Counter
from tqdm import tqdm
import sys

batch = sys.argv[1]
print(f"batch {batch}")
a = get_train_loader(batch_size=batch)
b = get_valid_loader(batch_size=batch)
c = get_test_loader(batch_size=batch)


def analyze(gen):
    cnt = None
    for batch in tqdm(gen):
        _, y = batch
        ct = Counter(y.flatten())
        if not cnt:
            cnt = ct
        else:
            cnt += ct
    print(cnt)


def main():
    print('train ')
    analyze(a)
    print("valid")
    analyze(b)
    print("test")
    analyze(c)
