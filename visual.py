import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class Visualer(object):
    def __init__(self):
        self.train_log = 'log/train.txt'
        self.test_log = 'log/test.txt'

    def parse_loss(self, fileobj):
        losses = []
        for line in fileobj:
            line_list = line.split(' ')
            if len(line_list) == 4:
                loss = float(line_list[-1])
                losses.append(loss)
        return losses

    def parse_iou(self, fileobj):
        ious = defaultdict(list)
        for line in fileobj:
            if ':' in line:
                line_list = line.split(' ')
                clas = line_list[3].strip(':')
                if clas == '0':
                    continue
                val = float(line_list[4])
                ious[clas].append(val)
        mious = []
        for ele in zip(*ious.values()):
            miou = sum(ele)
            mious.append(miou)
        return mious

    def run(self):
        train_loss = self.parse_loss(open(self.train_log, 'r'))
        test_loss = self.parse_loss(open(self.train_log, 'r'))
        miou = self.parse_iou(open(self.test_log, 'r'))
    


if __name__ == "__main__":
    v = Visualer()
    v.run()