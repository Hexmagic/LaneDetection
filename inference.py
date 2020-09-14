from argparse import ArgumentParser
import pdb

import torch
import cv2
from torchvision.transforms import ToTensor
import os
from torch.autograd import Variable
import time
from util.label_util import label_to_color_mask
from tqdm import tqdm

def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[690:, :, :]
    img = cv2.resize(img, (846, 255), interpolation=cv2.INTER_LINEAR)
    return ToTensor()(img)


def main():
    parser = ArgumentParser('Inference')
    parser.add_argument('--model', type=str, default='weights/best.pt')
    parser.add_argument('--img', type=str, default='sample')
    parser.add_argument('--video', type=str)
    parser.add_argument('--output', type=str, default='output')
    arg = parser.parse_args()
    if not os.path.exists(arg.output):
        os.mkdir(arg.output)
    model = torch.load(arg.model).cuda()
    s = time.time()

    with torch.no_grad():
        model.eval()
        cnt = 0
        for filename in tqdm(os.listdir(arg.img)):
            cnt += 1
            path = os.path.join(arg.img, filename)
            tensor = Variable(read_img(path)).cuda()
            tensor = tensor.unsqueeze(0)
            label = model(tensor)
            label = torch.argmax(label, dim=1)
            mask = label_to_color_mask(label.cpu())
            cv2.imwrite(f'{arg.output}/{filename}', mask[0])
        e = time.time()
        print(f"{1/((e-s)/cnt)} FPS")


if __name__ == '__main__':
    main()