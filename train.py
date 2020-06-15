import os
import shutil
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from math import sqrt
from model.unet_plus import Unet
from model.deeplabv3_plus import DeeplabV3Plus
from util.dataset import LaneDataSet
from util.label_util import label_to_color_mask
from util.loss import DiceLoss, FocalLoss
from util.metric import compute_iou


class Trainer(object):
    def __init__(self, args):
        self.lr = args.lr
        self.args = args
        if args.visdom:
            from visdom import Visdom

            self.visdom = Visdom()
        self.loss_func1 = BCEWithLogitsLoss().cuda()
        self.loss_func2 = DiceLoss().cuda()

    def encode(self, labels):
        """
        将传入的预测标签转换为彩色标签
        @param labels
        """
        rst = []
        for ele in labels:
            ele = np.argmax(ele, axis=0)
            rst.append(label_to_color_mask(ele))
        return rst

    def adjust_lr(self, opt, epoch):
        base_lr = self.args.lr
        lr = base_lr - (epoch % 5) * 1e-4 - sqrt(epoch)*1e-5
        for param in opt.param_groups:
            param['lr'] = lr

    def visual(self, img, sig, mask, total_mask_loss):
        """
        使用visdom可视化，需要启动visdom服务 python -m visdom.server
        """
        pred = sig.cpu().detach().numpy().copy()
        pred = np.array(self.encode(pred))
        pred = pred.transpose((0, 3, 1, 2))

        ground_truth = mask.cpu().detach().numpy().copy()
        ground_truth = np.array(self.encode(ground_truth))
        ground_truth = ground_truth.transpose((0, 3, 1, 2))
        if self.visdom:
            self.visdom.images(img, win="Image", opts=dict(title="colorimg"))
            self.visdom.images(pred, win="Pred", opts=dict(title="pred"))
            self.visdom.images(ground_truth,
                               win="GroudTruth",
                               opts=dict(title="label"))
            self.visdom.line(
                total_mask_loss,
                win="train_iter_loss",
                opts=dict(title="train iter loss"),
            )

    def train(self, net, epoch, optimizer):
        net.train()
        total_mask_loss = []
        bce_loss = []
        dice_loss = []
        data_set = LaneDataSet("train",
                               multi_scale=self.args.multi_scale,
                               wid=self.args.wid)
        dataprocess = tqdm(
            DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.batch_size,
                pin_memory=True,
                #collate_fn=data_set.collate_fn,
            ),
            dynamic_ncols=True,
        )
        i = 0

        dataprocess.set_description_str("epoch:{}".format(epoch))
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        for i, batch_item in enumerate(dataprocess):
            image, mask = batch_item
            image, mask = Variable(image).cuda(), Variable(mask).cuda()
            optimizer.zero_grad()
            out = net(image)
            sig = torch.sigmoid(out)

            loss1 = 0.7 * self.loss_func1(out, mask)
            bce_loss.append(loss1.item())
            loss2 = 0.3 * self.loss_func2(sig, mask)
            dice_loss.append(loss2.item())
            mask_loss = loss1 + loss2  # + loss_func3(out, mask)
            mask_loss.backward()

            total_mask_loss.append(mask_loss.item())

            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            if i % 5 == 0:
                dataprocess.set_postfix_str(
                    "t:{:.4f},l1:{:.4f},l2:{:.4f} ".format(
                        np.mean(total_mask_loss), np.mean(bce_loss),
                        np.mean(dice_loss)))
                # dataprocess.set_postfix_str("mask_loss:{:.7f}".format(
                #     np.mean(total_mask_loss)))
                if self.args.visdom:
                    self.visual(image, sig, mask, total_mask_loss)

            optimizer.step()

        mean_loss = f"Epoch {epoch}  Mean loss {np.mean(total_mask_loss)}"
        print(mean_loss)
        print("Train MIOU")
        self.mean_iou(epoch, result)

    def mean_iou(self, epoch, result):
        """
        计算miou,只计算1-7一共7个类别的iou
        """
        miou = 0
        for i in range(1, 8):
            result_string = "{}: {:.4f} \n".format(
                i, result["TP"][i] / result["TA"][i])
            print(result_string)
            miou += result["TP"][i] / result["TA"][i]
        return miou / 7

    def valid(self, net, epoch):
        net.eval()
        total_mask_loss = []
        bce_loss = []
        dice_loss = []
        data_set = LaneDataSet("val",
                               multi_scale=self.args.multi_scale,
                               wid=self.args.wid)
        dataprocess = tqdm(
            DataLoader(
                data_set,
                batch_size=4,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                #collate_fn=data_set.collate_fn,
            ),
            dynamic_ncols=True,
        )
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        # loss_func3 = FocalLoss().cuda(device=self.ids[0])
        dataprocess.set_description_str("epoch:{}".format(epoch))
        i = 0
        for batch_item in dataprocess:
            i += 1
            image, mask = batch_item
            if torch.cuda.is_available():
                image, mask = (Variable(image).cuda(), Variable(mask).cuda())
            out = net(image)
            sig = torch.sigmoid(out)
            loss1 = 0.7 * self.loss_func1(out, mask)
            bce_loss.append(loss1.item())
            loss2 = 0.3 * self.loss_func2(sig, mask)
            dice_loss.append(loss2.item())
            mask_loss = loss1 + loss2  # + loss_func3(out, mask)

            total_mask_loss.append(mask_loss.item())
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            if i % 10 == 0:
                dataprocess.set_postfix_str(
                    "t:{:.4f},l1:{:.4f},l2:{:.4f} ".format(
                        np.mean(total_mask_loss), np.mean(bce_loss),
                        np.mean(dice_loss)))
        print("Test MIOU")
        return self.mean_iou(epoch, result)

    def run(self):
        if self.args.weights:
            net = torch.load(self.args.weights)
        else:
            net = DeeplabV3Plus(8).cuda()
        optimizer = torch.optim.AdamW(net.parameters(),
                                      lr=self.args.lr,
                                      weight_decay=5e-4)
        last_MIOU = 0.0
        for epoch in range(self.args.epochs):
            self.adjust_lr(optimizer, epoch)
            self.train(net, epoch, optimizer)
            with torch.no_grad():
                miou = self.valid(net, epoch)
            if miou > last_MIOU:
                msg = f"miou {miou} > last_MIOU {last_MIOU},save model"
                print(msg)
                torch.save(net, os.path.join(os.getcwd(), f"weights/best.pt"))
                last_MIOU = miou
        torch.save(net, f"weights/last.pt")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    if not os.path.exists("weights"):
        os.mkdir("weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weights", type=str, help="预训练模型")
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--wid", type=int, default=846)
    parser.add_argument("--lr", type=float, help="基础学习率，默认6e-4", default=6e-4)
    args = parser.parse_args()
    trainer = Trainer(args=args)
    trainer.run()


if __name__ == "__main__":
    main()
