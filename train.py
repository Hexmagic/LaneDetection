import os
import shutil
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from tqdm import tqdm
from visdom import Visdom

from model.deeplabv3_plus import DeeplabV3Plus
from model.unet_plus import Unet
from setting import LOGPATH, MEMORY, MODELNAME, SIZE1, SIZE2, SIZE3
from data.datagener import (get_test_loader, get_train_loader,
                            get_valid_loader, one_hot)
from util.gpu import wait_gpu
from data.label_util import label_to_color_mask
from util.loss import DiceLoss, FocalLoss
from util.metric import compute_iou


class Trainer(object):
    def __init__(self, gpu, lr, loss, optim, memory=MEMORY, model='deeplab'):
        self.model = model
        self.lr = lr
        self.optim = optim
        self.gpu = gpu
        self.loss = loss
        self.ids = self.bootstrap(memory)
        if loss == 'bce+dice':
            self.loss_func1 = BCEWithLogitsLoss().cuda(device=self.ids[0])
            self.loss_func2 = DiceLoss().cuda(device=self.ids[0])
        else:
            self.loss_func2 = DiceLoss().cuda(device=self.ids[0])
            self.loss_func1 = FocalLoss().cuda(device=self.ids[0])
            
        plt = sys.platform
        self.visdom = Visdom() if plt == 'win32' or plt == 'darwin' else None
        self.trainF = open(os.path.join(LOGPATH, f'{self.model}_train.txt'),
                           'w')
        self.testF = open(os.path.join(LOGPATH, f'{self.model}_test.txt'), 'w')

    def bootstrap(self, memory):
        '''
        获取可用的GPU
        '''

        if self.gpu:
            ids = list(map(int, self.gpu.split(",")))
        else:
            ava_gpu_index = wait_gpu(need=memory)
            ids = ava_gpu_index
        print(f"Use Device  {ids} Train")
        return ids

    def adjust_lr_adam(self, optimizer, epoch):
        '''
        根据epoch衰减学习率
        '''
        if epoch == 0:
            lr = self.lr
        elif epoch == 2:
            lr = self.lr * 0.9
        elif epoch == 5:
            lr = self.lr * 0.7
        elif epoch == 8:
            lr = self.lr * 0.4
        elif epoch == 13:
            lr = self.lr * 0.5
        elif epoch == 18:
            lr = self.lr * 0.6
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_lr_sgd(self, optimizer, epoch):
        '''
        根据epoch衰减学习率
        '''
        if epoch == 0:
            lr = 5e-3
        elif epoch == 2:
            lr = 5e-3
        elif epoch == 5:
            lr = 3e-3
        elif epoch == 8:
            lr = 2e-3
        elif epoch == 13:
            lr = 1e-3
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def encode(self, labels):
        '''
        将传入的预测标签转换为彩色标签
        @param labels
        '''
        rst = []
        for ele in labels:
            ele = np.argmax(ele, axis=0)
            rst.append(label_to_color_mask(ele))
        return rst

    def visual(self, img, sig, mask, total_mask_loss):
        '''
        使用visdom可视化，需要启动visdom服务 python -m visdom.server
        '''
        pred = sig.cpu().detach().numpy().copy()
        pred = np.array(self.encode(pred))
        pred = pred.transpose((0, 3, 1, 2))

        ground_truth = mask.cpu().detach().numpy().copy()
        ground_truth = np.array(self.encode(ground_truth))
        ground_truth = ground_truth.transpose((0, 3, 1, 2))

        self.visdom.images(img, win='Image', opts=dict(title='colorimg'))
        self.visdom.images(pred, win='Pred', opts=dict(title='pred'))
        self.visdom.images(ground_truth,
                           win='GroudTruth',
                           opts=dict(title='label'))
        self.visdom.line(total_mask_loss,
                         win='train_iter_loss',
                         opts=dict(title='train iter loss'))

    def train(self, net, epoch, dataLoader, optimizer):
        net.train()
        total_mask_loss = []
        bce_loss = []
        dice_loss = []
        dataprocess = tqdm(dataLoader, dynamic_ncols=True)
        i = 0

        dataprocess.set_description_str("epoch:{}".format(epoch))
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        for i, batch_item in enumerate(dataprocess):
            if i % 100 == 0:
                self.trainF.flush()
            image, mask, _ = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            optimizer.zero_grad()
            out = net(image)
            sig = torch.sigmoid(out)
            if self.loss == 'bce+dice':
                loss1 = 0.7 * self.loss_func1(out, mask)
                bce_loss.append(loss1.item())
                loss2 = 0.3 * self.loss_func2(sig, mask)
                dice_loss.append(loss2.item())
                mask_loss = loss1 + loss2  #+ loss_func3(out, mask)
            else:
                loss1 = 0.7 * self.loss_func1(sig, mask)
                bce_loss.append(loss1.item())
                loss2 = 0.3 * self.loss_func2(sig, mask)
                dice_loss.append(loss2.item())
                mask_loss = loss1 + loss2  #+ loss_func3(out, mask)
            mask_loss.backward()
            total_mask_loss.append(mask_loss.item())
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            if i % 5 == 0:
                dataprocess.set_postfix_str("t:{:.4f},l1:{:.4f},l2:{:.4f} ".format(
                np.mean(total_mask_loss), np.mean(dice_loss),
                np.mean(bce_loss)))
                # dataprocess.set_postfix_str("mask_loss:{:.7f}".format(
                #     np.mean(total_mask_loss)))
                if self.visdom:
                    self.visual(image, sig, mask, total_mask_loss)

            self.trainF.write(f'Epoch {epoch} loss {mask_loss.item()}\n')
            optimizer.step()

        mean_loss = f"Epoch {epoch}  Mean loss {np.mean(total_mask_loss)}"
        print(mean_loss)
        self.trainF.write(mean_loss)
        self.trainF.flush()
        print("Train MIOU")
        self.mean_iou(epoch, result)

    def mean_iou(self, epoch, result):
        '''
        计算miou,只计算1-7一共7个类别的iou
        '''
        miou = 0
        for i in range(1, 8):
            result_string = "{}: {:.4f} \n".format(
                i, result["TP"][i] / result["TA"][i])
            self.testF.write(f'Epoch {epoch} IOU {result_string}')
            print(result_string)
            miou += result["TP"][i] / result["TA"][i]
        return miou / 7

    def valid(self, net, epoch, dataLoader):
        net.eval()
        total_mask_loss = []
        bce_loss = []
        dice_loss = []
        dataprocess = tqdm(dataLoader, dynamic_ncols=True)
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        #loss_func3 = FocalLoss().cuda(device=self.ids[0])
        for batch_item in dataprocess:
            image, mask, _ = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            out = net(image)
            sig = torch.sigmoid(out)
            if self.loss == 'bce+dice':
                loss1 = 0.7 * self.loss_func1(out, mask)
                bce_loss.append(loss1.item())
                loss2 = 0.3 * self.loss_func2(sig, mask)
                dice_loss.append(loss2.item())
                mask_loss = loss1 + loss2  #+ loss_func3(out, mask)
                # mask_loss = 0.7 * self.loss_func1(
                #     out, mask) + 0.3 * self.loss_func2(
                #         sig, mask)  #+ loss_func3(out, mask)
            else:
                loss1 = 0.7 * self.loss_func1(sig, mask)
                bce_loss.append(loss1.item())
                loss2 = 0.3 * self.loss_func2(sig, mask)
                dice_loss.append(loss2.item())
                mask_loss = loss1 + loss2
            #mask_loss = loss_func1(out, mask) + loss_func2(
            #sig, mask)  #+ loss_func3(out, mask)
            total_mask_loss.append(mask_loss.item())
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_postfix_str("t:{:.4f},l1:{:.4f},l2:{:.4f} ".format(
                np.mean(total_mask_loss), np.mean(dice_loss),
                np.mean(bce_loss)))
            self.testF.write(f'Epoch {epoch} loss {mask_loss.item()}\n')

        self.testF.flush()
        print("Test MIOU")
        return self.mean_iou(epoch, result)

    def load_model(self):
        model_name = f'{self.model}_{MODELNAME}'
        if os.path.exists(model_name):
            print("train from load model")
            last_gpu_id = int(
                open(f'{self.model}_last_gpu.id', 'r').read().strip())
            net = torch.load(
                model_name,
                map_location={f'cuda:{last_gpu_id}': f"cuda:{self.ids[0]}"})
        else:
            print("train from scratch")
            if self.model == 'deeplab':
                print("Model Deeplab")
                net = DeeplabV3Plus(8).cuda(device=self.ids[0])
            elif self.model == 'unet++':
                print("Model Unet++")
                net = Unet(8).cuda(device=self.ids[0])
            with open(f'{self.model}_last_gpu.id', 'w') as f:
                f.write(str(self.ids[0]))
        return net

    def run(self, batchsize, shape, epochs):
        train_data_batch = get_train_loader(batchsize, shape)
        val_data_batch = get_valid_loader(batchsize, shape)
        net = self.load_model()
        if len(self.ids) > 1:
            print("Use Mutil GPU Train Model")
            net = DataParallel(net, device_ids=self.ids)
        if self.optim.lower() == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=0.01,
                                        momentum=0.90,
                                        weight_decay=0.01,
                                        nesterov=True)
            adjust_lr = self.adjust_lr_sgd
        elif self.optim.lower() == 'adam':
            optimizer = torch.optim.AdamW(net.parameters(), weight_decay=0.002)
            adjust_lr = self.adjust_lr_adam
        else:
            optimizer = torch.optim.RMSprop(net.parameters(),
                                        momentum=0.99,
                                        weight_decay=0.02)
            adjust_lr = self.adjust_lr_adam
        last_MIOU = 0.0

        for epoch in range(epochs):
            adjust_lr(optimizer, epoch)
            self.train(net, epoch, train_data_batch, optimizer)
            with torch.no_grad():
                miou = self.valid(net, epoch, val_data_batch)
            if miou > last_MIOU:
                msg = f"miou {miou} > last_MIOU {last_MIOU},save model"
                print(msg)
                self.testF.write(msg)
                torch.save(
                    net, os.path.join(os.getcwd(),
                                      f"{self.model}_laneNet.pth"))
                last_MIOU = miou
        torch.save(net, os.path.join(os.getcwd(), f"{self.model}finalNet.pth"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='输入GPU的ID,或者以逗号分隔的ID列表')
    parser.add_argument('--model',
                        type=str,
                        help='模型名称，deeplab或者unet++',
                        default='deeplab')
    parser.add_argument('--stage', type=int, help='训练阶段，默认为1', default=1)
    parser.add_argument('--optim',
                        type=str,
                        help='优化器，Adam或者SGD RMSPROP',
                        default='adam')
    parser.add_argument('--lr', type=float, help='基础学习率，默认6e-4', default=6e-4)
    parser.add_argument('--loss',
                        type=str,
                        help="损失函数选择，例如bce+dice或者dice+focal,默认bce+dice",
                        default='bce+dice')
    args = parser.parse_args()

    model = args.model
    assert model in ['unet', 'deeplab', 'unet++']

    stages = [SIZE1, SIZE2, SIZE3]
    stages = stages[(args.stage - 1):]
    for ele in stages:
        print(f"Train Size {ele} Stage {args.stage}")
        shape, batch, epoch = ele
        trainer = Trainer(memory=6 if batch == 2 else 9,
                          model=model,
                          optim=args.optim,
                          lr=args.lr,
                          gpu=args.gpu,
                          loss=args.loss)
        trainer.run(batch, shape, epoch)


if __name__ == '__main__':
    main()
