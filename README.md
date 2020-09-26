## 
车道线识别
![](demo/demo.gif)

## 环境

系统： Ubuntu20
Python：3.7.9
cuda: 10.2
pytorch: 1.6.0
GPU: RTX2060 SUPER

## 安装
执行命令：
```
pip install -r requirements.txt
```
## 数据
截取自百度车道线数据，我这里用其中一部分Road02的数据，下载后改成下面的目录结构：
```
data
├── Image
│   └── ColorImage
└── Label
    ├── Record015
    ├── Record016
    ├── Record018
    ├── Record019
    ├── Record020
    ├── Record021
    ├── Record022
    ├── Record023
    ├── Record024
    ├── Record025
    ├── Record026
    ├── Record027
    ├── Record028
    ├── Record029
    └── Record030
```

## 训练
执行下面的脚本
```shell
python train.py --batch_size 4 --wid 846 --epochs 50 --lr 0.0006 --back deeplab 
```
back还可以选unet(实际是unet++)，unet训练推断速度快，但是精度低，deeplab精度高，训练速度慢(约为unet速度的1/5甚至更慢)。
由于原图分辨率较大3384x1710,图片上面690个像素主要为天空，这里我们只截取下半部分(3384x1020)，所以可以先训练小分辨率，然后逐步增大分辨率
1. 第一阶段wid为846
2. 第二阶段wid为1692
3. 第三阶段wid为3384
经过三个阶段的训练deeplab的测试集mAP可以达到83%,Unet大概75%左右

## 推断
执行下面的推断代码，其中sample为存放测试图片的目录，可以自己随意挑选几张，output为推断结果目录
```
python inference.py --model weights/best.pt --folder sample --output output
```
DeepLab版本可以达10FPS,UNET版本没怎么测试，应该有30多FPS


