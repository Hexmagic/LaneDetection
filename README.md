## 
车道线识别

![](demo/demo.gif)

## 环境

系统： Ubuntu20
Python：3.7.9
cuda: 10.2
pytorch: 1.6.0
GPU: RTX2060 SUPER

## Install
执行命令：
```
pip install -r requirements.txt
```
## TrainVal Data Split

截取自百度车道线复赛的数据，我这里用其中Road02和Road04的数据，新建data目录进入，下载ColorImage_road02.zip,ColorImage_road04.zip和Gray_Label_New.zip 解压即可(zip文件可删除)，目录结构如下:

```
data
├── ColorImage_road02
│   └── ColorImage
├── ColorImage_road02.zip
├── ColorImage_road04
│   └── ColorImage
├── ColorImage_road04.zip
├── Gray_Label
│   ├── Label_road02
│   ├── Label_road03
│   └── Label_road04
└── Gray_Label_New.zip
```
由于4类别比例比较少(在所有分类中约占百分之一),所以为了训练和测试分布大致一致，这里使用split.py进行数据划分生产train.txt和val.txt
```
python split.py
```
## Training
打开visdom
```
python -m visdom.server
```
执行下面的脚本进行可视化训练,具体参数查看train.py,第一阶段只需要训练10个epoch左右即可得到0.75mAP
```shell
python train.py --batch_size 4  --epochs 15 --visdom 
```
back还可以选unet(实际是unet++)，unet训练推断速度快，但是精度低，deeplab精度高，训练速度慢(约为unet速度的1/5甚至更慢)。
由于原图分辨率较大3384x1710,图片上面690个像素主要为天空，这里我们只截取下半部分(3384x1020)，所以可以先训练小分辨率，然后逐步增大分辨率
1. 第一阶段wid为846
2. 第二阶段wid为1692
3. 第三阶段wid为3384
经过三个阶段的训练deeplab的测试集mAP可以达到83%,Unet大概75%左右

## Inference

预训练模型：

>之前做过的没有保存，由于资源限制，我现在只能提供训练第一阶段的Deeplab模型，训练了约为8个Epoch，mAP约为0.75,

百度网盘: [链接](https://pan.baidu.com/s/19TZyCihG7z105PLUFLMydA)  密码: ca8j

Google Drive: [链接](https://drive.google.com/file/d/1F_RI45eOuT0CPHml_5s_I6hoG1rBX_Pd/view?usp=sharing)

执行下面的推断代码，其中sample为存放测试图片的目录，可以自己随意挑选几张，output为推断结果目录
```
python inference.py --model weights/best.pt --folder sample --output output
```
DeepLab版本可以达10FPS,UNET版本没怎么测试，应该有30多FPS


