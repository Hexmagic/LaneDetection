import sys
RoadList = ['Image_Data/Road02', 'Image_Data/Road03', 'Image_Data/Road04']
MEMORY = 6
EPOCH = 10
CSV_PATH = 'data_list'
DATAROOT = '/root/data/LaneSeg' if sys.platform != 'win32' else "D:\Compressed"
LOGPATH = 'log'
MODELNAME = 'laneNet.pth'
SIZE1 = [846, 255,2]
SIZE2 = [1128, 340,2]
SIZE3 = [1692, 510,1]


