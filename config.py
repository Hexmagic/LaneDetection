import sys
RoadList = ['Image_Data/Road02', 'Image_Data/Road03', 'Image_Data/Road04']
MEMORY = 6
EPOCH = 20
CSV_PATH = 'data_list'
DATAROOT = '/root/data/LaneSeg' if sys.platform != 'win32' else "D:\Compressed"
LOGPATH = 'log'
MODELNAME = 'laneNet.pth'
