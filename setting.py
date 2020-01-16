import sys
import os
RoadList = ['Image_Data/Road02', 'Image_Data/Road03', 'Image_Data/Road04']
MEMORY = 6
CSV_PATH = 'data_list'
if not os.path.exists(CSV_PATH):
    os.mkdir(CSV_PATH)
DATAROOT = '/root/data/LaneSeg' if sys.platform != 'win32' else "D:\Compressed"
LOGPATH = 'log'
MODELNAME = 'laneNet.pth'
SIZE1 = [[846, 255], 2, 10]
SIZE2 = [[1128, 340], 2, 5]
SIZE3 = [[1692, 510], 1, 10]
