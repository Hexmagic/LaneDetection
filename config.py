import sys
import os
RoadList = ['Image_Data/Road02', 'Image_Data/Road04']
MEMORY = 6
CSV_PATH = 'data_list'
if not os.path.exists(CSV_PATH):
    os.mkdir(CSV_PATH)

LOGPATH = 'log'
MODELNAME = 'laneNet.pth'
PREDICT_PATH = 'predict'
if not os.path.exists(PREDICT_PATH):
    os.mkdir(PREDICT_PATH)
SIZE1 = [[846, 255], 2, 15]
SIZE2 = [[1128, 340], 2, 10]
SIZE3 = [[1692, 510], 1, 15]
