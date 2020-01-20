import numpy as np
import cv2

GRAYMAP = {
    0: [
        0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202,
        231, 230, 228, 229, 233, 212, 223
    ],
    1: [200, 204, 209],
    2: [201, 203],
    3: [217],
    4: [210],
    5: [214],
    6: [220, 221, 222, 224, 225, 226],
    7: [205, 227, 250]
}


def mask_to_label(gray_mask):
    label = np.zeros_like(gray_mask, dtype=np.uint8)
    for k, v in GRAYMAP.items():
        for ele in v:
            label[gray_mask == ele] = k
    return label


def label_to_mask(label):
    labelMap = {0: 0, 1: 209, 2: 203, 3: 217, 4: 210, 5: 214, 6: 224, 7: 227}
    mask = np.zeros_like(label, dtype=np.uint8)
    for k, v in labelMap.items():
        mask[label == k] = v
    return mask


def label_to_color_mask(label):
    colroMap = {
        0: [0, 0, 0],
        1: [70, 130, 180],
        2: [0, 0, 142],
        3: [153, 153, 153],
        4: [128, 64, 128],
        5: [190, 153, 153],
        6: [0, 0, 230],
        7: [255, 128, 0]
    }
    channel = np.zeros_like(label, dtype=np.uint8)
    mask = cv2.merge([channel, channel, channel])
    for k, v in colroMap.items():
        mask[label == k] = v
    return mask
