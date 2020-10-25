from math import sqrt
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio
from imgaug.augmenters import segmentation
from imgaug.augmenters.arithmetic import Multiply
from imgaug.augmenters.blend import BlendAlpha
from imgaug.augmenters.geometric import Jigsaw
from imgaug.augmenters.meta import ChannelShuffle, OneOf
from imgaug.augmenters.size import CropAndPad
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from numpy.lib.type_check import imag
from torchvision.transforms.transforms import RandomCrop, RandomHorizontalFlip


def aug_image_and_segmap(image, segmap):
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []
    seq = iaa.Sequential(
        [

            #iaa.Sharpen((0.1, 0.3)),  # sharpen the image
            iaa.Affine(
                rotate=(-10, 15)
            ),  # rotate by -45 to 45 degrees (affects segmaps)
            #iaa.ElasticTransformation(
            #    alpha=10, sigma=5
            #),  # apply water effect (affects segmaps)
            iaa.ChannelShuffle(),
            # iaa.BlendAlphaRegularGrid(iaa.Multiply(0.0,0.5)),
            iaa.Fliplr(0.5),
            iaa.CropAndPad(),
            #iaa.Jigsaw(),
            iaa.OneOf(
                [
                    #iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
                    iaa.CoarseDropout((0.02,0.1), size_percent=0.07, random_state=2),
                ]

            )

            #iaa.AdditiveGaussianNoise( random_state=3),
        ],
        random_order=True,
    )
    image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
    return image_aug, segmap_aug.arr[:, :, 0]


if __name__ == "__main__":
    import cv2

    image = cv2.imread(
        "/home/mix/Github/LaneDetection/data/Image/ColorImage/Record015/Camera 5/170927_065535168_Camera_5.jpg")
    segmap = cv2.imread(
        "/home/mix/Github/LaneDetection/data/Label/Record015/Camera 5/170927_065535168_Camera_5_bin.png", 0
    )
    image, segmap = aug_image_and_segmap(image, segmap)
    cv2.imwrite("test.jpg", image)
    cv2.imwrite("test.png", segmap)
