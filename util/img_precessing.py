from torchvision.transforms import ToTensor, Normalize
import imgaug as iaa
import numpy as np


def image_augmentation(ori_img):
    random_seed = np.random.randint(0, 10)
    if random_seed > 5:
        seq = iaa.Sequential(
            [
                iaa.OneOf(
                    [
                        iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                        iaa.GaussianBlur(sigma=(0, 1.0)),
                    ]
                )
            ]
        )
        ori_img = seq.augment_image(ori_img)
    return ori_img

