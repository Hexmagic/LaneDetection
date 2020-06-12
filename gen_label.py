import os
from pathlib import Path
import shutil
from tqdm import tqdm
from random import shuffle
from typing import List


def write_file(img_ids: List[str], filename):
    with open(filename, "w") as f:
        for line in img_ids:
            f.write(line + "\n")


def main():
    if not os.path.exists("data/labels"):
        os.mkdir("data/labels")
        os.mkdir("data/images")
        path = Path("data")
        jpg_globs = list(path.glob("./*/*/*/*/*.jpg"))
        for jpg in tqdm(jpg_globs, desc="移动图片"):
            # print(jpg)
            shutil.copyfile(str(jpg), f"data/images/{jpg.name}")
        lpath = Path("data/Gray_Label")
        gray_globs = list(lpath.glob("./*/*/*/*/*.png"))
        for png in tqdm(gray_globs, desc="移动标签"):
            shutil.copyfile(str(png), f"data/labels/{png.name.replace('_bin','')}")

    label_ids = os.listdir("data/labels")
    image_ids = set(os.listdir("data/images"))
    label_ids = list(
        filter(lambda x: x.replace(".png", ".jpg") in image_ids, label_ids)
    )
    shuffle(label_ids)
    length = len(label_ids)
    train, valid = label_ids[: int(length * 0.7)], label_ids[int(length * 0.7) :]
    write_file(train, "data/train.txt")
    write_file(valid, "data/val.txt")


if __name__ == "__main__":
    main()
