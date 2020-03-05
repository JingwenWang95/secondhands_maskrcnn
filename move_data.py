#!/usr/bin/env python3
import os
import re
import shutil

re_maskname = re.compile('(?P<frame_id>\d*)_(?P<object_class>\d*)_(?P<number>\d*).png')
re_imagename = re.compile('Color(?P<frame_id>\d*).(jpg|png)')

def parse_mask_filename(name):
    """
    :param name: mask file name
    :return: triple of integers: (frame, object, label)
    """
    m = re_maskname.search(name)
    if m is None:
        raise RuntimeError("mask file name could not be parsed")
    return (m.group('frame_id')), int(m.group('object_class')), int(m.group('number'))


def parse_rgb_filename(name):
    """
    :param name: image file name
    :return: frame index
    """
    m = re_imagename.search(name)
    if m is None:
        print(name)
        return None
    else:
        return m.group('frame_id')


if __name__ == "__main__":
    ROOT_DIR = "/media/jingwen/Data/secondhands/processed"
    OUT_DIR = "/media/jingwen/Data/secondhands/train/complete_dataset"
    f = open(os.path.join(ROOT_DIR, "correct.txt"))
    correct_dirs = [d.strip("\n") for d in f]
    for dir in correct_dirs:
        DATA_DIR = os.path.join(ROOT_DIR, dir)
        OUT_DATA_DIR = os.path.join(OUT_DIR, dir)
        RGB_DIR = os.path.join(DATA_DIR, "frames")
        MASK_DIR = os.path.join(RGB_DIR, "train_masks")
        OUT_RGB_DIR = os.path.join(OUT_DATA_DIR, "images")
        OUT_MASK_DIR = os.path.join(OUT_DATA_DIR, "annotations")

        if not os.path.exists(OUT_RGB_DIR):
            os.makedirs(OUT_RGB_DIR)
        if not os.path.exists(OUT_MASK_DIR):
            os.makedirs(OUT_MASK_DIR)

        for img in os.listdir(RGB_DIR):
            id = parse_rgb_filename(img)
            if id is not None:
                image_dir = os.path.join(RGB_DIR, img)
                out_image_dir = os.path.join(OUT_RGB_DIR, id + ".png")
                shutil.copy(image_dir, out_image_dir)

        for img in os.listdir(MASK_DIR):
            image_dir = os.path.join(MASK_DIR, img)
            out_image_dir = os.path.join(OUT_MASK_DIR, img)
            shutil.copy(image_dir, out_image_dir)
