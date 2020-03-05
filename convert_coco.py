#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = '/media/jingwen/Data/secondhands/train/complete_dataset/test'

INFO = {
    "description": "Secondhands Dataset",
    "year": 2020,
    "contributor": "Jingwen Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

CATEGORIES = [
    {
        'id': 1,
        'name': 'brush',
        'supercategory': 'brush',
    },
    {
        'id': 2,
        'name': 'cutter',
        'supercategory': 'cutter',
    },
    {
        'id': 4,
        'name': 'pliers',
        'supercategory': 'pliers',
    },
    {
        'id': 5,
        'name': 'spraybottle',
        'supercategory': 'spraybottle',
    },
    {
        'id': 11,
        'name': 'mallet',
        'supercategory': 'mallet',
    },
]


def filter_for_png(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for dir in os.listdir(ROOT_DIR):
        DATA_DIR = os.path.join(ROOT_DIR, dir)
        IMAGE_DIR = os.path.join(DATA_DIR, "images")
        ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")

        # filter for jpeg images
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_png(root, files)

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.relpath(image_filename, ROOT_DIR), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:
                        annotation_name = os.path.relpath(annotation_filename, DATA_DIR)
                        print(annotation_filename)
                        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_name][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

    with open('{}/secondhands_train.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()