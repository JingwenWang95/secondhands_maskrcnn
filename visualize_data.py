from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer

coco_jsonfile = "/media/jingwen/Data/secondhands/train/complete_dataset/train/secondhands_train.json"
img_root = "/media/jingwen/Data/secondhands/train/complete_dataset/train"

register_coco_instances("secondhands_testing", {}, coco_jsonfile, img_root)
secondhands_metadata = MetadataCatalog.get("secondhands_testing")
# print(secondhands_metadata)
dataset_dicts = DatasetCatalog.get("secondhands_testing")
print(len(dataset_dicts))
for n in range(0, len(dataset_dicts), 1000):
    for j in range(0, 20):
        d = dataset_dicts[n+j]
        img = cv2.imread(d["file_name"])
        print(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=secondhands_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        cv2.imshow("rr", img)
        cv2.waitKey(0)