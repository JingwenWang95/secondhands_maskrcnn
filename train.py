import logging
import os
from collections import OrderedDict
import torch

from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.utils.logger import setup_logger
from detectron2.modeling import GeneralizedRCNNWithTTA
import os

DETECTRON_DIR = "/home/jingwen/Vision/detectron2"
TRAIN_DIR = "/media/jingwen/Data/secondhands/train/complete_dataset/train"
TEST_DIR = "/media/jingwen/Data/secondhands/train/complete_dataset/test"


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def register_dataset():
    register_coco_instances("secondhands_train", {}, os.path.join(TRAIN_DIR, "secondhands_train.json"), TRAIN_DIR)
    register_coco_instances("secondhands_test", {}, os.path.join(TEST_DIR, "secondhands_test.json"), TEST_DIR)


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(
    os.path.join(DETECTRON_DIR, "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("secondhands_train",)
    cfg.DATASETS.TEST = ("secondhands_test",)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    # initialize from model zoo
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = (6000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 classes (brush, cutter, pliers, spraybottle, mallet)
    cfg.OUTPUT_DIR = "/media/jingwen/Data/secondhands/train/complete_dataset/results"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main():
    register_dataset()
    cfg = setup()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == "__main__":
    setup_logger()
    launch(
        main,
        num_gpus_per_machine=1
    )