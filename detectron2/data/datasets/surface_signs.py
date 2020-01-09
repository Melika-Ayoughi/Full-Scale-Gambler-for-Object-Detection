# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from pathlib import Path
from typing import Dict

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json


"""
This file contains functions to parse kitt-format surface signs annotations into dicts in the
"Detectron2 format".
"""

__all__ = ["load_ssigns", "register_ssigns_instances"]


def register_ssigns_instances(json_file: str, class_name: Dict[int, str], dataset_name: str):
    """
    Register surface_signs in json annotation format for detection.
    """
    register_coco_instances(dataset_name, {}, str(json_file), "")
    split = json_file.split(".json")[0].split("_"[-1])
    MetadataCatalog.get(dataset_name).set(
        thing_classes=class_name, dirname="", split=split
    )


def load_ssigns(json_file, dataset_name=None):
    """
    Load a json file in coco's annotation format.

    Args:
        json_file (str): full path to the ssigns json annotation file.
        dataset_name (str): the name of the dataset (e.g., "ssigns_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    return load_coco_json(json_file, "", dataset_name)


def get_class_name(label_def_file: Path) -> Dict[int, str]:
    with label_def_file.open() as f:
        content = [x.strip() for x in f.readlines()]
    class_names = [line_i.replace("/", "_") for line_i in content]
    class_dict = {idx + 1: class_name for idx, class_name in enumerate(class_names)}
    return class_dict


def visualize():
    """
        Test the surface_sign json dataset loader.

        Usage:
            python -m detectron2.data.datasets.surface_signs \
                path/to/json dataset_name output_dir visulization_limit
        """
    import sys
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from PIL import Image
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[2])

    dicts = load_ssigns(sys.argv[1])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = Path(sys.argv[3]) / "data-vis"
    dirname.mkdir(parents=True, exist_ok=True)
    num_vis = 0
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        if num_vis < int(sys.argv[4]):
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(d)
            fpath = dirname / d["file_name"].replace("/", "_")
            vis.save(str(fpath))
            num_vis += 1


if __name__ == "__main__":
    visualize()
