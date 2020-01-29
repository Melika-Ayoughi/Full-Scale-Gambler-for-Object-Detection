# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
from pathlib import Path
import json
import time
import tqdm
from typing import Dict, List

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import (
    read_image,
    shift_pad_input,
    filter_prediction_with_gt,
)
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.visualizer import Visualizer


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    print(cfg)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 infer")
    parser.add_argument(
        "--config-file", metavar="FILE", required=True, help="path to config file"
    )
    parser.add_argument(
        "--input_file", required=True, help="A file with a list of input images path"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="A file or directory to save output visualizations. ",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--input_size", default=[800, 800], help="The input_size after padding"
    )
    parser.add_argument(
        "--shift_range",
        type=int,
        default=8,
        help="If shift_range > 0, inference would be made to shifted input image (0-shift_range) "
        "and pad zeros to the input_size",
    )
    parser.add_argument(
        "--plot_output",
        action="store_true",
        help="Whether or not to plot the predictions",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def read_gt_box(
    data_path: Path
) -> np.ndarray:  # as [xmin,ymin, xmax, ymax] in absolute position
    gt_box_file = data_path / "bbox.json"
    with gt_box_file.open() as f:
        gt_boxes = json.load(f)
    box = np.array(gt_boxes["bbox"])
    return box


def main(args):
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)
    cpu_device = torch.device("cpu")
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    input_size = args.input_size
    shift_range = args.shift_range
    if args.input_file:
        with Path(args.input_file).open() as file:
            path_list = [Path(session) for session in map(str.strip, file) if session]

        output_folder = Path(args.output)
        output_folder.mkdir(exist_ok=True, parents=True)
        for path in tqdm.tqdm(path_list, disable=not args.output):
            image_name = path / "lri_1refl_height_filtered" / "image_COMBINED.png"
            img = read_image(str(image_name), format="BGR")
            gt_box_original = (
                read_gt_box(path) if shift_range > 0 else None
            )  # "XYXY_ABS"
            out_i_folder = output_folder / path.parts[-3] / path.parts[-2] / path.name
            out_i_folder.mkdir(exist_ok=True, parents=True)

            logger.info(f"{str(path)}")
            for shifted_img, gt_box, shift_i in shift_pad_input(
                img, gt_box_original, input_size, shift_range=shift_range
            ):
                start_time = time.time()
                predictions = predictor(shifted_img)
                time_spent = time.time() - start_time
                instances = predictions["instances"].to(cpu_device)

                result_folder = (
                    out_i_folder / f"{shift_i[0]}_{shift_i[1]}"
                    if shift_range > 0
                    else out_i_folder
                )
                result_folder.mkdir(exist_ok=True, parents=True)
                output_json_file = result_folder / "result.json"
                if gt_box is not None:
                    instances = filter_prediction_with_gt(instances, gt_box)
                results: List[Dict] = instances_to_coco_json(instances, -1)
                if len(results) == 0:
                    logger.info(
                        f"shift {shift_i[0]}_{shift_i[1]} detected {len(instances)} instances in {time_spent:.2f}s"
                    )
                with output_json_file.open("w") as f:
                    json.dump(results, f)
                if args.plot_output:
                    out_filename = result_folder / "predicted.png"
                    visualizer = Visualizer(shifted_img, metadata)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances
                    )
                    vis_output.save(str(out_filename))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(get_parser().parse_args())
