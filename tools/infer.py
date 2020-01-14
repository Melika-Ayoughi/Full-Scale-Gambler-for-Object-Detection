# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
from pathlib import Path
import json
import time
import tqdm

import torch

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
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
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    print(cfg)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 infer ")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="A file with a list of input images path")
    parser.add_argument(
        "--output",
        required=True,
        help="A file or directory to save output visualizations. "
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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


def main(args):
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)
    cpu_device = torch.device("cpu")
    metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

    if args.input_file:
        with Path(args.input_file).open() as file:
            image_names = [str(Path(session) / "lri_1refl" / "image_COMBINED.png")
                           for session in map(str.strip, file) if session]

        output_folder = Path(args.output)
        output_folder.mkdir(exist_ok=True, parents=True)
        for path in tqdm.tqdm(image_names, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = predictor(img)
            num_predictions = len(predictions["instances"])
            time_spent = time.time() - start_time
            logger.info(f"{path}: detected {num_predictions} instances in {time_spent:.2f}s")

            instances = predictions["instances"].to(cpu_device)

            out_i_folder = output_folder / Path(path).parents[1].name
            out_i_folder.mkdir(exist_ok=True, parents=True)
            output_json_file = out_i_folder / "result.json"
            results = instances_to_coco_json(instances, -1)
            with output_json_file.open("w") as f:
                json.dump(results, f)
            if args.plot_output:
                out_filename = out_i_folder / "predicted.png"
                visualizer = Visualizer(img, metadata)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                vis_output.save(str(out_filename))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(get_parser().parse_args())
