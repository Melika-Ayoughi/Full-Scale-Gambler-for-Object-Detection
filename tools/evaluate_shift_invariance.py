import argparse
import json
from typing import Tuple, List

import os
import numpy as np
import cv2 as cv
from pathlib import Path

from detectron2.utils.logger import setup_logger

logger = setup_logger(name=__name__)


def box_center(box: np.ndarray) -> np.ndarray:
    """
    Return box center.
    Box coordinates need to be in [x,y,width,height] format
    """
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def crop_img_center(img: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    y, x, _ = img.shape
    start_x = x // 2 - (crop_width // 2)
    start_y = y // 2 - (crop_height // 2)
    return img[start_y : start_y + crop_height, start_x : start_x + crop_width, :]


def bbox_intersection(box_a: np.ndarray, box_b: np.ndarray) -> int:
    ixmin = np.maximum(box_a[0], box_b[0])
    iymin = np.maximum(box_a[1], box_b[1])
    ixmax = np.minimum(box_a[0] + box_a[2], box_b[0] + box_b[2])
    iymax = np.minimum(box_a[1] + box_a[3], box_b[1] + box_b[3])
    i_width = np.maximum(ixmax - ixmin, 0.0)
    i_height = np.maximum(iymax - iymin, 0.0)
    return i_width * i_height


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Intersection over union of two boxes.
    Box coordinates need to be in [x,y,width,height] format
    """
    inter = bbox_intersection(box_a, box_b)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    assert np.any(area_b >= 0) or np.any(area_a >= 0)
    union = area_a + area_b - inter
    return inter / union


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of shift invariance")
    parser.add_argument(
        "--input_list",
        required=True,
        type=Path,
        help="File with a list of samples to analyze. Each sample is a folder containing bbox predictions",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Path to output directory"
    )
    parser.add_argument(
        "--show_debug_images",
        required=False,
        action="store_true",
        help="Output additional debug images",
    )

    return parser.parse_args()


class Prediction:
    def __init__(self, category_id: int, score: float, iou: float, bbox: np.ndarray):
        self.category_id = category_id
        self.score = score
        self.iou = iou
        self.bbox = bbox


def read_results(filename: Path) -> Tuple[Prediction, np.ndarray, np.ndarray]:
    with filename.open("r") as file:
        try:
            results_dict_list = json.load(file)
        except FileNotFoundError:
            logger.error(f"{filename} not found.")
        except json.JSONDecodeError as e:
            logger.error(
                "JSONDecodeError raised while reading detection results:" + e.msg
            )

    if not results_dict_list:
        return None, None, None
    dict_i = results_dict_list[0]
    prediction = Prediction(
        int(dict_i["category_id"]),
        float(dict_i["score"]),
        float(dict_i["iou"]),
        dict_i["bbox"],
    )
    return prediction, dict_i["gt_box"], dict_i["anchor_box"]


class ShiftData:
    def __init__(self, shift_offset: Tuple[int, int], result_filename: Path):
        self.shift_offset = shift_offset
        self.prediction, self.gt_box, self.anchor_box = read_results(result_filename)
        self.anchor_gt_iou = (
            compute_iou(self.gt_box, self.anchor_box) if self.anchor_box else 0
        )
        self.anchor_gt_offset = (
            box_center(self.gt_box) - box_center(self.anchor_box)
            if self.anchor_box
            else 0
        )


class Sample:
    """
    Class containing all the information of one validation sample, defined as
    the set of all shifts of a given ground truth object.
    """

    def __init__(self, sample_folder: Path):
        self.sample_folder = sample_folder
        self.sample_unique_id = "-".join(str(self.sample_folder).split("/")[-3:])
        self.shifts: List[ShiftData] = []
        self.statistics = {}
        self.prediction_scores = []
        self.prediction_gt_ious = []
        self.img_score_shift_variance: np.ndarray = np.array([])
        self.img_iou_shift_variance: np.ndarray = np.array([])
        self.has_predictions = False
        self._load_data()

    def _load_data(self) -> None:
        """
        Load all predictions and ground truths, for all shifts
        """
        for _, shift_folders, __ in os.walk(str(self.sample_folder)):
            for folder in shift_folders:
                result_filename = self.sample_folder / folder / "result.json"
                shift_offset = (int(folder.split("_")[0]), int(folder.split("_")[1]))
                shift_data = ShiftData(shift_offset, result_filename)
                self.has_predictions = self.has_predictions or shift_data.prediction
                self.shifts.append(shift_data)

    def compute_statistics(self) -> None:
        """
        Extracts the statistics from the sample data.
        For example, how the score of the prediction varies for different shifts
        """

        def compute_stats(name: str, data: List):
            self.statistics[f"{name}_mean"] = np.mean(data)
            self.statistics[f"{name}_stddev"] = np.std(data)
            self.statistics[f"{name}_diff"] = np.max(data) - np.min(data)
            self.statistics[f"{name}_diff_norm"] = (
                np.max(data) - np.min(data)
            ) / np.mean(data)

        self.prediction_scores = [
            shift.prediction.score if shift.prediction else 0 for shift in self.shifts
        ]
        self.prediction_gt_ious = [
            shift.prediction.iou if shift.prediction else 0 for shift in self.shifts
        ]

        compute_stats("score", self.prediction_scores)
        compute_stats("iou", self.prediction_gt_ious)
        self.statistics["score_iou_correlation"] = np.corrcoef(
            self.prediction_scores, self.prediction_gt_ious
        )[0, 1]

        for shift in self.shifts:
            if shift.prediction:
                self.statistics["gt_width"] = shift.gt_box[2]
                self.statistics["gt_height"] = shift.gt_box[3]
                break

    def compute_visualizations(self) -> None:
        """
        Creates visualizations of how statistics change for shifted inputs.
        For example, it creates a map of size {shift_range, shift_range} that for each shift
        shows the score of the prediction in that place.
        """
        max_shift_x = max([shift.shift_offset[1] for shift in self.shifts]) + 1
        max_shift_y = max([shift.shift_offset[0] for shift in self.shifts]) + 1

        self.img_score_shift_variance = (np.array(self.prediction_scores) * 255).astype(
            int
        )
        self.img_score_shift_variance = self.img_score_shift_variance.reshape(
            (max_shift_y, max_shift_x)
        )
        self.img_iou_shift_variance = (np.array(self.prediction_gt_ious) * 255).astype(
            int
        )
        self.img_iou_shift_variance = self.img_iou_shift_variance.reshape(
            (max_shift_y, max_shift_x)
        )

    def log_debug_visualization(self, output_dir: Path) -> None:
        """
        Creates and logs additional visualizations that allow for an in depth analysis of results.
        For example, it draws the anchor box related to the prediction for each shift.
        Assumes that images for predictions are available at this path: {sample_folder}/box_{x}_{y}/predicted.png
        """
        (output_dir / self.sample_unique_id).mkdir(exist_ok=True)
        crop_size = 100
        for shift in self.shifts:
            img = cv.imread(
                str(
                    self.sample_folder
                    / f"{shift.shift_offset[0]}_{shift.shift_offset[1]}/predicted.png"
                )
            )
            if shift.gt_box:
                gt_box = np.array(shift.gt_box).astype(int)
                cv.rectangle(
                    img,
                    (gt_box[0], gt_box[1]),
                    (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]),
                    (0, 255, 0),
                )
            if shift.anchor_box:
                anchor_box = np.array(shift.anchor_box).astype(int)
                cv.rectangle(
                    img,
                    (anchor_box[0], anchor_box[1]),
                    (anchor_box[0] + anchor_box[2], anchor_box[1] + anchor_box[3]),
                    (255, 255, 255),
                )
            img_crop = crop_img_center(img, crop_size, crop_size)
            cv.imwrite(
                str(
                    output_dir
                    / self.sample_unique_id
                    / f"_debug_{shift.shift_offset[0]}_{shift.shift_offset[1]}.png"
                ),
                img_crop,
            )

    def save_visualizations(self, output_dir: Path) -> None:
        cv.imwrite(
            str(output_dir / self.sample_unique_id) + "_score.png",
            self.img_score_shift_variance,
        )
        cv.imwrite(
            str(output_dir / self.sample_unique_id) + "_iou.png",
            self.img_iou_shift_variance,
        )

    def save_statistics(self, output_dir: Path) -> None:
        statistics_filename = output_dir / "statistics.txt"
        if not statistics_filename.exists():
            with statistics_filename.open("w") as file_statistics:
                file_statistics.write("sample_folder\t")
                print(*self.statistics, sep="\t", file=file_statistics)

        file_statistics = statistics_filename.open("a")
        file_statistics.write(f"{self.sample_folder}")
        for stat in self.statistics:
            file_statistics.write(f"\t{self.statistics[stat]}")
        file_statistics.write("\n")

        anchor_statistics_filename = output_dir / "detailed_statistics.txt"
        file_anchor_statistics = anchor_statistics_filename.open("a")
        file_anchor_statistics.write(f"{self.sample_folder}")
        for shift in self.shifts:
            if shift.prediction:
                file_anchor_statistics.write(
                    f"{self.sample_folder}\t{shift.shift_offset[0]}\t{shift.shift_offset[1]}"
                    f"\t{shift.prediction.score}\t{shift.prediction.iou}\t{shift.anchor_gt_iou}"
                    f"\t{shift.anchor_gt_offset[0]}\t{shift.anchor_gt_offset[1]}"
                    f"\t{shift.gt_box[2]}\t{shift.gt_box[3]}\n"
                )


def aggregate_results(samples: List[Sample], output_dir: Path) -> None:
    score_diffs, iou_diffs = [], []
    for sample in samples:
        if sample.has_predictions:
            score_diffs.append(sample.statistics["score_diff"])
            iou_diffs.append(sample.statistics["iou_diff"])
    with (output_dir / "summary.txt").open("w") as file_summary:
        avg_score_diff = np.mean(score_diffs)
        avg_iou_diff = np.mean(iou_diffs)
        file_summary.write(f"avg_score_diff: {avg_score_diff}\n")
        file_summary.write(f"avg_iou_diff: {avg_iou_diff}\n")


def evaluate_shift_invariance(
    input_list: Path, output_dir: Path, show_debug_images: bool
) -> None:
    if not input_list.exists():
        logger.error(f"Input file {input_list} not found, aborting.")
        return
    with input_list.open("r") as input_file:
        input_folders = [x.strip() for x in input_file.readlines()]

    samples = []
    for i_sample, sample_folder in enumerate(input_folders):
        logger.info(
            f"Analyzing sample {i_sample}/{len(input_folders)}: {sample_folder}"
        )
        sample = Sample(Path(sample_folder))
        if not sample.has_predictions:
            continue
        sample.compute_statistics()
        sample.save_statistics(output_dir)
        sample.compute_visualizations()
        sample.save_visualizations(output_dir)
        if show_debug_images:
            sample.log_debug_visualization(output_dir)
        samples.append(sample)

    aggregate_results(samples, output_dir)


if __name__ == "__main__":
    args = parse_args()
    evaluate_shift_invariance(args.input_list, args.output_dir, args.show_debug_images)
