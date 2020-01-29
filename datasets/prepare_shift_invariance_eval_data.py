import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

from PIL import Image

from detectron2.data.datasets.surface_signs import (
    dataset_read,
    get_category,
    get_label_id,
    TILE_SIZE,
)


def argument_parser() -> Any:
    cl_parser = argparse.ArgumentParser(
        description="prepare the data for evaluation of shift-invariance"
    )
    cl_parser.add_argument(
        "-s",
        "--split_file",
        required=False,
        default="uca_split_unweighted.txt",
        help="file with a list of session names",
    )
    cl_parser.add_argument(
        "-l",
        "--label_def_file",
        required=False,
        default="label_def_class_agnostic.txt",
        help="label def file",
    )
    cl_parser.add_argument(
        "--output_path",
        required=False,
        default="/media/deepstorage01/sse_sessions/shift_invariance_eval",
        help="output directory",
    )
    cl_parser.add_argument(
        "--crop_size", required=False, default=(640, 640), help="The crop size",
    )
    return vars(cl_parser.parse_args())


def box_to_absolute(
    bbox_coord: List[float], img_width: int, img_height: int
) -> Tuple[List[float], Tuple[int, int]]:
    # bbox_coord [xmin, ymin,xmax,ymax]
    bbox_coord = [
        bbox_coord[0] * img_width,
        bbox_coord[1] * img_height,
        bbox_coord[2] * img_width,
        bbox_coord[3] * img_height,
    ]

    box_center = (
        int((bbox_coord[2] + bbox_coord[0]) / 2),
        int((bbox_coord[3] + bbox_coord[1]) / 2),
    )
    return bbox_coord, box_center


def crop_pad_image(
    image: Image,
    box_center: Tuple[int, int],
    crop_size: Tuple[int, int],
    box: List[float],
) -> Tuple[Image, List[float]]:
    left = int(box_center[0] - crop_size[0] / 2)
    top = int(box_center[1] - crop_size[1] / 2)
    right = int(box_center[0] + crop_size[0] / 2)
    bottom = int(box_center[1] + crop_size[1] / 2)
    output = image.crop((left, top, right, bottom))
    cropped_box = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
    return output, cropped_box


def get_eval_data(
    split_file: str, label_def_file: str, output_path: str, crop_size: Tuple[int, int]
) -> None:
    """
    The function goes through all tiles in the validation part specified by the split_file
    1. random sample two instances of box per tile
    2. crop a fixed size(specified by the crop_size) around the selected instance from tile.
    3. the cropped image and transformed box would be saved accordingly.
    """
    dataset_dir = Path(__file__).parent / "surface_signs" / "annotations"
    split_dataset = dataset_read(dataset_dir / split_file)
    _, lookup_table = get_category(dataset_dir / label_def_file)

    for img_path in split_dataset["validation"]:
        image_name = img_path + "/lri_1refl_height_filtered/image_COMBINED.png"

        session_name = img_path.split("/tiles_s1500_o350/")[0].split("/")[-1]
        tile_name = img_path.split("/tiles_s1500_o350/")[-1]
        image = Image.open(image_name)

        gt_file = Path(img_path) / "bbox.json"
        with open(gt_file) as file:
            detection_gt = json.load(file)  # List[Dict]
        if len(detection_gt) >= 2:
            instances_sampled = random.sample(detection_gt, 2)
            for idx, bbox_i in enumerate(instances_sampled):
                bbox_coord = bbox_i["bbox"]
                label_id = get_label_id(lookup_table, bbox_i["label"])
                abs_box, box_center = box_to_absolute(
                    bbox_coord, img_width=TILE_SIZE[0], img_height=TILE_SIZE[1]
                )
                crop_image, shifted_box = crop_pad_image(
                    image, box_center, crop_size, abs_box
                )
                box_name = f"box_{int(box_center[0])}_{int(box_center[1])}"
                output_folder = Path(output_path) / session_name / tile_name / box_name
                output_folder.mkdir(exist_ok=True, parents=True)
                output_image_folder = output_folder / "lri_1refl_height_filtered"
                output_image_folder.mkdir(exist_ok=True, parents=True)
                crop_image.save(str(output_image_folder / "image_COMBINED.png"))
                json_file = output_folder / "bbox.json"
                with json_file.open("w") as file:
                    instances = {"bbox": shifted_box, "categories": label_id - 1}
                    json.dump(instances, file, indent=2)


if __name__ == "__main__":
    get_eval_data(**argument_parser())
