import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

SPLIT_DICT = {"training": [], "validation": [], "test": []}


def argument_parser() -> Any:
    cl_parser = argparse.ArgumentParser(description="Convert kitt splits of surface signs to coco format")
    cl_parser.add_argument("-s", "--split_file", required=False,
                           default="uca_split_unweighted.txt",
                           help="file with a list of session names")
    cl_parser.add_argument("-l", "--label_def_file", required=False,
                           default="label_def_class_aware_150.txt",
                           help="output label def file")

    return vars(cl_parser.parse_args())


def dataset_read(split_file: Path) -> Dict[str, List[str]]:
    comment_char = "#"
    sections: Dict[str, List[str]] = copy.deepcopy(SPLIT_DICT)
    current_section: List[str] = []
    with split_file.open() as file:
        for line in map(str.strip, file):
            if len(line) > 2 and line[0] == "[" and line[-1] == "]":
                current_section = sections[line[1:-1]]
            elif line and (line[0] != comment_char):
                current_section.append(line)
    return sections


def kitt2coco(bbox_coord: List[float], img_width: int, img_height: int) -> List[float]:
    # bbox_coord [xmin, ymin,xmax,ymax]
    bbox_coord = [bbox_coord[0] * img_width, bbox_coord[1] * img_height,
                  bbox_coord[2] * img_width, bbox_coord[3] * img_height]
    box_width = bbox_coord[2] - bbox_coord[0]
    box_height = bbox_coord[3] - bbox_coord[1]
    return [bbox_coord[0], bbox_coord[1], box_width, box_height]


def get_label_id(lookup_table: Dict[str, int], label_name: str, default_value=-1) -> int:
    if label_name == "":
        return default_value
    if label_name in lookup_table:
        return lookup_table[label_name]
    else:
        parent_label_name = "/".join(label_name.split("/")[:-1])
        return get_label_id(lookup_table, parent_label_name)


def get_category(label_def_file: Path) -> Tuple[List[Dict], Dict]:
    """
    Returns the list of categories, and a lookup table that for each
    category/sub_category/value in {label_def_file}, contains that assigned label id.
    """
    with label_def_file.open() as f:
        content = [x.strip() for x in f.readlines()]
    categories = []
    lookup_table = {}
    for idx, line_i in enumerate(content):
        category = {"supercategory": "ssigns", "id": idx + 1, "name": line_i.replace("/", "_")}
        categories.append(category)
        for word_i in line_i.split("|"):
            if word_i:
                lookup_table.update({word_i.strip(): category["id"]})
    return categories, lookup_table


def convert_ssigns_coco_format(split_file: str, label_def_file: str) -> None:
    dataset_dir = Path(__file__).parent / "surface_signs" / "annotations"
    split_dataset = dataset_read(dataset_dir / split_file)
    split_file_name = split_file.split(".")[0]
    annotations = copy.deepcopy(SPLIT_DICT)
    images = copy.deepcopy(SPLIT_DICT)
    img_id = 0
    box_id = 0
    tile_width = 800
    tile_height = 1500
    categories, lookup_table = get_category(dataset_dir / label_def_file)
    output_file_suffix = label_def_file.lstrip("label_def_").split(".")[0]

    for split in split_dataset:
        for img_path in split_dataset[split]:
            image_name = img_path + "/lri_1refl_height_filtered/image_COMBINED.png"
            img_dict = {"license": 0, "file_name": image_name, "coco_url": "",
                        "height": tile_height, "width": tile_width, "data_captured": "", "id": img_id}
            images[split].append(img_dict)
            gt_file = Path(img_path) / "bbox.json"
            with open(gt_file) as file:
                detection_gt = json.load(file)  # List[Dict]
            for bbox_i in detection_gt:
                bbox_coord = bbox_i["bbox"]
                coco_box = kitt2coco(bbox_coord, img_width=tile_width, img_height=tile_height)
                box_area = coco_box[2] * coco_box[3]
                label_id = get_label_id(lookup_table, bbox_i["label"])
                if label_id >= 0:
                    # ignore the box which is not included in the label_def_file
                    annotations_dict = {"segmentation": [], "area": box_area, 'iscrowd': 0,
                                        'image_id': img_id, 'bbox': coco_box, 'category_id': label_id,
                                        'id': box_id}
                    annotations[split].append(annotations_dict)
                    box_id += 1
            img_id += 1

        json_file = dataset_dir / f"{output_file_suffix}_{split_file_name}_{split}.json"
        with json_file.open("w") as file:
            instances = {"annotations": annotations[split], "images": images[split], "categories": categories}
            json.dump(instances, file, indent=2)


if __name__ == "__main__":
    convert_ssigns_coco_format(**argument_parser())
