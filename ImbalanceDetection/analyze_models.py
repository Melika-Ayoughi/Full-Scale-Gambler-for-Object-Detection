from detectron2.engine import default_setup, launch, default_argument_parser
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from imbalancedetection.build import build_detector, build_gambler
from imbalancedetection.config import add_gambler_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import set_global_cfg, global_cfg
from detectron2.evaluation import (
    load_old_inference_results,
    Analyzer,
    print_csv_format,
)
from collections import OrderedDict
from train_net import GANTrainer
from detectron2.utils.events import EventStorage
import numpy as np
import os
import cv2
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from itertools import islice
import matplotlib.pyplot as plt
from fvcore.common.file_io import PathManager
import json
import math
import copy
from collections import defaultdict
import statistics
import torch
from ImbalanceDetection.imbalancedetection.gambler_heads import list_N_AK_H_W_to_NsumHWA_K

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


def get_topk_different_imgs(imgid_to_ap_ours, imgid_to_ap_base, k, order="desc"):
    imgid_to_diff_aps = {}
    for imgid in imgid_to_ap_ours.keys():
        imgid_to_diff_aps[imgid] = imgid_to_ap_ours[imgid] - imgid_to_ap_base[imgid]

    if order == "desc":
        sorted_imgid_to_diff_aps = {k: v for k, v in sorted(imgid_to_diff_aps.items(), key=lambda item: item[1], reverse=True)}
    elif order == "asc":
        sorted_imgid_to_diff_aps = {k: v for k, v in sorted(imgid_to_diff_aps.items(), key=lambda item: item[1])}
    else:
        raise Exception("not a possible order option!")

    print(f"sorted mapping of image ids to AP{sorted_imgid_to_diff_aps}")

    return dict(islice(sorted_imgid_to_diff_aps.items(), k))


def plot_aps(cfg, args, ours, baseline, sort="frequency"):

    catname_to_ap = {}
    for (name, ap) in ours['bbox'].items():
        # print(f"name:{name}:ap:{ap}")
        if name.startswith("AP-"):
            catname_to_ap[name.strip("AP-")] = ap
        else:
            catname_to_ap[name] = ap

    catname_to_ap_baseline = {}
    for (name, ap) in baseline['bbox'].items():
        # print(f"name:{name}:ap:{ap}")
        if name.startswith("AP-"):
            catname_to_ap_baseline[name.strip("AP-")] = -ap
        else:
            catname_to_ap_baseline[name] = -ap

    # frequency in train set is important
    if sort == "frequency":
        histogram = np.load(os.path.join(cfg.OUTPUT_DIR, f'histogram_{cfg.DATASETS.TRAIN[0]}.npy'))
        ind_sorted = np.argsort(histogram)[::-1]
        extra = len(catname_to_ap) - cfg.MODEL.RETINANET.NUM_CLASSES
        # shift the original ind_sorted by extra indices and add them in the beginning
        ind_sorted = list(range(extra)) + list(ind_sorted + extra)

    elif sort == "size":
        catid_to_abs_areas = defaultdict(list)
        catid_to_rel_areas = defaultdict(list)
        train_data = list(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
        for img in train_data:
            for ann in img['annotations']:
                abs_area = ann['bbox'][2] * ann['bbox'][3]
                rel_area = abs_area / (img['height'] * img['width'])
                catid_to_abs_areas[ann['category_id']].append(abs_area)
                catid_to_rel_areas[ann['category_id']].append(rel_area)

        catid_to_abs_area = {catid: statistics.mean(abs_areas) for (catid, abs_areas) in catid_to_abs_areas.items()}
        catid_to_rel_area = {catid: statistics.mean(rel_areas) for (catid, rel_areas) in catid_to_rel_areas.items()}

        sorted_catid_to_abs_area = {k: v for k, v in
                                    sorted(catid_to_abs_area.items(), key=lambda item: item[1], reverse=True)}
        sorted_catid_to_rel_area = {k: v for k, v in
                                    sorted(catid_to_rel_area.items(), key=lambda item: item[1], reverse=True)}

        extra = len(catname_to_ap) - cfg.MODEL.RETINANET.NUM_CLASSES

        ind_sorted = np.asarray(list(sorted_catid_to_rel_area))
        ind_sorted = list(range(extra)) + list(ind_sorted + extra)

    elif sort == "ap":
        cat_to_ap_base = copy.deepcopy(catname_to_ap_baseline)

        ind = 0
        catid_to_ap_base = {}
        for (catname, ap) in cat_to_ap_base.items():
            if not catname.startswith("AP"):
                catid_to_ap_base[ind] = -ap
                ind += 1
        
        sorted_catid_to_ap_base = {k: v for k, v in sorted(catid_to_ap_base.items(), key=lambda item: float('-inf') if math.isnan(item[1]) else item[1], reverse=True)}
        extra = len(catname_to_ap) - cfg.MODEL.RETINANET.NUM_CLASSES
        ind_sorted = np.asarray(list(sorted_catid_to_ap_base))
        ind_sorted = list(range(extra)) + list(ind_sorted + extra)

    elif sort == "class_entropy":

        catid_to_countinstances = dict.fromkeys(range(cfg.MODEL.RETINANET.NUM_CLASSES),0)
        catid_to_imgs = defaultdict(set)
        train_data = list(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
        for img in train_data:
            for ann in img['annotations']:
                catid_to_countinstances[ann['category_id']] += 1
                catid_to_imgs[ann['category_id']].add(img['image_id'])

        catid_to_entropies = {catid: count/len(catid_to_imgs[catid]) for (catid, count) in catid_to_countinstances.items()}
        sorted_catid_to_entropies = {k: v for k, v in sorted(catid_to_entropies.items(), key=lambda item: item[1], reverse=True)}

        extra = len(catname_to_ap) - cfg.MODEL.RETINANET.NUM_CLASSES
        ind_sorted = np.asarray(list(sorted_catid_to_entropies))
        ind_sorted = list(range(extra)) + list(ind_sorted + extra)

    else:
        ind_sorted = list(range(len(catname_to_ap)))

    fig = plt.figure(figsize=(30, 8))
    plt.bar(range(len(catname_to_ap)), width=0.5, height=np.array(list(catname_to_ap.values()))[ind_sorted],
            color='#3DA4AB')
    plt.bar(range(len(catname_to_ap_baseline)), width=0.5, height=np.array(list(catname_to_ap_baseline.values()))[ind_sorted], color='red')
    plt.xticks(range(len(catname_to_ap)), np.array(list(catname_to_ap.keys()))[ind_sorted], rotation=90, fontsize=10)
    fig.savefig(os.path.join(args.output, f"by{sort}_ap_compare_{args.dir_ours.split('/')[-2]}_{args.dir_baseline.split('/')[-2]}.png"))
    plt.close()

    fig = plt.figure(figsize=(30, 8))
    catname_to_ap_diffs = {name: (catname_to_ap[name] + catname_to_ap_baseline[name]) for (name, ap) in
                           catname_to_ap.items()}
    plt.bar(range(len(catname_to_ap)), width=0.5, height=np.array(list(catname_to_ap_diffs.values()))[ind_sorted], color='green')  # alpha=0.3
    plt.xticks(range(len(catname_to_ap)), np.array(list(catname_to_ap.keys()))[ind_sorted], rotation=90, fontsize=10)
    fig.savefig(os.path.join(args.output, f"by{sort}_ap_diffs_{args.dir_ours.split('/')[-2]}_{args.dir_baseline.split('/')[-2]}.png"))
    plt.close()


def setup(args):
    cfg = get_cfg()
    add_gambler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
    set_global_cfg(cfg)
    return cfg


def main(args):
    cfg = setup(args)
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")

    detector_ours = build_detector(cfg)
    gambler_ours = build_gambler(cfg)
    DetectionCheckpointer(detector_ours, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        os.path.join(args.dir_ours, "model_0099999.pth"), resume=args.resume
    )
    # DetectionCheckpointer(gambler_ours, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     os.path.join(args.dir_ours, "gambler_models/model_final.pth"), resume=args.resume
    # )

    detector_baseline = build_detector(cfg)
    gambler_baseline = build_gambler(cfg)
    DetectionCheckpointer(detector_baseline, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        os.path.join(args.dir_baseline, "model_final.pth"), resume=args.resume
    )
    # DetectionCheckpointer(gambler_baseline, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.GAMBLER_HEAD.WEIGHTS, resume=args.resume
    # )

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)

    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        test_data_loader = GANTrainer.build_test_loader(cfg, dataset_name)
        train_data_loader = GANTrainer.build_train_loader(cfg)

        evaluator_ours = GANTrainer.build_evaluator(cfg, dataset_name, os.path.join(args.dir_ours, "inference"))
        evaluator_baseline = GANTrainer.build_evaluator(cfg, dataset_name, os.path.join(args.dir_baseline, "inference"))
        analyzer = Analyzer(args.output)
        # visualize_inference(detector, gambler, train_data_loader, args.source)

        if args.per_image_ap:
            # import json
            # file_path = os.path.join("./output/gambler/2_full_coco/smoothing/fromscratch/focal_noimage/inference",
            #                          "coco_instances_results_mine.json")
            # with PathManager.open(file_path, "r") as f:
            #     imgid_to_pred_ours = json.load(f)
            #
            # from collections import defaultdict
            # _imgid_to_pred = defaultdict(list)
            # for prediction in imgid_to_pred_ours:
            #     _imgid_to_pred[prediction["image_id"]].append(prediction)
            # imgid_to_pred_ours = _imgid_to_pred
            #
            # file_path = os.path.join(
            #     "./output/retinanet_baselines/simple_retinanet/simple_retina_fullcoco/all_layers_3scales/inference",
            #     "coco_instances_results.json")
            # with PathManager.open(file_path, "r") as f:
            #     imgid_to_pred_base = json.load(f)
            #
            # from collections import defaultdict
            # _imgid_to_pred_base = defaultdict(list)
            # for prediction in imgid_to_pred_base:
            #     _imgid_to_pred_base[prediction["image_id"]].append(prediction)
            # imgid_to_pred_base = _imgid_to_pred_base
            #
            # # imgid_to_topk_APs={k: v for k, v in sorted(imgid_to_topk_APs.items(), key=lambda item: item[1])}
            # args.conf_threshold = 0.2
            # a = dict(islice(imgid_to_topk_APs.items(), 200))
            #
            # for dic in tqdm.tqdm(dicts):
            #     if dic["image_id"] in a.keys():
            #         img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
            #         basename = os.path.basename(dic["file_name"])
            #         predictions_ours = create_instances(imgid_to_pred_ours[dic["image_id"]], img.shape[:2])
            #         # predictions_ours.remove("pred_masks")
            #         vis = Visualizer(img, metadata)
            #         vis_pred = vis.draw_instance_predictions(predictions_ours).get_image()
            #
            #         vis = Visualizer(img, metadata)
            #         vis_gt = vis.draw_dataset_dict(dic).get_image()
            #
            #         predictions_base = create_instances(imgid_to_pred_base[dic["image_id"]], img.shape[:2])
            #         # predictions_base.remove("pred_masks")
            #         vis = Visualizer(img, metadata)
            #         vis_pred_baseline = vis.draw_instance_predictions(predictions_base).get_image()
            #
            #         concat = np.concatenate((vis_pred, vis_pred_baseline, vis_gt), axis=1)
            #         cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])

            with EventStorage(0) as storage:
                analyzer.find_ap_per_img(detector_ours, test_data_loader, evaluator_ours)
                imgid_to_ap_ours, imgid_to_pred_ours = analyzer.imgid_to_AP, analyzer.imgid_to_pred
                analyzer.save()
    
                analyzer.find_ap_per_img(detector_baseline, test_data_loader, evaluator_baseline)
                imgid_to_ap_base, imgid_to_pred_base = analyzer.imgid_to_AP, analyzer.imgid_to_pred
                analyzer.save()
    
                imgid_to_topk_APs = get_topk_different_imgs(imgid_to_ap_ours, imgid_to_ap_base, k=args.k, order=args.sort_order)
    
                if args.sort_order == "desc":
                    logger.info(f"The topk most different APs where our model outperformed")
                elif args.sort_order == "asc":
                    logger.info(f"The topk most different APs where baseline outperformed")
    
                for dic in tqdm.tqdm(dicts):
                    if dic["image_id"] in imgid_to_topk_APs.keys():
    
                        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
                        basename = os.path.basename(dic["file_name"])
                        predictions_ours = create_instances(imgid_to_pred_ours[dic["image_id"]], img.shape[:2])
                        predictions_ours.remove("pred_masks")
                        vis = Visualizer(img, metadata)
                        vis_pred = vis.draw_instance_predictions(predictions_ours).get_image()
    
                        vis = Visualizer(img, metadata)
                        vis_gt = vis.draw_dataset_dict(dic).get_image()
    
                        predictions_base = create_instances(imgid_to_pred_base[dic["image_id"]], img.shape[:2])
                        predictions_base.remove("pred_masks")
                        vis = Visualizer(img, metadata)
                        vis_pred_baseline = vis.draw_instance_predictions(predictions_base).get_image()
    
                        concat = np.concatenate((vis_pred, vis_pred_baseline, vis_gt), axis=1)
                        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])

        if args.compare_aps:
            with EventStorage(0) as storage:
                if os.path.isfile(os.path.join(args.dir_baseline, "results.json")):
                    with PathManager.open(os.path.join(args.dir_baseline, "results.json"), "r") as f:
                        results_baseline = json.load(f)
                else:
                    results_baseline = load_old_inference_results(test_data_loader, evaluator_baseline)
                    with PathManager.open(os.path.join(args.dir_baseline, "results.json"), "w") as f:
                        json.dump(results_baseline, f, indent=2)

                if os.path.isfile(os.path.join(args.dir_ours, "results.json")):
                    with PathManager.open(os.path.join(args.dir_ours, "results.json"), "r") as f:
                        results_ours = json.load(f)
                else:
                    results_ours = load_old_inference_results(test_data_loader, evaluator_ours)
                    with PathManager.open(os.path.join(args.dir_ours, "results.json"), "w") as f:
                        json.dump(results_ours, f, indent=2)



            plot_aps(cfg, args, results_ours, results_baseline, sort=args.sort)
    
            # results_ours_dict = OrderedDict()
            # results_baseline_dict = OrderedDict()
            #
            # results_ours_dict[dataset_name] = results_ours
            # results_baseline_dict[dataset_name] = results_baseline
            #
            # logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            # print_csv_format(results_ours_dict)
            # print_csv_format(results_baseline_dict)

        if args.with_gambler:

            with EventStorage(0) as storage:
                with torch.no_grad():
                    # with open(os.path.join(args.output, "statistics_gambler_fg.csv"), "a") as my_csv:
                    #     my_csv.write(f"img, "
                    #                  f"the highest score predicted, "
                    #                  f"gambler score before normalization, "
                    #                  f"gambler score after normalization, "
                    #                  f"Gt id, "
                    #                  f"predicted class id, "
                    #                  f"loss\n")
                    # with open(os.path.join(args.output, "statistics_gambler_highloss.csv"), "a") as my_csv:
                    #     my_csv.write(f"img, "
                    #                  f"the highest score predicted, "
                    #                  f"gambler score before normalization, "
                    #                  f"gambler score after normalization, "
                    #                  f"Gt id, "
                    #                  f"predicted class id, "
                    #                  f"loss\n")
                    # with open(os.path.join(args.output, "statistics_baseline_fg.csv"), "a") as my_csv:
                    #     my_csv.write(f"img, "
                    #                  f"the highest score predicted, "
                    #                  f"Gt id, "
                    #                  f"predicted class id, "
                    #                  f"loss\n")
                    # with open(os.path.join(args.output, "statistics_baseline_highloss.csv"), "a") as my_csv:
                    #     my_csv.write(f"img, "
                    #                  f"the highest score predicted, "
                    #                  f"Gt id, "
                    #                  f"predicted class id, "
                    #                  f"loss\n")
                    with torch.no_grad():
                        with open(os.path.join(args.output, f"{args.dir_ours.split('/')[-2]}_one_image_allanchors.csv"), "a") as my_csv:
                            my_csv.write(f"img, "
                                         f"the highest score predicted, "
                                         f"gambler score before normalization, "
                                         f"gambler score after normalization, "
                                         f"Gt id, "
                                         f"predicted class id, "
                                         f"loss\n")

                    for img_indx, inputs in enumerate(train_data_loader):
                        # if img_indx > 2:
                        #     break
                        input_images, generated_output, gt_classes, _, loss_dict = detector_ours(inputs)
                        mask = torch.ones_like(gt_classes)
                        gambler_loss_dict, weights, betting_map = gambler_ours(input_images,
                                                                               generated_output['pred_class_logits'],
                                                                               gt_classes,
                                                                               mask,
                                                                               detach_pred=False)  # (N,AK,H,W)




                        gt_classes = gt_classes.flatten() #torch.Size([37632])
                        pred_class_logits = list_N_AK_H_W_to_NsumHWA_K(generated_output['pred_class_logits'], cfg.MODEL.RETINANET.NUM_CLASSES) #torch.Size([37632, 80])
                        # pred_class_sigmoids = (torch.sigmoid(pred_class_logits) - 0.5) * 256
                        pred_class_sigmoids = torch.sigmoid(pred_class_logits)

                        betting_map = list_N_AK_H_W_to_NsumHWA_K(betting_map, num_classes=1).flatten() # torch.Size([37632])
                        weights = weights.flatten() # torch.Size([37632])
                        loss_before_weighting = list_N_AK_H_W_to_NsumHWA_K(gambler_loss_dict['NAKHW_loss'], num_classes=1).flatten() # torch.Size([37632])

                        # foreground_idxs = (gt_classes >= 0) & (gt_classes != cfg.MODEL.RETINANET.NUM_CLASSES)
                        # filtered_anchor_ind = foreground_idxs
                        # print(f"number of filtered anchors to be displayed: {filtered_anchor_ind.sum()}")
                        #
                        # f_betting_map = betting_map[filtered_anchor_ind]
                        # f_weights = weights[filtered_anchor_ind]
                        # f_gt_classes = gt_classes[filtered_anchor_ind]
                        # f_pred_class_sigmoids = pred_class_sigmoids[filtered_anchor_ind, :]
                        # f_loss_before_weighting = loss_before_weighting[filtered_anchor_ind]

                        #  for over all chosen anchors
                        for pred_class_sigmoid, gambler_score_bef, gambler_score_aft, matched_gt, loss in zip(pred_class_sigmoids, betting_map, weights, gt_classes, loss_before_weighting):
                            with open(os.path.join(args.output, f"{args.dir_ours.split('/')[-2]}_one_image_allanchors.csv"), "a") as my_csv:
                                my_csv.write(f"{inputs[0]['file_name'].split('/')[-1]}, "
                                             f"{pred_class_sigmoid.max()}, "
                                             f"{gambler_score_bef}, "
                                             f"{gambler_score_aft}, "
                                             f"{matched_gt}, "
                                             f"{pred_class_sigmoid.argmax()}, "
                                             f"{loss}\n")

                        # foreground_idxs = (gt_classes >= 0) & (gt_classes != cfg.MODEL.RETINANET.NUM_CLASSES)
                        # filtered_anchor_ind = (loss_before_weighting >= (0.1 * loss_before_weighting.max()))
                        # print(f"number of filtered anchors to be displayed: {filtered_anchor_ind.sum()}")
                        #
                        # f_betting_map = betting_map[filtered_anchor_ind]
                        # f_weights = weights[filtered_anchor_ind]
                        # f_gt_classes = gt_classes[filtered_anchor_ind]
                        # f_pred_class_sigmoids = pred_class_sigmoids[filtered_anchor_ind, :]
                        # f_loss_before_weighting = loss_before_weighting[filtered_anchor_ind]
                        #
                        # #  for over all chosen anchors
                        # for pred_class_sigmoid, gambler_score_bef, gambler_score_aft, matched_gt, loss in zip(
                        #         f_pred_class_sigmoids, f_betting_map, f_weights, f_gt_classes, f_loss_before_weighting):
                        #     with open(os.path.join(args.output, "statistics_gambler_highloss.csv"), "a") as my_csv:
                        #         my_csv.write(f"{inputs[0]['file_name'].split('/')[-1]}, "
                        #                      f"{pred_class_sigmoid.max()}, "
                        #                      f"{gambler_score_bef}, "
                        #                      f"{gambler_score_aft}, "
                        #                      f"{matched_gt}, "
                        #                      f"{pred_class_sigmoid.argmax()}, "
                        #                      f"{loss}\n")

                        ##############################################################################################

                        # input_images, generated_output, gt_classes, loss_dict = detector_baseline(inputs)
                        #
                        # gt_classes = gt_classes.flatten()  # torch.Size([37632])
                        # pred_class_logits = list_N_AK_H_W_to_NsumHWA_K(generated_output['pred_class_logits'],
                        #                                                cfg.MODEL.RETINANET.NUM_CLASSES)  # torch.Size([37632, 80])
                        # # pred_class_sigmoids = (torch.sigmoid(pred_class_logits) - 0.5) * 256
                        # pred_class_sigmoids = torch.sigmoid(pred_class_logits)
                        # # loss_before_weighting = list_N_AK_H_W_to_NsumHWA_K(gambler_loss_dict['NAKHW_loss'],
                        # #                                                    num_classes=1).flatten()  # torch.Size([37632])
                        #
                        # foreground_idxs = (gt_classes >= 0) & (gt_classes != cfg.MODEL.RETINANET.NUM_CLASSES)
                        # filtered_anchor_ind = foreground_idxs
                        # print(f"number of filtered anchors to be displayed: {filtered_anchor_ind.sum()}")
                        #
                        # # f_betting_map = betting_map[filtered_anchor_ind]
                        # # f_weights = weights[filtered_anchor_ind]
                        # f_gt_classes = gt_classes[filtered_anchor_ind]
                        # f_pred_class_sigmoids = pred_class_sigmoids[filtered_anchor_ind, :]
                        # # f_loss_before_weighting = loss_before_weighting[filtered_anchor_ind]
                        #
                        # #  for over all chosen anchors
                        # for pred_class_sigmoid, matched_gt in zip(f_pred_class_sigmoids, f_gt_classes):
                        #     with open(os.path.join(args.output, "statistics_baseline_fg.csv"), "a") as my_csv:
                        #         my_csv.write(f"{inputs[0]['file_name'].split('/')[-1]}, "
                        #                      f"{pred_class_sigmoid.max()}, "
                        #                      f"{matched_gt}, "
                        #                      f"{pred_class_sigmoid.argmax()}\n")

                        # # foreground_idxs = (gt_classes >= 0) & (gt_classes != cfg.MODEL.RETINANET.NUM_CLASSES)
                        # filtered_anchor_ind = (loss_before_weighting >= (0.1 * loss_before_weighting.max()))
                        # print(f"number of filtered anchors to be displayed: {filtered_anchor_ind.sum()}")
                        #
                        # f_betting_map = betting_map[filtered_anchor_ind]
                        # f_weights = weights[filtered_anchor_ind]
                        # f_gt_classes = gt_classes[filtered_anchor_ind]
                        # f_pred_class_sigmoids = pred_class_sigmoids[filtered_anchor_ind, :]
                        # f_loss_before_weighting = loss_before_weighting[filtered_anchor_ind]
                        #
                        # #  for over all chosen anchors
                        # for pred_class_sigmoid, gambler_score_bef, gambler_score_aft, matched_gt, loss in zip(
                        #         f_pred_class_sigmoids, f_betting_map, f_weights, f_gt_classes, f_loss_before_weighting):
                        #     with open(os.path.join(args.output, "statistics_gambler_highloss.csv"), "a") as my_csv:
                        #         my_csv.write(f"{inputs[0]['file_name'].split('/')[-1]}, "
                        #                      f"{pred_class_sigmoid.max()}, "
                        #                      f"{gambler_score_bef}, "
                        #                      f"{gambler_score_aft}, "
                        #                      f"{matched_gt}, "
                        #                      f"{pred_class_sigmoid.argmax()}, "
                        #                      f"{loss}\n")
                        break

        if args.vis_bets:
            from train_net import visualize_training_
            with torch.no_grad():
                with EventStorage(0) as storage:
                    for img_indx, inputs in enumerate(train_data_loader):
                        # if img_indx > 2:
                        #     break
                        input_images, generated_output, gt_classes, _, loss_dict = detector_ours(inputs)
                        mask = torch.ones_like(gt_classes)
                        gambler_loss_dict, weights, betting_map = gambler_ours(input_images,
                                                                               generated_output['pred_class_logits'],
                                                                               gt_classes,
                                                                               mask,
                                                                               detach_pred=False)  # (N,AK,H,W)
                        visualize_training_(gt_classes.clone().detach(), gambler_loss_dict["NAKHW_loss"],
                                            weights, input_images, storage)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--sort-order", default="desc", type=str, help="if desc: best of our model, asc: best of baseline")
    parser.add_argument("--k", default=100, type=int, help="top k results")
    parser.add_argument("--per-image-ap", action="store_true", help="if present: finds the most different images")
    parser.add_argument("--compare-aps", action="store_true", help="if present: draws the diagram of difference of ap")
    parser.add_argument("--dir-ours", required=True, help="input directory of our model")
    parser.add_argument("--dir-baseline", required=True, help="input directory of baseline model")
    parser.add_argument("--sort", default="frequency", type=str, help="frequency, size, ap, class_entropy")
    parser.add_argument("--with-gambler", action="store_true", help="if present: analyzes bettingmaps & ... ")
    parser.add_argument("--vis-bets", action="store_true", help="if present: visualizes bettingmaps & ... ")
    args = parser.parse_args()

    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )