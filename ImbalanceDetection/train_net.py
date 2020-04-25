from detectron2.engine import TrainerBase, default_setup, launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
import torch
from typing import Any, Dict, List
import logging
import time
import os
from imbalancedetection.build import build_detector, build_gambler
from imbalancedetection.config import add_gambler_config
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.solver import build_lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from detectron2.layers import cat
import torch.nn.functional as F
from detectron2.config import set_global_cfg, global_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    inference_and_visualize,
    print_csv_format,
    verify_results,
)
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from collections import OrderedDict
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.events import get_event_storage
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def N_AK_H_W_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def reverse_N_AK_H_W_to_N_HWA_K(tensor, N, H, W, K):
    tensor = tensor.reshape(N, H, W, -1, K)  # Size=(N,H,W,A,K)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.view(N, -1, H, W)
    return tensor


def permute_all_cls_to_NHWAxFPN_K_and_concat(box_cls, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [N_AK_H_W_to_N_HWA_K(x, num_classes) for x in box_cls]
    # box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, num_classes)
    return box_cls


def reverse_permute_all_cls_to_N_HWA_K_and_concat(tensor, num_fpn_layers, N, H, W, num_classes=80):
    tensor = tensor.reshape(N, -1, num_classes) # (n,h*w*a,k)
    tensor = torch.chunk(tensor, num_fpn_layers, dim=1)  # todo not sure about this
    tensor_prime = [reverse_N_AK_H_W_to_N_HWA_K(x, N, H, W, num_classes) for x in tensor]
    return tensor_prime


def permute_all_weights_to_N_HWA_K_and_concat(weights, num_classes=80, normalize_w= False):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    weights_flattened = [N_AK_H_W_to_N_HWA_K(w, num_classes) for w in weights] # Size=(N,HWA,K)
    weights_flattened = [w + global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_TEMPERATURE for w in weights_flattened]
    if normalize_w is True:
        weights_flattened = [w / torch.sum(w, dim=[1,2], keepdim=True) for w in weights_flattened] #normalize by wxh only for now #todo experiment with normalizing across anchors -> distribute bets among scales maybe some are more important
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    weights_flattened = cat(weights_flattened, dim=1).reshape(-1, num_classes)
    return weights_flattened


def normalize_to_01(input):
    _max = torch.max(input)
    _min = torch.min(input)
    return (input-_min)/(_max-_min)


def find_max_location(tensor_in):
    return (tensor_in == torch.ones_like(tensor_in) * tensor_in.max()).nonzero()


def find_min_location(tensor_in):
    return (tensor_in == torch.ones_like(tensor_in) * tensor_in.min()).nonzero()


def prepare_loss_grid(loss, num_scales):
    loss = torch.chunk(loss, num_scales, dim=1)  # todo hard coded scales
    y_list = []
    for scale, _y in enumerate(loss):
        # print(torch.sum(_y))
        # storage.put_scalar("loss_per_scale/" + str(scale), torch.sum(_y))
        y_list.append(make_grid(_y, nrow=2, pad_value=1))
    loss_grid = torch.cat(y_list, dim=1)
    return loss_grid


def prepare_gt_grid(gt_classes, batch, num_scales, H, W, device):
    a = torch.ones(gt_classes.shape) * 0.5  # gray foreground by default
    a[gt_classes == -1] = 1  # white unmatched
    a[gt_classes == 80] = 0  # black background
    gt_classes = a.to(device)
    gt = gt_classes.reshape(batch, H, W, -1, 1)  # (n, h, w, anchors, c)
    gt = torch.chunk(gt, num_scales, dim=3)  # todo hard coded scales #todo [0] is wrong /GAMBLER_OUT_CHANNELS is also wrong
    gt_list = []
    for _gt in gt:
        _gt = _gt.squeeze(dim=3)
        _gt = _gt.permute(0, 3, 1, 2)
        gt_list.append(make_grid(_gt, nrow=2, pad_value=1))
    gt_grid = torch.cat(gt_list, dim=1)
    return gt_grid


def prepare_input_images(input_images, num_scales, device):
    pixel_mean = torch.Tensor(global_cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(global_cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    denormalizer = lambda x: (x * pixel_std) + pixel_mean  # todo maybe normalized image is better
    input_images = denormalizer(input_images)
    input_grid = make_grid(input_images / 255., nrow=2, pad_value=1)
    if num_scales > 1:
        input_grid = input_grid.repeat(1, num_scales, 1)
    return input_grid


def prepare_betting_map(betting_map, batch, num_scales, H, W, input_grid=None, heatmap_mode=True):
    # betting_map = normalize_to_01(betting_map)
    # betting_map = betting_map * 27500
    betting_map = betting_map[:, 0].reshape(batch, H, W, -1, 1)  # (n,w,h,a,c) #todo if weights are only per anchor and not per class
    bm = torch.chunk(betting_map, num_scales, dim=3) #todo hardcoded scales
    bm_list = []
    for scale, _bm in enumerate(bm):
        # Create heatmap image in red channel
        # storage.put_scalar("weights_per_scale/" + str(scale), torch.sum(_bm))
        _bm = _bm.squeeze(dim=3)
        _bm = _bm.permute(0, 3, 1, 2)
        if heatmap_mode is True:
            g_channel = torch.zeros_like(_bm)
            b_channel = torch.zeros_like(_bm)
            _bm = torch.cat((_bm, g_channel, b_channel), dim=1)
        bm_list.append(make_grid(_bm, nrow=2, pad_value=1))
    betting_map_grid = torch.cat(bm_list, dim=1)

    if input_grid is not None and heatmap_mode is True:
    # blend the heatmap with input
        cm = plt.get_cmap('jet')
        betting_map_grid_heatmap = cm(betting_map_grid[0, :, :].detach().cpu())
        blended = betting_map_grid_heatmap.transpose(2, 0, 1)[:3, :, :] * 0.5 + input_grid.cpu().numpy() * 0.5
        return blended
    if heatmap_mode is True:
        cm = plt.get_cmap('jet')
        betting_map_grid_heatmap = cm(betting_map_grid[0, :, :].detach().cpu())
        heatmap = betting_map_grid_heatmap.transpose(2, 0, 1)[:3, :, :]
        return heatmap

    return betting_map_grid.cpu().numpy()


def visualize_training(gt_classes, loss, betting_map, input_images, storage):
    device = torch.device(global_cfg.MODEL.DEVICE)
    anchor_scales = global_cfg.MODEL.ANCHOR_GENERATOR.SIZES
    if len(loss) > 1:
        raise Exception("The code still does not support the full FPN layers!")
    loss = loss[0] #todo: change for multiple fpn layers
    [n, _, h, w] = loss.shape

    # Prepare loss *****************************************************************************************************
    loss_grid = prepare_loss_grid(loss, len(anchor_scales[0]))

    # Prepare ground truth *********************************************************************************************
    gt_grid = prepare_gt_grid(gt_classes, n, len(anchor_scales[0]), h, w, device)

    # Prepare input images *********************************************************************************************
    input_grid = prepare_input_images(input_images, 1, device)

    # Prepare betting map **********************************************************************************************
    bets_and_input = prepare_betting_map(normalize_to_01(betting_map), n, len(anchor_scales[0]), h, w, input_grid=input_grid, heatmap_mode=False)

    all_vis = []
    all_vis.extend([bets_and_input, (normalize_to_01(loss_grid)).cpu().numpy(), input_grid.cpu().numpy()])
    vis = np.concatenate((bets_and_input, (normalize_to_01(loss_grid)).cpu().numpy(), input_grid.cpu().numpy()), axis=2) #todo: visualize gt as well
    storage.put_image("all", vis)
    for i, vis in enumerate(all_vis):
        all_vis[i] = vis.transpose(1, 2, 0)  # numpy images are (W,H,C)
    return all_vis


class GANTrainer(TrainerBase):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()

        # the .train() function sets the train_mode on
        self.gambler_model = build_gambler(cfg).train()
        self.detection_model = build_detector(cfg).train()

        # self.detection_checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        DetectionCheckpointer(self.detection_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
        if cfg.MODEL.GAMBLER_HEAD.LOAD_PRETRAINED_GAMBLER is True:
            DetectionCheckpointer(self.gambler_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.GAMBLER_HEAD.WEIGHTS, resume=True)

        self.gambler_optimizer = self.build_optimizer_gambler(cfg, self.gambler_model)
        self.detection_optimizer = self.build_optimizer_detector(cfg, self.detection_model)

        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            self.gambler_model = DistributedDataParallel(
                self.gambler_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            self.detection_model = DistributedDataParallel(
                self.detection_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True,
            )

        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        self.detection_scheduler = self.build_lr_scheduler(cfg, self.detection_optimizer)
        self.gambler_scheduler = self.build_lr_scheduler(cfg, self.gambler_optimizer)

        self.detection_checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.detection_model,
            cfg.OUTPUT_DIR,
            optimizer=self.detection_optimizer,
            scheduler=self.detection_scheduler,
        )

        gambler_model_loc = os.path.join(cfg.OUTPUT_DIR, "gambler_models")
        os.makedirs(gambler_model_loc, exist_ok=True)

        self.gambler_checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.gambler_model,
            gambler_model_loc,
            optimizer=self.gambler_optimizer,
            scheduler=self.gambler_scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.max_iter_gambler = self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_ITERATIONS
        self.max_iter_detector = self.cfg.MODEL.GAMBLER_HEAD.DETECTOR_ITERATIONS
        # current training iteration of the gambler
        self.iter_G = 0
        # current training iteration of detector
        self.iter_D = 0

        det_hooks = self.build_hooks(
            self.detection_model,
            self.detection_optimizer,
            self.detection_scheduler,
            self.detection_checkpointer,
        )
        self.register_hooks(det_hooks)
        gamb_hooks = self.build_hooks_gambler(
            self.gambler_model,
            self.gambler_optimizer,
            self.gambler_scheduler,
            self.gambler_checkpointer
        )

        self.register_hooks(gamb_hooks)
        self.gambler_loss_lambda = cfg.MODEL.GAMBLER_HEAD.GAMBLER_LAMBDA
        self.regression_loss_lambda = cfg.MODEL.GAMBLER_HEAD.REGRESSION_LAMBDA
        self.gambler_outside_lambda = cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTSIDE_LAMBDA
        self.vis_period = cfg.MODEL.GAMBLER_HEAD.VIS_PERIOD
        self.device = self.cfg.MODEL.DEVICE

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @classmethod
    def build_optimizer_gambler(cls, cfg, gambler_model) -> torch.optim.Optimizer:
        """
        Returns:
            torch.optim.Optimizer:

        It returns the optimizer for the gambler to be able to control gambler and detector separately
        """
        params: List[Dict[str, Any]] = []
        # todo ONLY SELECT THE GAMBLER PARAMETERS
        for key, value in gambler_model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.MODEL.GAMBLER_HEAD.BASE_LR
            weight_decay = cfg.MODEL.GAMBLER_HEAD.WEIGHT_DECAY
            if key.endswith("norm.weight") or key.endswith("norm.bias"):
                weight_decay = cfg.MODEL.GAMBLER_HEAD.WEIGHT_DECAY_NORM
            elif key.endswith(".bias"):
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.MODEL.GAMBLER_HEAD.BASE_LR * cfg.MODEL.GAMBLER_HEAD.BIAS_LR_FACTOR
                weight_decay = cfg.MODEL.GAMBLER_HEAD.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        gambler_optimizer = torch.optim.SGD(params, lr, momentum=cfg.MODEL.GAMBLER_HEAD.MOMENTUM)

        # logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
        # logger.info("Gambler Optimizer:\n{}".format(gambler_optimizer))
        return gambler_optimizer

    @classmethod
    def build_optimizer_detector(cls, cfg, detection_model) -> torch.optim.Optimizer:
        """
        Returns:
            torch.optim.Optimizer:

        It returns the optimizer for the detector to be able to control gambler and detector separately.
        It uses the old configs for the detector with the same naming conventions
        """
        params: List[Dict[str, Any]] = []
        for key, value in detection_model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if key.endswith("norm.weight") or key.endswith("norm.bias"):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key.endswith(".bias"):
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        detector_optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
        # logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
        # logger.info("Detector Optimizer:\n{}".format(detector_optimizer))
        return detector_optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        """
                Returns:
                    iterable

                It now calls :func:`detectron2.data.build_detection_train_loader`.
                Overwrite it if you'd like a different data loader.
                """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

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
        # if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        #     evaluator_list.append(
        #         SemSegEvaluator(
        #             dataset_name,
        #             distributed=True,
        #             num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        #             ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
        #             output_dir=output_folder,
        #         )
        #     )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        # if evaluator_type == "coco_panoptic_seg":
        #     evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # elif evaluator_type == "cityscapes":
        #     assert (
        #             torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesEvaluator(dataset_name)
        # elif evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_and_visualize(cls, cfg, detector, gambler, evaluators=None, mode="dataloader"):

        logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            test_data_loader = cls.build_test_loader(cfg, dataset_name) #todo changed to train loader cause it's not really testing!
            train_data_loader = cls.build_train_loader(cfg)

            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            inference_and_visualize(detector, gambler, train_data_loader, mode)

            from detectron2.utils.events import EventStorage
            with EventStorage(0) as storage:
                results_i = inference_on_dataset(detector, test_data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_hooks(self, model, optimizer, scheduler, checkpointer):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.DetectorLRScheduler(optimizer, scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        from imbalancedetection.modelling.unet import UNet
        from imbalancedetection.gambler_heads import GamblerHeads

        # Perform evaluation only if the model is a meta architecture
        # if comm.get_world_size() > 1:
        #     if not (isinstance(model.module, UNet) or isinstance(model.module, GamblerHeads)):
        #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        # elif comm.get_world_size() == 1:
        #     if not (isinstance(model, UNet) or isinstance(model, GamblerHeads)):
        #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    def build_hooks_gambler(self, model, optimizer, scheduler, checkpointer):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            # hooks.IterationTimer(),
            hooks.GamblerLRScheduler(optimizer, scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        from imbalancedetection.modelling.unet import UNet
        from imbalancedetection.gambler_heads import GamblerHeads

        # Perform evaluation only if the model is a meta architecture
        # if comm.get_world_size() > 1:
        #     if not (isinstance(model.module, UNet) or isinstance(model.module, GamblerHeads)):
        #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        # elif comm.get_world_size() == 1:
        #     if not (isinstance(model, UNet) or isinstance(model, GamblerHeads)):
        #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # if comm.is_main_process():
        #     # run writers in the end, so that evaluation metrics are written
        #     ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        if metrics_dict == {}:
            logging.debug(" There are no metrics to print!")
            return
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time/gambler_iter" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time/gambler_iter") for x in all_metrics_dict])
                self.storage.put_scalar("data_time/gambler_iter", data_time)
            elif "data_time/detector_iter" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time/detector_iter") for x in all_metrics_dict])
                self.storage.put_scalar("data_time/detector_iter", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            if "loss_gambler" in all_metrics_dict[0]:
                self.storage.put_scalar("loss_detector", metrics_dict["loss_detector"])
            else:
                total_losses_reduced = sum(loss for loss in metrics_dict.values())
                self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.detection_checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    def softmax_ce_gambler_loss(self, predictions, betting_map, gt_classes):

        # weighting the loss with the output of the gambler
        # todo: ignore for retinanet - dimension doesn't change
        loss_func = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        [B, C, _, _] = predictions.shape
        pred_class_logits = predictions.reshape(B, C, -1).detach()
        # Find places with highest CE
        betting_map = betting_map.squeeze().reshape(betting_map.shape[0], -1)

        # Regularize the betting map
        betting_map = betting_map / (torch.sum(betting_map, dim=1)).reshape(betting_map.shape[0], 1).expand(
            betting_map.shape)
        loss_gambler = -torch.mean(loss_func(pred_class_logits, gt_classes) * betting_map.reshape(betting_map.shape[0],
                                                                                                  -1)) * self.gambler_loss_lambda
        return loss_gambler

    @staticmethod
    def sigmoid_gambler_loss(pred_class_logits, weights, gt_classes, normalize_w=False, detach_pred=False, reduction="sum"):
        '''

        Args:
            pred_class_logits: list of tensors, each tensor is [batch, #class_categories * anchors per location, w, h]
            weights: [batch, #anchors per location/#classes/ classes*anchors per location, w, h]
            gt_classes: [batch, #all_anchors = w * h * anchors per location]
            normalize_w:
            detach_pred:

        Returns:

        '''
        [N, AK, H, W] = pred_class_logits[0].shape
        if detach_pred is True:
            pred_class_logits = [p.detach() for p in pred_class_logits]

        num_classes = global_cfg.MODEL.RETINANET.NUM_CLASSES

        pred_class_logits = permute_all_cls_to_NHWAxFPN_K_and_concat(
            pred_class_logits, num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        gt_classes = gt_classes.flatten() #todo next: change the shape of gt to the proper shape
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        p = torch.sigmoid(pred_class_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes_target, reduction="none") # (N x R, K)
        p_t = p * gt_classes_target + (1 - p) * (1 - gt_classes_target)

        mode = global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_LOSS_MODE
        alpha = global_cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        gamma = global_cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA

        if mode == "focal":
            gambler_loss = ce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * gt_classes_target + (1 - alpha) * (1 - gt_classes_target)
                gambler_loss = alpha_t * gambler_loss
        elif mode == "sigmoid":
            gambler_loss = ce_loss
        else:
            logging.error("No mode it selected for the retinanet loss!!")
            gambler_loss = None

        # ignore the invalid ids in loss
        valid_loss = torch.zeros_like(gambler_loss)  # todo: backprop
        valid_loss[valid_idxs, :] = gambler_loss[valid_idxs, :]

        gambler_loss = reverse_permute_all_cls_to_N_HWA_K_and_concat(valid_loss, 1, N, H, W, num_classes)  # N,AK,H,W_loss #todo num fpn layers

        if global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "B1HW":
            gambler_loss = [torch.sum(l, dim=[1], keepdim=True) for l in gambler_loss]  # aggregate over classes and anchors
            NAKHW_loss = [l.clone().detach() for l in gambler_loss]
            gambler_loss = permute_all_cls_to_NHWAxFPN_K_and_concat(gambler_loss, num_classes=1)
            # loss = [torch.sum(l, dim=[1,2], keepdim=True) for l in loss] # aggregate over classes and anchors #todo seperate classes and anchors
        elif global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BCHW":
            gambler_loss = [torch.sum(l, dim=[1], keepdim=True) for l in gambler_loss]  # aggregate over anchors
        elif global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BAHW":
            gambler_loss = [torch.sum(l, dim=[2], keepdim=True) for l in gambler_loss]  # aggregate over classes
        elif global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BCAHW":
            gambler_loss = gambler_loss  # do nothing

        storage = get_event_storage()
        with open(os.path.join(global_cfg.OUTPUT_DIR, "weights‌.csv"), "a") as my_csv:
            max_loss = []
            max_weights = []
            for i in range(N):
                assert len(NAKHW_loss) == 1, "csv write does not work for full fpn layer!"
                max_loss.append(NAKHW_loss[0][i, :, :, :].max().detach().cpu().numpy())
                max_weights.append(weights[i, :, :, :].max().detach().cpu().numpy())
                my_csv.write(
                    f"iteration: {str(storage.iter)}, image: {str(i)}, max weight: {max_weights[-1]},‌ max loss: {max_loss[-1]}, "
                    f"loss where weight is max: {NAKHW_loss[0][i, :, :, :].flatten()[weights[i, :, :, :].argmax()].item()}, weight where loss is max: {weights[i, :, :, :].flatten()[NAKHW_loss[0][i, :, :, :].argmax()].item()},"
                    f"weight argmax: {weights[i, :, :, :].argmax()}, loss argmax: {NAKHW_loss[0][i, :, :, :].argmax()}\n")

        storage.put_scalar("sum/max_loss", np.sum(np.array(max_loss)))  # sum over the batch
        storage.put_scalar("sum/max_weight", np.sum(np.array(max_weights)))  # sum over the batch

        weights = permute_all_weights_to_N_HWA_K_and_concat([weights], 1, normalize_w)  # todo hardcoded 3: scales
        gambler_loss = -weights * gambler_loss

        if reduction == "mean":
            gambler_loss = gambler_loss.mean()
        elif reduction == "sum":
            gambler_loss = gambler_loss.sum()

        with open(os.path.join(global_cfg.OUTPUT_DIR, "weights‌.csv"), "a") as my_csv:
            my_csv.write(f"sum loss after weighting: {gambler_loss}\n")

        loss_before_weighting = [loss.sum() for loss in NAKHW_loss]

        return NAKHW_loss, sum(loss_before_weighting)/max(1, num_foreground), gambler_loss, weights.data

    def calc_log_metrics(self, betting_map, weights, loss_dict, loss_gambler, loss_before_weighting, data_time):

        loss_dict.update({"loss_box_reg": loss_dict["loss_box_reg"] * self.regression_loss_lambda})
        loss_gambler = loss_gambler * self.gambler_loss_lambda
        loss_dict.update({"loss_gambler": loss_gambler})
        loss_dict.update({"loss_before_weighting": loss_before_weighting})
        loss_detector = loss_dict["loss_box_reg"] + loss_dict["loss_cls"] - self.gambler_outside_lambda * loss_dict["loss_gambler"]
        loss_dict.update({"loss_detector": loss_detector})

        self._detect_anomaly(loss_detector, loss_dict)
        self._detect_anomaly(loss_gambler, loss_dict)

        loss_dict["gambler_bets/sum"] = torch.sum(betting_map)
        loss_dict["gambler_bets/max"] = torch.max(betting_map)
        loss_dict["gambler_bets/mean"] = torch.mean(betting_map)
        loss_dict["gambler_bets/median"] = torch.median(betting_map)
        loss_dict["visualized weights/sum"] = torch.sum(weights)
        loss_dict["visualized weights/max"] = torch.max(weights)
        loss_dict["visualized weights/mean"] = torch.mean(weights)
        loss_dict["visualized weights/median"] = torch.median(weights)
        loss_dict["data_time/gambler_iter"] = data_time

        return loss_dict

    @staticmethod
    def prepare_input_gambler(input_images, generated_output):
        stride = 16  # todo: stride depends on feature map layer
        input_images = F.interpolate(input_images, scale_factor=1 / stride, mode='bilinear')
        sigmoid_predictions = torch.sigmoid(generated_output['pred_class_logits'][0])

        if global_cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-0.5, 0.5]:
            scaled_prob = (sigmoid_predictions - 0.5)
            input_images = (input_images / 256.0)
        elif global_cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-128, 128]:
            scaled_prob = (sigmoid_predictions - 0.5) * 256

        # concatenate along the channel
        gambler_input = torch.cat((input_images, scaled_prob), dim=1)  # (N,3+AK,H,W)
        return gambler_input, input_images

    def run_step(self):
        """
        Overwrites the run_step() function of SimpleTrainer(TrainerBase) to be compatible with GAN learning
        """
        logger = setup_logger(output=global_cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)

        assert self.gambler_model.training, "[GANTrainer] gambler model was changed to eval mode!"
        assert self.detection_model.training, "[GANTrainer] detector model was changed to eval mode!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        input_images, generated_output, gt_classes, loss_dict = self.detection_model(data)
        gambler_input, input_images = self.prepare_input_gambler(input_images, generated_output)

        if self.iter_G < self.max_iter_gambler:
            logger.info(f"Iteration {self.iter} in Gambler")
            betting_map = self.gambler_model(gambler_input)  # (N,AK,H,W)
            y, loss_before_weighting, loss_gambler, weights = self.sigmoid_gambler_loss(generated_output['pred_class_logits'], betting_map, gt_classes, normalize_w=self.cfg.MODEL.GAMBLER_HEAD.NORMALIZE, detach_pred=True)

            if self.vis_period > 0 and self.storage.iter % self.vis_period == 0:
                visualize_training(gt_classes, y, weights, input_images, self.storage)

            metrics_dict = self.calc_log_metrics(betting_map, weights, loss_dict, loss_gambler, loss_before_weighting, data_time)

            self.gambler_optimizer.zero_grad()
            metrics_dict["loss_gambler"].backward()
            self.gambler_optimizer.step()
            self.iter_G += 1
            if self.iter_G == self.max_iter_gambler:
                logger.info("Finished training Gambler")

        elif self.iter_D < self.max_iter_detector:
            logger.info(f"Iteration {self.iter} in Detector")
            betting_map = self.gambler_model(gambler_input)  # (N,AK,H,W)
            y, loss_before_weighting, loss_gambler, weights = self.sigmoid_gambler_loss(generated_output['pred_class_logits'], betting_map, gt_classes, normalize_w=self.cfg.MODEL.GAMBLER_HEAD.NORMALIZE, detach_pred=False)

            if self.vis_period > 0 and self.storage.iter % self.vis_period == 0:
                visualize_training(gt_classes, y, weights, input_images, self.storage)

            metrics_dict = self.calc_log_metrics(betting_map, weights, loss_dict, loss_gambler, loss_before_weighting, data_time)

            self.detection_optimizer.zero_grad()
            metrics_dict["loss_detector"].backward()
            torch.nn.utils.clip_grad_norm_(self.detection_model.parameters(), 10)
            self.detection_optimizer.step()
            self.iter_D += 1
            if self.iter_D == self.max_iter_detector:
                logger.info("Finished training Detector")
                self.iter_G = 0
                self.iter_D = 0

        else:
            metrics_dict = {}
            logger.debug("Neither D_iter nor G_iter! Debugging with fixed detector!")
            self.iter_G = 0
            self.iter_D = 0
        self._write_metrics(metrics_dict)


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

    if args.eval_only:
        detector = build_detector(cfg) #.train .test
        DetectionCheckpointer(detector, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = GANTrainer.test(cfg, detector)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    elif args.eval_visualize:
        detector = build_detector(cfg)
        gambler = build_gambler(cfg)
        DetectionCheckpointer(detector, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        DetectionCheckpointer(gambler, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.GAMBLER_HEAD.WEIGHTS, resume=args.resume
        )
        res = GANTrainer.test_and_visualize(cfg, detector, gambler, mode=args.source)
        return res

    trainer = GANTrainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )