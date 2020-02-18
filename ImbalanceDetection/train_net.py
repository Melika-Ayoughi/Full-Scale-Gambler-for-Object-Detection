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
    print_csv_format,
    verify_results,
)
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from collections import OrderedDict
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.events import get_event_storage

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

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        # todo check that checkpoints of gambler and detector are not overwritten by eachother

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
        # gamb_hooks = self.build_hooks(
        #     self.gambler_model,
        #     self.gambler_optimizer,
        #     self.gambler_scheduler,
        #     self.gambler_checkpointer
        # )

        # self.register_hooks(gamb_hooks)
        # if comm.is_main_process():
        #     self.writer = SummaryWriter(self.cfg.OUTPUT_DIR)
        self.gambler_loss_lambda = cfg.MODEL.GAMBLER_HEAD.GAMBLER_LAMBDA
        self.regression_loss_lambda = cfg.MODEL.GAMBLER_HEAD.REGRESSION_LAMBDA
        self.vis_period = cfg.MODEL.GAMBLER_HEAD.VIS_PERIOD

    @classmethod
    def build_optimizer_gambler(self, cfg, gambler_model) -> torch.optim.Optimizer:
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
    def build_optimizer_detector(self, cfg, detection_model) -> torch.optim.Optimizer:
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
    def build_train_loader(self, cfg):
        """
                Returns:
                    iterable

                It now calls :func:`detectron2.data.build_detection_train_loader`.
                Overwrite it if you'd like a different data loader.
                """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

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
            hooks.LRScheduler(optimizer, scheduler),
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
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time/gambler" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time/gambler") for x in all_metrics_dict])
                self.storage.put_scalar("data_time/gambler", data_time)
            elif "data_time/detector" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time/detector") for x in all_metrics_dict])
                self.storage.put_scalar("data_time/detector", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
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

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

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

    def test(self, cfg, model, evaluators=None):
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
            data_loader = self.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = self.build_evaluator(cfg, dataset_name)
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

    def visualize_training(self, betting_map, input_images):
        storage = get_event_storage()

        for batch in range(betting_map.shape[0]):
            bet_Img = betting_map[batch, :, :, :].squeeze()[None, :, :]
            input_Img = input_images[batch, :, :, :].squeeze()
            temp = torch.cat((bet_Img, bet_Img, bet_Img), dim=0)
            both = torch.cat((temp, input_Img), 2)
            storage.put_image("input_betting_map", both)
            break # only the first image in the batch

    def run_step(self):
        """
        Overwrites the run_step() function of SimpleTrainer(TrainerBase) to be compatible with GAN learning
        """
        logger = setup_logger(output=global_cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)

        assert self.gambler_model.training, "[GANTrainer] gambler model was changed to eval mode!"
        assert self.detection_model.training, "[GANTrainer] detector model was changed to eval mode!"

        # 1 forward pass
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        input_images, generated_output, gt_classes, loss_dict = self.detection_model(data)

        stride = 16
        # input_images = input_images[:, :, ::stride, ::stride]
        input_images = F.interpolate(input_images, scale_factor=1 / stride) # todo: stride depends on feature map layer
        # input_images = F.max_pool2d(input_images, kernel_size=1, stride=16)
        # concatenate along the channel
        gambler_input = torch.cat((input_images, generated_output['pred_class_logits'][0]), dim=1)

        if self.iter_G < self.max_iter_gambler:
            logger.info(f"Iteration {self.iter} in Gambler")

            # self.detection_model.eval()
            # self.gambler_model.train()

            betting_map = self.gambler_model(gambler_input.detach())
            logger.debug(f"Gambler bets: max:{torch.max(betting_map)} min:{torch.min(betting_map)} mean: :{torch.mean(betting_map)}")

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(betting_map, input_images)

            # weighting the loss with the output of the gambler
            # todo: ignore for retinanet - dimension doesn't change
            loss_func = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
            [B,C,_,_] = generated_output['pred_class_logits'][0].shape
            pred_class_logits = generated_output['pred_class_logits'][0].reshape(B, C, -1).detach()
            # Find places with highest CE
            betting_map = betting_map.squeeze().reshape(betting_map.shape[0], -1)

            # Regularize the betting map
            betting_map = betting_map / (torch.sum(betting_map, dim=1)).reshape(betting_map.shape[0], 1).expand(betting_map.shape)
            loss_gambler = -torch.mean(loss_func(pred_class_logits, gt_classes) * betting_map.reshape(betting_map.shape[0], -1)) * self.gambler_loss_lambda

            loss_dict.update({"loss_box_reg": loss_dict["loss_box_reg"] * self.regression_loss_lambda})
            loss_detector = sum(loss for loss in loss_dict.values()) - loss_gambler
            self._detect_anomaly(loss_detector, loss_dict)
            loss_dict.update({"loss_gambler": loss_gambler})
            self._detect_anomaly(loss_gambler, loss_dict)
            metrics_dict = loss_dict
            metrics_dict["data_time/gambler_iter"] = data_time

            self.gambler_optimizer.zero_grad()
            loss_gambler.backward()
            self.gambler_optimizer.step()

            self.iter_G += 1
            if self.iter_G == self.max_iter_gambler:
                logger.info("Finished training Gambler")

            # self.detection_model.train()
        else:
            logger.info(f"Iteration {self.iter} in Detector")

            # self.detection_model.train()
            # self.gambler_model.eval()

            betting_map = self.gambler_model(gambler_input)
            # logger.debug(f"Gambler bets: {betting_map}")
            logger.debug(f"Gambler bets: max:{torch.max(betting_map)} min:{torch.min(betting_map)} mean: :{torch.mean(betting_map)}")

            # weighting the loss with the output of the gambler
            loss_func = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
            [B, C, _, _] = generated_output['pred_class_logits'][0].shape
            pred_class_logits = generated_output['pred_class_logits'][0].reshape(B, C, -1)
            # Find places with highest CE
            betting_map = betting_map.squeeze().reshape(betting_map.shape[0], -1)

            # Regularize the betting map
            betting_map = betting_map / (torch.sum(betting_map, dim=1)).reshape(betting_map.shape[0], 1).expand(betting_map.shape)
            loss_gambler = -torch.mean(loss_func(pred_class_logits, gt_classes) * betting_map.reshape(betting_map.shape[0], -1)) * self.gambler_loss_lambda

            loss_dict.update({"loss_box_reg": loss_dict["loss_box_reg"] * self.regression_loss_lambda})
            loss_detector = sum(loss for loss in loss_dict.values()) - loss_gambler
            self._detect_anomaly(loss_detector, loss_dict)
            loss_dict.update({"loss_gambler": loss_gambler})
            self._detect_anomaly(loss_gambler, loss_dict)
            metrics_dict = loss_dict
            metrics_dict["data_time/detector_iter"] = data_time
            self._write_metrics(metrics_dict)

            self.detection_optimizer.zero_grad()
            loss_detector.backward()
            torch.nn.utils.clip_grad_norm_(self.detection_model.parameters(), 10)
            self.detection_optimizer.step()
            self.iter_D += 1
            if self.iter_D == self.max_iter_detector:
                logger.info("Finished training Detector")
                self.iter_G = 0
                self.iter_D = 0

            # self.gambler_model.train()

        self._write_metrics(metrics_dict)


def setup(args):
    cfg = get_cfg()
    add_gambler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
    set_global_cfg(cfg)
    return cfg


def main(args):
    cfg = setup(args)

    # todo for now ignore this evaluation only
    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

    trainer = GANTrainer(cfg)
    # todo resume function maybe for the pretrained models we need it
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