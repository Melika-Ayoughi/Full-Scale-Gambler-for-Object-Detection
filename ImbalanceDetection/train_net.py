from detectron2.engine import TrainerBase, default_setup, launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
import torch
from typing import Any, Dict, List
import logging
import time

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
from detectron2.evaluation import verify_results
from detectron2.layers import cat
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GANTrainer(TrainerBase):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()

        # the .train() function sets the train_mode on
        self.gambler_model = build_gambler(cfg, 83, 80).train() #todo
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
                self.detection_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
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
        # self.gambler_checkpointer = DetectionCheckpointer(
        #     # Assume you want to save checkpoints together with logs/statistics
        #     self.gambler_model,
        #     cfg.OUTPUT_DIR,
        #     optimizer=self.gambler_optimizer,
        #     scheduler=self.gambler_scheduler,
        # )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        # self.register_hooks(self.build_hooks())

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

        logger = logging.getLogger(__name__)
        logger.info("Gambler Optimizer:\n{}".format(gambler_optimizer))
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
        logger.info("Detector Optimizer:\n{}".format(detector_optimizer))
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

    # def build_hooks(self, model, optimizer, scheduler, checkpointer):
    #     """
    #     Build a list of default hooks, including timing, evaluation,
    #     checkpointing, lr scheduling, precise BN, writing events.
    #
    #     Returns:
    #         list[HookBase]:
    #     """
    #     cfg = self.cfg.clone()
    #     cfg.defrost()
    #     cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
    #
    #     ret = [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(optimizer, scheduler),
    #         hooks.PreciseBN(
    #             # Run at the same freq as (but before) evaluation.
    #             cfg.TEST.EVAL_PERIOD,
    #             self.model,
    #             # Build a new data loader to not affect training
    #             self.build_train_loader(cfg),
    #             cfg.TEST.PRECISE_BN.NUM_ITER,
    #         )
    #         if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(model)
    #         else None,
    #     ]
    #
    #     # Do PreciseBN before checkpointer, because it updates the model and need to
    #     # be saved by checkpointer.
    #     # This is not always the best: if checkpointing has a different frequency,
    #     # some checkpoints may have more precise statistics than others.
    #     if comm.is_main_process():
    #         ret.append(hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
    #
    #     def test_and_save_results():
    #         self._last_eval_results = self.test(self.cfg, self.model)
    #         return self._last_eval_results
    #
    #     # Do evaluation after checkpointer, because then if it fails,
    #     # we can use the saved checkpoint to debug.
    #     ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
    #
    #     if comm.is_main_process():
    #         # run writers in the end, so that evaluation metrics are written
    #         ret.append(hooks.PeriodicWriter(self.build_writers()))
    #     return ret

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
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

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

    def test(self):
        # can check DefaultTrainer
        raise NotImplementedError

    def run_step(self):

        """
        Overwrites the run_step() function of SimpleTrainer(TrainerBase) to be compatible with GAN learning
        """
        # assert self.model.training, "[DefaultTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """

        """
        If your want to do something with the losses, you can wrap the model.
        """
        # todo make a function for the forward pass that is duplicate
        # todo losses need to be dictionaries

        for _ in range(self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_ITERATIONS):

            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start

            # A forward pass of the whole model

            input_images, generated_output, proposals, losses = self.detection_model(data)

            input_images = F.max_pool2d(input_images, kernel_size=1, stride=8)
            # if proposals[0].has("gt_boxes"):
            #     assert proposals[0].has("gt_classes")
            #     gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            betting_map = self.gambler_model(generated_output)

            # weighting the loss with the output of the gambler
            weighted_loss = torch.nn.CrossEntropyLoss(weight=betting_map, reduction="none") # todo: test, does this work?
            loss_gambler = weighted_loss(fake["pred_class_logits"], gt_classes)

            ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
            loss_detector = ce_loss(fake["pred_class_logits"], gt_classes) # todo: for visualization purposes

            self.gambler_optimizer.zero_grad()
            loss_gambler.backward()
            self.gambler_optimizer.step()

        for _ in self.cfg.MODEL.GAMBLER_HEAD.DETECTOR_ITERATIONS:

            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            # A forward pass of the whole model (maybe I need to change the structure)
            fake, proposals, losses = self.detection_model(data)

            if proposals[0].has("gt_boxes"):
                assert proposals[0].has("gt_classes")
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            betting_map = self.gambler_model(fake)
            # weighting the loss with the output of the gambler
            weighted_loss = torch.nn.CrossEntropyLoss(weight=betting_map,
                                                      reduction="none")  # todo: test, does this work?
            loss_gambler = weighted_loss(fake["pred_class_logits"], gt_classes)

            ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
            loss_detector = ce_loss(fake["pred_class_logits"], gt_classes) - loss_gambler # for visualization purposes

            self.detection_optimizer.zero_grad()
            loss_detector.backward()
            self.detection_optimizer.step()

        # todo write and keep the losses, detect anomaly
        # loss_dict = self.model(data)
        # losses = sum(loss for loss in loss_dict.values())
        # self._detect_anomaly(losses, loss_dict)
        #
        # metrics_dict = loss_dict
        # metrics_dict["data_time"] = data_time
        # self._write_metrics(metrics_dict)


def setup(args):
    cfg = get_cfg()
    add_gambler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
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