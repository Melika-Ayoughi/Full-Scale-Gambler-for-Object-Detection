# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

from detectron2.data import DatasetCatalog
from detectron2.utils.comm import is_main_process
from itertools import chain
from detectron2.config import global_cfg
from detectron2.data import detection_utils as utils
import tqdm
import matplotlib.pyplot as plt

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            from detectron2.config import global_cfg
            if global_cfg.MODEL.GAMBLER_ON is True:
                _, _, _, outputs = model(inputs)
            else:
                outputs = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_and_visualize(detector, gambler, data_loader=None, mode="dataloader"):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    import torch.nn.functional as F
    from train_net import GANTrainer, visualize_training
    import os
    from detectron2.utils.events import EventStorage
    from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
    from detectron2.engine import hooks

    # periodic_writer = hooks.PeriodicWriter([TensorboardXWriter(global_cfg.OUTPUT_DIR)])
    os.makedirs(os.path.join(global_cfg.OUTPUT_DIR, "images"), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("start visualization")

    def output(vis, filepath):
        print("Saving to {} ...".format(filepath))
        # vis.save(filepath)
        plt.imsave(filepath, vis)

    if mode == "dataloader":
        assert data_loader is not None, "dataloader should not be none in dataloader mode"
        with EventStorage(0) as storage:
            with torch.no_grad():
                for idx, inputs in tqdm.tqdm(enumerate(data_loader)):
                    input_images, generated_output, gt_classes, loss_dict = detector(inputs)
                    stride = 16
                    input_images = F.interpolate(input_images, scale_factor=1 / stride, mode='bilinear')  # todo: stride depends on feature map layer
                    sigmoid_predictions = torch.sigmoid(generated_output['pred_class_logits'][0])
                    scaled_prob = (sigmoid_predictions - 0.5) * 256
                    gambler_input = torch.cat((input_images, scaled_prob), dim=1)
                    betting_map = gambler(gambler_input)

                    y, loss_before_weighting, loss_gambler, weights = GANTrainer.sigmoid_gambler_loss(generated_output['pred_class_logits'], betting_map, gt_classes, normalize_w=global_cfg.MODEL.GAMBLER_HEAD.NORMALIZE)
                    vis = visualize_training(gt_classes, y, weights, input_images, storage)
                    output(vis, os.path.join(global_cfg.OUTPUT_DIR, "images", str(idx) + ".jpg"))
                    # for writer in periodic_writer._writers:
                    #     writer.write()
                    storage.step()
                    torch.cuda.synchronize()
                # for writer in periodic_writer._writers:
                #     writer.close()
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in global_cfg.DATASETS.TRAIN]))
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            input_images, generated_output, gt_classes, loss_dict = detector(img(2, 0, 1))
            stride = 16
            input_images = F.interpolate(input_images, scale_factor=1 / stride,
                                         mode='bilinear')  # todo: stride depends on feature map layer
            sigmoid_predictions = torch.sigmoid(generated_output['pred_class_logits'][0])
            scaled_prob = (sigmoid_predictions - 0.5) * 256
            gambler_input = torch.cat((input_images, scaled_prob), dim=1)
            betting_map = gambler(gambler_input)

            y, loss_before_weighting, loss_gambler, weights = GANTrainer.sigmoid_gambler_loss(generated_output['pred_class_logits'], betting_map, gt_classes, normalize_w=global_cfg.MODEL.GAMBLER_HEAD.NORMALIZE)
            visualize_training(gt_classes, y, weights, input_images)
            torch.cuda.synchronize()


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
