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
from collections import defaultdict
import copy
import os
import json
from pathlib import Path

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


class Analyzer:
    def __init__(self, output_dir) -> None:
        super().__init__()
        self._imgid_to_pred = defaultdict(list)
        self._imgid_to_AP = {}
        self._output_dir = Path(output_dir)

    def reset(self):
        self._imgid_to_pred.clear()
        self._imgid_to_AP.clear()

    def save(self):
        os.makedirs(self._output_dir, exist_ok=True)
        file_path = self._output_dir / "imgid_to_predictions.json"
        with file_path.open("w") as f:
            json.dump(self._imgid_to_pred, f, indent=2)

        file_path = self._output_dir / "imgid_to_AP.json"
        with file_path.open("w") as f:
            json.dump(self._imgid_to_AP, f, indent=2)

        # with PathManager.open(file_path, "r") as f:
        #     predictions = json.load(f)
        self.reset()

    def find_ap_per_img(self, model, data_loader, evaluator):
        num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        total = len(data_loader)  # inference data loader must have a fixed length
        evaluator.reset()

        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                print(f"loading image{inputs[0]['file_name']}")
                from detectron2.config import global_cfg
                if global_cfg.MODEL.GAMBLER_ON is True:
                    _, _, _, outputs = model(inputs)
                else:
                    outputs = model(inputs)
                torch.cuda.synchronize()
                evaluator.process(inputs, outputs)

                res = evaluator.evaluate()
                for prediction in evaluator._coco_results:
                    self._imgid_to_pred[prediction["image_id"]].append(prediction)

                self._imgid_to_AP[inputs[0]['image_id']] = res['bbox']['AP']
                evaluator.reset()

    @property
    def imgid_to_pred(self):
        return copy.deepcopy(self._imgid_to_pred)

    @property
    def imgid_to_AP(self):
        return copy.deepcopy(self._imgid_to_AP)


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


def load_old_inference_results(data_loader, evaluator):
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
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    evaluator.reset()
    #instad of process: evaluator.process(inputs, outputs), load predictions from files

    results = evaluator.evaluate_from_file()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def visualize_inference(detector, gambler, data_loader=None, mode="dataloader"):
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
    from train_net import visualize_training_, visualize_per_image
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
                    if idx > 3:
                        break

                    input_images, generated_output, gt_classes, loss_dict = detector(inputs)
                    gambler_loss_dict, weights, betting_map = gambler(input_images,
                                                                      generated_output['pred_class_logits'],
                                                                      gt_classes,
                                                                      detach_pred=True)  # (N,AK,H,W)

                    visualize_per_image(inputs, gt_classes.clone().detach(), gambler_loss_dict["NAKHW_loss"],
                                        weights, input_images, storage)
                    visualize_training_(gt_classes.clone().detach(), gambler_loss_dict["NAKHW_loss"],
                                        weights, input_images, storage)

                    # for i, vis in enumerate(all_vis):
                    #     # save .png to get rid of jpeg artifacts
                    #     output(vis, os.path.join(global_cfg.OUTPUT_DIR, "images", inputs[0]["file_name"].rsplit('/', 1)[1].split('.', 1)[0] + '_' + str(i) +'.png'))
                    # for writer in periodic_writer._writers:
                    #     writer.write()
                    storage.step()
                    torch.cuda.synchronize()
                # for writer in periodic_writer._writers:
                #     writer.close()



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
