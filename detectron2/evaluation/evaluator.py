# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

from detectron2.utils.comm import is_main_process


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


def inference_and_visualize(detector, gambler, data_loader, evaluator):
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
    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt
    import os

    def visualize_training(gt_classes, betting_map, input_images):
        # storage = get_event_storage()
        device = torch.device(global_cfg.MODEL.DEVICE)
        anchor_scales = global_cfg.MODEL.ANCHOR_GENERATOR.SIZES

        # [n, _, w, h] = y.shape
        # y = torch.chunk(y, len(anchor_scales[0]), dim=1)  # todo hard coded scales #todo [0] is wrong
        # y_list = []
        # for _y in y:
        #     y_list.append(make_grid(_y, nrow=2, pad_value=1))
        # loss_grid = torch.cat(y_list, dim=1)

        # a = torch.ones(gt_classes.shape) * 0.5  # gray foreground by default
        # a[gt_classes == -1] = 1  # white unmatched
        # a[gt_classes == 80] = 0  # black background
        # gt_classes = a.to(device)
        #
        # gt = gt_classes.reshape(n, w, h, -1, 1)  # (n, w, h, anchors, c)
        #
        # gt = torch.chunk(gt, len(anchor_scales[0]), dim=3)  # todo hard coded scales #todo [0] is wrong
        # gt_list = []
        # for _gt in gt:
        #     _gt = _gt.squeeze(dim=3)
        #     _gt = _gt.permute(0, 3, 1, 2)
        #     gt_list.append(make_grid(_gt, nrow=2, pad_value=1))
        # gt_scales = torch.cat(gt_list, dim=1)

        # save_image(gt / 1., os.path.join(global_cfg.OUTPUT_DIR, str(self.iter) + "iter_gt.jpg"), nrow=2)

        pixel_mean = torch.Tensor(global_cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
        pixel_std = torch.Tensor(global_cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
        denormalizer = lambda x: (x * pixel_std) + pixel_mean
        input_images = denormalizer(input_images)

        input_grid = make_grid(input_images / 255., nrow=2)
        # save_image(input_images/255., global_cfg.OUTPUT_DIR + "/test.jpg", nrow=2)
        bm = torch.chunk(betting_map, global_cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS, dim=1)  # todo hard coded scales
        bm_list = []

        for _bm in bm:
            # Create heatmap image in red channel
            g_channel = torch.zeros_like(_bm)
            b_channel = torch.zeros_like(_bm)
            _bm = torch.cat((_bm, g_channel, b_channel), dim=1)
            bm_list.append(make_grid(_bm, nrow=2))
        betting_map_grid = torch.cat(bm_list, dim=1)

        # blended = betting_map_grid*0.5 + input_grid*0.5
        # storage.put_image("blended", blended)
        cm = plt.get_cmap('jet')
        betting_map_grid_heatmap = cm(betting_map_grid[0, :, :].detach().cpu())
        blended = betting_map_grid_heatmap[:, :, :3] * 0.5 + input_grid.cpu().numpy().transpose(1, 2, 0) * 0.5
        fig = plt.figure()
        plt.imshow(blended)
        fig.savefig(os.path.join(global_cfg.OUTPUT_DIR, "AP_" + str(storage.iter) + ".pdf"))
        plt.imshow(betting_map_grid.cpu().numpy().transpose(1, 2, 0))
        plt.imshow(input_grid.cpu().numpy().transpose(1, 2, 0))
        plt.show()
        fig = plt.figure()
        plt.bar(bins, height=a[:, 1].astype(float), color='#F6CD61')
        plt.xticks(bins, np.array(class_names)[ind_sorted], rotation=90, fontsize=5)
        storage = get_event_storage()
        storage.put_fig("AP", fig)
        fig.savefig(os.path.join(global_cfg.OUTPUT_DIR, "AP_" + str(storage.iter) + ".pdf"))
        plt.close('all')

        # storage.put_image("blended", blended)

        # loss_grid = make_grid(y, nrow=2)  # todo loss_grid range
        # storage.put_image("gt_bettingmap", torch.cat((gt_scales, betting_map_grid), dim=1))
        # all = torch.cat((input_grid, loss_grid), dim=1)
        # storage.put_image("loss", loss_grid)
        # storage.put_image("input", input_grid)

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(detector), inference_context(gambler), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            from detectron2.config import global_cfg
            input_images, generated_output, gt_classes, loss_dict = detector(inputs)
            stride = 16
            # input_images = input_images[:, :, ::stride, ::stride]
            input_images = F.interpolate(input_images, scale_factor=1 / stride,
                                         mode='bilinear')  # todo: stride depends on feature map layer
            # input_images = F.max_pool2d(input_images, kernel_size=1, stride=16)
            # concatenate along the channel
            sigmoid_predictions = torch.sigmoid(generated_output['pred_class_logits'][0])
            # print(f"min logits: {torch.min(sigmoid_predictions)} max logits: {torch.max(sigmoid_predictions)}")
            scaled_prob = (sigmoid_predictions - 0.5) * 256
            # print(f"min predictions: {torch.min(scaled_prob)} max predictions: {torch.max(scaled_prob)}")
            # print(f"min image input: {torch.min(input_images)} max input images: {torch.max(input_images)}")
            gambler_input = torch.cat((input_images, scaled_prob), dim=1)
            betting_map = gambler(gambler_input)
            visualize_training(gt_classes, betting_map, input_images)
            # _, _, _, outputs = detector(inputs)

            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, loss_dict)

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
