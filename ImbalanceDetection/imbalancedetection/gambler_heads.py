import logging
import torch
from torch import nn
from .build import GAMBLER_HEAD_REGISTRY
from .modelling.unet import UNet, UnetGenerator, LayeredUnet
from .modelling.pre_post_models import PreGamblerPredictions, PreGamblerImage, PostGamblerPredictions
from detectron2.utils.events import get_event_storage
import os
import numpy as np
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.config import global_cfg

logger = logging.getLogger(__name__)


def get_loss_upper_bound(nakhw, N):
    max_loss = torch.empty(N, 5)
    assert len(nakhw) == 5, "only works with 5 fpn layers"

    for i, layer in enumerate(nakhw):  # torch.Size([8, 3, 80, 5, 5])
        while len(layer.shape) > 1:
            layer, _ = layer.max(dim=1, keepdim=False)  # torch.Size([8])
        max_loss[:, i] = layer.data
    max_loss, _ = max_loss.max(dim=1, keepdim=False)  # torch.Size([8, 5]) -> torch.Size([8])
    # print(max_loss, torch.sum(max_loss))
    return torch.sum(max_loss)


def N_AK_H_W_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    if len(tensor.shape) == 4:
        N, _, H, W = tensor.shape  # N,A,K,H,W
    elif len(tensor.shape) == 5:
        N, _, _, H, W = tensor.shape  # N,A,K,H,W
    else:
        Exception("wrong dimensionality!")

    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def reverse_N_AK_H_W_to_N_HWA_K(tensor, N, H, W, K):
    tensor = tensor.reshape(N, H, W, -1, K)  # Size=(N,H,W,A,K)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.view(N, -1, H, W)
    return tensor


def reverse_N_A_K_H_W_to_N_HWA_K(tensor, N, H, W, K):
    tensor = tensor.reshape(N, H, W, -1, K)  # Size=(N,H,W,A,K)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    return tensor


def list_N_AK_H_W_to_NsumHWA_K(box_cls, num_classes=80):
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


def reverse_list_N_AK_H_W_to_NsumHWA_K(tensor, num_fpn_layers, N, H, W, num_classes=80):
    tensor = tensor.reshape(N, -1, num_classes)  # (n,h*w*a,k)
    tensor = torch.chunk(tensor, num_fpn_layers, dim=1)
    tensor_prime = [reverse_N_AK_H_W_to_N_HWA_K(x, N, H, W, num_classes) for x in tensor]
    return tensor_prime


def reverse_list_N_A_K_H_W_to_NsumHWA_K_(tensor, in_layers, N, H, W, num_scale=3, num_classes=80):
    tensor = tensor.reshape(N, -1, num_classes)  # (n,h*w*a,k)
    if len(in_layers) == 1:  # 1 fpn layer
        assert isinstance(H, int)
        tensor = torch.chunk(tensor, len(in_layers), dim=1)
        tensor_prime = [reverse_N_A_K_H_W_to_N_HWA_K(t, N, H, W, num_classes) for t in tensor]
    else:  # multiple fpn layers
        assert isinstance(H, list)
        tensor = torch.split(tensor, [h * w * num_scale for h, w in zip(H, W)], dim=1)
        tensor_prime = [reverse_N_A_K_H_W_to_N_HWA_K(t, N, h, w, num_classes) for t, h, w in zip(tensor, H, W)]
    return tensor_prime


def calc_cls_loss(pred,
                  gt_target,
                  mode: str = "none",
                  alpha: float = -1,
                  gamma: float = 2,
):

    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, gt_target, reduction="none")  # (N x R, K)
    p_t = p * gt_target + (1 - p) * (1 - gt_target)

    if mode == "focal":
        cls_loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * gt_target + (1 - alpha) * (1 - gt_target)
            cls_loss = alpha_t * cls_loss
    elif mode == "sigmoid":
        cls_loss = ce_loss
        # if alpha >= 0:
        #     alpha_t = alpha * gt_classes_target + (1 - alpha) * (1 - gt_classes_target)
        #     cls_loss = alpha_t * cls_loss
    else:
        raise Exception("No mode it selected for the retinanet loss!!")

    return cls_loss


def calc_gambler_loss(valid_loss,
                      weights,
                      N,
                      H,
                      W,
                      in_layers,
                      gambler_heads,

                      normalize_w: bool = True,
                      num_classes: int = 80,
                      gambler_output: str = "L_BAHW",
                      gamma: float = 1.0):
    """

    Args:
        cfg: config file
        valid_loss: tensor, classification loss filtered only for valid classes (fg +‌bg)
        weights: list(Tensor), betting map or output of the gambler
        N: int, batch size
        H: list(heights)
        W: list (widths)

    Returns:

    """
    if in_layers is None:
        in_layers = [80, 40, 20, 10, 5]

    if gambler_output == "B1HW":
        # gambler loss: ‌N,AK,H,W
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss, in_layers, N, H, W,
                                                        num_classes)
        # aggregate over classes and anchors
        cls_loss = [torch.sum(l, dim=[1, 2])[:, None, :, :] for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=1)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat(global_cfg, [weights], num_classes=1,
                                                                          normalize_w=normalize_w)
    elif gambler_output == "BCHW":
        # gambler loss: ‌N,AK,H,W
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss, in_layers, N, H, W,
                                                        num_classes)
        # aggregate over anchors
        cls_loss = [torch.sum(l, dim=[1], keepdim=True) for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=num_classes)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat(global_cfg, [weights], num_classes=num_classes,
                                                                          normalize_w=normalize_w)
    elif gambler_output == "BAHW":
        # gambler loss: ‌N,AK,H,W
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss, in_layers, N, H, W,
                                                        num_classes)
        # aggregate over classes
        cls_loss = [torch.sum(l, dim=[2], keepdim=True) for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=1)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat(global_cfg, [weights], num_classes=1,
                                                                          normalize_w=normalize_w)
    elif gambler_output == "BCAHW":
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss, in_layers, N, H, W,
                                                        num_classes)
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=num_classes)
    elif gambler_output == "L_BCAHW":
        # gambler loss: ‌N,AK,H,W
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss,
                                                        in_layers,
                                                        N,
                                                        [80, 40, 20, 10, 5],
                                                        [80, 40, 20, 10, 5],
                                                        num_scale=len(global_cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]),
                                                        num_classes=num_classes)
        NAKHW_loss = [l.clone().detach().data for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=num_classes)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat_(global_cfg, weights, num_classes=num_classes,
                                                                           normalize_w=normalize_w)
    elif gambler_output == "L_BAHW":
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss,
                                                        in_layers,
                                                        N,
                                                        H,
                                                        W,
                                                        num_scale=len(global_cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]),
                                                        num_classes=num_classes)
        # aggregate over classes
        cls_loss = [torch.sum(l, dim=[2]) for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=1)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat_(global_cfg, weights, num_classes=1,
                                                                           normalize_w=normalize_w)
    elif gambler_output == "L_B1HW":
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss,
                                                        in_layers,
                                                        N,
                                                        H,
                                                        W,
                                                        num_scale=len(global_cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]),
                                                        num_classes=num_classes)
        # aggregate over classes and anchors
        cls_loss = [torch.sum(l, dim=[1, 2])[:, None, :, :] for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=1)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat_(global_cfg, weights, num_classes=1,
                                                                           normalize_w=normalize_w)
    elif gambler_output == "L_BAHW_extendtobatch":
        cls_loss = reverse_list_N_A_K_H_W_to_NsumHWA_K_(valid_loss,
                                                        in_layers,
                                                        N,
                                                        H,
                                                        W,
                                                        num_scale=len(global_cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]),
                                                        num_classes=num_classes)
        # aggregate over classes
        cls_loss = [torch.sum(l, dim=[2]) for l in cls_loss]
        NAKHW_loss = [l.clone().detach() for l in cls_loss]
        cls_loss = list_N_AK_H_W_to_NsumHWA_K(cls_loss, num_classes=1)
        weights = gambler_heads.permute_all_weights_to_N_HWA_K_and_concat_(global_cfg, weights, num_classes=1,
                                                                           normalize_w=normalize_w)

    gambler_loss = -(weights ** gamma) * cls_loss
    gambler_loss = gambler_loss.sum()

    return gambler_loss, NAKHW_loss, weights


calc_cls_loss_jit = torch.jit.script(calc_cls_loss)
# calc_gambler_loss_jit = torch.jit.script(calc_gambler_loss)


class GamblerHeads(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_IN_CHANNELS
        self.out_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS
        self.bilinear = cfg.MODEL.GAMBLER_HEAD.BILINEAR_UPSAMPLING

    @staticmethod
    def permute_all_weights_to_N_HWA_K_and_concat(cfg, weights, num_classes=80, normalize_w=False):
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
        weights_flattened = [N_AK_H_W_to_N_HWA_K(w, num_classes) for w in weights]  # Size=(N,HWA,K)
        weights_flattened = [w + cfg.MODEL.GAMBLER_HEAD.GAMBLER_TEMPERATURE for w in weights_flattened]
        if normalize_w is True:
            weights_flattened = [w / torch.sum(w, dim=[1, 2], keepdim=True) for w in weights_flattened]
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        weights_flattened = cat(weights_flattened, dim=1).reshape(-1, num_classes)
        return weights_flattened

    @staticmethod
    def permute_all_weights_to_N_HWA_K_and_concat_(cfg, weights, num_classes=80, normalize_w=False):
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
        weights_flattened = [N_AK_H_W_to_N_HWA_K(w, num_classes) for w in weights]  # Size=(N,HWA,K)
        weights_flattened = [w + cfg.MODEL.GAMBLER_HEAD.GAMBLER_TEMPERATURE for w in weights_flattened]
        if normalize_w is True:
            sum_all_layers = 0
            for w in weights_flattened:
                if cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "L_BAHW_extendtobatch":
                    sum_all_layers = sum_all_layers + torch.sum(w, dim=[0, 1, 2], keepdim=True)
                else:
                    sum_all_layers = sum_all_layers + torch.sum(w, dim=[1, 2], keepdim=True)

            weights_flattened = [w / sum_all_layers for w in weights_flattened]  # normalize by anchor, class and layer
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        weights_flattened = cat(weights_flattened, dim=1).reshape(-1, num_classes)
        return weights_flattened


@GAMBLER_HEAD_REGISTRY.register()
class UnetGambler(GamblerHeads):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gambler = UNet(self.in_channels, self.out_channels, bilinear=self.bilinear)
        self.to(self.device)

    def forward(self, input):
        return self.gambler(input)

    def gambler_loss(self, pred_class_logits, weights, gt_classes, normalize_w=False, detach_pred=False, reduction="sum"):
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

        num_classes = self.cfg.MODEL.RETINANET.NUM_CLASSES

        pred_class_logits = list_N_AK_H_W_to_NsumHWA_K(
            pred_class_logits, num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        gt_classes = gt_classes.flatten()
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        p = torch.sigmoid(pred_class_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes_target, reduction="none") # (N x R, K)
        p_t = p * gt_classes_target + (1 - p) * (1 - gt_classes_target)

        mode = self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_LOSS_MODE
        alpha = self.cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        gamma = self.cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA

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
        valid_loss = torch.zeros_like(gambler_loss)  # todo: check backprop
        valid_loss[valid_idxs, :] = gambler_loss[valid_idxs, :]

        gambler_loss = reverse_list_N_AK_H_W_to_NsumHWA_K(valid_loss, 1, N, H, W, num_classes)  # N,AK,H,W_loss

        if self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "B1HW":
            gambler_loss = [torch.sum(l, dim=[1], keepdim=True) for l in gambler_loss]  # aggregate over classes and anchors
            NAKHW_loss = [l.clone().detach() for l in gambler_loss]
            gambler_loss = list_N_AK_H_W_to_NsumHWA_K(gambler_loss, num_classes=1)
            # loss = [torch.sum(l, dim=[1,2], keepdim=True) for l in loss] # aggregate over classes and anchors #todo seperate classes and anchors
        elif self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BCHW":
            gambler_loss = [torch.sum(l, dim=[1], keepdim=True) for l in gambler_loss]  # aggregate over anchors
        elif self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BAHW":
            gambler_loss = [torch.sum(l, dim=[2], keepdim=True) for l in gambler_loss]  # aggregate over classes
        elif self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "BCAHW":
            gambler_loss = gambler_loss  # do nothing

        storage = get_event_storage()
        with open(os.path.join(self.cfg.OUTPUT_DIR, "weights‌.csv"), "a") as my_csv:
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

        weights = self.permute_all_weights_to_N_HWA_K_and_concat([weights], 1, normalize_w)
        gambler_loss = -weights * gambler_loss

        if reduction == "mean":
            gambler_loss = gambler_loss.mean()
        elif reduction == "sum":
            gambler_loss = gambler_loss.sum()

        with open(os.path.join(self.cfg.OUTPUT_DIR, "weights‌.csv"), "a") as my_csv:
            my_csv.write(f"sum loss after weighting: {gambler_loss}\n")

        loss_before_weighting = [loss.sum() for loss in NAKHW_loss]

        return NAKHW_loss, sum(loss_before_weighting)/max(1, num_foreground), gambler_loss, weights.data


@GAMBLER_HEAD_REGISTRY.register()
class LayeredUnetGambler(GamblerHeads):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.mode = cfg.MODEL.GAMBLER_HEAD.GAMBLER_LOSS_MODE
        self.alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA

        self.normalize_w = cfg.MODEL.GAMBLER_HEAD.NORMALIZE
        self.gambler_output = cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT
        self.in_layers = cfg.MODEL.GAMBLER_HEAD.IN_LAYERS
        self.gamma = cfg.MODEL.GAMBLER_HEAD.GAMBLER_GAMMA

        image_mode = cfg.MODEL.GAMBLER_HEAD.IMAGE_MODE #conv or downsample
        self.image_channels = cfg.MODEL.GAMBLER_HEAD.IMAGE_CHANNELS
        g_in_channels = cfg.MODEL.GAMBLER_HEAD.FIXED_CHANNEL

        self.pregamblerimage = PreGamblerImage(image_mode, out_channel=self.image_channels)
        self.pregamblerpredictions = PreGamblerPredictions(self.in_channels, out_channel=g_in_channels, num_conv=1, shared=True)
        logger.debug(f"Number of Channels from predictions {g_in_channels} and image {self.image_channels} don't match!!")
        self.layered_gambler = LayeredUnet(g_in_channels, self.image_channels, bilinear=self.bilinear)
        self.postgamblerpredictions = PostGamblerPredictions(in_channel=None, out_channel=self.out_channels, num_conv=1, shared=False)

        # todo # Initialization
        # for modules in [self.pregamblerpredictions, self.layered_gambler, self.postgamblerpredictions]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        #             torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        #             # torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             torch.nn.init.constant_(layer.bias, 0)

        if self.cfg.MODEL.GAMBLER_HEAD.INIT == "default":
            import math
            prior_prob = cfg.MODEL.GAMBLER_HEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for modules in [self.postgamblerpredictions]:
                for layer in modules.modules():
                    if isinstance(layer, nn.Conv2d):
                        torch.nn.init.constant_(layer.bias, bias_value)

        self.to(self.device)

    def forward(self, image, pred_class_logits, gt_classes, mask, detach_pred):
        """

        Args:
            image: the image batch tensor
            pred_class_logits:
            gt_classes:
            mask:
            detach_pred:

        Returns:

        """
        input, image = self.preprocess_input(pred_class_logits, image)

        # prepare the input:
        if self.image_channels == 0:
            im = None
        else:
            im = self.pregamblerimage(image)

        pred = self.pregamblerpredictions(input)
        out1 = self.layered_gambler(pred, im)  # out1 = ['p7', 'p6', 'p5', 'p4', 'p3']
        betting_map = self.postgamblerpredictions(out1)  # out2 = ['p3', 'p4', 'p5', 'p6', 'p7']
        loss_dict, weights = self.gambler_loss(pred_class_logits, betting_map, gt_classes, mask, detach_pred=detach_pred)

        return loss_dict, weights, betting_map

    def gambler_loss(self, pred_class_logits, weights, gt_classes, mask, detach_pred=False):
        """
        Args:
            pred_class_logits: list of tensors, each tensor is [batch, #class_categories * anchors per location, w, h]
            weights: [batch, #anchors per location/#classes/ classes*anchors per location, w, h]
            gt_classes: [batch, #all_anchors = w * h * anchors per location]
            mask: [batch, #all_anchors = w * h * anchors per location]
            detach_pred: if True predictions are detached from the computation graph (for gambler training)

        Returns:
            loss_dict: dict = {"NAKHW_loss",
                     "loss_before_weighting",
                     "gambler_loss"}
            weights: normalized betting map
        """

        assert self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "L_BAHW" or \
               self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "L_B1HW" or \
               self.cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "L_BAHW_extendtobatch", "does not support other shapes!"

        def get_N_H_W(pred_class_logits):
            H = []
            W = []
            for i in range(len(pred_class_logits)):
                H.append(pred_class_logits[i].shape[2])
                W.append(pred_class_logits[i].shape[3])
            N = pred_class_logits[i].shape[0]
            return N, H, W

        [N, H, W] = get_N_H_W(pred_class_logits)

        if detach_pred is True:
            pred_class_logits = [p.detach() for p in pred_class_logits]

        num_classes = self.cfg.MODEL.GAMBLER_HEAD.NUM_CLASSES

        pred_class_logits = list_N_AK_H_W_to_NsumHWA_K(
            pred_class_logits, num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        gt_classes = gt_classes.flatten()
        num_all_anchors = len(gt_classes)
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
        num_foreground = foreground_idxs.sum()
        num_background = (gt_classes == num_classes).sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        cls_loss = calc_cls_loss_jit(pred_class_logits, gt_classes_target, self.mode, self.alpha, self.focal_gamma)

        # ignore the invalid ids in loss
        valid_loss = torch.zeros_like(cls_loss)
        valid_loss[valid_idxs, :] = cls_loss[valid_idxs, :]

        gambler_heads = GamblerHeads(self.cfg)
        #  mask out parts of gambler bets for the loss (for low quality anchors)

        mask = reverse_list_N_A_K_H_W_to_NsumHWA_K_(mask,
                                                    global_cfg.MODEL.GAMBLER_HEAD.IN_LAYERS,
                                                    N,
                                                    H,
                                                    W,
                                                    num_scale=3,
                                                    num_classes=1)
        # weights = weights * mask
        for i, mask_layer in enumerate(mask):
            weights[i] = weights[i] * mask_layer.squeeze()

        gambler_loss, NAKHW_loss, weights = calc_gambler_loss(valid_loss,
                                                              weights,
                                                              N,
                                                              H,
                                                              W,
                                                              self.in_layers,
                                                              gambler_heads,
                                                              normalize_w=self.normalize_w,
                                                              num_classes=num_classes,
                                                              gambler_output=self.gambler_output,
                                                              gamma=self.gamma)

        storage = get_event_storage()
        storage.put_scalar("loss_gambler/lower_bound", -get_loss_upper_bound(NAKHW_loss, N))


        if self.mode == "focal":
            loss_before_weighting = [loss.sum() for loss in NAKHW_loss]
            loss_before_weighting = sum(loss_before_weighting) / max(1, num_foreground)
        elif self.mode == "sigmoid":
            loss_before_weighting = [loss.sum() for loss in NAKHW_loss]
            loss_before_weighting = sum(loss_before_weighting) / num_all_anchors
        else:
            raise Exception("No mode it selected for the retinanet loss!!")

        loss_dict = {"NAKHW_loss": NAKHW_loss,
                     "loss_before_weighting": loss_before_weighting,
                     "gambler_loss": gambler_loss}

        return loss_dict, weights.clone().detach()

    def preprocess_input(self, pred_class_logits, input_images):
        if len(self.cfg.MODEL.RETINANET.IN_FEATURES) > 1:
            logging.debug("Mode: multiple FPN layers")
            if self.cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-0.5, 0.5]:
                scaled_prob = [torch.sigmoid(pred) - 0.5 for pred in pred_class_logits]
            elif self.cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-128, 128]:
                scaled_prob = [(torch.sigmoid(pred) - 0.5) * 256 for pred in pred_class_logits]
            else:
                scaled_prob = [ torch.sigmoid(pred) for pred in pred_class_logits]
            return scaled_prob, input_images
        else:
            logging.debug("Mode: one FPN layer")
            if input_images is None:
                raise Exception("one fpn layer always needs the input image for concatenation!")
            stride = 16  # todo: stride depends on feature map layer
            input_images = F.interpolate(input_images, scale_factor=1 / stride, mode='bilinear')
            sigmoid_predictions = torch.sigmoid(pred_class_logits[0])

            if self.cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-0.5, 0.5]:
                scaled_prob = (sigmoid_predictions - 0.5)
                input_images = (input_images / 256.0)
            elif self.cfg.MODEL.GAMBLER_HEAD.DATA_RANGE == [-128, 128]:
                scaled_prob = (sigmoid_predictions - 0.5) * 256

            # concatenate along the channel
            gambler_input = torch.cat((input_images, scaled_prob), dim=1)  # (N,3+AK,H,W)
            return gambler_input, input_images

@GAMBLER_HEAD_REGISTRY.register()
class UnetLaurence(GamblerHeads):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        num_downs = 5 #before: 6
        ngf = 64 #??
        norm_layer = nn.BatchNorm2d
        use_dropout = False
        kernel_size = 3 #before: 4
        pool = False
        self.gambler = UnetGenerator(self.in_channels, self.out_channels, num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, kernel_size=kernel_size, pool=pool)
        self.to(self.device)

    def forward(self, input):
        return self.gambler(input)