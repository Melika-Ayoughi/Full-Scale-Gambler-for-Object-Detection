import logging
import torch
from torch import nn
from .build import GAMBLER_HEAD_REGISTRY
from .modelling.unet import UNet, UnetGenerator

logger = logging.getLogger(__name__)


class GamblerHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()


@GAMBLER_HEAD_REGISTRY.register()
class SimpleGambler(GamblerHeads):
    # todo need to make sure the dimension doesn't change with this cnn, so maybe i need to use a unet or sth similar
    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)

        in_channel = 100 # todo read from configs cfg.
        out_channel = 100 # todo read from configs
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )
        self.to(self.device)

    def forward(self, input):
        return self.layers(input)


@GAMBLER_HEAD_REGISTRY.register()
class UnetGambler(UNet):
    def __init__(self, cfg, bilinear=True): # todo read from cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        in_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_IN_CHANNELS # 81 + 3
        # also weighting the bg class cause it's easier for now cause it matches the ce loss of detection
        out_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS #used to be 81
        '''
        if cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "C":
            out_channels = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        elif cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "R":
            out_channels = 200 # todo: fix number of proposals
        elif cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT == "CR":
            out_channels = 200 * cfg.MODEL.ROI_HEADS.NUM_CLASSES # todo
        '''
        super().__init__(in_channels, out_channels, bilinear)
        self.to(self.device)

    def forward(self, input):
        return super().forward(input)


@GAMBLER_HEAD_REGISTRY.register()
class UnetLaurence(UnetGenerator):
    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)
        in_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_IN_CHANNELS
        out_channels = cfg.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS
        num_downs = 2**6 #todo
        ngf = 64 #todo??
        norm_layer = nn.BatchNorm2d
        use_dropout = False
        kernel_size = 4
        pool = False
        super().__init__(in_channels, out_channels, num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, kernel_size=kernel_size, pool=pool)
        self.to(self.device)

    def forward(self, input):
        return super().forward(input)