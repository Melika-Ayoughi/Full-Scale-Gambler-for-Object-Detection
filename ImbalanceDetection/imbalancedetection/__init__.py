from .build import META_ARCH_REGISTRY, GAMBLER_HEAD_REGISTRY, build_detector, build_gambler
from .config import add_gambler_config
from .gambler_heads import GamblerHeads, UnetGambler, LayeredUnetGambler, UnetLaurence, calc_cls_loss, calc_gambler_loss, calc_cls_loss_jit