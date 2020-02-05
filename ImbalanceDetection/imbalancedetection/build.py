from detectron2.utils.registry import Registry
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole detection model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
GAMBLER_HEAD_REGISTRY = Registry("GAMBLER_HEAD")
GAMBLER_HEAD_REGISTRY.__doc__ = """
Registry for gambler-architectures, i.e. the whole gambler model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_detector(cfg):
    """
    Returns the detection model
    Build the detector model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    It can for example be a ``faster rcnn`` or ``retinanet``
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)


def build_gambler(cfg):
    """
    Returns the gambler model
    Build the gambler model architecture, defined by ``cfg.MODEL.GAMBLER_HEAD``.
    Note that it does not load any weights from ``cfg``.
    It can for example be a ``SimpleGambler``
    """
    gambler_arch = cfg.MODEL.GAMBLER_HEAD.NAME
    return GAMBLER_HEAD_REGISTRY.get(gambler_arch)(cfg)
