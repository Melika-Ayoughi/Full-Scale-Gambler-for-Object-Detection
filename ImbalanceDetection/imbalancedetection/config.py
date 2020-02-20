from detectron2.config import CfgNode as CN


def add_gambler_config(cfg):
    """
    Add config for gambler/ gambler-detection head.
    """
    _C = cfg

    _C.MODEL.GAMBLER_ON = True
    _C.MODEL.GAMBLER_HEAD = CN()
    # VARIANTS: SimpleGambler - UnetGambler
    _C.MODEL.GAMBLER_HEAD.NAME = "UnetGambler"
    # number of iterations the gambler is trained before the detector is trained
    _C.MODEL.GAMBLER_HEAD.GAMBLER_ITERATIONS = 20
    # number of iterations the detector is trained before the gambler is trained
    _C.MODEL.GAMBLER_HEAD.DETECTOR_ITERATIONS = 40
    # VARIANTS: C(per class prediction) R(per region prediction) CR(both)
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT = "C"
    # Number of input channels to the gambler module (num_classes + 3 (input channel RGB))
    _C.MODEL.GAMBLER_HEAD.GAMBLER_IN_CHANNELS = 83
    # Number of output channels of the gambler module (desired betting map channels)
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS = 1
    # Hyperparameter lambda that multiplies the gambler loss
    _C.MODEL.GAMBLER_HEAD.GAMBLER_LAMBDA = 1000
    # Hyperparameter lambda that multiplies the regression loss
    _C.MODEL.GAMBLER_HEAD.REGRESSION_LAMBDA = 1
    # The period (in terms of steps) for minibatch visualization at train time.
    # Set to 0 to disable.
    _C.MODEL.GAMBLER_HEAD.VIS_PERIOD = 300
    # initialize all these values to the default of the detector but can be changed later
    _C.MODEL.GAMBLER_HEAD.BASE_LR = cfg.SOLVER.BASE_LR
    _C.MODEL.GAMBLER_HEAD.BIAS_LR_FACTOR = cfg.SOLVER.BIAS_LR_FACTOR
    _C.MODEL.GAMBLER_HEAD.WEIGHT_DECAY = _C.SOLVER.WEIGHT_DECAY
    _C.MODEL.GAMBLER_HEAD.WEIGHT_DECAY_NORM = _C.SOLVER.WEIGHT_DECAY_NORM
    _C.MODEL.GAMBLER_HEAD.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY_BIAS
    _C.MODEL.GAMBLER_HEAD.MOMENTUM = _C.SOLVER.MOMENTUM
