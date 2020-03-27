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
    # If this option is turned on, pretrained model will be loaded into gambler
    _C.MODEL.GAMBLER_HEAD.LOAD_PRETRAINED_GAMBLER = False
    # Loading pretrained weights for the gambler from this location
    _C.MODEL.GAMBLER_HEAD.WEIGHTS = "./output/gambler/focal_plus_gambler/gambler_models/model_0042749.pth"
    # number of iterations the gambler is trained before the detector is trained
    _C.MODEL.GAMBLER_HEAD.GAMBLER_ITERATIONS = 100
    # number of iterations the detector is trained before the gambler is trained
    _C.MODEL.GAMBLER_HEAD.DETECTOR_ITERATIONS = 200
    # VARIANTS: BCHW(input to gambler is (B, C, H, W)), BCAHW (input to gambler is (B, CxA, H, W))
    _C.MODEL.GAMBLER_HEAD.GAMBLER_INPUT = "BCAHW"
    # VARIANTS: B1HW BCHW BAHW BCAHW
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUTPUT = "BAHW"
    # Number of input channels to the gambler module (num_classes + 3 (input channel RGB))
    _C.MODEL.GAMBLER_HEAD.GAMBLER_IN_CHANNELS = 883  # 3(scales) x 80(classes) + 3(RGB)
    # Number of output channels of the gambler module (desired betting map channels)
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUT_CHANNELS = 11
    # Hyperparameter lambda that multiplies the gambler loss
    _C.MODEL.GAMBLER_HEAD.GAMBLER_LAMBDA = 2200
    # Hyperparameter lambda that multiplies the regression loss
    _C.MODEL.GAMBLER_HEAD.REGRESSION_LAMBDA = 1
    # Hyperparameter lambda that multiplies the gambler loss from outside
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUTSIDE_LAMBDA = 1
    _C.MODEL.GAMBLER_HEAD.GAMBLER_TEMPERATURE = 0.03
    # classification loss used in the gambler can be either "sigmoid" or "focal"
    _C.MODEL.GAMBLER_HEAD.GAMBLER_LOSS_MODE = "focal"
    # normalizing the weights of the gambler, turned off for sanity check that gambler is learning
    _C.MODEL.GAMBLER_HEAD.NORMALIZE = True
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
