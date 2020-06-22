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
    _C.MODEL.GAMBLER_HEAD.GAMBLER_KAPPA = 2200
    # Hyperparameter lambda that multiplies the regression loss
    _C.MODEL.GAMBLER_HEAD.REGRESSION_LAMBDA = 1
    # Hyperparameter lambda that multiplies the gambler loss from outside
    _C.MODEL.GAMBLER_HEAD.GAMBLER_OUTSIDE_LAMBDA = 1.0
    _C.MODEL.GAMBLER_HEAD.GAMBLER_TEMPERATURE = 0.03
    # classification loss used in the gambler can be either "sigmoid" or "focal"
    _C.MODEL.GAMBLER_HEAD.GAMBLER_LOSS_MODE = "focal"
    # detector loss can be either cls+reg-gambler or weighted_cls_with_gambler+reg
    _C.MODEL.GAMBLER_HEAD.DETECTOR_LOSS_MODE = "cls+reg-gambler"
    # normalizing the weights of the gambler, turned off for sanity check that gambler is learning
    _C.MODEL.GAMBLER_HEAD.NORMALIZE = True
    # adjusting the range of the data given to the gambler
    _C.MODEL.GAMBLER_HEAD.DATA_RANGE = [-128, 128]
    # If True, in unet gambler upsampling is done with bilinear interpolation, o.w. with TransposeConv
    _C.MODEL.GAMBLER_HEAD.BILINEAR_UPSAMPLING = True
    # original image is "downsample" or "conv" and then concatenated with predictions
    _C.MODEL.GAMBLER_HEAD.IMAGE_MODE = "downsample"
    # If image mode is downsample, image_channels has to be 3, otherwise it's a hyperparam
    _C.MODEL.GAMBLER_HEAD.IMAGE_CHANNELS = 3
    # Number of fixed channels going into gambler
    _C.MODEL.GAMBLER_HEAD.FIXED_CHANNEL = 32
    # Number of classes predicted by detector
    _C.MODEL.GAMBLER_HEAD.NUM_CLASSES = 80
    # Feature layer sizes going into gambler
    _C.MODEL.GAMBLER_HEAD.IN_LAYERS = [80, 40, 20, 10, 5]
    # if True all images are saved in "images" folder, otherwise only in tensorboard
    _C.MODEL.GAMBLER_HEAD.SAVE_VIS_FILES = False
    # {{{He/Xavier}_{uniform/normal}}/random}_{unet/unet+prepost}_{bias0/biasrand}
    _C.MODEL.GAMBLER_HEAD.INIT = "random"
    # prior probability on the last layer of gambler
    _C.MODEL.GAMBLER_HEAD.PRIOR_PROB = 0.01
    # gambler optimizer: sgd or adam
    _C.MODEL.GAMBLER_HEAD.OPTIMIZER = "sgd"
    # bettingmap goes to the power of gamma to control focus on bets, if 0 -> normal bce loss
    _C.MODEL.GAMBLER_HEAD.GAMBLER_GAMMA = 1
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
