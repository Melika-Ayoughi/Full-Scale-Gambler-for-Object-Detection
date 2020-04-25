import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class SizeEstimator(object):

    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for name, param in self.model.named_parameters():
            print(name, param.size())

        # for i in range(1, len(mods)):
        #     m = mods[i]
        #     p = list(m.parameters())
        #     for j in range(len(p)):
            sizes.append(np.array(param.size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True).to("cuda:0")
        l_names = list(self.model._modules)
        # mods = list(self.model.modules())
        out_sizes = []
        outputs = []
        for name in l_names:
            m = self.model._modules[name]
            if name.startswith("up"):
                out = m(outputs.pop(), outputs.pop())
            else:
                out = m(input_)
            outputs.append(out)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits * 2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    @staticmethod
    def bits_to_megabytes(bits):
        return (bits / 8) / (1024 ** 2) # 1024**2 ~= 10**6

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.input_bits + self.forward_backward_bits

        total_megabytes = self.bits_to_megabytes(total)
        return total_megabytes, total

from detectron2.config import get_cfg
from imbalancedetection.config import add_gambler_config
from detectron2.engine import TrainerBase, default_setup, launch, default_argument_parser
from detectron2.config import set_global_cfg, global_cfg
from imbalancedetection.gambler_heads import UnetGambler, UnetLaurence

def setup(args):
    cfg = get_cfg()
    add_gambler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imbalance detection")
    set_global_cfg(cfg)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    model = UnetGambler(cfg)
    # model = UnetLaurence(cfg)
    se = SizeEstimator(model, input_size=(8, 80*3+3, 80, 80))
    print(se.estimate_size())
    # Returns
    # (size in megabytes, size in bits)
    # (408.2833251953125, 3424928768)

    print(se.param_bits)  # bits taken up by parameters
    print(se.forward_backward_bits)  # bits stored for forward and backward
    print(se.input_bits)  # bits for input