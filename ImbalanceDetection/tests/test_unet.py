from unittest import TestCase
import torch
from imbalancedetection.modelling import UNet


class TestUNet(TestCase):
    def test_forward(self):
        unet = UNet(83, 80)
        in1 = torch.randn(8, 83, 96, 148)
        out1 = unet(in1)
        assert(out1.shape == (8, 80, 96, 148))
