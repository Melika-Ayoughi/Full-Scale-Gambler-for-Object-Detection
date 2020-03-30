from train_net import permute_to_N_HWA_K, reverse_permute_to_N_HWA_K, permute_all_cls_to_N_HWA_K_and_concat, reverse_permute_all_cls_to_N_HWA_K_and_concat
from unittest import TestCase
import torch


class TestReshapes(TestCase):
    def test_permute_to_n_HWA_K(self):
        '''
        test if permute_to_n_HWA_K is revertible
        '''
        n, a, h, w, num_classes= 8, 3, 28, 40, 80
        tensor = torch.randn((n, num_classes*a, h, w))
        changed = [permute_to_N_HWA_K(x, num_classes) for x in [tensor]]
        tensor_prime = [reverse_permute_to_N_HWA_K(x, n, h, w, num_classes) for x in changed]
        assert (tensor_prime[0]==tensor).all()

    def test_reverse_permute_all_cls_to_N_HWA_K_and_concat(self):
        n, a, h, w, num_classes = 8, 3, 28, 40, 80
        num_fpn_layers = 1
        tensor = torch.randn((n, num_classes*a, h, w))
        changed = permute_all_cls_to_N_HWA_K_and_concat([tensor], num_classes)
        tensor_prime = reverse_permute_all_cls_to_N_HWA_K_and_concat(changed, num_fpn_layers, n, h, w, num_classes)
        assert (tensor_prime[0] == tensor).all() #todo only suppports 1 fpn layer
