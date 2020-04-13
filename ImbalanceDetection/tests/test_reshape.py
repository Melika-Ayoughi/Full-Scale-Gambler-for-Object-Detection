from train_net import N_AK_H_W_to_N_HWA_K, reverse_N_AK_H_W_to_N_HWA_K, permute_all_cls_to_NHWAxFPN_K_and_concat, reverse_permute_all_cls_to_N_HWA_K_and_concat
from unittest import TestCase
import torch
from detectron2.layers import cat

class TestReshapes(TestCase):
    def test_permute_to_n_HWA_K(self):
        '''
        test if permute_to_n_HWA_K is revertible
        '''
        n, a, h, w, num_classes= 8, 3, 28, 40, 80
        tensor = torch.randn((n, num_classes*a, h, w))
        changed = [N_AK_H_W_to_N_HWA_K(x, num_classes) for x in [tensor]]
        tensor_prime = [reverse_N_AK_H_W_to_N_HWA_K(x, n, h, w, num_classes) for x in changed]
        assert (tensor_prime[0]==tensor).all()

    def test_reverse_permute_all_cls_to_N_HWA_K_and_concat(self):
        n, a, h, w, num_classes = 8, 3, 28, 40, 80
        num_fpn_layers = 1
        tensor = torch.randn((n, num_classes*a, h, w))
        changed = permute_all_cls_to_NHWAxFPN_K_and_concat([tensor], num_classes)
        tensor_prime = reverse_permute_all_cls_to_N_HWA_K_and_concat(changed, num_fpn_layers, n, h, w, num_classes)
        assert (tensor_prime[0] == tensor).all() #todo only suppports 1 fpn layer

    def test_reverse_cat(self):
        n, a, h, w, num_classes = 8, 3, 28, 40, 80
        num_fpn_layers = 7

        list_tensor = [torch.randn((n, h * w * a, num_classes)) for _ in range(num_fpn_layers)]
        tensor_tensor = cat(list_tensor, dim=1)
        new_list_tensor = torch.chunk(tensor_tensor, num_fpn_layers, dim=1)
        assert ([(new_list_tensor[i]==list_tensor[i]).all() for i in range(num_fpn_layers)])
