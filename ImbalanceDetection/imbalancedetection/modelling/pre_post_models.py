import torch
import torch.nn as nn
import torch.nn.functional as F


class PreGamblerPredictions(nn.Module):

    def __init__(self, in_channel, out_channel, num_conv, shared=True):
        super().__init__()
        if shared:
            if num_conv == 1:
                self.model = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1))
            else:
                self.model = nn.Sequential(
                    nn.Conv2d(in_channel, 256, kernel_size=1),
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.Conv2d(128, out_channel, kernel_size=1)
                )
                assert len(self.model._modules) == num_conv
        #todo
        # else:
        #     if num_conv == 1:
        #     else:
        #     assert len(self.model._modules) == num_conv

    def forward(self, layered_input):
        gambler_in = []
        for pred in layered_input:
            gambler_in.append(self.model(pred))
        return gambler_in


class PostGamblerPredictions(nn.Module):

    def __init__(self, in_channel, out_channel, num_conv, shared=True):
        super().__init__()
        #todo shared
        # if shared:
        #     if num_conv == 1:
        #         self.model = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1))
        #     else:
        #         self.model = nn.Sequential(
        #             nn.Conv2d(in_channel, 256, kernel_size=1),
        #             nn.Conv2d(256, 128, kernel_size=1),
        #             nn.Conv2d(128, out_channel, kernel_size=1)
        #         )
        #         assert len(self.model._modules) == num_conv
        if not shared:
            if num_conv == 1:
                self.p7 = nn.Conv2d(1024, out_channel, kernel_size=1)
                self.p6 = nn.Conv2d(512, out_channel, kernel_size=1)
                self.p5 = nn.Conv2d(256, out_channel, kernel_size=1)
                self.p4 = nn.Conv2d(128, out_channel, kernel_size=1)
                self.p3 = nn.Conv2d(64, out_channel, kernel_size=1)
            #todo else:
            # assert len(self.model._modules) == num_conv
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, layered_output):
        gambler_out = []
        gambler_out.append(self.p3(layered_output[4]))
        gambler_out.append(self.p4(layered_output[3]))
        gambler_out.append(self.p5(layered_output[2]))
        gambler_out.append(self.p6(layered_output[1]))
        gambler_out.append(self.p7(layered_output[0]))
        # for pred in reversed(layered_output):
        #     gambler_out.append(self.model(pred))
        for i, output in enumerate(gambler_out):
            gambler_out[i] = self.sigmoid(output)

        return gambler_out
    

class PreGamblerImage(nn.Module):
    def __init__(self, image_mode, out_channel):
        super().__init__()
        self.image_mode = image_mode
        if self.image_mode == "conv":
            #todo need to decide on the architecture
            self.model_image = DoubleConv(3, out_channel)

    def forward(self, input_images):
        if self.image_mode == "downsample":
            stride = 8
            return F.interpolate(input_images, scale_factor=1 / stride, mode='bilinear')
        elif self.image_mode == "conv":
            return self.model_image(input_images)
        else:
            raise Exception("Only these two modes are available")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
