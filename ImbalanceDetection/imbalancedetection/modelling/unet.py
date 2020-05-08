""" Parts of the U-Net model
taken from https://github.com/milesial/Pytorch-UNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class UNet(nn.Module):
    """
    Args:
        n_channels: int number of input channels
        n_classes: int number of output channels
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # todo nn.sequential make it prettier
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        considering N is batch_size.
        Args:
            x: Tensor (N, C_in, H, W)

        Returns:
            logits: Tensor (N, C_out, H_out, W_out)

        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return self.sigmoid(logits)


class LayeredUnet(nn.Module):
    """
    Args:
        n_channels: int number of input channels
        n_classes: int number of output channels
    """
    def __init__(self, pred_channels, img_channels, bilinear=True):
        super(LayeredUnet, self).__init__()

        # # todo # Initialization
        # for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv2d):
        #             torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             torch.nn.init.constant_(layer.bias, 0)
        #
        # # Use prior in model initialization to improve stability
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.inc = DoubleConv(pred_channels+img_channels, 64)
        self.down1 = DownCat(pred_channels, 64, 128)
        self.down2 = DownCat(pred_channels, 128, 256)
        self.down3 = DownCat(pred_channels, 256, 512)
        self.down4 = DownCat(pred_channels, 512, 1024)
        self.up1 = UpCat(1024, 512, bilinear)
        self.up2 = UpCat(512, 256, bilinear)
        self.up3 = UpCat(256, 128, bilinear)
        self.up4 = UpCat(128, 64, bilinear)
        # self.outc = OutConv(64, out_classes)

        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, layered_x, image):
        """
        considering N is batch_size.
        Args:
            layered_x: List(Tensor (N, C_i, H_i, W_i))
                each Tensor is predictions from the corresponding FPN layer
                starting from P3 (80x80) to P7 (5x5)
            image: Tensor(N, C(variable),

        Returns:
            logits: Tensor (N, C_out, H_out, W_out)

        """
        assert image.shape[2] == layered_x[0].shape[2] and image.shape[3] == layered_x[0].shape[3]
        layered_output = []

        x1 = self.inc(torch.cat((layered_x[0], image), dim=1))
        print(f"x1: {x1.shape}")
        x2 = self.down1(layered_x[1], x1)
        print(f"x2: {x2.shape}")
        x3 = self.down2(layered_x[2], x2)
        print(f"x3: {x3.shape}")
        x4 = self.down3(layered_x[3], x3)
        print(f"x4: {x4.shape}")
        x5 = self.down4(layered_x[4], x4)
        print(f"x5: {x5.shape}")
        layered_output.append(x5)
        o1 = self.up1(x5, x4)
        print(f"o1: {o1.shape}")
        layered_output.append(o1)
        o2 = self.up2(o1, x3)
        print(f"o2: {o2.shape}")
        layered_output.append(o2)
        o3 = self.up3(o2, x2)
        print(f"o3: {o3.shape}")
        layered_output.append(o3)
        o4 = self.up4(o3, x1)
        print(f"o4: {o4.shape}")
        layered_output.append(o4)

        return layered_output


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


class DownCat(nn.Module):
    """Downscaling with maxpool then concat with predictions then double conv"""

    def __init__(self, pred_channels, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(pred_channels+in_channels, out_channels)

    def forward(self, pred, x):
        out1 = self.maxpool(x)
        print(f"prediction channels: {pred.shape} unet channels: {out1.shape}")
        out2 = torch.cat([pred, out1], dim=1)
        return self.conv(out2)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpCat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            raise Exception("have not tested this branch!")
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, kernel_size=4, pool=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, pool=pool)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, pool=pool)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pool=pool)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pool=pool)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, pool=pool, emb=True)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, pool=pool)  # add the outermost layer
        self.features = []
        self.counter = 0

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, kernel_size=4, pool=False, emb=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.emb = emb
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        use_bias = True
        if pool:
            maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=2,
                                        padding=1, bias=use_bias)
            if pool:
                down = [downrelu, downconv, maxpool, downnorm]
            else:
                down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.1)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            output = self.model(x)
            return output
        else:   # add skip connections
            output = self.model(x)
            features = torch.cat([x, output], 1)
            return features