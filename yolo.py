import torch
import torch.nn as nn

# Tuple: (out_channels, kernel_size, stride)
# List: ["B" (block), num_repears]
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",       # Scale prediction
    (256, 1, 1),
    "U",       # Up-sample
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        # if bn_act will be used, then bias is uneeded
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    # in scale prediciton output, we dont want to use leaky relu & bn
    def forward(self,x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __int__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in num_repeats:
            self.layers += [
                CNNBlock(channels, channels//2, kernel_size=1),
                CNNBlock(channels//2, channels, kernel_size=3, padding=1)
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def foward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
   def __init__(self, in_channels, num_classes):
       super().__init__()
       self.pred = nn.Sequential(
           CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
           CNNBlock(2*in_channels,(num_classes + 5)*3, bn_act=False, kernel_size=1),
       )

class YOLOv3(nn.Module):

    pass





