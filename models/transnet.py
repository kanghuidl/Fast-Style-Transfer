import torch
import torch.nn as nn
import torch.nn.functional as F


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()

        model = [
            Conv2d(3, 32, kernel_size=9, stride=1), # /1
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            Conv2d(32, 64, kernel_size=3, stride=2), # /2
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            Conv2d(64, 128, kernel_size=3, stride=2), # /2
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        ]

        for i in range(5):
            model += [ResBlock(128)]

        model += [
            UpConv2d(128, 64, kernel_size=3, stride=1, upsample=2), # *2
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            UpConv2d(64, 32, kernel_size=3, stride=1, upsample=2), # *2
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            UpConv2d(32, 3, kernel_size=9, stride=1), # *1
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2d, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1), # /1
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1), # /1
            nn.InstanceNorm2d(in_channels, affine=True)
        )

    def forward(self, x):
        return F.relu(x + self.model(x), inplace=True)


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpConv2d, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.upsample = upsample

    def forward(self, x):
        if self.upsample is not None:
            x = F.interpolate(x, scale_factor=self.upsample)
        return self.model(x)
