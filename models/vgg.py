import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(pretrained=True).features[:23]
        for param in features.parameters():
            param.requires_grad = False
        self.features = features

    def forward(self, x):
        results = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [3, 8, 15, 22]:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
