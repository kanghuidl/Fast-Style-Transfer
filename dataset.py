import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.transform = transform
        self.style = sorted(glob.glob(os.path.join(root, phase + 'A', '*.jpg')))
        self.content = sorted(glob.glob(os.path.join(root, phase + 'B', '*.jpg')))

    def __getitem__(self, index):
        style = Image.open(self.style[264])
        content = Image.open(self.content[index])

        if self.transform is not None:
            style = self.transform(style)
            content = self.transform(content)
        else:
            style = np.array(style)
            content = np.array(content)

        return style, content

    def __len__(self):
        return len(self.content)
