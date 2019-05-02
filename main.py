import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv

from models import Vgg16
from models import TransNet
from dataset import ImageFolder
from torch.utils.data import DataLoader


data = os.path.expanduser('~/.torch/datasets/vangogh2photo')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=data)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1e5)
parser.add_argument('--content_weight', type=float, default=1e0)
cfg = parser.parse_args()
print(cfg)

def gram_matrix(x):
    (b, c, h, w) = x.size()
    ft = x.view(b, c, w * h)
    ft_t = torch.transpose(ft, 1, 2)
    gram = torch.bmm(ft, ft_t) / (c * h * w)
    return gram

dataset = ImageFolder(cfg.data, transform=tv.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

style, content = iter(dataloader).next()

vgg16 = Vgg16().to(device)
transform = TransNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(transform.parameters(), lr=cfg.lr)

plt.ion()
for epoch in range(cfg.epochs):
    print('Epoch: {}/{}'.format(epoch + 1, cfg.epochs))

    for i, (s, x) in enumerate(dataloader):
        optimizer.zero_grad()

        s = s.to(device)
        x = x.to(device)
        y = transform(x)

        fts_s = vgg16(s)
        fts_x = vgg16(x)
        fts_y = vgg16(y)

        style_loss = 0
        for ft_y, ft_s in zip(fts_y, fts_s):
            gm_y = gram_matrix(ft_y)
            gm_s = gram_matrix(ft_s)
            style_loss += criterion(gm_y, gm_s)
        style_loss *= cfg.style_weight

        content_loss = criterion(fts_y.relu2_2, fts_x.relu2_2)
        content_loss *= cfg.content_weight

        total_loss = style_loss + content_loss
        total_loss.backward()
        optimizer.step()

        print(
            '[{}/{}]'.format(epoch + 1, cfg.epochs) +
            '[{}/{}]'.format(i + 1, len(dataloader)) + ', ' +
            'style_loss: {:.4f}, content_loss: {:.4f}, total_loss: {:.4f}'.format(style_loss.item(), content_loss.item(), total_loss.item())
        )

        content = content.to(device)
        outputs = transform(content)

        content = content.detach().cpu()
        outputs = outputs.detach().cpu()

        display = torch.cat((style[:3], content[:3], outputs[:3]), 0)
        display = tv.utils.make_grid(display, nrow=3)
        display = display.numpy().transpose(1, 2, 0)
        plt.imshow(display)
        plt.pause(0.1)

plt.ioff()
plt.show()
