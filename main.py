import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Vgg16
from models import TransNet
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


data = os.path.expanduser('~/.torch/datasets/vangogh2photo')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=data)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=1)
cfg = parser.parse_args()
print(cfg)

def gram_matrix(x):
    (b, c, h, w) = x.size()
    ft = x.view(b, c, w * h)
    ft_t = torch.transpose(ft, 1, 2)
    gram = torch.bmm(ft, ft_t) / (c * h * w)
    return gram

dataset = ImageFolder(cfg.data, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

vgg16 = Vgg16().to(device)
transform = TransNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(transform.parameters(), lr=cfg.lr)

# plt.ion()
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

        content_loss = criterion(fts_y.relu2_2, fts_x.relu2_2)

        total_loss = style_loss + content_loss
        total_loss.backward()
        optimizer.step()

        print(
            '[{}/{}]'.format(epoch + 1, cfg.epochs) +
            '[{}/{}]'.format(i + 1, len(dataloader)) + ', ' +
            'style_loss: {:.4f}, content_loss: {:.4f}, total_loss: {:.4f}'.format(style_loss.item(), content_loss.item(), total_loss.item())
        )

        # if i % 100 == 99:
        #     fake = net_G(fixed_noises)
        #     fake = fake.detach().cpu()
        #     fake = tv.utils.make_grid(fake)
        #     fake = fake.numpy().transpose(1, 2, 0)
        #     plt.imshow(fake)
        #     plt.pause(0.1)

# plt.ioff()
# plt.show()
