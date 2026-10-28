"""Residual U-Net: output = clamp(y - net(y), 0, 1)."""
import torch
import torch.nn as nn


def block(cin, cout, norm):
    N = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), N(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), N(cout), nn.ReLU(inplace=True))


class ResidualUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), norm="batch"):
        super().__init__()
        self.downs = nn.ModuleList()
        cin = 1
        for c in channels:
            self.downs.append(block(cin, c, norm))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for c in reversed(channels[:-1]):
            self.ups.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(block(2 * c, c, norm))
            cin = c
        self.head = nn.Conv2d(cin, 1, 1)

    def forward(self, y):
        skips, h = [], y
        for i, d in enumerate(self.downs):
            h = d(h)
            if i < len(self.downs) - 1:
                skips.append(h)
                h = self.pool(h)
        for up, dec in zip(self.ups, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        return (y - self.head(h)).clamp(0, 1)
