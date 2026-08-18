import torch
from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        self.lateral3 = nn.Conv2d(512, out_channels, 1)
        self.lateral4 = nn.Conv2d(1024, out_channels, 1)
        self.lateral5 = nn.Conv2d(2048, out_channels, 1)

        self.output3 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.output4 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.output5 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)

        p4 = self.lateral4(c4) + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest",
        )

        p3 = self.lateral3(c3) + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest",
        )

        p3 = self.output3(p3)
        p4 = self.output4(p4)
        p5 = self.output5(p5)

        return [p3, p4, p5]