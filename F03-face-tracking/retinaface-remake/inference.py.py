import torch
from torch import nn
import torch.nn.functional as F
class PredictionHead(nn.Module):
    def __init__(
        self,
        in_channels=256,
        num_anchors=2,
    ):
        super().__init__()

        self.classification = nn.Conv2d(
            in_channels,
            num_anchors * 2,
            kernel_size=1,
        )

        self.box_regression = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=1,
        )

        self.landmark_regression = nn.Conv2d(
            in_channels,
            num_anchors * 10,
            kernel_size=1,
        )

    def forward(self, feature):
        return {
            "classification": self.classification(feature),
            "boxes": self.box_regression(feature),
            "landmarks": self.landmark_regression(feature),
        }