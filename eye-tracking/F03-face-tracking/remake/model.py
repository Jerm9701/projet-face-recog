from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IRBlock(nn.Module):
    """
    Improved residual block inspired by the ArcFace paper:
        BN -> Conv -> BN -> PReLU -> Conv(stride possibly 2) -> BN
    The downsampling stride is applied on the second convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class IRResNet(nn.Module):
    """
    Face-recognition ResNet adapted to 112x112 inputs.

    Differences from an ImageNet ResNet:
      - 3x3 first convolution with stride 1
      - no max-pooling
      - PReLU activations
      - 512-dimensional face embedding
      - output setting similar to Option-E:
            BN -> Dropout -> FC -> BN
    """

    def __init__(
        self,
        blocks: Sequence[int] = (3, 4, 14, 3),
        embedding_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.stage1 = self._make_stage(64, 64, blocks[0], first_stride=2)
        self.stage2 = self._make_stage(64, 128, blocks[1], first_stride=2)
        self.stage3 = self._make_stage(128, 256, blocks[2], first_stride=2)
        self.stage4 = self._make_stage(256, 512, blocks[3], first_stride=2)

        # 112 -> 56 -> 28 -> 14 -> 7
        self.output_bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features_bn = nn.BatchNorm1d(embedding_dim)

        # Common face-recognition practice: do not learn an affine bias after
        # the final embedding normalization.
        nn.init.constant_(self.features_bn.weight, 1.0)
        self.features_bn.weight.requires_grad = False

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        layers = [IRBlock(in_channels, out_channels, stride=first_stride)]
        layers.extend(
            IRBlock(out_channels, out_channels, stride=1)
            for _ in range(num_blocks - 1)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None and module is not self.features_bn:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.output_bn(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features_bn(x)
        return x


def iresnet50(embedding_dim: int = 512, dropout: float = 0.4) -> IRResNet:
    # 3 + 4 + 14 + 3 is the commonly used ArcFace-style IResNet-50 layout.
    return IRResNet(
        blocks=(3, 4, 14, 3),
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


class ArcMarginProduct(nn.Module):
    """
    ArcFace classification head.

    For the target class:
        cos(theta) -> cos(theta + margin)
    Then all logits are scaled by s and passed to cross-entropy.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        )
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
