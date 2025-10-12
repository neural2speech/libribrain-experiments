"""
EEGNet-v3 (Lawhern et al., 2018) - PyTorch implementation

The code keeps the layer-ordering and default hyper-parameters of the
Keras reference while adapting the tensor layout used in this repo:
    input  (B, C, T)  ->  reshape to (B, 1, C, T) for 2-D convolutions
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DepthwiseConv2d(nn.Conv2d):
    """Depth-wise conv: one filter per input-channel."""

    def __init__(self, in_channels, kernel_size, depth_multiplier=1, **kw):
        super().__init__(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            groups=in_channels,
            bias=False,
            **kw,
        )


class _SeparableConv2d(nn.Module):
    """TF/Keras-style separable conv = depth-wise + point-wise."""

    def __init__(self, in_ch, out_ch, kernel_size, **kw):
        super().__init__()
        self.depth = _DepthwiseConv2d(in_ch, kernel_size, **kw)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding="same", bias=False)

    def forward(self, x):
        return self.point(self.depth(x))


class EEGNet(nn.Module):
    def __init__(
        self,
        chans: int = 306,
        seq_len: int = 200,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kern_len: int = 64,
        dropout: float = 0.5,
        norm_rate: float = 0.25,
    ):
        super().__init__()

        F2 = F1 * D  # Lawhern et al.'s default

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_len), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            _DepthwiseConv2d(
                F1, kernel_size=(chans, 1), depth_multiplier=D, padding="valid"
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2
        self.block2 = nn.Sequential(
            _SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), padding="same"),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # classifier
        n_features = F2 * ((seq_len // 4) // 8)  # C_out * T_out
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, num_classes, bias=True),
        )

        # optional weight-norm constraint (max-norm) on the dense layer
        nn.init.uniform_(self.classifier[1].weight, -0.01, 0.01)
        self.classifier[1].weight.data = torch.clip(
            self.classifier[1].weight.data, -norm_rate, norm_rate
        )

    def forward(self, x):  # x: (B, C, T)
        x = x.unsqueeze(1)  #       -> (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)  # logits
