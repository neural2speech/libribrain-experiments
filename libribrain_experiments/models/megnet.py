# libribrain_experiments/models/megnet.py

from __future__ import annotations
import torch
import torch.nn as nn

class LayerNormOverChannels(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):  # (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class MEGNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 306,
        seq_len: int = 125,
        num_classes: int = 39,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kern_length: int = 32,
        sep_temporal: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
        pre_norm: str | None = "instance",
        pre_bn_eps: float = 1e-5,
        pre_bn_momentum: float = 0.1,
    ):
        super().__init__()
        C, T = int(input_dim), int(seq_len)
        self.C, self.T = C, T
        self.F1 = int(F1)
        self.D  = int(D)
        self.F2 = int(F2) if F2 is not None else int(F1 * D)
        self.pool1 = int(pool1)
        self.pool2 = int(pool2)
        self.dropout_p = float(dropout)

        # pre-norm on (B, C, T)
        kind = (pre_norm or "").lower()
        if kind == "batch":
            self.pre_norm = nn.BatchNorm1d(C, eps=pre_bn_eps, momentum=pre_bn_momentum, affine=True)
        elif kind == "layer":
            self.pre_norm = LayerNormOverChannels(C)
        elif kind == "instance":
            self.pre_norm = nn.InstanceNorm1d(C, affine=True, eps=pre_bn_eps, momentum=0.1, track_running_stats=False)
        else:
            self.pre_norm = nn.Identity()

        # Block 1
        # IMPORTANT: use padding="same" for temporal convs (even kernels handled correctly)
        self.conv_temporal = nn.Conv2d(
            in_channels=1, out_channels=self.F1,
            kernel_size=(1, int(kern_length)), stride=(1, 1), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1)

        # Depthwise spatial conv over channels (C,1), no padding on height
        self.conv_depthwise_spatial = nn.Conv2d(
            in_channels=self.F1, out_channels=self.F1 * self.D,
            kernel_size=(C, 1), stride=(1, 1), padding=(0, 0),
            groups=self.F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.F1 * self.D)
        self.act1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, self.pool1), stride=(1, self.pool1))
        self.drop1 = nn.Dropout(self.dropout_p)

        # Block 2 (Separable)
        # Depthwise temporal with SAME padding
        self.conv_sep_depthwise = nn.Conv2d(
            in_channels=self.F1 * self.D, out_channels=self.F1 * self.D,
            kernel_size=(1, int(sep_temporal)), stride=(1, 1), padding="same",
            groups=self.F1 * self.D, bias=False
        )
        # Pointwise 1x1
        self.conv_sep_pointwise = nn.Conv2d(
            in_channels=self.F1 * self.D, out_channels=self.F2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, self.pool2), stride=(1, self.pool2))
        self.drop2 = nn.Dropout(self.dropout_p)

        # infer feature dim (no Lazy)
        with torch.no_grad():
            dummy = torch.zeros(1, C, T)   # fixed size known
            feat = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.pre_norm(x)
        x = x.unsqueeze(1)                    # (B, 1, C, T)

        # Block 1
        x = self.conv_temporal(x)             # (B, F1, C, T)
        x = self.bn1(x)
        x = self.conv_depthwise_spatial(x)    # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.avgpool1(x)                  # (B, F1*D, 1, T//pool1)
        x = self.drop1(x)

        # Block 2
        x = self.conv_sep_depthwise(x)        # (B, F1*D, 1, T//pool1)
        x = self.conv_sep_pointwise(x)        # (B, F2,   1, T//pool1)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.avgpool2(x)                  # (B, F2, 1, (T//pool1)//pool2)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)     # (B, F2 * reduced_T)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        return self.classifier(feats)
