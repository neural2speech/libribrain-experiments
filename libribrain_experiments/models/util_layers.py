import torch
import torch.nn as nn


class Permute(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class SelectChannels(nn.Module):
    """Keep only the sensors given in channels.

    Input : (B, C_in, T)
    Output: (B, len(channels), T)
    """
    def __init__(self, channels: list[int]):
        super().__init__()
        self.register_buffer("idx", torch.tensor(channels, dtype=torch.long))

    def forward(self, x):
        return torch.index_select(x, 1, self.idx)


class LSTMBlock(nn.Module):
    """1-D LSTM returning the last hidden state (for the last layer).

    Use with `batch_first=True` and remember to permute the
    (B, C, T) tensor to (B, T, C) beforehand.
    """
    def __init__(self, **cfg):
        super().__init__()
        # default to batch_first so the YAML looks like PyTorch defaults
        cfg.setdefault("batch_first", True)
        self.lstm = nn.LSTM(**cfg)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers * dir, B, H)
        return h_n[-1]              # (B, H)


class FlattenTime(nn.Module):
    """(B, C, T) -> (B*T, C) so we can reuse SequenceClassificationModule."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FlattenTime expects (B,C,T), got {x.shape}")
        B, C, T = x.shape
        return x.permute(0, 2, 1).reshape(B * T, C)


class BatchNorm1dLastDim(nn.Module):
    """
    BatchNorm over the *last* dim of an (B, T, H) tensor by reshaping
    to (B*T, H) semantics. This is what you already used for in_norm.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):  # x: (B, T, H)
        B, T, H = x.shape
        x = x.reshape(B * T, H)
        x = self.bn(x)
        return x.reshape(B, T, H)


class LayerNormOverChannels(nn.Module):
    """
    LayerNorm across channels for (B, C, T): normalize each time step
    independently using the C dimension statistics.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):  # x: (B, C, T)
        # move channels to last dim, LN over that, then move back
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class BatchNormLastDim(nn.Module):
    """Apply BatchNorm1d over the last dimension of a (B, T, H) tensor."""
    def __init__(self, num_features, **bn_kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **bn_kwargs)

    def forward(self, x):
        # x: (B, T, H) -> (B, H, T) -> BN -> (B, T, H)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x
