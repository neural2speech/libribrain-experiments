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
