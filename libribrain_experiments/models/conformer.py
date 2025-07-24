"""
Conformer-based MEG classifier

• Accepts raw windows of shape (B, 306, T)
• Passes a linear "frontend" -> Conformer encoder -> mean-pool -> classifier
"""

import torch, torch.nn as nn

# PyTorch >= 2.2 has nn.Conformer, older versions can fall back to torchaudio
try:
    from torch.nn import Conformer           # PyTorch 2.2 / 2.3
except ImportError:
    from torchaudio.models import Conformer  # Torchaudio >= 2.2


class ConformerSpeech(nn.Module):
    def __init__(
        self,
        input_dim: int = 306,      # sensors
        seq_len: int = 125,        # time steps in the HDF5 window (0-0.8 s)
        hidden_size: int = 256,    # Conformer model size
        ffn_dim: int = 512,        # FFN expansion
        num_heads: int = 4,
        num_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        num_classes: int = 2,      # speech / silence
        use_preproj: bool = True,  # whether to add a projection layer
        size: str = None,          # Model size to use
        in_norm: str = None,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
    ):
        super().__init__()

        # If size passed, set the correct hparams
        if size is not None:
            if size.lower()[0] == "s":
                hidden_size = 144
                ffn_dim = 576
                num_layers = 16
                num_heads = 4
            elif size.lower()[0] == "m":
                hidden_size = 256
                ffn_dim = 1024
                num_layers = 16
                num_heads = 4
            elif size.lower()[0] == "l":
                hidden_size = 512
                ffn_dim = 2048
                num_layers = 17
                num_heads = 8
            else:
                raise ValueError(
                    f"Unknown Conformer size: {size}"
                )

        #  Front‑end projection
        self._pre_kind = None
        if isinstance(use_preproj, str):
            self._pre_kind = use_preproj.lower()
        elif use_preproj:           # legacy True: "linear"
            self._pre_kind = "linear"

        if self._pre_kind == "linear":
            self.preproj = nn.Linear(input_dim, hidden_size)
        elif self._pre_kind == "conv1d":
            # Conv over the time dimension, kernel=1 ⇒ just channel mixing
            self.preproj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        elif self._pre_kind == "conv2d":
            # Treat sensors as “height”, time as “width”
            self.preproj = nn.Conv2d(1, hidden_size, kernel_size=(input_dim, 1))
        else:
            if input_dim != hidden_size:
                raise ValueError(
                    "When use_preproj=False you must set hidden_size == input_dim"
                )
            self.preproj = nn.Identity()
        self.in_dropout = nn.Dropout(in_dropout) if in_dropout > 0 else nn.Identity()

        if in_norm == "layer":
            self.in_norm = nn.LayerNorm(hidden_size)
        elif in_norm == "batch":
            self.in_norm = nn.BatchNorm1d(hidden_size)
        else:
            self.in_norm = nn.Identity()

        self.encoder = Conformer(
            input_dim=hidden_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #  Apply projection   (input comes in as (B, C, T))
        if self._pre_kind == "linear":
            x = self.preproj(x.transpose(1, 2))  # (B, T, H)
        elif self._pre_kind == "conv1d":
            x = self.preproj(x)                  # (B, H, T)
            x = x.transpose(1, 2)                # (B, T, H)
        elif self._pre_kind == "conv2d":
            x = x.unsqueeze(1)                   # (B,1,C,T)
            x = self.preproj(x)                  # (B,H,1,T)
            x = x.squeeze(2).transpose(1, 2)     # (B, T, H)
        else:  # Identity
            x = x.transpose(1, 2)                # (B, T, C=H)

        x = self.in_dropout(x)
        x = self.in_norm(x)

        # Encoder needs "lengths" -> here all windows have identical length
        lengths = torch.full(
            (x.size(0),), x.size(1), dtype=torch.long, device=x.device
        )

        x, _ = self.encoder(x, lengths)  # (B, hidden)
        x = self.out_dropout(x).mean(dim=1)
        return self.classifier(x)
