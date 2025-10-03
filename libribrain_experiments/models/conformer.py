"""
Conformer-based MEG classifier

- Accepts raw windows of shape (B, 306, T)
- Passes a linear "frontend" -> Conformer encoder -> mean-pool -> classifier
"""

import math
import torch
import torch.nn as nn

# PyTorch >= 2.2 has nn.Conformer, older versions can fall back to torchaudio
try:
    from torch.nn import Conformer           # PyTorch 2.2 / 2.3
except ImportError:
    from torchaudio.models import Conformer  # Torchaudio >= 2.2

from libribrain_experiments.models.configurable_modules.conformer_rope import \
    ConformerRoPE


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


class VlaaiCNNStack(nn.Module):
    """
    VLAAI extractor stack:
      [Conv1d -> LayerNorm(channels) -> LeakyReLU -> ZeroPad1d((0, k-1))] x L
    Preserves time length by padding *after* each conv (as in the TF ref).

    Input:  (B, C_in, T)
    Output: (B, T, C_out)  <- already transposed for your Conformer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters=(256, 256, 256, 128, 128),
        kernels=8,                    # int or list[int], same length as filters if list
        use_layernorm: bool = True,   # matches paper default
        negative_slope: float = 0.01, # LeakyReLU slope
    ):
        super().__init__()
        if isinstance(kernels, int):
            kernels = [kernels] * len(filters)
        if len(filters) != len(kernels):
            raise ValueError("'filters' and 'kernels' must have the same length")

        layers = []
        c_in = in_channels
        for c_out, k in zip(filters, kernels):
            conv = nn.Conv1d(c_in, c_out, kernel_size=k, bias=True, padding=0, stride=1)
            norm = LayerNormOverChannels(c_out) if use_layernorm else nn.Identity()
            act  = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
            pad  = nn.ConstantPad1d((0, k - 1), 0.0)  # pad end so time length stays constant
            layers.append(nn.ModuleDict({"conv": conv, "norm": norm, "act": act, "pad": pad}))
            c_in = c_out

        self.layers = nn.ModuleList(layers)

        # If the final filters don't match desired out_channels (Conformer hidden_size),
        # adapt with a cheap 1x1 conv. If they match, this is Identity.
        last_c = filters[-1] if len(filters) > 0 else in_channels
        self.proj = nn.Conv1d(last_c, out_channels, kernel_size=1) if last_c != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        for layer in self.layers:
            x = layer["conv"](x)          # (B, C', T - k + 1)
            x = layer["norm"](x)          # LayerNorm over channels per time step
            x = layer["act"](x)
            x = layer["pad"](x)           # (B, C', T)
        x = self.proj(x)                  # (B, hidden, T)
        return x.transpose(1, 2)          # -> (B, T, hidden) for the Conformer


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)           # (L, H)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)               # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)          # (1, T, H)


class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        pos = torch.arange(T, device=x.device)
        return x + self.emb(pos)[None, :, :]         # (1, T, H)


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
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        num_classes: int = 2,      # speech / silence
        use_preproj: bool = True,  # whether to add a projection layer
        size: str = None,          # Model size to use
        pre_norm: str | None = None,  # None | "batch" | "layer" | "instance"
        pre_dropout: float = 0.0,
        in_norm: str = None,
        in_bn_eps: float = 1e-5,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
        # Optional BN hyperparams (useful for small batches)
        pre_bn_eps: float = 1e-5,
        pre_bn_momentum: float = 0.1,
        pos_enc: str | None = None,      # None | "sin" | "learned"
        max_pos_embeddings: int | None = 4096,
        pos_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        rope_base: int = 10000,
        # conv1d projection
        conv_kernel_size: int | tuple = 1,
        # VLAAI extractor knobs (used only when use_preproj="vlaai")
        vlaai_filters: tuple = (256, 256, 256, 128, 128),
        vlaai_kernels: int | tuple = 8,
        vlaai_use_layernorm: bool = True,
        vlaai_negative_slope: float = 0.01,
    ):
        super().__init__()

        # If size passed, set the correct hparams
        if size is not None:
            if size.lower()[0] == "t":
                hidden_size = 128
                ffn_dim = 512
                num_layers = 8
                num_heads = 4
            elif size.lower()[0] == "s":
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

        # Pre-projection normalization on (B, C, T)
        self._pre_norm = pre_norm
        if pre_norm == "batch":
            self.pre_norm = nn.BatchNorm1d(
                input_dim, eps=pre_bn_eps, momentum=pre_bn_momentum,
                affine=True
            )
        elif pre_norm == "layer":
            self.pre_norm = LayerNormOverChannels(input_dim)
        elif pre_norm == "instance":
            # per-sample, per-channel across time; robust for tiny batches
            self.pre_norm = nn.InstanceNorm1d(
                input_dim, affine=True, eps=pre_bn_eps, momentum=0.1,
                track_running_stats=False
            )
        else:
            self.pre_norm = nn.Identity()
        self.pre_dropout = nn.Dropout(pre_dropout) if pre_dropout > 0 else nn.Identity()

        #  Front-end projection
        self._pre_kind = None
        if isinstance(use_preproj, str):
            self._pre_kind = use_preproj.lower()
        elif use_preproj:           # legacy True: "linear"
            self._pre_kind = "linear"

        if self._pre_kind == "linear":
            self.preproj = nn.Linear(input_dim, hidden_size)
        elif self._pre_kind == "conv1d":
            # Conv over the time dimension, kernel=1 -> just channel mixing
            if conv_kernel_size > 1:
                self.preproj = nn.Conv1d(
                    input_dim, hidden_size, kernel_size=conv_kernel_size,
                    padding="same",
                )
            else:
                self.preproj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        elif self._pre_kind == "conv2d":
            # conv_kernel_size can be int or (Kh, Kt). If int, use same Kh==Kt.
            if isinstance(conv_kernel_size, (tuple, list)):
                Kh, Kt = int(conv_kernel_size[0]), int(conv_kernel_size[1])
            else:
                Kh = Kt = int(conv_kernel_size)

            # Clamp to valid ranges
            Kh = max(1, min(input_dim, Kh))
            Kt = max(1, Kt)

            # Manual SAME-padding amounts (height=sensors, width=time)
            ph_top  = (Kh - 1) // 2
            ph_bot  = (Kh - 1) - ph_top
            pw_left = (Kt - 1) // 2
            pw_rght = (Kt - 1) - pw_left

            self._height_same_pad = (ph_top, ph_bot)
            self._time_same_pad   = (pw_left, pw_rght)

            # 2D conv that preserves H and W after our manual padding
            self.preproj = nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=(Kh, Kt),
                stride=(1, 1),
                padding=(0, 0),   # we pad manually
                bias=True,
            )
            # else:
            #     self.preproj = nn.Conv2d(1, hidden_size, kernel_size=(input_dim, 1))
        elif self._pre_kind == "vlaai":
            # One VLAAI block (extractor stack) as the projection
            self.preproj = VlaaiCNNStack(
                in_channels=input_dim,
                out_channels=hidden_size,
                filters=vlaai_filters,
                kernels=vlaai_kernels,
                use_layernorm=vlaai_use_layernorm,
                negative_slope=vlaai_negative_slope,
            )
        else:
            if input_dim != hidden_size:
                raise ValueError(
                    "When use_preproj=False you must set hidden_size == input_dim"
                )
            self.preproj = nn.Identity()
        self.in_dropout = nn.Dropout(in_dropout) if in_dropout > 0 else nn.Identity()

        self._in_norm = in_norm
        if in_norm == "layer":
            self.in_norm = nn.LayerNorm(hidden_size)
        elif in_norm == "batch":
            self.in_norm = BatchNormLastDim(hidden_size)
        elif in_norm == "instance":
            # per-sample, per-channel across time; robust for tiny batches
            self.in_norm = nn.InstanceNorm1d(
                input_dim, affine=True, eps=in_bn_eps, momentum=0.1,
                track_running_stats=False
            )
        else:
            self.in_norm = nn.Identity()

        # Positional encoding module
        if max_pos_embeddings is None:
            max_pos_embeddings = seq_len
        pe = pos_enc.lower() if isinstance(pos_enc, str) else None
        if   pe == "sin":
            self.pos_enc = SinusoidalPE(hidden_size, max_pos_embeddings)
        elif pe == "learned":
            self.pos_enc = LearnedPE(hidden_size, max_pos_embeddings)
        else:
            self.pos_enc = nn.Identity()
        self.pos_dropout = nn.Dropout(pos_dropout) if pos_dropout > 0 else nn.Identity()

        if pe == "rope":
            self.encoder = ConformerRoPE(
                input_dim=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_layers=num_layers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
                convolution_first=convolution_first,
                rope_max_seq_len=rope_max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.encoder = Conformer(
                input_dim=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_layers=num_layers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
                convolution_first=convolution_first,
            )
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # NEW: normalize raw sensors
        x = self.pre_norm(x)
        x = self.pre_dropout(x)

        #  Apply projection   (input comes in as (B, C, T))
        if self._pre_kind == "linear":
            x = self.preproj(x.transpose(1, 2))  # (B, T, H)
        elif self._pre_kind == "conv1d":
            x = self.preproj(x)                  # (B, H, T)
            x = x.transpose(1, 2)                # (B, T, H)
        elif self._pre_kind == "conv2d":
            # x: (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
            # F.pad order for NCHW is (W_left, W_right, H_top, H_bottom)
            pl, pr = self._time_same_pad
            pt, pb = self._height_same_pad
            x = torch.nn.functional.pad(x, (pl, pr, pt, pb))
            x = self.preproj(x)        # (B, hidden, C, T)  -- H preserved
            x = x.mean(dim=2)          # collapse sensor axis -> (B, hidden, T)
            x = x.transpose(1, 2)      # (B, T, hidden)
            # x = x.unsqueeze(1)                   # (B,1,C,T)
            # x = self.preproj(x)                  # (B,H,1,T)
            # x = x.squeeze(2).transpose(1, 2)     # (B, T, H)
        elif self._pre_kind == "vlaai":
            x = self.preproj(x)                  # already (B, T, H)
        else:  # Identity
            x = x.transpose(1, 2)                # (B, T, C=H)

        x = self.in_dropout(x)
        x = self.in_norm(x)

        # Add absolute PE AFTER in_norm so Batch/LayerNorm doesn't wash it out
        x = self.pos_enc(x)
        x = self.pos_dropout(x)

        # Encoder needs "lengths" -> here all windows have identical length
        lengths = torch.full(
            (x.size(0),), x.size(1), dtype=torch.long, device=x.device
        )

        x, _ = self.encoder(x, lengths)  # (B, hidden)
        x = self.out_dropout(x).mean(dim=1)
        return self.classifier(x)
