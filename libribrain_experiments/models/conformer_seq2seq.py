"""
Conformer encoder + Transformer decoder for per-time-step VAD labelling.

Returns logits of shape (B*T, num_classes) so it can be used with the
existing CE/BCE losses without touching anything else.
"""
import math
import warnings

import torch
import torch.nn as nn

try:
    from torch.nn import Conformer  # PyTorch >= 2.2
except ImportError:
    from torchaudio.models import Conformer  # Torchaudio >= 2.2


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ConformerSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim: int = 306,      # sensors
        seq_len: int = 125,        # time steps in the HDF5 window (0-0.8 s)
        hidden_size: int = 256,    # Conformer model size
        ffn_dim: int = 512,        # FFN expansion
        num_heads: int = 4,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        num_classes: int = 2,      # speech / silence
        use_preproj: bool = True,  # whether to add a projection layer
        size: str = None,          # Model size to use
        in_norm: str = None,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
        init_from_encoder: str | dict | None = None,
        load_modules: list[str] | tuple[str, ...] = ("preproj", "in_norm", "encoder"),
        strict_shapes: bool = False,
        freeze_loaded: bool = False,
    ):
        super().__init__()

        # If size passed, set the correct hparams
        if size is not None:
            if size.lower()[0] == "s":
                hidden_size = 144
                ffn_dim = 576
                num_enc_layers = 16
                num_dec_layers = 16
                num_heads = 4
            elif size.lower()[0] == "m":
                hidden_size = 256
                ffn_dim = 1024
                num_enc_layers = 16
                num_dec_layers = 16
                num_heads = 4
            elif size.lower()[0] == "l":
                hidden_size = 512
                ffn_dim = 2048
                num_enc_layers = 17
                num_dec_layers = 17
                num_heads = 8
            else:
                raise ValueError(
                    f"Unknown Conformer size: {size}"
                )

        # Front-end projection
        self._pre_kind = None
        if isinstance(use_preproj, str):
            self._pre_kind = use_preproj.lower()
        elif use_preproj:           # legacy True: "linear"
            self._pre_kind = "linear"

        if self._pre_kind == "linear":
            self.preproj = nn.Linear(input_dim, hidden_size)
        elif self._pre_kind == "conv1d":
            # Conv over the time dimension, kernel=1 -> just channel mixing
            self.preproj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        elif self._pre_kind == "conv2d":
            # Treat sensors as "height", time as "width"
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

        # Encoder
        self.encoder = Conformer(
            input_dim=hidden_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_enc_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )

        # Decoder
        # self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_enc = PositionalEncoding(hidden_size, max_len=5000)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # Classifier
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

        # causal mask (square) re-used every forward pass
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1),
            persistent=False,
        )

        # Optionally load pretrained encoder weights
        if init_from_encoder is not None:
            self._init_from_encoder_weights(
                init_from_encoder,
                load_modules=tuple(load_modules),
                strict_shapes=bool(strict_shapes),
                freeze_loaded=bool(freeze_loaded),
            )

    @staticmethod
    def _to_state_dict(obj):
        """Accept a path, a full checkpoint dict, or a raw state_dict."""
        if isinstance(obj, (str, bytes)):
            ckpt = torch.load(obj, map_location="cpu")
        elif isinstance(obj, dict):
            ckpt = obj
        else:
            raise TypeError(
                "init_from_encoder must be a path or a dict (state_dict/checkpoint)"
            )
        # Lightning checkpoints often store weights under 'state_dict'
        return ckpt.get("state_dict", ckpt)

    @staticmethod
    def _extract_submodule(
        sd: dict[str, torch.Tensor], subname: str
    ) -> dict[str, torch.Tensor]:
        """
        From a (possibly prefixed) state_dict, collect tensors for a given submodule,
        returning keys relative to that submodule.

        Works with keys like:
          'preproj.weight'
          'model.preproj.weight'
          'modules_list.0.conformer.preproj.weight'
          'backbone.encoder.layers.0....' (for subname='encoder')
        """
        out = {}
        token = f".{subname}."
        for k, v in sd.items():
            if k.startswith(subname + "."):
                out[k[len(subname) + 1 :]] = v
            elif token in k:
                # keep the suffix after ".{subname}."
                out[k.split(token, 1)[1]] = v
        return out

    def _init_from_encoder_weights(
        self,
        src: str | dict,
        load_modules: tuple[str, ...] = ("preproj", "in_norm", "encoder"),
        strict_shapes: bool = False,
        freeze_loaded: bool = False,
    ):
        sd = self._to_state_dict(src)

        loaded_any = False
        for name, module in (
            ("preproj", self.preproj),
            ("in_norm", self.in_norm),
            ("encoder", self.encoder),
        ):
            if name not in load_modules:
                continue
            if isinstance(module, nn.Identity):
                continue  # nothing to load
            sub_sd = self._extract_submodule(sd, name)
            if not sub_sd:
                warnings.warn(
                    f"[ConformerSeq2Seq] No weights found for '{name}' in checkpoint - skipping."
                )
                continue
            try:
                missing, unexpected = module.load_state_dict(
                    sub_sd, strict=strict_shapes
                )
                if missing:
                    warnings.warn(
                        f"[ConformerSeq2Seq] Missing keys when loading '{name}': {missing}"
                    )
                if unexpected:
                    warnings.warn(
                        f"[ConformerSeq2Seq] Unexpected keys when loading '{name}': {unexpected}"
                    )
                loaded_any = True
            except RuntimeError as e:
                if strict_shapes:
                    raise
                warnings.warn(
                    f"[ConformerSeq2Seq] Shape mismatch while loading '{name}': {e}\n - skipped"
                )

        if not loaded_any:
            warnings.warn(
                "[ConformerSeq2Seq] No encoder-side weights loaded (check prefixes or shapes)."
            )

        if freeze_loaded:
            for pname, p in self.named_parameters():
                if pname.startswith(("preproj", "in_norm", "encoder")):
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, T)

        Returns
        -------
        logits : Tensor, shape (B*T, num_classes)
        """
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

        B, T, H = x.shape

        x = self.in_dropout(x)
        x = self.in_norm(x)

        lengths = torch.full((B,), T, device=x.device)
        enc, _ = self.encoder(x, lengths)  # (B,T,H)

        tgt = torch.zeros_like(enc)  # (B,T,H)
        # tgt = self.start_token.expand(B, -1, -1).repeat(1, T, 1)
        tgt = self.pos_enc(tgt)
        tgt_mask = self._causal_mask[:T, :T]  # causal autoregressive

        dec = self.decoder(
            tgt,
            enc,
            tgt_mask=tgt_mask,  # Causal mask
            # tgt_mask=None,        # Bidirectional decoder
            memory_key_padding_mask=None,
        )  # (B,T,H)

        logits = self.out_dropout(self.classifier(dec))  # (B,T,num_classes)
        logits = logits.permute(0, 2, 1)                 # (B,num_classes,T)
        return logits.reshape(B * T, -1)                 # (B*T,num_classes)
