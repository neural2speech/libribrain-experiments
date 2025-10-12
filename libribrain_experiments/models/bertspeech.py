# libribrain_experiments/models/bertspeech.py

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# Reuse the same helpers used by Conformer
from libribrain_experiments.models.util_layers import (
    BatchNormLastDim,          # BN over last dim of (B, T, H)
    LayerNormOverChannels,     # LN over channels for (B, C, T)
)
from libribrain_experiments.models.conformer import (
    VlaaiCNNStack,
    SinusoidalPE,
    LearnedPE,
)


class WhisperConvStem(nn.Module):
    """
    Whisper feature-extraction stem: two Conv1d layers with kernel=3, GELU;
    the second conv downsamples time by 2 (stride=2). Padding=1 keeps lengths
    compatible with 'same' convs: T -> T (conv1), then T -> ceil(T/2) (conv2).
    Input:  (B, C_in, T)
    Output: (B, H, T')   where T' = ceil(T/2)
    """
    def __init__(self, in_channels: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.act2  = nn.GELU()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample_factor = 2

    def forward(self, x):                 # x: (B, C, T)
        x = self.drop(self.act1(self.conv1(x)))   # (B, H, T)
        x = self.drop(self.act2(self.conv2(x)))   # (B, H, T//2)
        return x



class BertSpeech(nn.Module):
    """BERT encoder for MEG with modular frontend + normalizations.

    Parameters
    ----------
    input_dim : int
        # sensors (C).
    seq_len : int, default 125
        Expected time-steps (T). Used for manual pos-encoding shape and BERT config.
    num_classes : int, default 2
    hidden_size : int, default 768
        BERT hidden size (H). Must match projection output.
    emb_size : int, default 768
        First MLP width for use_preproj='linear'.
    factor_size : int, default 128
        Bottleneck size for use_preproj='linear'.
    num_heads : int, default 12
    num_layers : int, default 4
    use_preproj : str|bool, default 'linear'
        One of {'linear','conv1d','conv2d', False}. If False, identity (requires input_dim==hidden_size).
    pre_norm : None | 'batch' | 'layer' | 'instance', default None
        Normalization over raw (B, C, T).
    pre_dropout : float, default 0.0
    in_norm : None | 'batch' | 'layer', default None
        Normalization over sequence embeddings (B, T, H).
    in_dropout : float, default 0.0
    out_dropout : float, default 0.0
    pre_bn_eps : float, default 1e-5
    pre_bn_momentum : float, default 0.1
    pos_enc : None | 'sin' | 'learned', default None
        External positional encoding added before BERT.
    add_manual_pos : bool, default True
        If True, add a learnable `(1, T, H)` positional tensor before feeding to BERT.
        Note: BERT will ALSO add its own positional embeddings; set False to rely only on BERT's.
    disable_bert_pos_when_external : bool, default True
        If True and any external PE is used, zero & freeze BERT position/token embeddings.
        vlaai_* : knobs for the VLAAI extractor used when use_preproj='vlaai'
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 125,
        num_classes: int = 2,
        hidden_size: int = 768,
        emb_size: int = 768,
        factor_size: int = 128,
        num_heads: int = 12,
        num_layers: int = 4,
        use_preproj: str | bool = "linear",
        pre_norm: str | None = None,
        pre_dropout: float = 0.0,
        in_norm: str | None = None,
        in_dropout: float = 0.0,
        out_dropout: float = 0.0,
        # Optional BN hyperparams (useful for small batches)
        pre_bn_eps: float = 1e-5,
        pre_bn_momentum: float = 0.1,
        # Positional encoding options
        pos_enc: str | None = None,      # None | "sin" | "learned"
        max_pos_embeddings: int | None = 4096,
        pos_dropout: float = 0.0,
        add_manual_pos: bool = True,
        disable_bert_pos_when_external: bool = True,
        # VLAAI extractor knobs (used only when use_preproj="vlaai")
        vlaai_filters: tuple = (256, 256, 256, 128, 128),
        vlaai_kernels: int | tuple = 8,
        vlaai_use_layernorm: bool = True,
        vlaai_negative_slope: float = 0.01,
        # Whisper options
        whisper_size: str | None = None,  # "tiny"|"base"|"small"|"medium"|"large"
        whisper_stem_dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.add_manual_pos = bool(add_manual_pos)
        self.disable_bert_pos_when_external = bool(disable_bert_pos_when_external)

        # Apply Whisper size presets
        WHISPER_SIZES = {
            "tiny":   dict(hidden_size=384,  num_layers=4,  num_heads=6),
            "base":   dict(hidden_size=512,  num_layers=6,  num_heads=8),
            "small":  dict(hidden_size=768,  num_layers=12, num_heads=12),
            "medium": dict(hidden_size=1024, num_layers=24, num_heads=16),
            "large":  dict(hidden_size=1280, num_layers=32, num_heads=20),
        }
        if whisper_size is not None:
            ws = WHISPER_SIZES[whisper_size.lower()]
            hidden_size = self.hidden_size = ws["hidden_size"]
            num_layers  = ws["num_layers"]
            num_heads   = ws["num_heads"]

        # Pre-normalization over raw sensors (B, C, T)
        self._pre_norm = pre_norm
        if pre_norm == "batch":
            self.pre_norm = nn.BatchNorm1d(
                input_dim, eps=pre_bn_eps, momentum=pre_bn_momentum, affine=True
            )
        elif pre_norm == "layer":
            self.pre_norm = LayerNormOverChannels(input_dim)
        elif pre_norm == "instance":
            # per-sample, per-channel across time (robust for tiny batches)
            self.pre_norm = nn.InstanceNorm1d(
                input_dim, affine=True, eps=pre_bn_eps, momentum=0.1, track_running_stats=False
            )
        else:
            self.pre_norm = nn.Identity()
        self.pre_dropout = nn.Dropout(pre_dropout) if pre_dropout > 0 else nn.Identity()

        # Frontend projection to hidden_size (B, T, H)
        self._pre_kind = None
        if isinstance(use_preproj, str):
            self._pre_kind = use_preproj.lower()
        elif use_preproj:  # legacy True -> "linear" MLP
            self._pre_kind = "linear"

        self.preproj_conv1d = None
        self.preproj_conv2d = None
        self.preproj_vlaai = None
        self.preproj_whisper = None
        self.embedding = self.factorization = self.projection = None
        if self._pre_kind == "linear":
            # Apply on (B, T, C) with MLP -> (B, T, H)
            self.embedding = nn.Linear(input_dim, emb_size)
            self.factorization = nn.Linear(emb_size, factor_size)
            self.projection = nn.Linear(factor_size, hidden_size)
            # conv layers not used in this mode
        elif self._pre_kind == "conv1d":
            # Conv over time: (B, C, T) -> (B, H, T) -> (B, T, H)
            self.preproj_conv1d = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        elif self._pre_kind == "conv2d":
            # Treat sensors as "height", time as "width"
            self.preproj_conv2d = nn.Conv2d(1, hidden_size, kernel_size=(input_dim, 1))
        elif self._pre_kind == "vlaai":
            # VLAAI extractor stack adapted to output hidden_size channels
            self.preproj_vlaai = VlaaiCNNStack(
                in_channels=input_dim,
                out_channels=hidden_size,
                filters=vlaai_filters,
                kernels=vlaai_kernels,
                use_layernorm=vlaai_use_layernorm,
                negative_slope=vlaai_negative_slope,
            )
        elif self._pre_kind == "whisper":
            self.preproj_whisper = WhisperConvStem(input_dim, hidden_size, dropout=whisper_stem_dropout)
            self.downsample_factor = 2
        else:
            # Identity: requires shapes to match
            if input_dim != hidden_size:
                raise ValueError("use_preproj=False requires input_dim == hidden_size")

        # In-normalization over (B, T, H)
        self._in_norm = in_norm
        if in_norm == "layer":
            self.in_norm = nn.LayerNorm(hidden_size)
        elif in_norm == "batch":
            self.in_norm = BatchNormLastDim(hidden_size)
        else:
            self.in_norm = nn.Identity()
        self.in_dropout = nn.Dropout(in_dropout) if in_dropout > 0 else nn.Identity()

        # Positional encoding module
        if max_pos_embeddings is None:
            max_pos_embeddings = self.seq_len
        pe_kind = pos_enc.lower() if isinstance(pos_enc, str) else None
        if pe_kind == "sin":
            self.pos_enc = SinusoidalPE(hidden_size, max_pos_embeddings)
        elif pe_kind == "learned":
            self.pos_enc = LearnedPE(hidden_size, max_pos_embeddings)
        else:
            self.pos_enc = nn.Identity()
        self.pos_dropout = nn.Dropout(pos_dropout) if pos_dropout > 0 else nn.Identity()

        # Optional extra learnable (1,T,H) table
        if self.add_manual_pos:
            self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))
            nn.init.normal_(self.pos_encoding, std=0.02)
        else:
            self.register_parameter("pos_encoding", None)

        # BERT encoder
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=seq_len,
            vocab_size=1,         # dummy (we use inputs_embeds)
            pad_token_id=0,
        )
        self.encoder = BertModel(config)

        # Optional manual positional parameter (match T,H)
        # if self.add_manual_pos:
        #     self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        #     nn.init.normal_(self.pos_encoding, std=0.02)  # small init like BERT
        # else:
        #     self.register_parameter("pos_encoding", None)

        # If any external PE is active, remove BERT's absolute & token-type embeddings' influence.
        self._external_pe_active = (pe_kind in {"sin", "learned"}) or self.add_manual_pos
        if self._external_pe_active and self.disable_bert_pos_when_external:
            with torch.no_grad():
                # Zero them so they do not bias the sequence
                self.encoder.embeddings.position_embeddings.weight.zero_()
                self.encoder.embeddings.token_type_embeddings.weight.zero_()
            # Freeze
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False

        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns logits: (B, num_classes)
        """
        # pre-norm on raw sensors
        x = self.pre_norm(x)          # (B, C, T)
        x = self.pre_dropout(x)

        # projection to (B, T, H)
        if self._pre_kind == "linear":
            x = x.transpose(1, 2)     # (B, T, C)
            x = self.embedding(x)
            x = self.factorization(x)
            x = self.projection(x)    # (B, T, H)
        elif self._pre_kind == "conv1d":
            x = self.preproj_conv1d(x)   # (B, H, T)
            x = x.transpose(1, 2)        # (B, T, H)
        elif self._pre_kind == "conv2d":
            x = x.unsqueeze(1)           # (B, 1, C, T)
            x = self.preproj_conv2d(x)   # (B, H, 1, T)
            x = x.squeeze(2).transpose(1, 2)  # (B, T, H)
        elif self._pre_kind == "vlaai":
            x = self.preproj_vlaai(x)    # (B, T, H)
        elif self._pre_kind == "whisper":
            x = self.preproj_whisper(x)  # (B, H, T') with T'~ceil(T/2)
            x = x.transpose(1, 2)        # -> (B, T', H)
        else:  # identity
            x = x.transpose(1, 2)        # (B, T, H==C)

        # in-norm on sequence embeddings
        x = self.in_dropout(x)
        x = self.in_norm(x)              # (B, T, H)

        # (optional) manual positional encoding
        if self.add_manual_pos:
            # handle any small T drift defensively
            T = x.size(1)
            if self.pos_encoding.size(1) != T:
                pe = self.pos_encoding[:, :T, :]
            else:
                pe = self.pos_encoding
            x = x + pe
        x = self.pos_enc(x)
        x = self.pos_dropout(x)

        # Add absolute PE AFTER in_norm so Batch/LayerNorm doesn't wash it out
        x = self.pos_enc(x)
        x = self.pos_dropout(x)

        # BERT encoder
        outputs = self.encoder(inputs_embeds=x)
        seq = outputs.last_hidden_state          # (B, T, H)

		# pool + classify
        pooled = self.out_dropout(seq).mean(dim=1)
        return self.classifier(pooled)
