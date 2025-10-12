"""Conformer with RoPE positional encoding.

Based in:
- https://github.com/pytorch/audio Conformer implementation
- Using `torchtune.modules.RotaryPositionalEmbeddings`
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE from torchtune
from torchtune.modules import RotaryPositionalEmbeddings

__all__ = ["ConformerRoPE"]


# Utilities (same behavior as torchaudio)
def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    lengths: (B,)
    returns: key_padding_mask (B, T) with True for PAD positions (like PyTorch
    MHA)
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    # dtype doesn't matter (bool result), but keep device the same
    ar = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
    padding_mask = ar.expand(batch_size, max_length) >= lengths.unsqueeze(1)
    return padding_mask


# Conformer submodules (mirroring torchaudio)
class _ConvolutionModule(nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input
            channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution
            layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each
            convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm.
            (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            (
                nn.GroupNorm(num_groups=1, num_channels=num_channels)
                if use_group_norm
                else nn.BatchNorm1d(num_channels)
            ),
            nn.SiLU(),
            nn.Conv1d(
                num_channels, input_dim, kernel_size=1, stride=1, padding=0, bias=bias
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> conv expects (B, C, T)
        y = self.layer_norm(x)
        y = y.transpose(1, 2)
        y = self.sequential(y)
        return y.transpose(1, 2)


class _FeedForwardModule(nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(x)


class _MultiheadSelfAttentionRoPE(nn.Module):
    """MHA with RoPE applied to Q and K.

    Expects/returns time-major (T, B, D) to match ConformerLayer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even.")

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.drop = nn.Dropout(dropout)

        # RoPE expects [b, s, n_h, h_d]
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
        )

    @staticmethod
    def _shape_bsnd(x: torch.Tensor, B: int, T: int, H: int, Hd: int) -> torch.Tensor:
        # (B, T, D) -> (B, T, H, Hd)
        return x.view(B, T, H, Hd)

    def forward(
        self,
        x_tbd: torch.Tensor,  # (T, B, D)
        key_padding_mask: Optional[
            torch.Tensor
        ] = None,  # (B, T), True for PAD (mask out keys)
        input_pos: Optional[
            torch.Tensor
        ] = None,  # (B, T) positions; if None use 0..T-1
    ) -> torch.Tensor:
        T, B, D = x_tbd.shape
        x_btd = x_tbd.transpose(0, 1)  # (B, T, D)

        qkv = self.qkv(x_btd)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        H, Hd = self.num_heads, self.head_dim

        # reshape to (B, T, H, Hd)
        q = self._shape_bsnd(q, B, T, H, Hd)
        k = self._shape_bsnd(k, B, T, H, Hd)
        v = self._shape_bsnd(v, B, T, H, Hd)

        # positions
        if input_pos is None:
            # default 0..T-1 for every sequence in batch
            input_pos = torch.arange(T, device=x_btd.device).unsqueeze(0).expand(B, T)

        # apply RoPE to Q and K (keeps shape (B, T, H, Hd))
        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        # move heads to batch: (B, H, T, Hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention scores: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if key_padding_mask is not None:
            # key mask: True=PAD -> set scores to -inf where keys are PAD
            # expand to (B, 1, 1, T) to mask the key dimension
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,T)
            scores = scores.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        # context: (B, H, T, Hd)
        ctx = torch.matmul(attn, v)

        # back to (B, T, D)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(ctx)  # (B, T, D)
        return out.transpose(0, 1)  # (T, B, D)


class ConformerLayerRoPE(nn.Module):
    r"""Conformer layer that constitutes Conformer with RoPE support.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution
            layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than
            ``BatchNorm1d`` in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead
            of the attention module. (Default: ``False``)
        rope_max_seq_len (int, optional): Maximum expected sequence length for
            the model, if exceeded the cached freqs will be recomputed
        rope_base (int, optional): The base for the geometric progression used
            to compute the rotation angles
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        rope_max_seq_len: int = 4096,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = _MultiheadSelfAttentionRoPE(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            rope_max_seq_len=rope_max_seq_len,
            rope_base=rope_base,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, x_tbd: torch.Tensor) -> torch.Tensor:
        residual = x_tbd
        x_btd = x_tbd.transpose(0, 1)
        x_btd = self.conv_module(x_btd)  # (B, T, D)
        return residual + x_btd.transpose(0, 1)

    def forward(
        self,
        x_tbd: torch.Tensor,  # (T, B, D)
        key_padding_mask: Optional[torch.Tensor],  # (B, T) True=PAD
        input_pos: Optional[torch.Tensor] = None,  # (B, T) optional positions
    ) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in
                self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        # FeedForward (pre)
        residual = x_tbd
        x = self.ffn1(x_tbd)
        x = x * 0.5 + residual

        # Conv first (optional)
        if self.convolution_first:
            x = self._apply_convolution(x)

        # Self-attention with RoPE
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask, input_pos=input_pos)
        x = self.self_attn_dropout(x)
        x = x + residual

        # Conv after (default)
        if not self.convolution_first:
            x = self._apply_convolution(x)

        # FeedForward (post)
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        # Final norm
        x = self.final_layer_norm(x)
        return x


class ConformerRoPE(nn.Module):
    """
    Conformer with Rotary Positional Embeddings (RoPE) in self-attention.

    API matches torchaudio.models.Conformer:
      - forward(input: (B, T, D), lengths: (B,)) -> (B, T, D), lengths

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's
            depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than
            ``BatchNorm1d`` in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead
            of the attention module. (Default: ``False``)
        rope_max_seq_len (int, optional): Maximum expected sequence length for
            the model, if exceeded the cached freqs will be recomputed
        rope_base (int, optional): The base for the geometric progression used
            to compute the rotation angles

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        rope_max_seq_len: int = 4096,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConformerLayerRoPE(
                    input_dim=input_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                    rope_max_seq_len=rope_max_seq_len,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x_btd: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element
                representing number of valid frames for i-th batch element in
                ``input``.

        x_btd: (B, T, D)
        lengths: (B,)

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element
                    representing number of valid frames for i-th batch element
                    in output frames.
        """
        key_padding_mask = _lengths_to_padding_mask(lengths)  # (B, T) True=PAD

        # default positions per sequence (B, T): 0..T-1
        # NOTE: RoPE handles packed batches if you provide per-token positions here.
        B, T, _ = x_btd.shape
        input_pos = torch.arange(T, device=x_btd.device).unsqueeze(0).expand(B, T)

        x_tbd = x_btd.transpose(0, 1)
        for layer in self.layers:
            x_tbd = layer(x_tbd, key_padding_mask=key_padding_mask, input_pos=input_pos)
        return x_tbd.transpose(0, 1), lengths
