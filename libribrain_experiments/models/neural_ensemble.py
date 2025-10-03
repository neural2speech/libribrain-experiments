from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn


class FrozenBackbone(nn.Module):
    """Wrap a ClassificationModule checkpoint as a plain nn.Module.

    - forwards raw (B,C,T) and returns logits (B,num_classes)
    - can be frozen (no grads, eval mode)
    - automatically uses no_grad() when frozen (but allows later unfreeze)
    """

    def __init__(self, ckpt_path: str, freeze: bool = True, strict: bool = False):
        super().__init__()
        # Lazy import here to break the import cycle
        from libribrain_experiments.models.configurable_modules.classification_module import (
            ClassificationModule,
        )

        # Load Lightning module; its forward is just the sequential pipeline
        model: ClassificationModule = ClassificationModule.load_from_checkpoint(
            ckpt_path, map_location="cpu", strict=strict
        )
        self.model = model

        # Infer output classes from final Linear if possible
        last = None
        for m in model.modules_list[::-1]:
            last = m
            if isinstance(m, nn.Linear):
                break
        if isinstance(last, nn.Linear):
            self.num_classes = last.out_features
        else:
            self.num_classes = None  # will be validated later

        # optionally freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._frozen = freeze

    def train(self, mode: bool = True):
        # If backbone remains frozen, keep it in eval; otherwise follow parent mode
        if self._frozen:
            self.model.eval()
            return self
        self.model.train(mode)
        return super().train(mode)

    def is_frozen(self) -> bool:
        # consider unfrozen if *any* param requires grad
        if any(p.requires_grad for p in self.model.parameters()):
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.is_frozen() else nullcontext()
        with ctx:
            return self.model(x)  # (B, C)


class NeuralEnsemble(nn.Module):
    """Trainable ensemble over N pretrained models' logits.

    Forward:
        x: (B, C_in, T) -> list logits_i: (B, C) -> stack (B, N, C)
        -> combiner head -> (B, C)

    Args
    ----
    members : list[dict]
        Each dict must contain:
          - checkpoint: str  (path to a ClassificationModule .ckpt)
        Optional per-member:
          - freeze: bool = True
    num_classes : int
        Number of classes (validated against members).
    head : str
        "per_class_affine" | "linear" | "mlp"
        - per_class_affine: y_c = b_c + sum_n W[n,c] * logit[n,c]
        - linear: flatten(N*C) -> Linear -> C
        - mlp:    flatten(N*C) -> Linear(h) -> GELU -> Dropout -> Linear(C)
    init : str
        "avg" | "zeros"
        - avg: initialize W so output is plain average of logits.
    freeze_backbones_default : bool
        Default freeze for members without explicit 'freeze'.
    mlp_hidden : int
        Hidden size for head="mlp".
    dropout : float
        Dropout inside MLP head.
    strict_shapes : bool
        Pass to PL load_from_checkpoint(strict=...) for backbones.
    """

    def __init__(
        self,
        *,
        members: list[dict],
        num_classes: int,
        head: str = "per_class_affine",
        init: str = "avg",
        freeze_backbones_default: bool = True,
        mlp_hidden: int = 512,
        dropout: float = 0.0,
        strict_shapes: bool = False,
    ):
        super().__init__()
        assert (
            isinstance(members, (list, tuple)) and len(members) >= 1
        ), "Provide at least one member."
        self.num_classes = int(num_classes)
        self.head_type = head.lower()
        self.dropout_p = float(dropout)

        # Load members
        backs = []
        for m in members:
            ckpt = m.get("checkpoint")
            if not isinstance(ckpt, str):
                raise ValueError("Each ensemble member needs 'checkpoint: <path>'")
            freeze = bool(m.get("freeze", freeze_backbones_default))
            backs.append(FrozenBackbone(ckpt, freeze=freeze, strict=strict_shapes))
        self.members = nn.ModuleList(backs)
        self.N = len(self.members)

        # Validate output sizes
        for idx, b in enumerate(self.members):
            if b.num_classes is not None and b.num_classes != self.num_classes:
                raise ValueError(
                    f"Member {idx} outputs {b.num_classes} classes but ensemble num_classes={self.num_classes}"
                )

        # Combiner head
        C = self.num_classes
        if self.head_type == "per_class_affine":
            # W: (N,C), b: (C,)
            self.W = nn.Parameter(torch.zeros(self.N, C))
            self.b = nn.Parameter(torch.zeros(C))
            if init == "avg":
                with torch.no_grad():
                    self.W.fill_(1.0 / self.N)
        elif self.head_type == "linear":
            self.flatten = nn.Flatten()  # (B, N*C)
            self.head = nn.Linear(self.N * C, C)
            if init == "avg":
                with torch.no_grad():
                    self.head.weight.zero_()
                    self.head.bias.zero_()
        elif self.head_type == "mlp":
            self.flatten = nn.Flatten()
            self.net = nn.Sequential(
                nn.Linear(self.N * C, mlp_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
                nn.Linear(mlp_hidden, C),
            )
            if init == "avg":
                with torch.no_grad():
                    for p in self.net.parameters():
                        if p.dim() == 2:
                            nn.init.xavier_uniform_(p, gain=1.0)
        else:
            raise ValueError(f"Unknown head: {head}")

    def train(self, mode: bool = True):
        # Default train/eval; but keep fully-frozen members in eval to disable dropout/BN updates
        super().train(mode)
        for b in self.members:
            b.train(mode and not b.is_frozen())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect logits from each backbone
        logits = [b(x) for b in self.members]  # list of (B,C)
        L = torch.stack(logits, dim=1)  # (B, N, C)

        if self.head_type == "per_class_affine":
            # y_bc = b_c + sum_n W[n,c] * L_bnc
            y = torch.einsum("bnc,nc->bc", L, self.W) + self.b
        elif self.head_type == "linear":
            y = self.head(self.flatten(L))
        else:  # mlp
            y = self.net(self.flatten(L))
        return y
