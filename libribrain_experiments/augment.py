"""
MEGAugment: SpecAugment-style data-augmentation for (C,T) MEG windows
=====================================================================

All sub-transforms are OFF by default.  Enable them through a config
dict (see example YAML below).

Implemented sub-transforms
--------------------------
- time_masking: zero out blocks of consecutive time-steps
- channel_masking: zero out complete sensor channels
- channel_shuffle: random permutation of a subset of channels
- bandstop_masking: notch-filter selected frequency bands
                    (needs SciPy, harmlessly skipped if unavailable)

Everything is done on-the-fly inside `__call__`, so no extra RAM is used.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from scipy.signal import butter, filtfilt


class MEGAugment:
    def __init__(
        self,
        *,
        time_mask: dict | None = None,
        channel_mask: dict | None = None,
        channel_shuffle: dict | None = None,
        bandstop_mask: dict | None = None,
        apply_mode: str = "random_one",  #  "random_one" | "all"
        sample_rate: int = 250,  # Hz
        device: str | torch.device | None = None,
    ):
        self.time_mask = time_mask or {}
        self.channel_mask = channel_mask or {}
        self.channel_shuffle = channel_shuffle or {}
        self.bandstop_mask = bandstop_mask or {}
        self.fs = sample_rate
        self.device = device
        if apply_mode not in {"random_one", "all"}:
            raise ValueError(
                "MEGAugment: apply_mode must be 'random_one' or 'all', "
                f"got {apply_mode!r}"
            )
        self.apply_mode = apply_mode

        # build the list of candidate ops once
        self._ops = []
        if self.time_mask.get("num", 0) > 0:
            self._ops.append(self._time_mask)
        if self.channel_mask.get("num", 0) > 0:
            self._ops.append(self._channel_mask)
        if self.channel_shuffle.get("p", 0.0) > 0:
            self._ops.append(self._channel_shuffle)
        if self.bandstop_mask and self.bandstop_mask.get("bands", []):
            self._ops.append(self._bandstop_mask)

        # nothing active → identity transform
        if not self._ops:
            self.__call__ = lambda x: x  # tiny shortcut

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        C, T = x.shape
        max_T = self.time_mask.get("T", int(0.1 * T))  # max width
        n_masks = self.time_mask["num"]
        for _ in range(n_masks):
            t = random.randint(0, max_T)
            t0 = random.randint(0, max(1, T - t))
            x[..., t0:t0 + t] = 0.0
        return x

    def _channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        C, _ = x.shape
        n_masks = self.channel_mask["num"]
        chans = random.sample(range(C), k=min(n_masks, C))
        x[chans, :] = 0.0
        return x

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        C, _ = x.shape
        p_shuffle = self.channel_shuffle.get("p", 0.0)
        k = int(round(C * p_shuffle))
        if k > 1:
            idx = random.sample(range(C), k)
            perm = idx.copy()
            random.shuffle(perm)
            x[idx, :] = x[perm, :]
        return x

    def _bandstop_mask(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.bandstop_mask.get("bands", [])
        order = self.bandstop_mask.get("order", 4)
        prob = self.bandstop_mask.get("p", 0.5)
        for fl, fh in bands:
            if random.random() < prob:
                b, a = butter(order, [fl, fh], fs=self.fs, btype="bandstop")
                x_np = filtfilt(b, a, x.cpu().numpy(), axis=-1)

                if x_np.strides[ -1 ] < 0 or not x_np.flags['C_CONTIGUOUS']:
                    x_np = np.ascontiguousarray(x_np)  # a cheap .copy()
                if x_np.dtype != np.float32:
                    x_np = x_np.astype(np.float32, copy=False)
                x = torch.as_tensor(x_np, device=x.device)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (C,T)
            One MEG window.

        Returns
        -------
        Tensor
            Augmented copy (NEW tensor, original untouched).
        """
        x = x.clone()

        # Apply according to the selected strategy
        if self.apply_mode == "all":
            for op in self._ops:
                x = op(x)
            return x

        # Pick one rabdom augmentation op and apply it (random_one)
        op = random.choice(self._ops)
        return op(x)
