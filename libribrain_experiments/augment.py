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

from collections import defaultdict
import random
from typing import Sequence, List, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
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

        # nothing active -> identity transform
        if not self._ops:
            self.__call__ = lambda x: x  # tiny shortcut

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        C, T = x.shape
        max_T = self.time_mask.get("T", int(0.1 * T))  # max width
        n_masks = self.time_mask["num"]
        for _ in range(n_masks):
            t = random.randint(0, max_T)
            t0 = random.randint(0, max(1, T - t))
            x[..., t0 : t0 + t] = 0.0
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

                if x_np.strides[-1] < 0 or not x_np.flags["C_CONTIGUOUS"]:
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


class DynamicGroupedDataset(Dataset):
    """
    Re-creates new random label-homogeneous groups every time
    `reshuffle()` is called (typically once per epoch).

    Parameters
    ----------
    base : the original single-sample dataset
    grouped_samples : int | Sequence[int]
        - int -> single size (old behavior)
        - list/tuple -> multiple sizes, e.g. [100,100,50,20]
          (one random pass per size; groups from all passes are concatenated)
    average : bool
        True  -> return mean(stack(group))    (same shape as base sample)
        False -> return cat(group, dim=0)     (time dimension x N)
    drop_remaining : bool
        When True skip incomplete groups.
    """

    def __init__(
        self,
        base: Dataset,
        grouped_samples: int | Sequence[int] = 100,
        average: bool = True,
        drop_remaining: bool = True,
    ):
        self.base = base
        # normalize to a non-empty list of sizes
        if isinstance(grouped_samples, (list, tuple)):
            self.group_sizes: List[int] = [
                int(k) for k in grouped_samples if int(k) >= 1
            ]
        else:
            self.group_sizes = [int(grouped_samples)]
        if not self.group_sizes:
            raise ValueError("DynamicGroupedDataset: no valid group sizes provided")

        self.average = average
        self.drop_remaining = drop_remaining

        # expose bookkeeping attributes so the rest of the code keeps working
        self.labels_sorted = getattr(base, "labels_sorted", None)
        self.channel_means = getattr(base, "channel_means", None)
        self.channel_stds = getattr(base, "channel_stds", None)

        self._build_groups()  # initial grouping

    def _build_groups(self) -> None:
        """Make one or more random passes (one per size) and form groups."""
        all_groups: List[List[int]] = []

        for k in self.group_sizes:
            perm = torch.randperm(len(self.base))
            bucket = defaultdict(list)  # label -> list[int]

            for i in perm:
                lbl = self.base[i][1].item()
                bucket[lbl].append(int(i))
                if len(bucket[lbl]) == k:
                    all_groups.append(bucket[lbl])
                    bucket[lbl] = []

            if not self.drop_remaining:
                all_groups.extend(g for g in bucket.values() if g)

        self.groups = all_groups

    def reshuffle(self) -> None:
        """Call this once per epoch to rebuild `self.groups` for all sizes."""
        self._build_groups()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        ids = self.groups[idx]
        xs = [self.base[i][0] for i in ids]
        x = torch.stack(xs).mean(0) if self.average else torch.cat(xs, dim=0)
        y = self.base[ids[0]][1]  # all labels identical
        return x, y


class DynamicGroupedDatasetPerClass(Dataset):
    """
    Like DynamicGroupedDataset, but supports per-class group sizes.

    Parameters
    ----------
    base : dataset yielding (x, y) with y in {0..C-1}
    per_class_sizes : Dict[int, Union[int, Sequence[int]]]
        Map from class-id -> group size(s). Each value can be an int or a list of ints.
        If a class id is missing, `default_sizes` is used.
    default_sizes : Union[int, Sequence[int]]
        Global fallback size(s) used for classes not present in `per_class_sizes`.
    average : bool
        True -> return mean over grouped samples; False -> concat in time.
    drop_remaining : bool
        If True, incomplete groups are discarded; otherwise kept.
    """

    def __init__(
        self,
        base: Dataset,
        per_class_sizes: Dict[int, Union[int, Sequence[int]]],
        *,
        default_sizes: Union[int, Sequence[int]] = 100,
        average: bool = True,
        drop_remaining: bool = True,
    ):
        self.base = base
        self.average = average
        self.drop_remaining = drop_remaining

        # Normalize sizes to {class_id: [int, ...]}
        def _norm_sizes(v) -> List[int]:
            if isinstance(v, (list, tuple)):
                s = [int(k) for k in v if int(k) >= 1]
            else:
                s = [int(v)]
            if not s:
                raise ValueError("DynamicGroupedDatasetPerClass: empty group sizes")
            return s

        self.default_sizes = _norm_sizes(default_sizes)
        self.per_class_sizes: Dict[int, List[int]] = {
            int(cid): _norm_sizes(sizes) for cid, sizes in per_class_sizes.items()
        }

        # Expose bookkeeping attributes for downstream code
        self.labels_sorted = getattr(base, "labels_sorted", None)
        self.channel_means = getattr(base, "channel_means", None)
        self.channel_stds = getattr(base, "channel_stds", None)

        self._build_groups()

    def _class_sizes(self, cid: int) -> List[int]:
        return self.per_class_sizes.get(cid, self.default_sizes)

    def _build_groups(self) -> None:
        # Collect indices per class once (speeds up multiple passes)
        per_class_indices: Dict[int, List[int]] = {}
        for i in range(len(self.base)):
            y = int(self.base[i][1])
            per_class_indices.setdefault(y, []).append(i)

        all_groups: List[List[int]] = []
        for cid, idxs in per_class_indices.items():
            if not idxs:
                continue
            idxs = torch.tensor(idxs)
            # One random pass *per size* requested for this class
            for k in self._class_sizes(cid):
                perm = idxs[torch.randperm(len(idxs))].tolist()
                bucket: List[int] = []
                for i in perm:
                    bucket.append(i)
                    if len(bucket) == k:
                        all_groups.append(bucket[:])
                        bucket = []
                if not self.drop_remaining and bucket:
                    all_groups.append(bucket[:])

        self.groups = all_groups

    def reshuffle(self) -> None:
        self._build_groups()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        ids = self.groups[idx]
        xs = [self.base[i][0] for i in ids]
        x = torch.stack(xs).mean(0) if self.average else torch.cat(xs, dim=0)
        y = self.base[ids[0]][1]
        return x, y
