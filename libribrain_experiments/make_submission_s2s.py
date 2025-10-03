#!/usr/bin/env python
"""
Seq2Seq submission: use the full window output, aggregate over overlap,
and emit one probability per LibriBrain segment_idx (stride=1).

Example:
python -m libribrain_experiments.make_submission_s2s \
  --data_path data/ --tmax 2.5 --split test \
  checkpoints/your_conformer_seq2seq.ckpt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from pnpl.datasets import LibriBrainCompetitionHoldout, LibriBrainSpeech
from torchmetrics import F1Score
from tqdm import tqdm

from libribrain_experiments.models.configurable_modules.classification_module import (
    ClassificationModule,
)
from libribrain_experiments.utils import SENSORS_SPEECH_MASK


@torch.inference_mode()
def seq2seq_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward a batch and return probabilities of shape (B, T),
    speech=1 per time-step.

    Expected model logits shape: (B*T, C) with C in {1,2}.
    """
    logits = model(x)  # (B*T, C) expected
    B, C, T = x.shape
    if logits.ndim != 2 or logits.shape[0] != B * T:
        raise RuntimeError(
            f"Expected (B*T, C) logits from seq2seq model; got {tuple(logits.shape)}"
        )
    logits = logits.view(B, T, -1)  # (B, T, C)

    if logits.size(-1) == 1:
        p = torch.sigmoid(logits[..., 0])  # (B, T)
    else:
        p = torch.softmax(logits, dim=-1)[..., 1]  # (B, T)
    return p


def safe_collate(batch):
    xs = [torch.as_tensor(b[0]).clone() for b in batch]
    T = max(t.shape[1] for t in xs)
    xs = [
        torch.nn.functional.pad(t, (0, T - t.shape[1])) if t.shape[1] < T else t
        for t in xs
    ]
    xs = torch.stack(xs, 0)  # (B, C, T)

    # labels present for non-holdout splits (array per window)
    ys = None
    if len(batch[0]) >= 2 and isinstance(batch[0][1], (np.ndarray, torch.Tensor)):
        ys = torch.stack([torch.as_tensor(b[1]) for b in batch])
    return xs, ys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Seq2Seq submission (aggregate full-window outputs).")
    parser.add_argument("checkpoint", help="Path to *.ckpt from the repo")
    parser.add_argument(
        "--output", default="submission_s2s.csv", help="Predictions output CSV"
    )
    parser.add_argument("--labels", default="labels.csv", help="Labels CSV (non-holdout)")
    parser.add_argument("--data_path", default="./data/", help="LibriBrain data root")
    parser.add_argument(
        "--split", default="holdout", choices=["holdout", "train", "validation", "test"]
    )
    parser.add_argument("--tmax", type=float, default=2.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sensor_mask", action="store_true")
    parser.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Skip manual (x-mnean)/std normalisation entirely",
    )
    parser.add_argument(
        "--no-train_stats",
        "--no_train_stats",
        action="store_false",
        dest="use_train_stats",
        help="Compute mean/std on the same split instead of the train split",
    )
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect standardization stats if requested
    chan_means = chan_stds = None
    if args.standardize and args.use_train_stats:
        print("Collecting training-set mean / std ...", end="", flush=True)
        train_tmp = LibriBrainSpeech(
            data_path=args.data_path,
            partition="train",
            tmin=0.0,
            tmax=args.tmax,
            standardize=True,
            preload_files=False,
        )
        chan_means = train_tmp.channel_means
        chan_stds = train_tmp.channel_stds
        if args.sensor_mask:
            chan_means = chan_means[SENSORS_SPEECH_MASK]
            chan_stds = chan_stds[SENSORS_SPEECH_MASK]
        del train_tmp

    # Load split with stride=1 so segment_idx matches EvalAI
    print("Loading LibriBrain dataset...")
    if args.split == "holdout":
        ds = LibriBrainCompetitionHoldout(
            data_path=args.data_path,
            tmin=0.0,
            tmax=args.tmax,
            standardize=False,
            clipping_boundary=None,
            task="speech",
        )
    else:
        ds = LibriBrainSpeech(
            data_path=args.data_path,
            partition=args.split,
            tmin=0.0,
            tmax=args.tmax,
            standardize=False,
            clipping_boundary=None,
            stride=1,  # IMPORTANT: segment_idx definition
        )

    M = len(ds)  # number of windows (segment_idx count)
    T = ds[0][0].shape[1]  # window length
    L_full = M + T - 1  # full absolute timeline covered by stride=1
    center = T // 2
    print(f"Windows (M)={M:,}, T={T}, full timeline L={L_full:,}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
    )

    # Load model
    model: ClassificationModule = ClassificationModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.eval().to(device)

    # Accumulators over absolute timeline
    sum_probs = np.zeros(L_full, dtype=np.float64)
    hit_count = np.zeros(L_full, dtype=np.int32)
    seg_true = []  # for metrics on non-holdout splits
    windows_seen = 0

    print("Running inference and aggregating...")
    for xs, ys in tqdm(loader, unit_scale=args.batch_size, desc="Batches"):
        if args.sensor_mask:
            xs = xs[:, SENSORS_SPEECH_MASK, :]

        if args.standardize and args.use_train_stats:
            cm = torch.as_tensor(chan_means, device=xs.device, dtype=torch.float32)
            cs = torch.as_tensor(chan_stds, device=xs.device, dtype=torch.float32)
            xs = (xs - cm[None, :, None]) / (cs[None, :, None])

        xs = torch.clamp(xs, -10.0, 10.0).to(device)

        p_bt = seq2seq_probs(model, xs)  # (B, T_batch)  - T_batch may be < T
        p_np = p_bt.cpu().numpy()
        B, T_batch = p_np.shape   # !! real length of these windows

        # aggregate into absolute timeline (stride=1: start = windows_seen + b)
        for b in range(B):
            s = windows_seen + b
            e = s + T
            if s >= L_full:  # guard if dataset length not multiple of batch size
                break
            slice_len = min(T_batch, L_full - s)   # <= real win length
            sum_probs[s : s + slice_len] += p_np[b, :slice_len]
            hit_count[s : s + slice_len] += 1

        # collect center labels (if present)
        if ys is not None:
            seg_true.extend(ys[:, center].cpu().tolist())

        windows_seen += B
    # end for batches

    # Final per-timepoint and per-segment tracks
    with np.errstate(divide="ignore", invalid="ignore"):
        full_track = sum_probs / np.maximum(hit_count, 1)

    # emit exactly one score per segment_idx: the center-aligned slice
    per_segment = full_track[center : center + M]  # length M == len(ds)
    assert len(per_segment) == M

    # Metrics on non-holdout
    if args.split != "holdout" and len(seg_true) == M:
        y_true = torch.tensor(seg_true, dtype=torch.long, device=device)
        y_pred = torch.tensor((per_segment >= 0.5).astype(np.int64), device=device)
        f1_macro = F1Score(task="multiclass", num_classes=2, average="macro").to(device)
        f1_value = f1_macro(y_pred, y_true).item()
        print(
            f"Sample-/segment-level F1-macro on the {args.split} split: {f1_value:.4f}"
        )

    # Write submission
    print(f"Writing {args.output} ...")
    if args.split == "holdout":
        # LibriBrain helper expects list[Tensor(1)]
        tensor_preds = [
            torch.tensor(float(p)).unsqueeze(0) for p in per_segment.tolist()
        ]
        ds.generate_submission_in_csv(tensor_preds, args.output)
    else:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["segment_idx", "speech_prob"])
            for idx, p in enumerate(per_segment.tolist()):
                w.writerow([idx, float(p)])
    print(f"File '{args.output}' with {len(per_segment):,} probabilities is ready.")

    # Optional: save labels for non-holdout
    if args.split != "holdout":
        print(f"Writing {args.labels} ...")
        with open(args.labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["segment_idx", "speech_label"])
            for idx, lbl in enumerate(seg_true):
                w.writerow([idx, int(lbl)])


if __name__ == "__main__":
    sys.exit(main())
