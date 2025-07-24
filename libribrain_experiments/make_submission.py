#!/usr/bin/env python
"""
Generate LibriBrain speech-detection submission from a repo checkpoint.

Examples
--------
Check the F1-macro scores in the test split using multiple checkpoints and
1.8-second windows:

```shell
python -m libribrain_experiments.make_submission \
    --data_path data/ \
    --tmax 1.8 \
    --split test \
    checkpoints/best-val_f1_macro-hpo-*.ckpt
```

It is recommended to check this split results before generating the final
submission, to confirm that everything is working as expected. Small deviations
in the scores may exist.

Generate a submission for the hold-out set using one checkpoint:

```shell
python -m libribrain_experiments.make_submission \
    --data_path data/ \
    --tmax 1.8 \
    --split holdout \
    checkpoints/best-val_f1_macro-hpo-0.ckpt
```

This will generate a "submissions.csv" file with the predictions.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import scipy.ndimage as ndi
import torch
from pnpl.datasets import (  # pip install pnpl
    LibriBrainCompetitionHoldout,
    LibriBrainSpeech,
)
from torchmetrics import F1Score
from tqdm import tqdm

from libribrain_experiments.models.configurable_modules.classification_module import (
    ClassificationModule,
)
from libribrain_experiments.utils import SENSORS_SPEECH_MASK


def majority_filter(x: np.ndarray, k: int) -> np.ndarray:
    """
    Median/majority vote in a k-sample sliding window.

    Parameters
    ----------
    x : np.ndarray
        1-D array of probabilities or binary decisions.
        Works with any numeric dtype.
    k : int
        Size of the sliding window.
        k >= 3 applies a classical median-filter (odd `k` preferred).
        Values < 3 bypass the filter and return x unchanged.

    Returns
    -------
    np.ndarray
        Filtered array, same shape and dtype as the input.
    """
    return ndi.median_filter(x, size=k, mode="nearest") if k >= 3 else x


def hysteresis(p: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Two-threshold (Schmitt-trigger) hysteresis on a probability track.

    A new speech segment starts when p rises above high and
    ends only after it has fallen below low.

    Parameters
    ----------
    p : np.ndarray
        1-D array of probabilities.
    low : float
        Leave-speech (lower) threshold.
    high : float
        Enter-speech (upper) threshold. Must satisfy high > low.

    Returns
    -------
    np.ndarray
        Binary mask (float /1) of the same length as p.
    """
    out = np.zeros_like(p, dtype=float)
    state = False
    for i, v in enumerate(p):
        state = (v > high) or (state and v > low)
        out[i] = 1.0 if state else 0.0
    return out


def enforce_min_run(vec: np.ndarray, min_len: int) -> np.ndarray:
    """
    Remove contiguous 1 runs shorter than a minimum length.

    Useful for deleting very short "blips" of speech.

    Parameters
    ----------
    vec : np.ndarray
        Binary 1-D mask (0/1 or bool).
    min_len : int
        Minimum allowed run length (in samples).
        If `min_len < 1` the input is returned unchanged.

    Returns
    -------
    np.ndarray
        Mask with short 1-runs set to 0.
    """
    if min_len < 1:
        return vec
    out = vec.copy()
    idx = np.flatnonzero(np.diff(np.r_[0, out, 0]))
    for s, e in zip(idx[::2], idx[1::2]):
        if e - s < min_len:
            out[s:e] = 0.0
    return out


def fill_short_gaps(vec: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Bridge short 0-gaps inside speech regions.

    Parameters
    ----------
    vec : np.ndarray
        Binary mask where 1 marks speech.
    max_gap : int
        Maximum gap length (in samples) to be filled.
        A value <= 0 disables gap filling.

    Returns
    -------
    np.ndarray
        Mask where 0-runs shorter than max_gap are turned into 1.
    """
    return 1 - enforce_min_run(1 - vec, max_gap) if max_gap > 0 else vec


@torch.inference_mode()
def predict_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-window speech probabilities.

    The helper converts the network's raw output into a 1-D tensor containing
    `speech=1` for every sample in the batch.

    It transparently supports the two heads used in this repo: binary head
    (sigmoid), and multiclass head (softmax).

    Parameters
    ----------
    model : torch.nn.Module
        A trained `ClassificationModule` or any model that maps
        `(B, C, T)` -> logits.
    x : torch.Tensor
        Input batch of shape `(B, C, T)` in channels-time format.

    Returns
    -------
    torch.Tensor
        A 1-D tensor of length `B` with speech probabilities in the range
        `[0, 1]`.
    """
    logits = model(x)
    if logits.dim() == 2 and logits.size(1) == 1:  # single-logit head
        probs = torch.sigmoid(logits).squeeze(-1)  # (B,)
    else:  # two-logit soft-max
        probs = torch.softmax(logits, dim=1)[:, 1]  # (B,)
    return probs


def safe_collate(batch):
    """
    Collate LibriBrain samples that can be different lengths.

    Parameters
    ----------
    batch : list
        A list of dataset samples, each structured as
        `(x, y)` or `(x, y, info)` where

        * `x`: tensor of shape `(C, T_i)`,
        * `y`: label array (may be missing on the hold-out set).

    Returns
    -------
    tuple
        `(xs, ys)` where
        * `xs`: zero-padded tensor of shape `(B, C, T_max)`,
        * `ys`: stacked labels or `None` if labels are not present.
    """
    xs_list = [torch.as_tensor(b[0]).clone() for b in batch]

    # longest window length inside this mini-batch
    max_len = max(t.shape[1] for t in xs_list)

    # pad (time-dim) so every tensor is [C, max_len]
    xs_pad = [
        (
            torch.nn.functional.pad(t, (0, max_len - t.shape[1]))
            if t.shape[1] < max_len
            else t
        )
        for t in xs_list
    ]
    xs = torch.stack(xs_pad, 0)  # (B, C, max_len)

    # LibriBrainSpeech has labels, competition hold-out does not
    if isinstance(batch[0][1], (int, float, torch.Tensor)):
        ys = torch.stack([torch.as_tensor(b[1]) for b in batch])
    else:  # dict / meta info
        ys = None

    return xs, ys


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry-point that writes a submission CSV (and optional diagnostics).

    The routine:

    1. Parses command-line arguments.
    2. Loads one or more checkpoints and runs inference.
    3. Optionally standardises and clips the input windows.
    4. Computes sample- and segment-level F1 scores when labels exist.
    5. Saves speech probabilities (and labels) to disk.

    Side effects:

    - Writes one `submission*.csv` per checkpoint.
    - Optionally writes `labels.csv` for non-hold-out splits.
    - Prints diagnostic metrics to stdout.

    Parameters
    ----------
    argv : list of str or None, optional
        Command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate LibriBrain submission CSV")
    parser.add_argument("checkpoint", nargs="+", help="Path to *.ckpt from the repo")
    parser.add_argument(
        "--output", default="submission.csv", help="Predictions output CSV file"
    )
    parser.add_argument("--labels", default="labels.csv", help="Labels output CSV file")
    parser.add_argument(
        "--data_path",
        default="./data/",
        help="Where LibriBrain data live / will be downloaded",
    )
    parser.add_argument(
        "--split",
        default="holdout",
        type=str,
        help="Split to load: holdout, train, validation, test.",
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Inference batch size"
    )
    parser.add_argument(
        "--sensor_mask", action="store_true", help="Apply the sensor mask"
    )
    parser.add_argument(
        "--tmax", default=0.5, type=float, help="Window size for the segments."
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help=(
            "Controls how far (in samples) you move the sliding window between consecutive samples."
            " We usually want `stride=1` to have one label per sample."
            " Use `stride=0` to use default stride (window size) in test and validation splits."
        ),
    )
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
    post = parser.add_argument_group("post-processing (disabled by default)")
    post.add_argument(
        "--median_win",
        type=int,
        default=0,
        help="Majority / median filter window (frames); 0 = skip",
    )
    post.add_argument(
        "--hys_low",
        type=float,
        default=None,
        help="Hysteresis: leave-speech threshold (prob)",
    )
    post.add_argument(
        "--hys_high",
        type=float,
        default=None,
        help="Hysteresis: enter-speech threshold (prob)",
    )
    post.add_argument(
        "--min_speech_len",
        type=int,
        default=0,
        help="Drop speech islands shorter than N frames",
    )
    post.add_argument(
        "--min_sil_len",
        type=int,
        default=0,
        help="Fill silence gaps shorter than N frames",
    )

    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.standardize and args.use_train_stats:
        print("Collecting training-set mean / std ...", end="", flush=True)
        train_ds_tmp = LibriBrainSpeech(
            data_path=args.data_path,
            partition="train",
            tmin=0.0,
            tmax=args.tmax,
            standardize=True,  # compute & cache stats
            preload_files=False,  # quickest way
        )
        chan_means = train_ds_tmp.channel_means
        chan_stds = train_ds_tmp.channel_stds
        if args.sensor_mask:  # keep stats in sync with masking
            chan_means = chan_means[SENSORS_SPEECH_MASK]
            chan_stds = chan_stds[SENSORS_SPEECH_MASK]
        del train_ds_tmp
    else:
        chan_means = None  # train stats or None
        chan_stds = None

    # competition hold-out dataset
    print("Loading LibriBrainCompetitionHoldout ...")
    split = args.split.lower()
    if split == "holdout":
        print("Loading Holdout split.")
        ds = LibriBrainCompetitionHoldout(
            data_path=args.data_path,
            tmin=0.0,
            tmax=args.tmax,
            task="speech",
        )
    else:
        print(f"Loading {split} split.")
        ds = LibriBrainSpeech(
            data_path=args.data_path,
            partition=split,
            tmin=0.0,
            tmax=args.tmax,
            # This is performed below to confirm normalization is working ok
            # because we need it for the holdout, which does not support it
            # channel_means=chan_means,
            # channel_stds=chan_stds,
            # Disable internal normalization
            standardize=False,
            clipping_boundary=None,
            stride=args.stride if args.stride > 0 else None,
        )
    if args.standardize and not args.use_train_stats:
        if split != "holdout":
            chan_means = ds.channel_means
            chan_stds = ds.channel_stds
        else:
            # This will be messy to implement, and will probably be left unused
            raise NotImplementedError(
                "Standarization with non-train stats in Holdout not implemented."
            )

    N = len(ds)  # = 560 638
    print(f"   {N:,} time-points to predict")

    # Figure out the window length that this dataset uses
    seq_len = ds[0][0].shape[1]
    center_offset = seq_len // 2
    first_pred_tp = center_offset
    last_pred_tp = N - center_offset - 1
    pred_count = last_pred_tp - first_pred_tp + 1
    print(f"tmax={args.tmax}s, {seq_len} samples")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
    )

    f1_values = []
    f1_segs = []
    for idx, checkpoint in enumerate(tqdm(args.checkpoint, desc="Checkpoint")):
        print(f"Loading checkpoint {idx}: {checkpoint}")
        model: ClassificationModule = ClassificationModule.load_from_checkpoint(
            checkpoint, map_location=device
        )
        model.eval().to(device)

        # container for all time-point probabilities (default 0.0 = 'silence')
        all_probs = [0.0] * N
        all_labels = [0.0] * N if split != "holdout" else None
        seg_true, seg_pred = [], []

        print(f"Generating {pred_count:,} model predictions ...")
        start_sample_idx = 0
        for xs, ys in tqdm(loader, unit_scale=args.batch_size, desc="Samples"):
            if args.sensor_mask:
                xs = xs[:, SENSORS_SPEECH_MASK, :]

            # manual (x-mean)/std `standardize` implementation
            if args.standardize and args.use_train_stats:
                chan_means = torch.as_tensor(
                    chan_means, device=xs.device, dtype=torch.float32
                )
                chan_stds = torch.as_tensor(
                    chan_stds, device=xs.device, dtype=torch.float32
                )
                xs = (xs - chan_means[None, :, None]) / (chan_stds[None, :, None])

            # clipping_boundary implementation
            xs = torch.clamp(xs, -10.0, 10.0)  # same as LibriBrain's default

            xs = xs.to(device)
            probs = predict_probs(model, xs).cpu().tolist()

            for i, p in enumerate(probs):
                tp_idx = start_sample_idx + i + center_offset
                if first_pred_tp <= tp_idx <= last_pred_tp:
                    all_probs[tp_idx] = float(p)
                    if ys is not None:  # non-holdout
                        lbl = float(ys[i, center_offset])  # central label
                        all_labels[tp_idx] = lbl
            start_sample_idx += len(probs)

            # keep per-segment predictions (only when labels available)
            if ys is not None:
                seg_targets = ys[:, center_offset].cpu().tolist()
                seg_preds = [1 if p >= 0.5 else 0 for p in probs]
                seg_true.extend(seg_targets)
                seg_pred.extend(seg_preds)

            # we can break once we have filled all predictable time-points
            if start_sample_idx >= (N - 2 * center_offset):
                break

        # Optional post-processing of the probability track
        scores = np.asarray(all_probs, dtype=float)  # raw probabilities

        # majority / median filter
        if args.median_win >= 3:
            scores = majority_filter(scores, args.median_win)

        # hysteresis or plain 0.5 cut-off -> binary mask
        if args.hys_low is not None and args.hys_high is not None:
            binary = hysteresis(scores, args.hys_low, args.hys_high)
        else:
            binary = (scores >= 0.5).astype(float)

        # duration constraints
        if args.min_speech_len > 0 or args.min_sil_len > 0:
            binary = enforce_min_run(binary, args.min_speech_len)
            binary = fill_short_gaps(binary, args.min_sil_len)

        all_probs = np.where(
            binary > 0,
            np.maximum(scores, 0.5),  # make sure the next 0.5 cut-off keeps them
            0.0,
        ).tolist()

        # convert to list[Tensor(1)] for helper
        tensor_preds = [torch.tensor(p).unsqueeze(0) for p in all_probs]

        # If we have the ground-truth (train/val/test) compute F1-macro
        if split != "holdout":
            eval_start = first_pred_tp
            eval_end = last_pred_tp + 1

            # ground truth
            y_true = torch.tensor(
                all_labels[eval_start:eval_end], dtype=torch.long, device=device
            )

            # binary predictions: prob >= 0.5 -> speech
            y_pred = torch.tensor(
                [1 if p >= 0.5 else 0 for p in all_probs[eval_start:eval_end]],
                dtype=torch.long,
                device=device,
            )

            f1_macro = F1Score(task="multiclass", num_classes=2, average="macro").to(
                device
            )
            f1_value = f1_macro(y_pred, y_true).item()
            f1_values.append(f1_value)
            print(f"Sample-level F1-macro on the {split} split: {f1_value:.4f}")

            # segment-level F1 (matches training/validation metric)
            seg_true_t = torch.tensor(seg_true, device=device)
            seg_pred_t = torch.tensor(seg_pred, device=device)
            f1_macro = F1Score(task="multiclass", num_classes=2, average="macro").to(
                device
            )
            f1_seg = f1_macro(seg_pred_t, seg_true_t).item()
            f1_segs.append(f1_seg)
            print(f"Segment-level F1-macro on the {split} split: {f1_seg:.4f}")

        # write submission
        if len(args.checkpoint) > 1:
            # If there are multiple files, add an index
            root, ext = os.path.splitext(args.output)
            output = f"{root}-{idx}{ext}"
        else:
            output = args.output
        print(f"Writing {output} ...")

        if split == "holdout":
            ds.generate_submission_in_csv(tensor_preds, output)
        else:
            # speech task
            with open(output, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["segment_idx", "speech_prob"])

                for idx_tensor, tensor in enumerate(tensor_preds):
                    # Ensure we extract the scalar float from tensor
                    speech_prob = (
                        tensor.item()
                        if isinstance(tensor, torch.Tensor)
                        else float(tensor)
                    )
                    writer.writerow([idx_tensor, speech_prob])
        print(
            f"File '{output}' with {len(tensor_preds):,} probabilities is ready for EvalAI."
        )

    if split != "holdout":
        # Save matching ground-truth labels
        print(f"Writing {args.labels} ...")
        with open(args.labels, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["segment_idx", "speech_label"])
            for idx, lbl in enumerate(all_labels):
                w.writerow([idx, lbl])

    # Show average scores if we evaluate more than one checkpoint
    if len(args.checkpoint) > 1 and len(f1_values) > 0:
        print()
        print("Number of checkpoints:", len(args.checkpoint))
        f1_value_avg = f"{np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}"
        print(f"Average sample-level F1-macro on the {split} split: {f1_value_avg}")
        f1_seg_avg = f"{np.mean(f1_segs):.4f} ± {np.std(f1_segs):.4f}"
        print(f"Average segment-level F1-macro on the {split} split: {f1_seg_avg}")


if __name__ == "__main__":
    sys.exit(main())
