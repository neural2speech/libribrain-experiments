#!/usr/bin/env python
"""
Generate LibriBrain phoneme-classification submissions or
obtain val/test scores with one (or several) checkpoints.

Now supports optional ensembling across multiple checkpoints.

Examples
--------
One CSV per checkpoint:

```shell
python -m libribrain_experiments.make_submission_phoneme \
    --data_path data/ \
    --split test \
    --mimic_holdout_style \
    checkpoints/best-val_f1_macro-hpo-0.ckpt \
    checkpoints/best-val_f1_macro-hpo-1.ckpt
```

Ensemble (avg of probabilities) + also write per-model CSVs:

```shell
python -m libribrain_experiments.make_submission_phoneme \
    --data_path data/ \
    --split test \
    --ensemble avg_probs \
    --ensemble_out submission_ens.csv \
    checkpoints/a.ckpt checkpoints/b.ckpt checkpoints/c.ckpt
```

Ensemble (avg of logits) with weights:

```shell
python -m libribrain_experiments.make_submission_phoneme \
    --data_path data/ \
    --split test \
    --ensemble avg_logits \
    --weights 0.5,0.3,0.2 \
    checkpoints/a.ckpt checkpoints/b.ckpt checkpoints/c.ckpt
```
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from pnpl.datasets import (
    GroupedDataset,
    LibriBrainCompetitionHoldout,
    LibriBrainPhoneme,
)
from pnpl.datasets.libribrain2025.speech_dataset_holdout import LibriBrainSpeechHoldout
from torch.utils.data import ConcatDataset
from torchmetrics import F1Score
from tqdm.auto import tqdm

from libribrain_experiments.models.configurable_modules.classification_module import (
    ClassificationModule,
)

# Avoid loading backbone models for ensembles
os.environ["LB_LOAD_MEMBER_WEIGHTS"] = "0"


@torch.inference_mode()
def forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass -> raw logits, shape (B, C). Supports single-logit heads
    but for phoneme classification we expect C=39.
    """
    logits = model(x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    return logits


@torch.inference_mode()
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over class dim."""
    if logits.size(1) == 1:  # binary fallback
        p1 = torch.sigmoid(logits.squeeze(1))
        logits = torch.stack([1.0 - p1, p1], dim=1)
    return torch.softmax(logits, dim=1)


def collate_pad(batch):
    """
    Batch LibriBrain samples of variable length.

    Handles both
    - hold-out split: x : Tensor(C, T)
    - val/test/train: (x, y) where y is a scalar label
    """
    if isinstance(batch[0], torch.Tensor):  # hold-out
        xs_list = [b.clone() for b in batch]  # x only
        ys = None
    else:  # labeled
        xs_list = [torch.as_tensor(b[0]).clone() for b in batch]
        ys = torch.stack([b[1] for b in batch])  # (B,)

    T_max = max(t.shape[-1] for t in xs_list)
    xs_pad = [torch.nn.functional.pad(t, (0, T_max - t.shape[-1])) for t in xs_list]
    xs = torch.stack(xs_pad, 0)  # (B, C, T_max)
    return xs, ys


def parse_weights(weights_arg: Optional[str], n_models: int) -> Optional[torch.Tensor]:
    if not weights_arg:
        return None
    parts = [float(x.strip()) for x in weights_arg.split(",")]
    if len(parts) != n_models:
        raise ValueError(
            f"--weights expects {n_models} comma-separated values, got {len(parts)}"
        )
    w = torch.tensor(parts, dtype=torch.float32)
    s = float(w.sum().item())
    if s <= 0:
        raise ValueError("All weights must be non-negative and sum > 0.")
    return w / s


def ensemble_predictions(
    probs_list: List[torch.Tensor],  # list of (N,C) prob tensors
    logits_list: List[torch.Tensor],  # list of (N,C) logits tensors
    method: str,
    weights: Optional[torch.Tensor] = None,
    hard_vote_one_hot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (probs_ens, labels_ens) where probs_ens is (N, C), labels_ens is (N,).
    """
    method = method.lower()
    n_models = len(probs_list)
    N, C = probs_list[0].shape

    if method in {"avg_probs", "avg", "mean"}:
        P = torch.stack(probs_list, dim=0)  # (M,N,C)
        if weights is not None:
            w = weights.view(-1, 1, 1)  # (M,1,1)
            P = (P * w).sum(dim=0)
        else:
            P = P.mean(dim=0)
        labels = P.argmax(dim=1)
        return P, labels

    if method == "geom_mean":
        # normalized product of probabilities (work in log-space)
        # avoid log(0)
        eps = 1e-12
        logs = [torch.log(p.clamp_min(eps)) for p in probs_list]  # (N,C)
        if weights is not None:
            # weighted sum of logs
            logs = torch.stack(logs, dim=0)  # (M,N,C)
            lw = weights.view(-1, 1, 1)
            L = (logs * lw).sum(dim=0)
        else:
            L = torch.stack(logs, dim=0).sum(dim=0)  # (N,C)
        P = torch.softmax(L, dim=1)
        labels = P.argmax(dim=1)
        return P, labels

    if method == "avg_logits":
        if not logits_list:
            raise ValueError("avg_logits requires collecting logits; got empty list.")
        L = torch.stack(logits_list, dim=0)  # (M,N,C)
        if weights is not None:
            w = weights.view(-1, 1, 1)
            L = (L * w).sum(dim=0)
        else:
            L = L.mean(dim=0)
        P = torch.softmax(L, dim=1)
        labels = P.argmax(dim=1)
        return P, labels

    if method == "majority_vote":
        # hard labels per model
        labels_m = [p.argmax(dim=1) for p in probs_list]  # list of (N,)
        labels_m = torch.stack(labels_m, dim=0)  # (M,N)
        if weights is None:
            # mode per column; if tie -> pick smallest class index
            # also produce a SOFT vote distribution by counts
            votes = torch.zeros(N, C, dtype=torch.float32, device=labels_m.device)
            for m in range(n_models):
                votes[torch.arange(N), labels_m[m]] += 1.0
        else:
            votes = torch.zeros(N, C, dtype=torch.float32, device=labels_m.device)
            for m in range(n_models):
                votes[torch.arange(N), labels_m[m]] += float(weights[m].item())

        labels = votes.argmax(dim=1)

        if hard_vote_one_hot:
            P = torch.zeros_like(votes)
            P[torch.arange(N), labels] = 1.0
        else:
            # soft vote distribution (counts normalized)
            s = votes.sum(dim=1, keepdim=True).clamp_min(1e-12)
            P = votes / s
        return P, labels

    if method == "max_conf":
        # probs_list: list of (N,C)
        Pstack = torch.stack(probs_list, dim=0)  # (M, N, C)

        # Get each model's max prob per sample: (M, N)
        max_vals, _ = Pstack.max(dim=2)

        # Best model index per sample: (N,)
        best_model = max_vals.argmax(dim=0)

        # Gather that model's full distribution for each sample -> (N, C)
        idx_n = torch.arange(Pstack.size(1), device=Pstack.device)  # (N,)
        P = Pstack[best_model, idx_n, :]  # pairwise gather

        labels = P.argmax(dim=1)
        return P, labels

    raise ValueError(f"Unknown ensemble method: {method}")


def force_reload_state(model, ckpt_path: str, strict_try: bool = True):
    """Re-apply the checkpoint's state_dict to the already-constructed model.

    This guarantees that any fine-tuned weights saved in the ensemble checkpoint
    overwrite whatever was loaded during __init__ (e.g., member backbones).
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob.get("state_dict", blob)
    # print("force_reload_state() -> pre:", model.modules_list[0].members[0].model.modules_list[0].encoder.conformer_layers[0].ffn1.sequential[0].weight[0:10])
    # First try strict; if that fails (code drift), fall back to non-strict.
    print("Reloading model...")
    if strict_try:
        try:
            model.load_state_dict(state, strict=True)
            # print("force_reload_state() post:", model.modules_list[0].members[0].model.modules_list[0].encoder.conformer_layers[0].ffn1.sequential[0].weight[0:10])
            return
        except Exception as e:
            print(
                f"[force_reload_state] strict=True failed ({e}); retrying with strict=False"
            )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            "[force_reload_state] Warning:",
            f"missing={len(missing)} unexpected={len(unexpected)}",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Phoneme submission / evaluation")
    parser.add_argument("checkpoint", nargs="+")
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "validation2",
            "test",
            "test2",
            "holdout",
            "validation+test",
            "test+validation",
        ],
        default="holdout",
    )
    parser.add_argument("--output", default="submission_phoneme.csv")
    parser.add_argument(
        "--force_reload_state",
        action="store_true",
        help="After constructing the model from checkpoint, re-apply the "
        "checkpoint's state_dict onto it. Useful for trained ensembles to "
        "ensure fine-tuned member weights are used at inference.",
    )
    parser.add_argument(
        "--labels", default="labels.csv", help="Write ground-truth labels (non-holdout)"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument(
        "--grouped_samples",
        type=int,
        default=100,
        help="How many single-trials to average when building validation/test samples.",
    )
    parser.add_argument(
        "--drop_remaining",
        action="store_true",
        help="Drop the remaining example when grouping.",
    )
    parser.add_argument(
        "--group_augment",
        type=int,
        default=1,
        help=(
            "For non-holdout splits only: number of independent random groupings "
            "to concatenate (each uses shuffle=True). Default 1 = no augmentation."
        ),
    )

    ens = parser.add_argument_group("ensembling")
    ens.add_argument(
        "--ensemble",
        default=[],
        nargs="*",
        choices=["avg_probs", "avg_logits", "geom_mean", "majority_vote", "max_conf"],
        help="If set (!= none) and multiple checkpoints are given, also write an ensemble CSV.",
    )
    ens.add_argument(
        "--ensemble_out",
        default=None,
        help="Path to write the ensemble CSV. Defaults to <output-root>-ensemble.csv",
    )
    ens.add_argument(
        "--weights",
        default=None,
        help="Optional comma-separated weights for checkpoints (used by avg_probs/avg_logits/geom_mean).",
    )
    ens.add_argument(
        "--hard_vote_one_hot",
        action="store_true",
        help="With majority_vote, output one-hot rows instead of soft vote distribution.",
    )

    std = parser.add_argument_group("normalization")
    std.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Skip (x-mean)/std",
    )
    std.add_argument(
        "--no-train_stats",
        dest="use_train_stats",
        action="store_false",
        help="Compute mean/std on the same split instead of TRAIN",
    )
    std.add_argument(
        "--mimic_holdout_style",
        action="store_true",
        help=(
            "On train/validation/test: standardize singles to that split's OWN "
            "stats, then group/average, then de-normalize back to raw, "
            "and finally (if --standardize) re-apply TRAIN z-score."
        ),
    )
    std.add_argument(
        "--destandardize",
        action="store_true",
        help="On holdout only: de-normalize the averaged standardized holdout",
    )

    args = parser.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    multi_ckpt = len(args.checkpoint) > 1
    do_ensemble = multi_ckpt and len(args.ensemble) > 0

    # optional weights
    weights = parse_weights(args.weights, len(args.checkpoint)) if do_ensemble else None

    # get channel statistics from train
    if args.standardize and args.use_train_stats:
        print("Collecting training-set mean / std ...")
        train_ds_tmp = LibriBrainPhoneme(
            data_path=args.data_path,
            partition="train",
            tmin=0.0,
            tmax=args.tmax,
            standardize=True,  # compute & cache stats
            preload_files=False,  # quickest way
        )
        chan_means = train_ds_tmp.channel_means
        chan_stds = train_ds_tmp.channel_stds
        train_label_order = list(train_ds_tmp.labels_sorted)
        print("Labels:", train_label_order)
        del train_ds_tmp
    else:
        chan_means = None  # train stats or None
        chan_stds = None

    # load dataset
    print("Loading LibriBrain dataset...")
    args.split = args.split.lower()
    own_mu = own_sd = None
    if args.split == "holdout":
        print("Loading Holdout split.")
        ds = LibriBrainCompetitionHoldout(
            data_path=args.data_path,
            tmin=0.0,
            tmax=args.tmax,
            standardize=False,  # it is already standardized
            channel_means=None,
            channel_stds=None,
            task="phoneme",
        )
        print(f"[holdout] {len(ds):,} averaged samples to predict")

        if args.destandardize:
            print("Collecting holdout-set mean / std ...")
            hodlout_ds_tmp = LibriBrainSpeechHoldout(
                data_path=args.data_path,
                tmin=0.0,
                tmax=args.tmax,
                # If we want to mimic holdout, we want OWN stats baked in now:
                standardize=True,
                clipping_boundary=None,
            )
            own_mu = hodlout_ds_tmp.channel_means
            own_sd = hodlout_ds_tmp.channel_stds
            del hodlout_ds_tmp
    else:
        if args.split in ["validation+test", "test+validation"]:
            base = LibriBrainPhoneme(
                data_path=args.data_path,
                include_run_keys=[
                    ["0", "11", "Sherlock1", "2"],
                    ["0", "12", "Sherlock1", "2"],
                ],
                tmin=0.0,
                tmax=args.tmax,
                # If we want to mimic holdout, we want OWN stats baked in now:
                standardize=True if args.mimic_holdout_style else args.standardize,
                channel_means=None,
                channel_stds=None,
                clipping_boundary=None,
            )
        elif args.split == "validation2":
            base = LibriBrainPhoneme(
                data_path=args.data_path,
                include_run_keys=[
                    ["0", "1", "Sherlock1", "1"],
                    ["0", "11", "Sherlock1", "2"],
                ],
                tmin=0.0,
                tmax=args.tmax,
                # If we want to mimic holdout, we want OWN stats baked in now:
                standardize=True if args.mimic_holdout_style else args.standardize,
                channel_means=None,
                channel_stds=None,
                clipping_boundary=None,
            )
        elif args.split == "test2":
            base = LibriBrainPhoneme(
                data_path=args.data_path,
                include_run_keys=[
                    ["0", "9", "Sherlock1", "1"],
                    ["0", "12", "Sherlock1", "2"],
                ],
                tmin=0.0,
                tmax=args.tmax,
                # If we want to mimic holdout, we want OWN stats baked in now:
                standardize=True if args.mimic_holdout_style else args.standardize,
                channel_means=None,
                channel_stds=None,
                clipping_boundary=None,
            )
        else:
            base = LibriBrainPhoneme(
                data_path=args.data_path,
                partition=args.split,
                tmin=0.0,
                tmax=args.tmax,
                standardize=True if args.mimic_holdout_style else args.standardize,
                channel_means=None,
                channel_stds=None,
                clipping_boundary=None,
            )
        # Save the split's own stats to be able to de-normalize after grouping
        own_mu = base.channel_means
        own_sd = base.channel_stds

        # If group_augment == 1 (default), keep current behavior.
        # If >1, create N different random groupings and concatenate them.
        if args.group_augment <= 1:
            ds = GroupedDataset(
                base,
                grouped_samples=args.grouped_samples,
                average_grouped_samples=True,
                drop_remaining=args.drop_remaining,
                shuffle=False,  # deterministic single grouping (unchanged default)
            )
            n_groups = 1
            total_len = len(ds)
        else:
            grouped_list = []
            for _ in range(args.group_augment):
                g = GroupedDataset(
                    base,
                    grouped_samples=args.grouped_samples,
                    average_grouped_samples=True,
                    drop_remaining=False,
                    shuffle=True,  # Important: different random grouping each time
                )
                grouped_list.append(g)
            ds = ConcatDataset(grouped_list)
            n_groups = args.group_augment
            total_len = sum(len(g) for g in grouped_list)

        print(
            f"[{args.split}] {total_len:,} averaged samples "
            f"({args.grouped_samples}x) from {len(base):,} singles "
            f"x {n_groups} grouping(s)"
        )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad,
    )

    n_classes = 39
    all_probs = []  # list of (N,C)
    all_logits = []  # list of (N,C) (for avg_logits)

    f1s = []
    # inference for each checkpoint (unchanged behavior + collect for ensemble)
    for ckpt_idx, ckpt in enumerate(args.checkpoint):
        print(f"\nCheckpoint {ckpt_idx}: {Path(ckpt).name}")
        model: ClassificationModule = ClassificationModule.load_from_checkpoint(
            ckpt, map_location=dev
        )
        if args.force_reload_state:
            force_reload_state(model, ckpt)

        model.eval().to(dev)

        probs_chunks, logits_chunks = [], []
        gts = []  # only last loop's gts kept (same across models)

        for xs, ys in tqdm(loader, desc=f"batches (model {ckpt_idx})"):
            xs = xs.to(dev)

            mu_x = xs.mean(dim=(0, 2)).cpu().numpy()
            std_x = xs.std(dim=(0, 2), unbiased=False).cpu().numpy()

            # de-normalize to raw (holdout style) if requested, then re-standardize with TRAIN stats
            if args.destandardize or args.mimic_holdout_style:
                mu_t = torch.as_tensor(own_mu, device=xs.device, dtype=xs.dtype)[
                    None, :, None
                ]
                sd_t = torch.as_tensor(own_sd, device=xs.device, dtype=xs.dtype)[
                    None, :, None
                ]
                xs = xs * sd_t + mu_t
                xs = torch.clamp(xs, -10.0, 10.0)

            if args.standardize and chan_means is not None and chan_stds is not None:
                mu_tr = torch.as_tensor(chan_means, device=xs.device, dtype=xs.dtype)[
                    None, :, None
                ]
                sd_tr = torch.as_tensor(chan_stds, device=xs.device, dtype=xs.dtype)[
                    None, :, None
                ]
                xs = (xs - mu_tr) / sd_tr

            mu_x = xs.mean(dim=(0, 2)).cpu().numpy()
            std_x = xs.std(dim=(0, 2), unbiased=False).cpu().numpy()

            l = forward_logits(model, xs).cpu()  # (B,C)
            p = logits_to_probs(l)  # (B,C)
            logits_chunks.append(l)
            probs_chunks.append(p)
            if ys is not None:
                gts.append(ys.cpu())

        logits = torch.cat(logits_chunks)  # (N,C)
        probs = torch.cat(probs_chunks)  # (N,C)
        if len(all_probs) == 0:
            # cache labels once (same order for all models)
            has_labels = len(gts) > 0
            if has_labels:
                labels_true = torch.cat(gts)  # (N,)

        all_probs.append(probs)
        all_logits.append(logits)

        # metrics on labelled splits
        if has_labels:
            f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")(
                probs.argmax(1), labels_true
            )
            print(f"Macro-F1 on {args.split}: {f1:.4f}")
            f1s.append(f1)

            # shuffled columns -> sanity check
            # perm = torch.randperm(n_classes)
            # f1_shuf = F1Score(task="multiclass", num_classes=n_classes, average="macro")(
            #     probs[:, perm].argmax(1), labels_true
            # )
            # print(f"Macro-F1 on {args.split}: {f1:.4f} (orig) | {f1_shuf:.4f} (columns shuffled)")

        # write CSV (per-model)
        if multi_ckpt:
            root, ext = os.path.splitext(args.output)
            out_file = f"{root}-{ckpt_idx}{ext}"
        else:
            out_file = args.output

        if args.split == "holdout" and hasattr(ds, "generate_submission_in_csv"):
            tensors = [row.clone() for row in probs]  # list[Tensor(39)]
            ds.generate_submission_in_csv(tensors, out_file)
        else:
            with open(out_file, "w", newline="") as f:
                w = csv.writer(f)
                header = ["segment_idx"] + [f"class_{i}" for i in range(n_classes)]
                w.writerow(header)
                for idx_row, row in enumerate(probs.numpy()):
                    w.writerow([idx_row, *row.tolist()])
        print(f"Wrote {len(probs):,} rows -> {out_file}")

        # optional: write ground truth
        if has_labels and not multi_ckpt:
            with open(args.labels, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["segment_idx", "phoneme_label"])
                for i, lbl in enumerate(labels_true.tolist()):
                    w.writerow([i, lbl])
            print(f"Wrote labels -> {args.labels}")

    if has_labels:
        print("")
        print(f"Mean results: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Ensemble phase
    if do_ensemble:
        print("\n== Ensembling ==")
        for ensemble in args.ensemble:
            P_ens, y_ens = ensemble_predictions(
                probs_list=all_probs,
                logits_list=all_logits,
                method=ensemble,
                weights=weights,
                hard_vote_one_hot=args.hard_vote_one_hot,
            )
            if has_labels:
                f1_ens = F1Score(
                    task="multiclass", num_classes=n_classes, average="macro"
                )(y_ens, labels_true)
                print(f"[Ensemble={ensemble}] Macro-F1 on {args.split}: {f1_ens:.4f}")

            # choose output path
            if args.ensemble_out:
                out_ens = args.ensemble_out
            else:
                root, ext = os.path.splitext(args.output)
                out_ens = f"{root}-ensemble-{ensemble}{ext}"

            if args.split == "holdout" and hasattr(ds, "generate_submission_in_csv"):
                tensors = [row.clone() for row in P_ens]  # list[Tensor(39)]
                ds.generate_submission_in_csv(tensors, out_ens)
            else:
                with open(out_ens, "w", newline="") as f:
                    w = csv.writer(f)
                    header = ["segment_idx"] + [f"class_{i}" for i in range(n_classes)]
                    w.writerow(header)
                    for idx_row, row in enumerate(P_ens.numpy()):
                        w.writerow([idx_row, *row.tolist()])
            print(f"[Ensemble={ensemble}] Wrote {len(P_ens):,} rows -> {out_ens}")

            if has_labels:
                labels_out = args.labels
                root, ext = os.path.splitext(labels_out)
                labels_out = f"{root}-ensemble{ext}"
                with open(labels_out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["segment_idx", "phoneme_label"])
                    for i, lbl in enumerate(labels_true.tolist()):
                        w.writerow([i, lbl])
                print(f"[Ensemble] Wrote labels -> {labels_out}")


if __name__ == "__main__":
    sys.exit(main())
