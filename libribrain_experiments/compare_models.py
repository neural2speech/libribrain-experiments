#!/usr/bin/env python3
"""
compare_models.py: compare two model variants across seeds and test significance.

Usage
-----
$ ./compare_models.py \
    results/final-speech-results/modelA-seed-*/*.json \
    results/final-speech-results/modelB-seed-*/*.json

Or, if you just glob them together:
$ ./compare_models.py results/final-speech-results/*/*.json

In the second case, the script will split the list of files in half:
the first half is Model A, the second half is Model B.

By default we test several validation metrics (same defaults as aggregate_metrics.py),
but you can override with --metric.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two model configs across seeds and run a significance test."
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help=(
            "JSON result files from both models. "
            "Either pass A-files then B-files, or pass a single glob and let the script split half/half."
        ),
    )
    p.add_argument(
        "--metric",
        "-m",
        nargs="+",
        default=[
            "val_f1_macro",
            "val_f1_micro",
            "val_class_1_f1",
            "val_bal_acc",
            "val_rocauc_macro",
            "val_rocauc_micro",
            "val_loss",
        ],
        help="Metric key(s) to compare (default: %(default)s).",
    )
    p.add_argument(
        "--suffix",
        "-s",
        default=None,
        help="Optional suffix to append to metric names (e.g. '_avg5').",
    )
    p.add_argument(
        "--no-half-split",
        action="store_true",
        help=(
            "If set, we will NOT auto-split the file list in half. "
            "Instead, we'll assume the user already passed Model A files first, "
            "then Model B files after a '--'. "
            "Example:\n"
            "  compare_models.py A/*.json -- B/*.json"
        ),
    )
    return p.parse_args()


def load_metric(path: Path, metric_name: str):
    with path.open() as f:
        data = json.load(f)
    return float(data[metric_name])


def split_files(
    files: list[Path], no_half_split: bool
) -> tuple[list[Path], list[Path]]:
    """
    Return (files_A, files_B).

    If no_half_split=False:
        - We split the *sorted* file list in half.
        - If odd length, we drop the middle file so both halves are same size.
    If no_half_split=True:
        - We interpret '--' style usage from argparse.
        - argparse won't keep the literal '--', so the user must call:
            compare_models.py A/*.json B/*.json --no-half-split
          In that mode, we just split at the halfway point *as given*,
          i.e. first chunk of consecutive files is Model A, second chunk is Model B.
          (This still behaves like "first half A / second half B", just without re-sorting.)
    """
    if not no_half_split:
        # sort for reproducibility
        files_sorted = sorted(files)
        n = len(files_sorted)
        if n < 2:
            raise ValueError("Need at least 2 result files to compare.")
        mid = n // 2

        # if odd, drop the middle file so groups are equal length
        if n % 2 == 1:
            # We'll prefer to drop the exact middle index
            drop_idx = mid
            files_kept = [f for i, f in enumerate(files_sorted) if i != drop_idx]
            n2 = len(files_kept)
            mid = n2 // 2
            files_A = files_kept[:mid]
            files_B = files_kept[mid:]
            sys.stderr.write(
                f"[WARN] Odd number of files ({n}); "
                f"dropped {files_sorted[drop_idx]} to balance groups to {len(files_A)} each.\n"
            )
        else:
            files_A = files_sorted[:mid]
            files_B = files_sorted[mid:]
        return files_A, files_B
    else:
        # "no_half_split" here just means: don't sort; split as-given
        n = len(files)
        if n < 2:
            raise ValueError("Need at least 2 result files to compare.")
        mid = n // 2
        if n % 2 == 1:
            drop_idx = mid
            files_kept = [f for i, f in enumerate(files) if i != drop_idx]
            n2 = len(files_kept)
            mid = n2 // 2
            files_A = files_kept[:mid]
            files_B = files_kept[mid:]
            sys.stderr.write(
                f"[WARN] Odd number of files ({n}); "
                f"dropped {files[drop_idx]} to balance groups to {len(files_A)} each.\n"
            )
        else:
            files_A = files[:mid]
            files_B = files[mid:]
        return files_A, files_B


def collect_metrics(files: list[Path], metric_name: str):
    """
    Extract metric_name from each file in 'files'.
    Returns list[float]. Skips files where metric is missing or unreadable,
    but warns.
    """
    vals = []
    bad = []
    for fp in files:
        try:
            vals.append(load_metric(fp, metric_name))
        except FileNotFoundError:
            sys.stderr.write(f"[WARN] file not found: {fp}\n")
        except KeyError:
            bad.append(fp)
        except Exception as ex:
            sys.stderr.write(f"[WARN] could not read {fp}: {ex}\n")

    if bad:
        sys.stderr.write(
            f"[WARN] metric '{metric_name}' not found in "
            f"{len(bad)} file(s): {[str(p) for p in bad]}\n"
        )
    return vals


def p_to_stars(p: float) -> str:
    """
    Map p-value to significance stars:
      ***  p < 0.001
      **   p < 0.01
      *    p < 0.05
      n.s. otherwise
    """
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def main() -> None:
    args = parse_args()

    files_A, files_B = split_files(args.files, args.no_half_split)

    if len(files_A) != len(files_B):
        sys.stderr.write(
            f"[ERROR] Unequal group sizes after split "
            f"({len(files_A)} vs {len(files_B)}). Cannot run paired test.\n"
        )
        sys.exit(1)

    print(f"# Model A files: {len(files_A)}")
    print(f"# Model B files: {len(files_B)}")

    for metric in args.metric:
        metric_name = metric + (args.suffix or "")

        vals_A = collect_metrics(files_A, metric_name)
        vals_B = collect_metrics(files_B, metric_name)

        if len(vals_A) != len(vals_B):
            sys.stderr.write(
                f"[ERROR] After loading metric '{metric_name}', "
                f"group sizes differ ({len(vals_A)} vs {len(vals_B)}). Skipping.\n"
            )
            continue

        if len(vals_A) < 2:
            sys.stderr.write(
                f"[WARN] Not enough samples for '{metric_name}' "
                f"({len(vals_A)}); need >=2 for a Wilcoxon test.\n"
            )
            continue

        A = np.asarray(vals_A, dtype=float)
        B = np.asarray(vals_B, dtype=float)

        mean_A = A.mean()
        std_A = A.std(ddof=1) if len(A) > 1 else 0.0
        mean_B = B.mean()
        std_B = B.std(ddof=1) if len(B) > 1 else 0.0

        # Wilcoxon signed-rank test (paired, two-sided)
        try:
            stat, pval = wilcoxon(A, B, alternative="two-sided")
        except ValueError as e:
            # e.g. all differences are zero
            stat, pval = (np.nan, 1.0)
            sys.stderr.write(
                f"[WARN] Wilcoxon failed for '{metric_name}': {e}. " "Setting p=1.0.\n"
            )

        diff = B - A  # positive means B > A
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1) if len(diff) > 1 else 0.0

        stars = p_to_stars(pval)

        print()
        print(f"Metric: {metric_name}")
        print(f"- Model A: {mean_A:.4f} ± {std_A:.4f}")
        print(f"- Model B: {mean_B:.4f} ± {std_B:.4f}")
        print(f"- (B - A): {mean_diff:.4f} ± {std_diff:.4f}")
        print(f"- Wilcoxon signed-rank: W={stat:.8f}, p={pval:.8f} [{stars}]")


if __name__ == "__main__":
    main()
