#!/usr/bin/env python3
"""
chance_significance.py: test whether a model's seed-wise scores are above chance.

Usage
-----
# Fixed chance for all metrics (e.g., binary F1-macro ~ 0.5)
$ ./chance_significance.py results/final-speech-results/modelA-seed-*/*.json \
    -m val_f1_macro val_bal_acc -c 0.5 --alt greater

# Compute chance as 1/C for macro metrics (e.g., 39-class F1-macro)
$ ./chance_significance.py results/final-phoneme-results/modelA-seed-*/*.json \
    -m val_f1_macro --classes 39 --alt greater

# With a suffix appended to metric names (e.g., "_avg5")
$ ./chance_significance.py results/*/*.json -m val_f1_macro -s _avg5 -c 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test if a model's seed-wise results are statistically above chance."
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON result files from one model/config (multiple seeds).",
    )
    p.add_argument(
        "--metric",
        "-m",
        nargs="+",
        default=[
            "val_f1_macro",
            "val_f1_micro",
            "val_bal_acc",
            "val_rocauc_macro",
            "val_rocauc_micro",
        ],
        help="Metric key(s) to test (default: %(default)s).",
    )
    p.add_argument(
        "--suffix",
        "-s",
        default=None,
        help="Optional suffix to append to metric names (e.g. '_avg5').",
    )
    p.add_argument(
        "--chance",
        "-c",
        type=float,
        default=None,
        help="Chance level to test against (e.g., 0.5 for binary). "
        "If omitted and --classes is provided, chance is set to 1/classes "
        "for metrics containing 'f1_macro' or 'accuracy'.",
    )
    p.add_argument(
        "--classes",
        type=int,
        default=None,
        help="Number of classes C; if --chance is not set and the metric name "
        "contains 'f1_macro' or 'accuracy', chance is set to 1/C.",
    )
    p.add_argument(
        "--alt",
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="Alternative hypothesis for Wilcoxon: "
        "'greater' tests median(model - chance) > 0 (default).",
    )
    return p.parse_args()


def load_metric(path: Path, metric_name: str):
    with path.open() as f:
        data = json.load(f)
    return float(data[metric_name])


def collect_metrics(files: list[Path], metric_name: str):
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


def default_chance_for_metric(metric_name: str, classes: int | None) -> float | None:
    """If --chance missing, infer chance as 1/C for macro-F1 / accuracy style names."""
    if classes is None:
        return None
    name = metric_name.lower()
    if "f1_macro" in name or "accuracy" in name or "acc" in name:
        return 1.0 / float(classes)
    return None


def p_to_stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def main() -> None:
    args = parse_args()

    files_sorted = sorted(args.files)
    n_files = len(files_sorted)
    if n_files < 2:
        sys.stderr.write("[ERROR] Need at least 2 result files (seeds) to test.\n")
        sys.exit(1)

    print(f"# Files (seeds): {n_files}")

    for metric in args.metric:
        metric_name = metric + (args.suffix or "")

        vals = collect_metrics(files_sorted, metric_name)
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            sys.stderr.write(
                f"[WARN] Not enough valid values for '{metric_name}' "
                f"({len(vals)}); need >=2.\n"
            )
            continue

        chance = args.chance
        if chance is None:
            chance = default_chance_for_metric(metric_name, args.classes)

        if chance is None:
            sys.stderr.write(
                f"[ERROR] Chance level is not specified for '{metric_name}'. "
                f"Provide --chance or --classes.\n"
            )
            continue

        X = np.asarray(vals, dtype=float)
        D = X - float(chance)

        mean = X.mean()
        std = X.std(ddof=1) if len(X) > 1 else 0.0
        mean_diff = D.mean()
        std_diff = D.std(ddof=1) if len(D) > 1 else 0.0

        # One-sample Wilcoxon vs. zero median (i.e., vs chance)
        try:
            stat, pval = wilcoxon(D, zero_method="wilcox", alternative=args.alt)
        except ValueError as e:
            stat, pval = (np.nan, 1.0)
            sys.stderr.write(
                f"[WARN] Wilcoxon failed for '{metric_name}': {e}. Setting p=1.0.\n"
            )

        stars = p_to_stars(pval)

        print()
        print(f"Metric: {metric_name}")
        print(f"- Mean ± SD: {mean:.4f} ± {std:.4f}  (n={len(X)})")
        print(f"- Chance:    {chance:.4f}")
        print(f"- Diff (mean - chance): {mean_diff:.4f} ± {std_diff:.4f}")
        print(
            f"- Wilcoxon one-sample (alt='{args.alt}'): W={stat:.8f}, p={pval:.8f} [{stars}]"
        )

        if args.alt == "greater":
            verdict = (
                "significantly above chance"
                if pval < 0.05
                else "not significantly above chance"
            )
        elif args.alt == "less":
            verdict = (
                "significantly below chance"
                if pval < 0.05
                else "not significantly below chance"
            )
        else:
            verdict = (
                "significantly different from chance"
                if pval < 0.05
                else "not significantly different from chance"
            )
        print(f"- Conclusion: {verdict}")


if __name__ == "__main__":
    main()
