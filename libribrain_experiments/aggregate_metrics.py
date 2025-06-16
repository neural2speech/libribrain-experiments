#!/usr/bin/env python
"""
aggregate_metrics.py: collect a metric from several result-JSON files.

Example
-------
$ libribrain_experiments/aggregate_metrics.py \
    results/final-speech-results/test-best-libribrain.speech-hpo-*/*.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        An object with the following attributes:

        files : list[pathlib.Path]
            One or more JSON files that contain the experiment
            results to be aggregated.
        metric : str
            Name of the key to extract from each JSON file
            (default is `"val_f1_macro"`).
    """
    p = argparse.ArgumentParser(
        description="Aggregate a validation metric over many JSON result files."
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Result files produced by LibriBrain experiments (JSON).",
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
            "val_jaccard_index",
            "val_loss",
        ],
        help="Key to extract from each JSON file (default: %(default)s).",
    )
    return p.parse_args()


def main() -> None:
    """
    Entry-point that orchestrates aggregation.

    Notes
    -----
    * Calls :func:`parse_args` to obtain user-supplied parameters.
    * Collects the specified metric from all input files,
      reports warnings for missing files or keys.
    * Prints the mean and (sample) standard deviation of the metric.
    * Exits with status code `1` if no valid values are found.
    """
    args = parse_args()

    values = []
    missing = []

    for metric in args.metric:
        for fp in args.files:
            try:
                with fp.open() as f:
                    data = json.load(f)
                values.append(float(data[metric]))
            except FileNotFoundError:
                print(f"[WARN] file not found: {fp}", file=sys.stderr)
            except KeyError:
                missing.append(fp)
            except Exception as ex:  # catch malformed JSON, etc.
                print(f"[WARN] could not read {fp}: {ex}", file=sys.stderr)

        if missing:
            print(
                f"[WARN] metric '{metric}' not found in "
                f"{len(missing)} file(s): {[str(p) for p in missing]}",
                file=sys.stderr,
            )

        if not values:
            print(
                "No metric values collected for '{metric}' - aborting.", file=sys.stderr
            )
            break

        mean = np.mean(values)
        stdev = np.std(values) if len(values) > 1 else 0.0

        print(
            f"{metric}: {mean:.4f} ± {stdev:.4f}  "
            f"(n={len(values)} from {len(args.files)} file(s))"
        )


if __name__ == "__main__":
    main()
