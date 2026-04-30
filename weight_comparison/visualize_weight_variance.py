"""Visualize per-weight mean weight magnitude across runs vs. a target connectome constraint.

For each input weights_*.csv file, extract the hidden-hidden RNN kernel
(the block following the comment line containing
"/ScannedRNN_0/SimpleCell_1/h/kernel"). For 4 randomly chosen neurons, plot
the mean weight magnitude (|w|) across runs with error bars showing the
standard deviation of the magnitudes, and overlay the target weight
magnitudes from cellstats_constraint_level1_test.npy.
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

KERNEL_TAG = "/ScannedRNN_0/SimpleCell_1/h/kernel"
TARGET_NPY = "cellstats_constraint_level1_test.npy"
N_NEURONS_PLOT = 4
GRID_ROWS, GRID_COLS = 2, 2


def load_kernel(csv_path):
    """Read the h/kernel block and return it in (neuron, input_weight) orientation.

    The training code in ppo_rnn.py saves weights via ``np.savetxt(file,
    np.transpose(curr_value))``, so each CSV row is a column of the in-memory
    kernel. We transpose back here so the returned array matches the
    orientation of the target npy (kernel[i, j] == returned[i, j]).
    """
    rows = []
    in_block = False
    with open(csv_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if in_block:
                    break
                if KERNEL_TAG in line:
                    in_block = True
                continue
            if in_block and line:
                rows.append(np.fromstring(line, sep=",", dtype=np.float32))
    if not rows:
        raise ValueError(f"No kernel block found in {csv_path}")
    return np.stack(rows, axis=0).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files", nargs="+", help="Paths to weights_*.csv files to compare."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output image filename."
    )
    parser.add_argument(
        "--target",
        default=os.path.join(os.path.dirname(__file__), TARGET_NPY),
        help=f"Path to target npy file (default: {TARGET_NPY} next to this script).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for neuron selection."
    )
    args = parser.parse_args()

    target = np.load(args.target).squeeze()
    if target.ndim != 2:
        sys.exit(f"Target must be 2D after squeeze, got shape {target.shape}.")

    kernels = []
    for path in args.csv_files:
        k = load_kernel(path)
        if k.shape[0] != target.shape[0]:
            sys.exit(
                f"Row mismatch: {path} kernel has {k.shape[0]} rows, target has {target.shape[0]}."
            )
        kernels.append(k)

    common_cols = min(target.shape[1], min(k.shape[1] for k in kernels))
    if any(k.shape[1] != target.shape[1] for k in kernels):
        print(
            f"Warning: kernel column counts {[k.shape[1] for k in kernels]} vs target "
            f"{target.shape[1]}; comparing only first {common_cols} weights per neuron.",
            file=sys.stderr,
        )
    kernels = np.stack([k[:, :common_cols] for k in kernels], axis=0)
    target_trimmed = target[:, :common_cols]

    abs_kernels = np.abs(kernels)
    mean_weights = np.mean(abs_kernels, axis=0)
    std_weights = np.std(abs_kernels, axis=0)
    abs_target = np.abs(target_trimmed)

    n_neurons, n_weights = mean_weights.shape
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(range(n_neurons), min(N_NEURONS_PLOT, n_neurons)))

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(10 * GRID_COLS, 6 * GRID_ROWS), sharex=True
    )
    x = np.arange(n_weights)
    mismatch_threshold = 0.05
    for ax, neuron_idx in zip(np.atleast_1d(axes).flat, selected):
        tgt = abs_target[neuron_idx]
        mean = mean_weights[neuron_idx]
        target_zero = tgt == 0
        mismatched = (~target_zero) & (np.abs(mean - tgt) > mismatch_threshold)
        bar_colors = np.where(target_zero, "crimson", "cornflowerblue")
        target_colors = np.where(mismatched, "gold", "forestgreen")
        ax.bar(
            x, mean, width=1.0,
            yerr=std_weights[neuron_idx],
            color=bar_colors, alpha=0.7,
            error_kw={"ecolor": "black", "elinewidth": 0.5, "capsize": 0},
        )
        ax.bar(
            x, tgt, width=1.0,
            color=target_colors, alpha=0.7,
        )
        legend_handles = [
            Patch(facecolor="cornflowerblue", alpha=0.7, label="Mean |weight| (target != 0)"),
            Patch(facecolor="crimson", alpha=0.7, label="Mean |weight| (target == 0)"),
            Patch(facecolor="forestgreen", alpha=0.7, label="|Target| (matched)"),
            Patch(facecolor="gold", alpha=0.7,
                  label=f"|Target| (|mean - target| > {mismatch_threshold})"),
        ]
        ax.set_title(f"Neuron {neuron_idx}")
        ax.set_xlabel("Weight #")
        ax.set_ylabel("Value")
        ax.legend(handles=legend_handles, loc="best", fontsize="small")
    for ax in list(np.atleast_1d(axes).flat)[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        f"Mean per-weight magnitude with std vs. target ({len(args.csv_files)} runs)"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
