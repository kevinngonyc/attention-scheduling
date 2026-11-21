#!/usr/bin/env python3
"""Plot naive vs. LPT kernel times from data/*/kernel_times.csv.

For each dataset, one figure is produced per x-variable (num_seqs, num_tiles,
imbalance) that actually varies. A variable is skipped when a single value accounts
for more than MODE_THRESHOLD of the rows (e.g. num_seqs = 64 in every row).

Usage:
    python3 plot_kernel_times.py [data_dir] [--show]

Figures are written next to each CSV as <dataset>/kernel_times_<x>.png.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

X_VARS = ["num_seqs", "num_tiles", "imbalance", "skew"]
SERIES = [  # (column, label, color)
    ("naive", "Naive", "#2a78d6"),
    ("lpt", "LPT", "#eb6834"),
]
MODE_THRESHOLD = 0.5  # skip x if the most common value covers > this fraction
X_LABELS = {"num_seqs": "Number of sequences", "num_tiles": "Number of tiles",
            "imbalance": "Imbalance (max / mean sequence length)", "skew": "Skew"}

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def is_varying(col: pd.Series) -> bool:
    values = col.round(3)
    if values.nunique() <= 1:
        return False
    mode_frac = values.value_counts().iloc[0] / len(values)
    return mode_frac <= MODE_THRESHOLD


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(df: pd.DataFrame, dataset: str, x: str, out_path: str):
    df = df.sort_values(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ends = [df[col].iloc[-1] for col, _, _ in SERIES]
    y_span = max(df[[c for c, _, _ in SERIES]].max().max(), 1e-9)
    # Direct-label line ends only when they won't collide (legend covers the rest)
    label_ends = abs(ends[0] - ends[1]) / y_span > 0.06
    for col, label, color in SERIES:
        ax.plot(df[x], df[col], color=color, linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                label=label)
        if label_ends:
            ax.annotate(label, (df[x].iloc[-1], df[col].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", color=TEXT, fontsize=9)

    if x == "num_seqs" and df[x].min() > 0 and df[x].max() / df[x].min() >= 8:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df[x].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel(X_LABELS[x], color=TEXT_MUTED)
    ax.set_ylabel("Kernel time (ms)", color=TEXT_MUTED)
    ax.set_title(f"{dataset}: kernel time vs. {X_LABELS[x].lower()}",
                 color=TEXT, loc="left", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=TEXT, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default="data")
    parser.add_argument("--show", action="store_true", help="also open figures interactively")
    args = parser.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "kernel_times.csv")))
    if not csvs:
        raise SystemExit(f"No kernel_times.csv files found under {args.data_dir}/")

    for csv in csvs:
        dataset = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv)
        xs = [x for x in X_VARS if x in df.columns and is_varying(df[x])]
        skipped = [x for x in X_VARS if x in df.columns and x not in xs]
        print(f"{dataset}: plotting {xs or 'nothing'}"
              + (f" (skipped {skipped})" if skipped else ""))
        for x in xs:
            out = os.path.join(os.path.dirname(csv), f"kernel_times_{x}.png")
            plot(df, dataset, x, out)
            print(f"  wrote {out}")

    if args.show:
        for png in sorted(glob.glob(os.path.join(args.data_dir, "*", "kernel_times_*.png"))):
            img = plt.imread(png)
            plt.figure(figsize=(6.4, 4.2))
            plt.imshow(img)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
