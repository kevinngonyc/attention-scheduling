#!/usr/bin/env python3
"""Gantt chart of per-tile SM traces: naive vs. LPT.

Reads data/<name>/sm_info_w_<w>_{naive,lpt}.csv (written by runSMInfoAttention)
and draws one row per SM, with one lane per co-resident block. Tiles from the
longest sequence are highlighted. Also prints a summary per schedule, including
how many SMs ran two long tiles at the same time.

Usage:
    python3 plot_gantt.py data/descending 3     # one workload
    python3 plot_gantt.py data/descending       # every workload with both traces

Writes data/<name>/gantt_w<w>.png.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

SCHEDULES = ["naive", "lpt"]
LONG_COLOR = "#eb6834"
OTHER_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    t0 = df["start_ns"].min()
    df["start_ms"] = (df["start_ns"] - t0) / 1e6
    df["end_ms"] = (df["end_ns"] - t0) / 1e6
    df["kv_len"] = df["kv_end"] - df["kv_start"]
    return df


def assign_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Per SM, pack tiles into lanes so no two tiles in a lane overlap."""
    df = df.sort_values(["smid", "start_ns", "end_ns"]).copy()
    lanes = []
    for _, group in df.groupby("smid", sort=False):
        lane_ends = []  # last end_ns in each lane
        for start, end in zip(group["start_ns"], group["end_ns"]):
            for lane, last_end in enumerate(lane_ends):
                if last_end <= start:
                    lane_ends[lane] = end
                    break
            else:
                lane = len(lane_ends)
                lane_ends.append(end)
            lanes.append(lane)
    df["lane"] = lanes
    return df


def max_concurrent(intervals) -> int:
    """Largest number of intervals open at the same instant (touching ends don't count)."""
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
                    key=lambda x: (x[0], x[1]))
    best = cur = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def summarize(name: str, df: pd.DataFrame, long_mask: pd.Series):
    long = df[long_mask]
    dur = long["end_ms"] - long["start_ms"]
    per_sm = long.groupby("smid")
    paired = sum(1 for _, g in per_sm if max_concurrent(zip(g["start_ns"], g["end_ns"])) >= 2)
    print(f"  {name:5s} makespan={df['end_ms'].max():7.3f} ms | "
          f"long tiles={len(long)} on {per_sm.ngroups} SMs, "
          f"SMs with 2+ long tiles at once={paired} | "
          f"long tile ms: first start={long['start_ms'].min():.3f} "
          f"last end={long['end_ms'].max():.3f} "
          f"mean dur={dur.mean():.3f} max dur={dur.max():.3f} | "
          f"SMs seen={df['smid'].nunique()} lanes/SM={df['lane'].max() + 1}")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_workload(data_dir: str, w: int):
    paths = {s: os.path.join(data_dir, f"sm_info_w_{w}_{s}.csv") for s in SCHEDULES}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"workload {w}: missing {', '.join(missing)}; skipped")
        return

    traces = {s: assign_lanes(load(p)) for s, p in paths.items()}
    max_kv = max(df["kv_len"].max() for df in traces.values())
    smids = sorted(set().union(*(set(df["smid"]) for df in traces.values())))
    row_of = {sm: i for i, sm in enumerate(smids)}  # SM ids may have gaps
    n_lanes = max(df["lane"].max() for df in traces.values()) + 1
    xmax = max(df["end_ms"].max() for df in traces.values())

    print(f"{data_dir} workload {w} (long tile = kv length {max_kv})")
    fig, axes = plt.subplots(2, 1, figsize=(11, 2 + 0.17 * len(smids) * 2),
                             sharex=True, dpi=150)
    lane_h = 1.0 / n_lanes
    for ax, sched in zip(axes, SCHEDULES):
        df = traces[sched]
        long_mask = df["kv_len"] == max_kv
        summarize(sched, df, long_mask)

        y = df["smid"].map(row_of) + df["lane"] * lane_h
        for is_long, color in ((False, OTHER_COLOR), (True, LONG_COLOR)):
            sel = long_mask == is_long
            ax.barh(y[sel], (df["end_ms"] - df["start_ms"])[sel], left=df["start_ms"][sel],
                    height=lane_h * 0.85, align="edge", color=color,
                    edgecolor="white", linewidth=0.3)

        makespan = df["end_ms"].max()
        ax.axvline(makespan, color=TEXT_MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"{sched.upper()}: makespan {makespan:.3f} ms",
                     loc="left", color=TEXT, fontsize=11)
        ax.set_yticks([i + 0.5 for i in range(len(smids))])
        ax.set_yticklabels(smids, fontsize=6)
        ax.set_ylim(len(smids), 0)
        ax.set_ylabel("SM id", color=TEXT_MUTED)
        style_axes(ax)

    axes[-1].set_xlabel("Time since first tile start (ms)", color=TEXT_MUTED)
    axes[-1].set_xlim(0, xmax * 1.02)
    axes[0].legend(handles=[Patch(color=LONG_COLOR, label=f"Long-sequence tile (kv={max_kv})"),
                            Patch(color=OTHER_COLOR, label="Other tiles")],
                   frameon=False, labelcolor=TEXT, loc="upper right", fontsize=8,
                   bbox_to_anchor=(1, 1.12), ncol=2)
    fig.suptitle(f"{os.path.basename(os.path.normpath(data_dir))} workload {w}: tiles per SM",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    out = os.path.join(data_dir, f"gantt_w{w}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="e.g. data/descending")
    parser.add_argument("workload", nargs="?", type=int, help="workload index (default: all)")
    args = parser.parse_args()

    if args.workload is not None:
        workloads = [args.workload]
    else:
        found = glob.glob(os.path.join(args.data_dir, "sm_info_w_*_naive.csv"))
        workloads = sorted(int(re.search(r"sm_info_w_(\d+)_naive", f).group(1)) for f in found)
        if not workloads:
            raise SystemExit(f"No sm_info_w_*_naive.csv files in {args.data_dir}")
    for w in workloads:
        plot_workload(args.data_dir, w)


if __name__ == "__main__":
    main()
