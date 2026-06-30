"""Build the energy arc of each set.

Extraction gives me one row of features per 30-second window. Here, I turn the raw
energy of those windows into the "arc": the shape of how a set's energy rises and
falls from start to finish.

Two things make the arcs comparable across sets:

  - energy_norm   per-set min-max of rms_energy, so a quietly mastered set and a
                  loud one are judged on their own shape (not volume).
  - t_norm        position in the set from 0 (first window) to 1 (last), so a
                  60-minute set and a 120-minute set line up on the same x-axis.

I also keep a smoothed version of the arc, because raw 30-second energy is jumpy
and the shape reads more clearly once it is gently averaged.

Outputs:
  - data/processed/arcs.parquet   the features plus the arc columns
  - figures/arcs_grid.png         every set's arc as a small-multiple panel
  - figures/mean_arc.png          all sets overlaid, with the average arc on top

Run:
    python -m src.build_arc
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # write figures to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config, resolve


def _minmax(s: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]; flat series collapse to zeros, not NaNs."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def add_arc_columns(df: pd.DataFrame, smooth_windows: int) -> pd.DataFrame:
    """Add t_norm, energy_norm and energy_smooth, computed within each set."""
    df = df.sort_values(["set_id", "window_index"]).copy()
    g = df.groupby("set_id")

    # Position in the set, 0 at the first window to 1 at the last.
    df["t_norm"] = g["window_index"].transform(
        lambda s: s / s.max() if s.max() > 0 else s * 0.0
    )
    # Per-set min-max so loudness differences between sets do not dominate.
    df["energy_norm"] = g["rms_energy"].transform(_minmax)
    # A centred rolling mean to make the arc shape readable.
    df["energy_smooth"] = g["energy_norm"].transform(
        lambda s: s.rolling(smooth_windows, center=True, min_periods=1).mean()
    )
    return df


def plot_arcs_grid(df: pd.DataFrame, out_path) -> None:
    """One small panel per set: smoothed energy against minutes into the set."""
    set_ids = sorted(df["set_id"].unique())
    ncols = 3
    nrows = math.ceil(len(set_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, set_id in zip(axes, set_ids):
        sub = df[df["set_id"] == set_id]
        minutes = sub["t_start_sec"] / 60.0
        ax.fill_between(minutes, sub["energy_smooth"], alpha=0.25)
        ax.plot(minutes, sub["energy_smooth"], linewidth=1.5)
        ax.set_title(set_id, fontsize=8)
        ax.set_xlabel("minutes", fontsize=7)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)

    # Hide any unused panels in the last row.
    for ax in axes[len(set_ids):]:
        ax.axis("off")

    fig.suptitle("Energy arc of each set (per-set normalised)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_mean_arc(df: pd.DataFrame, out_path, grid_points: int) -> None:
    """Overlay every set on a shared 0-1 timeline and draw the average arc.

    Each set has a different number of windows, so I interpolate every set onto
    the same normalised-time grid before averaging. The mean line is the
    headline result: the typical shape of a set."""
    grid = np.linspace(0, 1, grid_points)
    resampled = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for set_id, sub in df.groupby("set_id"):
        sub = sub.sort_values("t_norm")
        y = np.interp(grid, sub["t_norm"], sub["energy_smooth"])
        resampled.append(y)
        ax.plot(grid, y, color="grey", alpha=0.3, linewidth=1)

    stack = np.vstack(resampled)
    mean_arc = stack.mean(axis=0)
    lo, hi = np.percentile(stack, [25, 75], axis=0)

    ax.fill_between(grid, lo, hi, alpha=0.2, label="middle 50% of sets")
    ax.plot(grid, mean_arc, linewidth=3, label="average arc")
    ax.set_xlabel("position in set (0 = start, 1 = end)")
    ax.set_ylabel("relative energy")
    ax.set_title("The average shape of a DJ set")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_arc() -> None:
    cfg = load_config()
    interim_dir = resolve(cfg["paths"]["interim"])
    processed_dir = resolve(cfg["paths"]["processed"])
    fig_dir = resolve("figures")
    smooth_windows = cfg["arc"]["smooth_windows"]
    grid_points = cfg["arc"]["grid_points"]

    features = pd.read_parquet(interim_dir / "features.parquet")
    arcs = add_arc_columns(features, smooth_windows)

    out = processed_dir / "arcs.parquet"
    arcs.to_parquet(out, index=False)

    plot_arcs_grid(arcs, fig_dir / "arcs_grid.png")
    plot_mean_arc(arcs, fig_dir / "mean_arc.png", grid_points)

    print(f"Wrote {len(arcs)} windows with arc columns to {out}")
    print(f"Figures: {fig_dir / 'arcs_grid.png'}, {fig_dir / 'mean_arc.png'}")


if __name__ == "__main__":
    build_arc()
