"""Cluster windows into vibe profiles.

The arc tells me how loud a set is over time, but not what kind of sound is
filling that energy. Thus, clustering takes every 30-second window and
groups them by their sound (tempo, brightness, timbre, danceability) into a
handful of "vibe profiles", with no genre labels given.

Then, I do three things with the clusters:

  - Profile them. Each vibe gets a plain-language label from its average features.
  - Sanity-check them against the genre_hint column (which the model never sees).
  - Trace them through time to allow me to see a set move from one vibe to another.

Outputs:
  - data/processed/vibes.parquet   every window tagged with its vibe
  - figures/vibe_profiles.png      what each vibe sounds like (feature heatmap)
  - figures/vibe_clusters.png      the vibes in 2D (PCA projection)
  - figures/vibe_genre.png         vibe vs genre_hint cross-check
  - figures/vibe_journey.png       each set's path through the vibes over time

Run:
    python -m src.cluster_vibes
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, resolve

# Features that describe the *character* of a window, not its position in the set.
# mfcc_0 is essentially loudness, so I leave it out and let rms_energy carry that;
# mfcc_1..12 capture the timbre/texture.
CLUSTER_FEATURES = (
    ["bpm", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
     "zcr", "danceability", "rms_energy"]
    + [f"mfcc_{i}" for i in range(1, 13)]
)


def label_clusters(profile: pd.DataFrame) -> dict:
    """Turn each cluster's average features into a readable name."""
    energy, bright, tempo = profile["rms_energy"], profile["spectral_centroid"], profile["bpm"]

    def band(series, value, low, mid, high):
        lo, hi = series.quantile(1 / 3), series.quantile(2 / 3)
        return low if value <= lo else (high if value >= hi else mid)

    labels = {}
    for c in profile.index:
        e = band(energy, energy[c], "low-energy", "mid-energy", "high-energy")
        b = "dark" if bright[c] < bright.median() else "bright"
        labels[c] = f"{b} {e}"
    return labels


def cluster() -> None:
    cfg = load_config()
    n_vibes = cfg["cluster"]["n_vibes"]
    seed = cfg["cluster"]["random_state"]
    processed_dir = resolve(cfg["paths"]["processed"])
    fig_dir = resolve("figures")

    df = pd.read_parquet(processed_dir / "arcs.parquet").sort_values(
        ["set_id", "window_index"]
    ).reset_index(drop=True)

    # genre_hint lives in sets.csv, not the feature table. Join it on by id so
    # I can sanity-check the clusters against it (the model never sees it).
    sets = pd.read_csv(cfg["paths"]["sets"], comment="#")[["id", "genre_hint"]]
    df = df.merge(sets, left_on="set_id", right_on="id", how="left").drop(columns="id")

    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=n_vibes, random_state=seed, n_init=10)
    df["vibe"] = km.fit_predict(X)

    # Profile each vibe on interpretable features and give it a name.
    profile = df.groupby("vibe")[
        ["bpm", "rms_energy", "spectral_centroid", "danceability"]
    ].mean()
    profile["size"] = df.groupby("vibe").size()
    labels = label_clusters(profile)
    df["vibe_label"] = df["vibe"].map(labels)
    profile["label"] = profile.index.map(labels)

    print("Vibe profiles (averages per cluster):")
    print(profile.round(2).to_string())
    print("\nVibe vs genre_hint (share of each genre's windows):")
    crosstab = pd.crosstab(df["vibe_label"], df["genre_hint"], normalize="columns")
    print(crosstab.round(2).to_string())

    out = processed_dir / "vibes.parquet"
    df.to_parquet(out, index=False)

    _plot_profiles(df, labels, fig_dir / "vibe_profiles.png")
    _plot_pca(X, df["vibe"], labels, fig_dir / "vibe_clusters.png")
    _plot_genre(crosstab, fig_dir / "vibe_genre.png")
    _plot_journey(df, labels, fig_dir / "vibe_journey.png")
    print(f"\nWrote {len(df)} tagged windows to {out} and figures to {fig_dir}")


def _plot_profiles(df, labels, out_path):
    """Heatmap of standardised feature means per vibe: what each one sounds like."""
    feats = ["bpm", "rms_energy", "spectral_centroid", "spectral_bandwidth",
             "zcr", "danceability"]
    means = df.groupby("vibe")[feats].mean()
    # Standardise each column so colours compare across very different units.
    z = (means - means.mean()) / means.std()
    z.index = [labels[c] for c in z.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(z.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(z)), z.index, fontsize=9)
    ax.set_title("What each vibe sounds like (standardised feature means)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="above / below average")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_pca(X, vibes, labels, out_path):
    """Project the high-dimensional features to 2D so the clusters are visible."""
    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(np.unique(vibes)):
        mask = vibes == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, label=labels[c])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Vibe profiles in feature space (PCA)")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_genre(crosstab, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(crosstab.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(crosstab.shape[1]), crosstab.columns, rotation=40,
                  ha="right", fontsize=7)
    ax.set_yticks(range(crosstab.shape[0]), crosstab.index, fontsize=8)
    ax.set_title("Do genres land in different vibes? (column-normalised)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of genre's windows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_journey(df, labels, out_path):
    """Each set as a row: its vibe at every point in time, coloured by vibe."""
    set_ids = sorted(df["set_id"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(set_ids) + 1.5))
    for y, sid in enumerate(set_ids):
        sub = df[df["set_id"] == sid]
        ax.scatter(sub["t_start_sec"] / 60.0, [y] * len(sub),
                   c=[cmap(v % 10) for v in sub["vibe"]], s=14, marker="s")
    ax.set_yticks(range(len(set_ids)), set_ids, fontsize=7)
    ax.set_xlabel("minutes into set")
    ax.set_title("Each set's journey through the vibes")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                          color=cmap(c % 10), label=labels[c])
               for c in sorted(labels)]
    ax.legend(handles=handles, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cluster()
