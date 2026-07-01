"""Analyse a single uploaded set inside the app.

The pipeline (extract -> arc -> cluster) runs over the whole library. This
helper does the same work for one freshly uploaded audio file, reusing the exact
same feature extraction, arc construction, and the trained scaler + KMeans so
the new set's vibes are on the same footing as everything else.

It is intentionally kept out of the heavy import path: librosa only loads when
someone actually uploads a set.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.build_arc import add_arc_columns
from src.extract_features import process_set
from src.utils import load_config, resolve


def load_vibe_model() -> dict:
    """The scaler + KMeans + feature list + labels saved by cluster_vibes."""
    return joblib.load(resolve("outputs") / "vibe_model.joblib")


def analyse_audio(path, set_id: str = "uploaded") -> pd.DataFrame:
    """Turn one audio file into a per-window table tagged with arc + vibe."""
    cfg = load_config()
    rows = process_set(
        set_id, path,
        sr=cfg["extract"]["sample_rate"],
        win_s=cfg["extract"]["window_seconds"],
        hop_s=cfg["extract"]["hop_seconds"],
    )
    df = add_arc_columns(pd.DataFrame(rows), cfg["arc"]["smooth_windows"])

    vm = load_vibe_model()
    X = vm["scaler"].transform(df[vm["features"]])
    df["vibe"] = vm["kmeans"].predict(X)
    df["vibe_label"] = df["vibe"].map(vm["labels"])
    return df
