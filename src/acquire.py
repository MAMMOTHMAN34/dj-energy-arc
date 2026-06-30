"""Acquire public DJ sets.

I keep a list of public set URLs in `configs/sets.csv`. This script pulls the
best audio-only stream for each one with yt-dlp and saves it under `data/raw/`
named by its `id`, so the rest of the pipeline can find a set purely by id.

Audio files are large and licence-encumbered, so they are gitignored. Anyone
cloning the repo re-fetches them by running:

    python -m src.acquire

Requires the `ffmpeg` binary on PATH (brew install ffmpeg) for remuxing.
"""
from __future__ import annotations

import sys

import pandas as pd
import yt_dlp

from src.utils import load_config, resolve


def _already_have(raw_dir, set_id: str):
    """Return an existing file for this id, if one is already downloaded."""
    hits = list(raw_dir.glob(f"{set_id}.*"))
    return hits[0] if hits else None


def acquire() -> None:
    cfg = load_config()
    raw_dir = resolve(cfg["paths"]["raw"])
    sets = pd.read_csv(cfg["paths"]["sets"], comment="#")

    if sets.empty:
        sys.exit("configs/sets.csv has no rows. Add at least one set URL first.")

    audio_format = cfg["acquire"]["audio_format"]
    max_seconds = cfg["acquire"]["max_duration_minutes"] * 60

    for row in sets.itertuples(index=False):
        set_id, url = str(row.id), str(row.url)

        if "REPLACE_ME" in url:
            print(f"[skip] {set_id}: still the placeholder URL, edit sets.csv")
            continue

        existing = _already_have(raw_dir, set_id)
        if existing is not None:
            print(f"[have] {set_id}: already at {existing.name}")
            continue

        # outtmpl writes to data/raw/<id>.<ext>; yt-dlp fills the extension.
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(raw_dir / f"{set_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                }
            ],
            # Reject anything longer than the configured ceiling before it
            # downloads, so a stray multi-hour stream cannot fill the disk.
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration < {max_seconds}"
            ),
            "quiet": True,
            "no_warnings": True,
        }

        print(f"[get ] {set_id}: {url}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # one bad URL should not kill the batch
            print(f"[fail] {set_id}: {exc}")
            continue

    print("Done. Downloaded sets live in", raw_dir)


if __name__ == "__main__":
    acquire()
