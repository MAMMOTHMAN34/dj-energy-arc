"""Extract per-window features from each set.

I work energy, tempo and key out from the waveform myself with librosa.

For every set in `data/raw/` I cut the audio into fixed windows (30 seconds by
default) and, for each window, measure:

  - bpm                tempo of that stretch of the set
  - rms_energy         loudness/energy, the spine of the energy arc
  - spectral_centroid  "brightness": where the spectral weight sits
  - spectral_bandwidth spread of frequencies around that centre
  - spectral_rolloff   frequency below which most energy lives
  - zcr                zero-crossing rate, a rough noisiness/percussiveness cue
  - danceability       a pulse-clarity proxy: how steady/strong the beat is
  - key + mode         estimated musical key, via Krumhansl-Schmuckle profiles
  - mfcc_0..mfcc_12    timbre summary, the texture of the sound

The result is one tidy row per (set_id, window), saved to
`data/interim/features.parquet`. Every later stage joins onto this table.

Run:
    python -m src.extract_features            # all sets
    python -m src.extract_features set01      # just one set id
"""
from __future__ import annotations

import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import load_config, resolve

# The twelve pitch classes, indexed the way librosa orders chroma bins.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckle key profiles: how strongly each pitch class is expected
# to sound in a major vs a minor key. Correlating a window's averaged chroma
# against every rotation of these tells me the most likely key and mode.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# librosa renamed the tempo estimator across versions; import it once here so
# the rest of the code does not care which librosa is installed.
try:  # librosa >= 0.10
    from librosa.feature.rhythm import tempo as _tempo_fn
except ImportError:  # older librosa
    from librosa.beat import tempo as _tempo_fn


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Return (key, mode) for a window from its mean chroma vector."""
    best_corr, best_key, best_mode = -np.inf, "C", "major"
    for shift in range(12):
        for profile, mode in ((KS_MAJOR, "major"), (KS_MINOR, "minor")):
            rotated = np.roll(profile, shift)
            # Pearson correlation between the window's chroma and this candidate.
            corr = np.corrcoef(chroma_mean, rotated)[0, 1]
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, PITCH_CLASSES[shift], mode
    return best_key, best_mode


def danceability(onset_env: np.ndarray) -> float:
    """Pulse-clarity proxy in roughly [0, 1].

    A danceable stretch has a strong, regular beat, which shows up as a tall
    peak in the autocorrelation of the onset-strength envelope. I take that
    peak relative to the zero-lag energy as a simple, interpretable score."""
    if onset_env.size == 0:
        return 0.0
    ac = librosa.autocorrelate(onset_env)
    if ac[0] <= 0:
        return 0.0
    # Ignore lag 0 (always the max); the next-best peak is the beat periodicity.
    return float(ac[1:].max() / ac[0])


def window_features(y_win: np.ndarray, sr: int) -> dict:
    """Compute every feature for a single window of samples."""
    rms = librosa.feature.rms(y=y_win)
    centroid = librosa.feature.spectral_centroid(y=y_win, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y_win, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_win, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_win)
    onset_env = librosa.onset.onset_strength(y=y_win, sr=sr)
    bpm = float(np.atleast_1d(_tempo_fn(onset_envelope=onset_env, sr=sr))[0])

    chroma = librosa.feature.chroma_cqt(y=y_win, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, mode = estimate_key(chroma_mean)

    mfcc = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=13).mean(axis=1)

    feats = {
        "bpm": bpm,
        "rms_energy": float(rms.mean()),
        "spectral_centroid": float(centroid.mean()),
        "spectral_bandwidth": float(bandwidth.mean()),
        "spectral_rolloff": float(rolloff.mean()),
        "zcr": float(zcr.mean()),
        "danceability": danceability(onset_env),
        "key": key,
        "mode": mode,
    }
    feats.update({f"mfcc_{i}": float(v) for i, v in enumerate(mfcc)})
    return feats


def process_set(set_id: str, audio_path, sr: int, win_s: int, hop_s: int):
    """Slice one set into windows and return a list of feature rows."""
    # mono=True so energy/brightness describe the mix, not stereo placement.
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    win_len, hop_len = int(win_s * sr), int(hop_s * sr)

    rows = []
    # Stop one full window before the end so the last window is never a stub.
    starts = range(0, max(len(y) - win_len + 1, 0), hop_len)
    for w_idx, start in enumerate(tqdm(starts, desc=set_id, unit="win")):
        y_win = y[start : start + win_len]
        feats = window_features(y_win, sr)
        feats.update(
            {
                "set_id": set_id,
                "window_index": w_idx,
                "t_start_sec": round(start / sr, 1),
                "t_end_sec": round((start + win_len) / sr, 1),
            }
        )
        rows.append(feats)
    return rows


def find_audio(raw_dir, set_id: str):
    """Find the downloaded file for a set id, ignoring the .gitkeep marker."""
    hits = [p for p in raw_dir.glob(f"{set_id}.*") if p.suffix != ""]
    return hits[0] if hits else None


def extract(only: str | None = None) -> None:
    cfg = load_config()
    raw_dir = resolve(cfg["paths"]["raw"])
    interim_dir = resolve(cfg["paths"]["interim"])
    sr = cfg["extract"]["sample_rate"]
    win_s = cfg["extract"]["window_seconds"]
    hop_s = cfg["extract"]["hop_seconds"]

    sets = pd.read_csv(cfg["paths"]["sets"], comment="#")
    if only is not None:
        sets = sets[sets["id"].astype(str) == only]
        if sets.empty:
            sys.exit(f"No set with id '{only}' in configs/sets.csv")

    all_rows = []
    for set_id in sets["id"].astype(str):
        audio_path = find_audio(raw_dir, set_id)
        if audio_path is None:
            print(f"[skip] {set_id}: no audio in {raw_dir}, run src.acquire first")
            continue
        all_rows.extend(process_set(set_id, audio_path, sr, win_s, hop_s))

    if not all_rows:
        sys.exit("No features extracted. Download at least one set first.")

    features = pd.DataFrame(all_rows)
    # Lead with the identifying columns so the table reads naturally.
    lead = ["set_id", "window_index", "t_start_sec", "t_end_sec"]
    features = features[lead + [c for c in features.columns if c not in lead]]

    out = interim_dir / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"Wrote {len(features)} windows across "
          f"{features['set_id'].nunique()} set(s) to {out}")


if __name__ == "__main__":
    extract(sys.argv[1] if len(sys.argv) > 1 else None)
