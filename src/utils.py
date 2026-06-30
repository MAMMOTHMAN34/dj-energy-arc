"""Small shared helpers so every stage loads config and paths the same way."""
from __future__ import annotations

from pathlib import Path

import yaml

# The project root is two levels up from this file (src/utils.py -> project/).
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Read the central YAML config into a plain dict."""
    with open(ROOT / path, "r") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Turn a config-relative path (e.g. 'data/raw') into an absolute one
    and make sure the directory exists, so callers never trip on a missing
    folder."""
    p = ROOT / path
    p.mkdir(parents=True, exist_ok=True)
    return p
