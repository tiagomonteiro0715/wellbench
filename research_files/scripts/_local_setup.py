"""Local execution helpers shared by every research script.

These scripts were originally Google Colab notebooks. This module provides
the small amount of glue that lets them run on a normal machine:

* `DATA_DIR`     - folder where downloaded well-log files live.
* `CHECKPOINT_DIR` - folder where Optuna / training checkpoints land.
* `rss_mb()`     - cross-platform process-memory reading (replaces the
                   Linux-only ``resource`` module the Colab code used).

Each script calls `bootstrap()` once near the top: that switches the
working directory to `DATA_DIR` so existing ``gdown.download(url)`` calls
and bare-filename ``pd.read_csv("FOO.csv")`` calls keep working unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def bootstrap(checkpoint_subdir: str | None = None) -> tuple[Path, Path]:
    """Ensure data + checkpoint folders exist and chdir into the data folder.

    Returns ``(data_dir, checkpoint_dir)``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = CHECKPOINT_DIR / checkpoint_subdir if checkpoint_subdir else CHECKPOINT_DIR
    out_ckpt.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_DIR)
    return DATA_DIR, out_ckpt


def rss_mb() -> float:
    """Resident-set size of the current process in MB (cross-platform)."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
