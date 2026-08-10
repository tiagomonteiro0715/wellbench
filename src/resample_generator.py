from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .generator import PHYSICAL_BOUNDS


class _ResampleGenerator:
    """Shared plumbing for generators that resample an existing DataFrame.

    Unlike the physics and CTGAN generators, these are *non-parametric*: they
    have no forward model and no trained weights, so they need a source
    DataFrame to draw from. Pass real cleaned wells for the research protocol,
    or any DataFrame with a DEPTH column plus log columns.
    """

    def __init__(self, params: dict, source: pd.DataFrame):
        self.p = params
        self.source = source

    def _depth(self, depth: np.ndarray | None) -> np.ndarray:
        if depth is None:
            start, stop = self.p["depth_range"]
            depth = np.arange(start, stop, self.p["depth_step"])
        return np.asarray(depth, dtype=float)

    def _matrix(self) -> tuple[list[str], np.ndarray]:
        """Log columns and their finite rows, as a float matrix."""
        cols = [c for c in self.source.columns if c != "DEPTH"]
        X = self.source[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < 2:
            raise ValueError(
                f"source needs >=2 rows with all log columns finite, got {len(X)}"
            )
        return cols, X

    def _finish(self, values: np.ndarray, cols: list[str],
                depth: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(values, columns=cols)
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            if col in df.columns:
                df[col] = np.clip(df[col], lo, hi)
        df.insert(0, "DEPTH", depth)
        return df


class SMOTEGenerator(_ResampleGenerator):
    """Synthesise rows by SMOTE-style interpolation between nearest neighbours.

    For each output row: pick a source row at random, pick one of its
    ``smote_k_neighbors`` nearest neighbours, and interpolate between them by a
    uniform random factor. Neighbours are found on the raw (unscaled) log
    values, so columns with the widest numeric range dominate the distance.
    """

    def generate(self, seed: int, depth: np.ndarray | None = None) -> pd.DataFrame:
        depth = self._depth(depth)
        cols, X = self._matrix()
        rng = np.random.default_rng(seed)
        n = len(depth)

        k = min(int(self.p.get("smote_k_neighbors", 5)), len(X) - 1)
        base = rng.integers(0, len(X), size=n)
        # Column 0 of the query result is the point itself, so skip it.
        neighbours = cKDTree(X).query(X[base], k=k + 1)[1]
        picked = neighbours[np.arange(n), rng.integers(1, k + 1, size=n)]

        lam = rng.random((n, 1))
        return self._finish(X[base] + lam * (X[picked] - X[base]), cols, depth)


class SmoothedBootstrapGenerator(_ResampleGenerator):
    """Synthesise rows by resampling with replacement plus Gaussian jitter.

    Jitter is scaled to each column's standard deviation by
    ``bootstrap_bandwidth`` and shared across logs by
    ``bootstrap_noise_correlation``.
    """

    def generate(self, seed: int, depth: np.ndarray | None = None) -> pd.DataFrame:
        depth = self._depth(depth)
        cols, X = self._matrix()
        rng = np.random.default_rng(seed)
        n = len(depth)

        resampled = X[rng.integers(0, len(X), size=n)]

        # blended = sqrt(1-w) * independent + sqrt(w) * shared keeps each
        # column's jitter at unit variance (so bandwidth means what it did)
        # while giving any two logs a jitter correlation of exactly w.
        w = float(np.clip(self.p.get("bootstrap_noise_correlation", 0.25), 0.0, 1.0))
        blended = (np.sqrt(1.0 - w) * rng.normal(0, 1, resampled.shape)
                   + np.sqrt(w) * rng.normal(0, 1, (n, 1)))
        jitter = blended * (self.p.get("bootstrap_bandwidth", 0.2)
                            * np.nanstd(X, axis=0))

        return self._finish(resampled + jitter, cols, depth)
