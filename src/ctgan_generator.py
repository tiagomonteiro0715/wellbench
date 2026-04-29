from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .generator import PHYSICAL_BOUNDS


# CTGAN models were trained with slightly different column names than the
# physics-based generator. Rename on the way out so both generators produce a
# consistent schema that `PHYSICAL_BOUNDS` and `clean_well_data` understand.
_CTGAN_COLUMN_RENAMES = {
    "RHOB_combined": "RHOB",
    "RES_DEEP": "RT",
}

# Canonical output order, matching SyntheticWellLogGenerator.generate().
_COLUMN_ORDER_BASIC = ["DEPTH", "GR", "DT", "RHOB", "RT"]
_COLUMN_ORDER_PP = _COLUMN_ORDER_BASIC + ["HP", "OB", "DT_NCT", "PPP"]

_MODELS_DIR = Path(__file__).parent / "ctgan_models"


def _resolve_model_path(region_index: int, model_dir: Path | None = None) -> Path:
    directory = Path(model_dir) if model_dir is not None else _MODELS_DIR
    path = directory / f"ctgan_r{region_index}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"CTGAN checkpoint not found: {path}")
    return path


def _load_ctgan(path: Path):
    """Load a pickled CTGAN synthesizer.

    CTGAN.save uses torch.save under the hood, so torch is the loader. The
    heavy deps (torch, ctgan) are only imported here, keeping the base
    wellbench install lightweight.
    """
    try:
        import torch  # noqa: F401
        import ctgan  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CTGAN generation requires the optional 'ctgan' and 'torch' "
            "packages. Install with: pip install wellbench[ctgan]"
        ) from exc

    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


class CTGANSyntheticWellLogGenerator:
    """Generate synthetic well logs from a pre-trained CTGAN model.

    Mirrors the interface of `SyntheticWellLogGenerator`: call `.generate(seed,
    depth=None)` to get a DataFrame with a DEPTH column plus log columns. Since
    CTGAN produces i.i.d. tabular rows (no sequential structure), one row is
    drawn per depth sample and rows are ordered along the depth axis in
    sampling order — the depth column is an externally imposed index, not a
    learned variable.
    """

    def __init__(
        self,
        params: dict,
        model_path: str | Path | None = None,
        region_index: int | None = None,
    ):
        """
        params:        region parameter dict (same one used for the physics
                       generator). Supplies `depth_range`, `depth_step`, and
                       `has_pore_pressure`.
        model_path:    explicit path to a CTGAN .pkl checkpoint.
        region_index:  1-based index into the bundled ctgan_rN.pkl files.
                       Ignored if model_path is provided.
        """
        self.p = params

        if model_path is not None:
            self._model_path = Path(model_path)
        elif region_index is not None:
            self._model_path = _resolve_model_path(region_index)
        else:
            raise ValueError(
                "Provide either `model_path` or `region_index` to locate the "
                "CTGAN checkpoint."
            )

        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_ctgan(self._model_path)

    # ------------------------------------------------------------------
    def _set_seed(self, seed: int) -> None:
        """Make CTGAN.sample() deterministic for a given seed."""
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    def generate(self, seed: int, depth: np.ndarray | None = None) -> pd.DataFrame:
        self._ensure_loaded()

        if depth is None:
            start, stop = self.p["depth_range"]
            depth = np.arange(start, stop, self.p["depth_step"])
        depth = np.asarray(depth, dtype=float)

        self._set_seed(seed)
        samples = self._model.sample(len(depth))

        df = samples.rename(columns=_CTGAN_COLUMN_RENAMES).copy()

        # Clip to physical bounds so downstream code can trust the schema.
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            if col in df.columns:
                df[col] = np.clip(df[col].astype(float), lo, hi)

        df.insert(0, "DEPTH", depth)

        if self.p.get("has_pore_pressure", False):
            order = [c for c in _COLUMN_ORDER_PP if c in df.columns]
        else:
            order = [c for c in _COLUMN_ORDER_BASIC if c in df.columns]
        extras = [c for c in df.columns if c not in order]
        return df[order + extras].reset_index(drop=True)


def load_ctgan_generator(
    region_index: int,
    regions: list[dict] | None = None,
) -> CTGANSyntheticWellLogGenerator:
    """Convenience factory: pair REGION_N params with ctgan_rN.pkl (1-indexed)."""
    if regions is None:
        from .regions import ALL_REGIONS
        regions = ALL_REGIONS
    if not (1 <= region_index <= len(regions)):
        raise IndexError(
            f"region_index must be in 1..{len(regions)}, got {region_index}"
        )
    return CTGANSyntheticWellLogGenerator(
        params=regions[region_index - 1],
        region_index=region_index,
    )
