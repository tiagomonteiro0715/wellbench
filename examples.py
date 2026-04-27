"""
wellbench — usage examples
==========================

A single, runnable tour of the public API. Each ``example_*`` function
demonstrates one use case. Run the whole tour with::

    python examples.py

…or pick one with::

    python examples.py basic

The optional CTGAN example needs the ``[ctgan]`` extra::

    pip install wellbench[ctgan]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wellbench import (
    ALL_REGIONS,
    BENCHMARK_SEEDS,
    PHYSICAL_BOUNDS,
    REGION_1,
    REGION_4,
    SyntheticWellLogGenerator,
    clean_well_data,
    generate_benchmark,
)


# ---------------------------------------------------------------------------
# 1. Basic generation — one region, one seed
# ---------------------------------------------------------------------------
def example_basic() -> pd.DataFrame:
    """Generate a single synthetic well from REGION_1 (Missa Keswal)."""
    gen = SyntheticWellLogGenerator(REGION_1)
    df = gen.generate(seed=42)

    print(f"REGION_1 / seed=42 -> {len(df)} rows, columns: {list(df.columns)}")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 2. Custom depth axis — re-sample on an arbitrary array
# ---------------------------------------------------------------------------
def example_custom_depth() -> pd.DataFrame:
    """Pass an explicit depth array (e.g. to match a real well's sampling)."""
    gen = SyntheticWellLogGenerator(REGION_4)
    depth = np.linspace(120, 700, num=2000)  # metres
    df = gen.generate(seed=7, depth=depth)

    print(
        f"REGION_4 / custom depth -> {len(df)} rows, "
        f"depth {df.DEPTH.min():.1f}-{df.DEPTH.max():.1f} m"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Pore-pressure regions vs basic regions
# ---------------------------------------------------------------------------
def example_pore_pressure() -> None:
    """Regions 1-3 add HP, OB, DT_NCT, PPP. Regions 4-5 do not."""
    for idx, region in enumerate(ALL_REGIONS, start=1):
        df = SyntheticWellLogGenerator(region).generate(seed=42)
        has_pp = "PPP" in df.columns
        print(
            f"  Region {idx} ({region['zone']:<8}): "
            f"{len(df.columns)} columns, has_pore_pressure={has_pp}"
        )


# ---------------------------------------------------------------------------
# 4. Cleaning utility — remove sentinels, out-of-bounds and outliers
# ---------------------------------------------------------------------------
def example_cleaning() -> pd.DataFrame:
    """Inject sentinel/outlier values into a synthetic frame, then clean."""
    df = SyntheticWellLogGenerator(REGION_1).generate(seed=42).copy()

    df.loc[10:14, "GR"] = -999.25     # sentinel
    df.loc[20:21, "RHOB"] = 99.0      # out of physical bounds
    df.loc[30, "DT"] = 5_000.0        # extreme outlier

    cleaned = clean_well_data(df, label="demo", verbose=True)
    print(f"  before: {len(df)} rows, after: {len(cleaned)} rows")
    return cleaned


# ---------------------------------------------------------------------------
# 5. Inspect the bundled physical bounds
# ---------------------------------------------------------------------------
def example_show_bounds() -> None:
    print("Physical bounds applied to all generated/cleaned data:")
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        print(f"  {col:<7} [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# 6. Full benchmark — 5 regions x 3 seeds = 15 CSV files
# ---------------------------------------------------------------------------
def example_benchmark(output_dir: str = "benchmark_demo") -> list[Path]:
    """Reproduce the canonical 15-dataset benchmark."""
    paths = generate_benchmark(
        regions=ALL_REGIONS,
        seeds=BENCHMARK_SEEDS,
        output_dir=output_dir,
    )
    print(f"  wrote {len(paths)} CSV files to {output_dir}/")
    return paths


# ---------------------------------------------------------------------------
# 7. CTGAN-based generation (optional dep)
# ---------------------------------------------------------------------------
def example_ctgan() -> pd.DataFrame | None:
    """Sample from the bundled CTGAN checkpoint for region 1.

    Skipped silently if the optional 'ctgan' / 'torch' extras aren't installed.
    """
    try:
        from wellbench import load_ctgan_generator
    except ImportError:
        print("  wellbench[ctgan] not installed — skipping.")
        return None

    try:
        gen = load_ctgan_generator(region_index=1)
        df = gen.generate(seed=42)
    except ImportError as e:
        print(f"  optional CTGAN deps missing: {e}")
        return None

    print(f"  CTGAN region 1 / seed=42 -> {len(df)} rows, columns: {list(df.columns)}")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 8. Aligning synthetic depth to a real well's depth column (recipe)
# ---------------------------------------------------------------------------
def example_align_to_real(real_csv: str | Path) -> pd.DataFrame:
    """Read a real well's DEPTH column and emit a depth-aligned synthetic well.

    Useful when you want one synthetic row per real measurement, e.g. for
    side-by-side comparison plots.
    """
    real = pd.read_csv(real_csv, usecols=["DEPTH"])
    depth = real["DEPTH"].to_numpy()

    gen = SyntheticWellLogGenerator(REGION_1)
    synth = gen.generate(seed=42, depth=depth)

    print(
        f"  aligned to {real_csv}: {len(synth)} rows, "
        f"depth {depth.min():.1f}-{depth.max():.1f}"
    )
    return synth


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
EXAMPLES = {
    "basic": example_basic,
    "custom-depth": example_custom_depth,
    "pore-pressure": example_pore_pressure,
    "cleaning": example_cleaning,
    "bounds": example_show_bounds,
    "benchmark": example_benchmark,
    "ctgan": example_ctgan,
}


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    names = argv or list(EXAMPLES)

    for name in names:
        if name not in EXAMPLES:
            print(f"unknown example: {name!r}; valid: {list(EXAMPLES)}")
            continue
        print("=" * 70)
        print(f"  example: {name}")
        print("=" * 70)
        EXAMPLES[name]()
        print()


if __name__ == "__main__":
    main()
