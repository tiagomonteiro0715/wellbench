"""
main.py — depth-matched synthetic counterparts for the real wells in ``original_data/``
======================================================================================

Reads every real well in ``original_data/``, works out which region calibration it
belongs to, and writes one synthetic well per real well **on that well's own depth
axis** — same interval, same sampling, same row count — into ``synthetic/``.

This is the difference from the packaged ``wellbench`` CLI: that one emits each
region over the region's nominal ``depth_range`` (e.g. 500–4500 ft for regions
1–3), which is a basin-wide envelope rather than any particular borehole. Here the
depth array comes from the real file, so a synthetic well lines up row-for-row
with its real counterpart and can be diffed or plotted against it directly.

Usage::

    python main.py                                  # physics generator, seed 42
    python main.py --generator ctgan                # CTGAN baseline (needs [ctgan])
    python main.py --generator smote --seeds 42 123 # resampling baseline, 2 seeds
    python main.py --wells PINDORI-1 MINWAL-2       # only these wells
    python main.py --help

Output mirrors the ``real_<WELL>.xlsx`` naming of ``original_data/``::

    synthetic/
    ├── manifest.csv                                 # provenance for every file
    ├── synth_JOYAMAIR-4.xlsx
    ├── synth_MINWAL-2.xlsx
    ...
    └── synth_PINDORI-3.xlsx

With more than one ``--seeds`` value the seed is appended
(``synth_JOYAMAIR-4_seed_123.xlsx``) so the files don't collide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wellbench import (
    ALL_REGIONS,
    SMOTEGenerator,
    SmoothedBootstrapGenerator,
    SyntheticWellLogGenerator,
    clean_well_data,
)

ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = ROOT / "original_data"
DEFAULT_OUTPUT_DIR = ROOT / "synthetic"

# Which region calibration each real well belongs to. Mirrors REGION_CONFIGS in
# research_files/scripts/final_optimal_gan_based_generation.py — the same
# groupings the region parameters and CTGAN checkpoints were fitted on.
WELL_REGIONS = {
    "MISSA-KESWAL-01": 1,
    "MISSA-KESWAL-02": 1,
    "MISSA-KESWAL-03": 1,
    "PINDORI-1": 2,
    "PINDORI-2": 2,
    "PINDORI-3": 2,
    "JOYAMAIR-4": 3,
    "MINWAL-2": 3,
    "MINWAL-X-1": 3,
}

# Real files carry an inconsistent mix of vendor names for the same curve, plus a
# lot of columns the generators don't model. Map the ones we use onto the library
# schema (DEPTH, GR, DT, RHOB, RT, HP, OB, DT_NCT, PPP) and drop the rest.
_COLUMN_ALIASES = {
    "DEPTH": "DEPTH",
    "GR": "GR",
    "DT": "DT",
    "DT_NCT": "DT_NCT",
    "RHOB": "RHOB",
    "RHOB_COMBINED": "RHOB",
    "RES_DEEP": "RT",
    "RT": "RT",
    "HP": "HP",
    "OB": "OB",
    "PPP": "PPP",
}


def well_name(path: Path) -> str:
    """``original_data/real_PINDORI-1.xlsx`` -> ``PINDORI-1``."""
    return path.stem[len("real_"):] if path.stem.startswith("real_") else path.stem


def read_real_well(path: Path, verbose: bool = True) -> pd.DataFrame:
    """Read one real well and reduce it to the library's schema, cleaned.

    Cleaning matters for the depth axis: `clean_well_data` drops rows where every
    log is NaN, so the surviving DEPTH values are the interval the well actually
    logged rather than the full spreadsheet extent.
    """
    raw = pd.read_excel(path)

    keep = {}
    for col in raw.columns:
        target = _COLUMN_ALIASES.get(str(col).strip().upper())
        # First alias wins, so a file with both RHOB and RHOB_COMBINED keeps one.
        if target is not None and target not in keep:
            keep[target] = col

    if "DEPTH" not in keep:
        raise ValueError(f"{path.name}: no DEPTH column (found {list(raw.columns)})")

    df = raw[list(keep.values())].copy()
    df.columns = list(keep)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)

    return clean_well_data(df, label=f"{well_name(path)} REAL", verbose=verbose)


def build_generator(kind: str, region: dict, real: pd.DataFrame, region_index: int):
    """Instantiate one of the library's four generators for a region."""
    if kind == "physics":
        return SyntheticWellLogGenerator(region)
    if kind == "ctgan":
        from wellbench import load_ctgan_generator

        return load_ctgan_generator(region_index=region_index)
    if kind == "smote":
        return SMOTEGenerator(region, real)
    if kind == "bootstrap":
        return SmoothedBootstrapGenerator(region, real)
    raise ValueError(f"unknown generator: {kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Generate a synthetic counterpart for every real well in "
            "original_data/, on each well's own depth axis."
        ),
    )
    parser.add_argument(
        "-g", "--generator",
        default="physics",
        choices=["physics", "ctgan", "smote", "bootstrap"],
        help="Which wellbench generator to sample from (default: physics).",
    )
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds; one file per (well, seed) pair. Default: 42",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the .xlsx wells (default: {DEFAULT_OUTPUT_DIR.name}/).",
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=ORIGINAL_DATA_DIR,
        help=f"Directory of real .xlsx wells (default: {ORIGINAL_DATA_DIR.name}/).",
    )
    parser.add_argument(
        "-w", "--wells",
        nargs="+",
        metavar="NAME",
        help="Only these wells, by name (e.g. PINDORI-1). Default: all found.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress the per-well cleaning report.",
    )
    args = parser.parse_args(argv)

    files = sorted(args.input_dir.glob("*.xlsx"))
    if not files:
        print(f"No .xlsx files found in {args.input_dir}/", file=sys.stderr)
        return 1

    if args.wells:
        wanted = {w.upper() for w in args.wells}
        files = [f for f in files if well_name(f).upper() in wanted]
        if not files:
            print(f"No wells matched {sorted(wanted)}", file=sys.stderr)
            return 1

    unmapped = [well_name(f) for f in files if well_name(f) not in WELL_REGIONS]
    if unmapped:
        print(
            f"No region mapping for {unmapped} — add them to WELL_REGIONS.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  wellbench - depth-matched synthetic wells")
    print("=" * 72)
    print(f"  generator : {args.generator}")
    print(f"  seeds     : {args.seeds}")
    print(f"  wells     : {len(files)} from {args.input_dir.name}/")
    print(f"  output    : {args.output_dir}")
    print()

    rows = []
    for path in files:
        name = well_name(path)
        region_index = WELL_REGIONS[name]
        region = ALL_REGIONS[region_index - 1]

        real = read_real_well(path, verbose=not args.quiet)
        depth = real["DEPTH"].to_numpy(float)
        if len(depth) < 2:
            print(f"  ! {name}: only {len(depth)} usable rows, skipped")
            continue

        gen = build_generator(args.generator, region, real, region_index)

        for seed in args.seeds:
            synth = gen.generate(seed=seed, depth=depth)
            # Mirror original_data's real_<WELL>.xlsx. Only disambiguate by seed
            # when there is more than one, so the common case stays 1:1.
            suffix = f"_seed_{seed}" if len(args.seeds) > 1 else ""
            fname = f"synth_{name}{suffix}.xlsx"
            synth.to_excel(args.output_dir / fname, index=False)

            step = float(np.median(np.diff(depth))) if len(depth) > 1 else float("nan")
            rows.append({
                "file": fname,
                "well": name,
                "region": region_index,
                "region_name": region["name"],
                "generator": args.generator,
                "seed": seed,
                "n_rows": len(synth),
                "depth_min": round(float(depth.min()), 4),
                "depth_max": round(float(depth.max()), 4),
                "depth_step_median": round(step, 4),
                "depth_unit": region["depth_unit"],
                "columns": " ".join(synth.columns),
                "source_file": path.name,
            })
            print(
                f"  [{len(rows):>2}] {fname:<36} "
                f"region {region_index}  "
                f"{len(synth):>6} rows  "
                f"{depth.min():>9.2f}-{depth.max():>9.2f} {region['depth_unit']}"
            )

    if not rows:
        print("Nothing generated.", file=sys.stderr)
        return 1

    manifest = args.output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    print()
    print(f"Done - {len(rows)} .xlsx files + manifest.csv in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
