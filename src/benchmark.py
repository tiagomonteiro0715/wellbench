from __future__ import annotations

from pathlib import Path

from .generator import SyntheticWellLogGenerator
from .regions import ALL_REGIONS, BENCHMARK_SEEDS


def generate_benchmark(
    regions: list[dict] = ALL_REGIONS,
    seeds: list[int] = BENCHMARK_SEEDS,
    output_dir: str = "benchmark",
) -> list[Path]:
    """Generate the full 15-dataset benchmark (3 seeds x 5 regions).

    Returns the list of CSV file paths written.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    paths: list[Path] = []

    for region_idx, region in enumerate(regions, start=1):
        gen = SyntheticWellLogGenerator(region)
        for seed in seeds:
            df = gen.generate(seed)
            fname = f"region_{region_idx}_seed_{seed}.csv"
            fpath = out / fname
            df.to_csv(fpath, index=False)
            paths.append(fpath)
            print(f"  [{len(paths):>2}/15] {fname}  "
                  f"({len(df)} rows, {len(df.columns)} cols)")

    return paths
