from __future__ import annotations

import argparse
import sys

from .benchmark import generate_benchmark
from .regions import ALL_REGIONS, BENCHMARK_SEEDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wellbench",
        description="Generate physics-based synthetic well-log benchmark datasets.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="benchmark",
        help="Directory to write CSV files (default: benchmark)",
    )
    parser.add_argument(
        "-r", "--regions",
        type=int,
        nargs="+",
        choices=range(1, 6),
        metavar="N",
        help="Region numbers to generate (1-5). Default: all five.",
    )
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        nargs="+",
        help=f"Random seeds to use. Default: {BENCHMARK_SEEDS}",
    )

    args = parser.parse_args(argv)

    regions = ALL_REGIONS
    if args.regions:
        regions = [ALL_REGIONS[i - 1] for i in args.regions]

    seeds = args.seeds or BENCHMARK_SEEDS
    total = len(regions) * len(seeds)

    print("=" * 60)
    print("wellbench — Synthetic Well-Log Benchmark Generator")
    print("=" * 60)
    for i, r in enumerate(regions, start=1):
        print(f"  Region {i}: {r['name']}")
    print(f"  Seeds: {seeds}")
    print(f"  Total datasets: {total}")
    print()

    paths = generate_benchmark(regions=regions, seeds=seeds, output_dir=args.output_dir)

    print()
    print(f"Done — {len(paths)} CSV files written to {args.output_dir}/")
