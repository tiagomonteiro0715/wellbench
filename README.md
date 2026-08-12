<h1 align="center">wellbench</h1>

<p align="center">
  <em>Physics-calibrated synthetic well-log benchmarks for pore-pressure prediction research.</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.12-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-orange">
</p>

`wellbench` is a small Python package and a research codebase for generating reproducible, physically plausible synthetic well-log data. It ships **four interchangeable generators** — a deterministic physics forward-model, a CTGAN tabular baseline, and SMOTE and smoothed-bootstrap resampling baselines — all behind the same `gen.generate(seed, depth=...)` interface, plus **five regional calibrations** tuned with Optuna against real wells (Eastern Potwar Basin, Bering Sea, Volve / North Sea), a **CLI** that emits a canonical 5-regions × 3-seeds = 15-CSV benchmark, and a separate **research subdirectory** that reproduces the calibration, training, and TRTR/TSTR evaluation pipelines locally (no Colab).

Benchmark: https://huggingface.co/datasets/monteirot/wellbench


---

## Contents

- [What's in the box](#whats-in-the-box)
- [Install](#install)
- [Quickstart](#quickstart)
- [Command-line interface](#command-line-interface)
- [Depth-matched wells (`main.py`)](#depth-matched-wells-mainpy)
- [Regions](#regions)
- [Output schema](#output-schema)
- [Physics models](#physics-models)
- [CTGAN baseline](#ctgan-baseline)
- [Resampling baselines](#resampling-baselines)
- [Cleaning real or synthetic data](#cleaning-real-or-synthetic-data)
- [Defining a custom region](#defining-a-custom-region)
- [Research code (`research_files/`)](#research-code-research_files)
- [Calibration artifacts](#calibration-artifacts)
- [Repository layout](#repository-layout)
- [Public API](#public-api)
- [Documentation](#documentation)
- [Reproducibility](#reproducibility)
- [Intended use & limitations](#intended-use--limitations)
- [Citing](#citing)
- [License](#license)

---

## What's in the box

| Component                       | Where                              | What it does                                                                       |
|---------------------------------|------------------------------------|------------------------------------------------------------------------------------|
| Physics generator               | `src/wellbench/generator.py`        | Deterministic forward model (Athy + Wyllie + Archie + Eaton).                      |
| CTGAN generator                 | `src/wellbench/ctgan_generator.py`  | Optional GAN baseline with the same `.generate()` API.                             |
| Resampling generators           | `src/wellbench/resample_generator.py` | SMOTE and smoothed-bootstrap baselines with the same `.generate()` API.          |
| Five region calibrations        | `src/wellbench/regions.py`          | `REGION_1` … `REGION_5` parameter dicts, plus `ALL_REGIONS` and `BENCHMARK_SEEDS`. |
| Benchmark builder               | `src/wellbench/benchmark.py`        | `generate_benchmark(...)` → 15 CSVs (5 regions × 3 seeds).                         |
| CLI                             | `src/wellbench/cli.py`              | `wellbench` console script wrapping the benchmark builder.                         |
| CTGAN checkpoints               | `CTGAN_files/`, `src/wellbench/ctgan_models/`| One pickled model + Optuna-best hyperparameters per region.               |
| CTGAN evaluation reports        | `CTGAN_files/metrics/r{1..5}/`      | Per-region fidelity, correlation, spatial, SDMetrics, DCR, and Optuna trial logs.   |
| Physics calibration outputs     | `physical_model_files/`            | Optuna-tuned physics hyperparameters and softmax weights per region.               |
| Real wells                      | `original_data/`                   | Nine cleaned real wells (regions 1–3) as `.xlsx`.                                  |
| Depth-matched runner            | `main.py`                          | `synthetic/synth_<WELL>.xlsx` — one per real well, on that well's own depth axis.    |
| Research code (Colab-converted) | `research_files/`                  | Original notebooks + local-runnable scripts for generation and ML evaluation.      |
| Sphinx docs                     | `docs/`                            | Quickstart, regions reference, and autodoc API.                                    |
| Examples                        | `examples.py`                      | One runnable function per public entry point.                                      |

---

## Install

The package targets Python ≥ 3.12. Core deps are `numpy`, `pandas`, and `scipy`; the CTGAN baseline lazily pulls in `torch` and `ctgan` so the base install stays light.

From a clone of this repo:

```bash
pip install -e .
pip install -e ".[ctgan]"   # adds the GAN baseline (torch + ctgan)
pip install -e ".[data]"    # adds openpyxl, needed by main.py to read original_data/
pip install -e ".[docs]"    # to rebuild the Sphinx docs locally
```

The research subdirectory has its own pinned environment managed with [uv](https://docs.astral.sh/uv/) — see [Research code](#research-code-research_files) below.

---

## Quickstart

```python
from wellbench import SyntheticWellLogGenerator, REGION_1

gen = SyntheticWellLogGenerator(REGION_1)
df  = gen.generate(seed=42)
print(df.head())
#    DEPTH       GR        DT     RHOB       RT      HP      OB   DT_NCT     PPP
# 0  500.0  140.21   138.14   1.6418  12.738  645.94  926.85  137.43  615.33
# ...
```

Every public entry point has a runnable demo in [`examples.py`](examples.py):

```bash
python examples.py                  # run every example
python examples.py basic ctgan      # pick specific examples
```

---

## Command-line interface

The packaged `wellbench` console script reproduces the canonical 15-dataset benchmark — 5 regions × 3 seeds — and writes one CSV per `(region, seed)` pair:

```bash
wellbench                           # all 15 datasets -> ./benchmark/
wellbench -r 2 -s 99 200            # region 2, seeds 99 and 200 (2 CSVs)
wellbench -r 1 2 3                  # only the pore-pressure regions
wellbench -o my_data                # custom output directory
wellbench --help                    # full reference
```

Output filenames are deterministic: `region_<N>_seed_<S>.csv`. With the default seeds `[42, 123, 7777]` you get a tidy benchmark folder:

```
benchmark/
├── region_1_seed_42.csv
├── region_1_seed_123.csv
├── region_1_seed_7777.csv
├── region_2_seed_42.csv
...
└── region_5_seed_7777.csv
```

Equivalent Python:

```python
from wellbench import generate_benchmark
paths = generate_benchmark(output_dir="benchmark")   # list of 15 CSV paths
```

---

## Depth-matched wells (`main.py`)

The `wellbench` CLI emits each region over its *nominal* `depth_range` — a
basin-wide envelope (500–4500 ft for regions 1–3), not any particular borehole.
[`main.py`](main.py) does the other thing: it reads every real well in
[`original_data/`](original_data), works out which region calibration it belongs to,
and generates a synthetic counterpart **on that well's own depth axis** — same
interval, same sampling rate, same row count. The result lines up row-for-row with
the real well, so it can be diffed or plotted against it directly.

```bash
pip install -e ".[data]"                          # openpyxl, for .xlsx read/write

python main.py                                    # physics generator, seed 42
python main.py --generator ctgan                  # CTGAN baseline (needs [ctgan] too)
python main.py --generator smote --seeds 42 123   # resampling baseline, 2 seeds
python main.py --wells PINDORI-1 MINWAL-2         # just these wells
python main.py --help
```

All four generators are available via `--generator {physics,ctgan,smote,bootstrap}`.
The two resampling baselines draw from the real well that is being matched, so they
need no extra input.

Output lands in `synthetic/` as `.xlsx`, mirroring the `real_<WELL>.xlsx` naming of
`original_data/` so real and synthetic sit side by side:

```
synthetic/
├── manifest.csv                  # provenance for every file below
├── synth_JOYAMAIR-4.xlsx         # <- original_data/real_JOYAMAIR-4.xlsx
├── synth_MINWAL-2.xlsx
├── synth_MINWAL-X-1.xlsx
├── synth_MISSA-KESWAL-01.xlsx
├── synth_MISSA-KESWAL-02.xlsx
├── synth_MISSA-KESWAL-03.xlsx
├── synth_PINDORI-1.xlsx
├── synth_PINDORI-2.xlsx
└── synth_PINDORI-3.xlsx
```

The region number isn't in the filename — it's in `manifest.csv`, which records per
file the source well and region, the generator and seed, the row count, and the
realised depth min / max / median step, so a published folder carries its own
provenance. Passing more than one `--seeds` value appends the seed
(`synth_JOYAMAIR-4_seed_123.xlsx`) so the files don't collide.

Two details worth knowing:

- **Well → region mapping** lives in `WELL_REGIONS` in `main.py` and mirrors the
  groupings the calibrations were fitted on (Missa Keswal → region 1, Pindori →
  region 2, Joyamair / Minwal → region 3). Add an entry there for a new well.
- **The depth axis is the *cleaned* one.** Each real well goes through
  `clean_well_data` first, which drops rows where every log is `NaN`, so the depth
  span reflects the interval actually logged. For `MINWAL-2` that trims the bottom
  ~234 ft (9878 → 8360 rows); the other eight wells are unaffected.

`original_data/` holds the nine Potwar Basin wells backing regions 1–3. Regions 4
(Bering Sea) and 5 (Volve) are downloaded on demand by the research scripts and have
no local file, so `main.py` does not emit wells for them.

---

## Regions

| #  | Region                                         | Location                                | Depth                  | Pore pressure |
|----|------------------------------------------------|-----------------------------------------|------------------------|---------------|
| 1  | Missa Keswal (Zone 1)                          | Eastern Potwar Basin, Punjab, Pakistan  | 500–4500 ft, 0.5 ft    | yes           |
| 2  | PINDORI-1 (Zone 2)                             | Eastern Potwar Basin, Punjab, Pakistan  | 500–4500 ft, 0.5 ft    | yes           |
| 3  | JOYAMAIR-4 / MINWAL-2 (Zone 3)                 | Eastern Potwar Basin, Punjab, Pakistan  | 500–4500 ft, 0.5 ft    | yes           |
| 4  | IODP Expedition 323, Hole U1343E               | Bering Sea                              | 100–750 m, 0.1 m       | no            |
| 5  | Volve oil field                                | North Sea (Norway/UK)                   | 2800–4200 m, 0.15 m    | no            |

Convenience constants: `ALL_REGIONS` (the five in order) and `BENCHMARK_SEEDS` (`[42, 123, 7777]`).

---

## Output schema

| Column     | Always present | PP regions only | Units / range                  |
|------------|:--------------:|:---------------:|--------------------------------|
| `DEPTH`    | ✅             |                 | depends on region (ft or m)    |
| `GR`       | ✅             |                 | API, `[0, 200]`                |
| `DT`       | ✅             |                 | µs/ft, `[30, 180]`             |
| `RHOB`     | ✅             |                 | g/cc, `[1.2, 2.9]`             |
| `RT`       | ✅             |                 | Ω·m, `[0.01, 10 000]`          |
| `HP`       |                | ✅              | psi, hydrostatic pressure      |
| `OB`       |                | ✅              | psi, overburden                |
| `DT_NCT`   |                | ✅              | µs/ft, normal compaction trend |
| `PPP`      |                | ✅              | psi, pore pressure (Eaton)     |

Physics and resampling outputs are clipped to `wellbench.PHYSICAL_BOUNDS`, which downstream code can rely on (the CTGAN generator uses the wider `CTGAN_BOUNDS` — see [CTGAN baseline](#ctgan-baseline)):

```python
from wellbench import PHYSICAL_BOUNDS
for col, (lo, hi) in PHYSICAL_BOUNDS.items():
    print(f"{col:<7} [{lo}, {hi}]")
```

---

## Physics models

| Curve            | Closed-form                                                       |
|------------------|-------------------------------------------------------------------|
| Porosity (φ)     | Athy exponential compaction + layered sinusoids + Gaussian noise. |
| Sonic (`DT`)     | Wyllie time-average equation.                                     |
| Density (`RHOB`) | Bulk-density mixing law with a small lithology trend.             |
| Resistivity (`RT`) | Archie's equation.                                              |
| Gamma ray (`GR`) | Shale-volume linear mixing.                                       |
| Pore pressure (`PPP`) | Eaton's method on a normal compaction trend (regions 1–3).   |

The full source is in [`src/wellbench/generator.py`](src/wellbench/generator.py).

---

## CTGAN baseline

Five pre-trained CTGAN models — one per region — ship under [`src/wellbench/ctgan_models/`](src/wellbench/ctgan_models) (also mirrored in [`CTGAN_files/`](CTGAN_files)) and load lazily so the base install does not need PyTorch. Sample with the same `.generate()` interface as the physics generator:

```python
from wellbench import load_ctgan_generator

gen = load_ctgan_generator(region_index=1)        # ctgan_r1.pkl
df  = gen.generate(seed=42)
df.to_csv("ctgan_region1_seed42.csv", index=False)
```

Or point it at your own checkpoint:

```python
from wellbench import CTGANSyntheticWellLogGenerator, REGION_1

gen = CTGANSyntheticWellLogGenerator(
    params=REGION_1,
    model_path="my_models/ctgan_custom.pkl",
)
df = gen.generate(seed=0, depth=my_depth_array)
```

CTGAN samples are i.i.d. tabular rows; `wellbench` orders them along the depth axis you supply, renames columns to match the physics generator's schema, and clips to `CTGAN_BOUNDS`.

### Why CTGAN output uses its own bounds

`CTGAN_BOUNDS` is deliberately wider than the physics generator's `PHYSICAL_BOUNDS`,
and matches the envelope the checkpoints were trained and scored under (see
`research_files/scripts/final_optimal_gan_based_generation.py`):

| Column | `PHYSICAL_BOUNDS` | `CTGAN_BOUNDS`   |
|--------|-------------------|------------------|
| `GR`   | `[0, 200]`        | `[0, 350]`       |
| `DT`   | `[30, 180]`       | `[30, 200]`      |
| `RHOB` | `[1.2, 2.9]`      | `[1.0, 3.5]`     |
| `HP`   | `[500, 15 000]`   | `[0, 20 000]`    |
| `OB`   | `[2 000, 40 000]` | `[0, 50 000]`    |
| `PPP`  | `[0, 30 000]`     | `[-2 000, 20 000]` |

Clipping GAN samples with the tighter physics bounds visibly distorts the learned
marginals. Region 4 (Bering Sea) is the clearest case: real `DT` there runs to
~201 µs/ft, so a 180 µs/ft ceiling collapsed **over half of its samples onto the
bound** as a spike. Sampling against the training envelope keeps the distribution
continuous. If you need strictly physics-bounded rows, run the result through
`clean_well_data`, which applies `PHYSICAL_BOUNDS` and `NaN`s out what falls outside:

```python
from wellbench import clean_well_data, load_ctgan_generator

df = clean_well_data(load_ctgan_generator(region_index=4).generate(seed=42))
```

---

## Resampling baselines

Two cheap non-parametric baselines round out the generator set. Both are
*resampling* methods, so unlike the physics and CTGAN generators they take a
source DataFrame to draw from — real cleaned wells for the research protocol, or
any frame with a `DEPTH` column plus log columns:

```python
from wellbench import REGION_1, SMOTEGenerator, SmoothedBootstrapGenerator

real = clean_well_data(pd.read_csv("real_well.csv"))

SMOTEGenerator(REGION_1, real).generate(seed=42)              # k-NN interpolation
SmoothedBootstrapGenerator(REGION_1, real).generate(seed=42)  # resample + jitter
```

- **`SMOTEGenerator`** picks a random source row, picks one of its
  `smote_k_neighbors` nearest neighbours, and interpolates between them by a
  uniform random factor. Neighbours are found on raw (unscaled) log values, so
  the widest-range columns dominate the distance.
- **`SmoothedBootstrapGenerator`** resamples rows with replacement and adds
  Gaussian jitter scaled by `bootstrap_bandwidth × std(column)`. The jitter is
  shared across logs by `bootstrap_noise_correlation`: any two columns get a
  jitter correlation of exactly `w`, while each column stays at unit variance.

All three per-region hyperparameters (`smote_k_neighbors`,
`bootstrap_bandwidth`, `bootstrap_noise_correlation`) are Optuna-calibrated and
live in the region dicts alongside the physics parameters.

Both baselines match real *marginals* very closely but weaken vertical
structure — resampled rows are drawn i.i.d., so autocorrelation length collapses
relative to the physics generator. Use them as a fidelity floor, not as a
substitute for depth-correlated logs.

## Cleaning real or synthetic data

`clean_well_data` applies the same physical-bounds and outlier rules to any DataFrame with a `DEPTH` column plus log columns:

```python
import pandas as pd
from wellbench import clean_well_data

raw     = pd.read_csv("real_well.csv")
cleaned = clean_well_data(raw, outlier_std=5, label="real_A", verbose=True)
cleaned.to_csv("real_well_clean.csv", index=False)
```

It will (1) drop any `SPHI` column, (2) replace sentinel values (`-999`, `-999.25`) with `NaN`, (3) `NaN` out values outside physical bounds, (4) `NaN` out values further than `outlier_std` σ from the mean, and (5) drop rows where every log column is `NaN`.

---

## Defining a custom region

A region is just a `dict` of physical parameters. Copy a built-in and override what you need:

```python
from wellbench import REGION_1, SyntheticWellLogGenerator

my_region = {
    **REGION_1,
    "name":         "My field",
    "depth_range": (1_000, 3_000),
    "depth_step":   0.25,
    "depth_unit":   "ft",
    "phi0":         0.42,           # surface porosity
    "compaction_c": 0.0006,         # Athy exponential decay
}
df = SyntheticWellLogGenerator(my_region).generate(seed=42)
```

Required keys cover porosity (`phi0`, `compaction_c`, `phi_layer_amp`, …), Wyllie / Archie / RHOB / GR transforms, and — when `has_pore_pressure=True` — the hydrostatic, overburden, NCT, and Eaton parameters. See [`src/wellbench/regions.py`](src/wellbench/regions.py) for five fully-worked examples.

---

## Research code (`research_files/`)

The [`research_files/`](research_files) subdirectory is the research codebase that produced the calibration artifacts and the comparative ML study. It contains:

- **`notebooks/`** — the original Colab notebooks (kept verbatim):
  - `final_optimal_Physics_based_generation.ipynb` — Optuna search over physics parameters.
  - `final_optimal_GAN_based_generation.ipynb` — CTGAN training + Optuna over generator hyperparameters.
  - `final_POFM_TRTR_ML_models_study.ipynb` / `final_POFM_TSTR_classic_ML_models_study.ipynb` — train-real / test-real and train-synthetic / test-real benchmarks for the **p**hysics-**o**ptimised **f**orward **m**odel.
  - `final_CTGAN_TRTR_ML_models_study.ipynb` / `final_CTGAN_TSTR_classic_ML_models_study.ipynb` — same protocols against the CTGAN baseline.
- **`scripts/`** — the same code as `.py` files, patched to run on a normal Linux / macOS / Windows machine (no Google Drive mounts, no `/content/...` paths, cross-platform memory tracking via `psutil` instead of Linux-only `resource`).
- **`pyproject.toml` + `uv.lock`** — a fully-pinned, 210-package environment (CPU PyTorch by default).

Run it locally with [uv](https://docs.astral.sh/uv/):

```bash
cd research_files
uv sync                                                     # one-shot env setup
uv run python scripts/final_optimal_physics_based_generation.py
uv run python scripts/final_optimal_gan_based_generation.py
uv run python scripts/final_pofm_trtr_ml_models_study.py
uv run python scripts/final_ctgan_tstr_classic_ml_models_study.py
# ...etc
uv run jupyter lab notebooks/                               # run the original notebooks
```

The first run downloads source well-log files via `gdown` into `research_files/data/` and caches them. Long-running Optuna studies persist to `research_files/checkpoints/<benchmark>/` — delete that folder for a clean re-run. See [`research_files/README.md`](research_files/README.md) for the full Colab-vs-local diff.

---

## Calibration artifacts

The Optuna search results that the package ships with are stored at the repository root for easy inspection (and re-bundled into the wheel where appropriate):

- [`physical_model_files/`](physical_model_files) — five `hp_optimised_r{N}.json` files (best physics hyperparameters per region) plus five `softmax_weights_r{N}.json` files (per-objective weighting used to fold JS-divergence and Wasserstein-1 into a single Optuna objective).
- [`CTGAN_files/`](CTGAN_files) — five `ctgan_r{N}.pkl` model checkpoints and five `best_params_r{N}.json` files with the corresponding training hyperparameters.
- [`CTGAN_files/metrics/r{N}/`](CTGAN_files/metrics) — the full evaluation report behind each checkpoint:

  | File                            | Contents                                                                        |
  |---------------------------------|---------------------------------------------------------------------------------|
  | `summary_r{N}.md`               | Human-readable run card: wall time, best JSD, trial count, and all tables below. |
  | `table3_r{N}.csv`               | Per-column JSD / Wasserstein means, σ and 95% CI across seeds *and* wells.       |
  | `table3_paper_r{N}.csv`         | The same numbers pre-formatted as `mean ± σ` for publication.                    |
  | `fidelity_r{N}.csv`             | Per-column JSD, WD, Kolmogorov–Smirnov and Anderson–Darling statistics.          |
  | `baseline_real_vs_real_r{N}.csv`| Real-vs-real floor for the same metrics — the irreducible between-well spread.    |
  | `correlation_r{N}.csv`          | Real vs synthetic Pearson correlation for every column pair, plus abs error.      |
  | `spatial_r{N}.csv`              | Autocorrelation length and variogram sill, real vs synthetic.                     |
  | `multivariate_r{N}.json`        | Sliced-Wasserstein, MMD (RBF, with permutation *p*), energy distance, Frobenius.  |
  | `sdmetrics_r{N}.json`           | SDMetrics quality / diagnostic / DCR scores, with `_shapes_` and `_trends_` CSVs. |
  | `dcr_breakdown_r{N}.json`       | Distance-to-closest-record vs a random-data baseline (privacy proxy).            |
  | `trials_r{N}.csv`               | Every Optuna trial: params, objective value, duration, state.                     |
  | `best_params_per_fold_r{N}.json`| Winning hyperparameters per leave-one-well-out fold.                             |
  | `resource_usage_r{N}.csv`       | Wall clock, host RSS and GPU peak per phase (Optuna / bench / extended eval).     |

Calibration uses **Jensen–Shannon divergence** and **Wasserstein-1** on each log column's marginal (synthetic vs. real). Optuna minimises a softmax-weighted sum of the two across `GR`, `DT`, `RHOB`, `log RT`, and (for pore-pressure regions) `PPP`.

### Reading the CTGAN numbers

The two fidelity tables answer different questions and disagree by design:

- `fidelity_r{N}.csv` scores **one** synthetic draw against the **pooled** real data for the region (JSD ≈ 0.005–0.013 for region 1 — near-perfect marginals).
- `table3_r{N}.csv` is the leave-one-well-out protocol: train on the other wells, score against the held-out one, over 4 seeds. Region 1 JSD lands at 0.17–0.44 there.

The gap is generalisation to an unseen well, not a regression. `baseline_real_vs_real_r{N}.csv` is the reference for it: one real well scored against another gives JSD 0.17–0.46 for the same columns, so the held-out CTGAN numbers sit at roughly the between-well spread of the real data. Also note `spatial_r{N}.csv` — synthetic autocorrelation length is 1 sample against ~50 for real, the expected consequence of i.i.d. row sampling.

---

## Repository layout

```
wellbench/
├── src/
│   └── wellbench/
│       ├── __init__.py
│       ├── benchmark.py          # generate_benchmark
│       ├── cli.py                # `wellbench` console script
│       ├── ctgan_generator.py    # CTGANSyntheticWellLogGenerator, load_ctgan_generator
│       ├── ctgan_models/         # bundled CTGAN checkpoints (ctgan_r1.pkl, ...)
│       ├── generator.py          # SyntheticWellLogGenerator, PHYSICAL_BOUNDS, clean_well_data
│       ├── regions.py            # REGION_1 … REGION_5, ALL_REGIONS, BENCHMARK_SEEDS
│       └── resample_generator.py # SMOTEGenerator, SmoothedBootstrapGenerator
├── CTGAN_files/                  # CTGAN checkpoints + best Optuna params
│   └── metrics/r{1..5}/          # per-region evaluation reports
├── physical_model_files/         # physics best params + softmax weights
├── original_data/                # nine real wells (regions 1-3), .xlsx
├── docs/                         # Sphinx + autodoc + Napoleon
├── research_files/               # Colab-converted research codebase (uv-managed)
├── main.py                       # depth-matched wells -> synthetic/synth_<WELL>.xlsx
├── examples.py                   # runnable tour of the public API
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

## Public API

```python
from wellbench import (
    # Physics generator
    SyntheticWellLogGenerator,
    PHYSICAL_BOUNDS,
    SENTINEL_VALUES,
    clean_well_data,

    # CTGAN generator (needs the [ctgan] extra at runtime)
    CTGANSyntheticWellLogGenerator,
    load_ctgan_generator,
    CTGAN_BOUNDS,

    # Resampling generators (need a source DataFrame)
    SMOTEGenerator,
    SmoothedBootstrapGenerator,

    # Benchmark
    generate_benchmark,

    # Region presets
    ALL_REGIONS, BENCHMARK_SEEDS,
    REGION_1, REGION_2, REGION_3, REGION_4, REGION_5,
)
```

All four generators expose the same surface:

```python
gen.generate(seed: int, depth: np.ndarray | None = None) -> pandas.DataFrame
```

---

## Documentation

Full Sphinx docs live under [`docs/`](docs). Build locally with:

```bash
pip install -e ".[docs]"
cd docs
make html             # POSIX
.\make.bat html       # Windows
# open _build/html/index.html
```

The docs cover the quickstart, every region's parameters, and an autodoc API reference.

---

## Reproducibility

- **Determinism.** Every generator method takes a `seed`. The same `(region, seed, depth)` triple produces identical output across runs and platforms. The CTGAN sampler also seeds NumPy and PyTorch (CPU and CUDA) before each `.generate()` call.
- **Frozen region parameters.** Region parameters are plain dictionaries — dump the dict alongside any CSV you publish to record exactly which calibration produced it.
- **Pinned research env.** `research_files/uv.lock` pins the entire dependency graph (210 packages, CPU PyTorch by default) so `uv sync` is bit-for-bit reproducible.
- **Hardware.** The physics generator and CLI run comfortably on a single CPU core. CTGAN training (in `research_files/`) was done on a single NVIDIA GPU; CTGAN *inference* uses only the bundled checkpoints and stays on CPU.

---

## Intended use & limitations

**Intended use.** Methods research — pore-pressure prediction model development, robustness evaluation, data augmentation for small real-well training sets, teaching, and software testing.

**Out of scope.**
- *Operational drilling decisions.* Outputs capture distributional properties of real basins, not actual pressures at any surveyed location. Do not use `wellbench` to plan a real well.
- *Privacy proxy.* The physics generator is parametric and does not memorise individual real samples, but the bundled CTGAN checkpoints were trained on cleaned real wells and have not been audited for membership inference. Treat them as research artefacts.

**Known limits.**
- Calibration covers five basins only — extrapolation outside those settings is the user's responsibility.
- The CTGAN baseline samples i.i.d. rows; depth ordering is imposed post-hoc, so vertical correlation is weaker than in real logs and weaker than in the physics generator.
- Pore pressure uses Eaton's method on a normal compaction trend; under-compaction is captured, but other overpressure mechanisms (kerogen maturation, tectonic loading) are not.
- Region calibrations optimise marginals of log columns, not joint distributions or cross-column residuals.

---

## Citing

If `wellbench` helps your research, please cite the software release:

```bibtex
@software{wellbench,
  author  = {Monteiro, Tiago and contributors},
  title   = {wellbench: Physics-calibrated synthetic well-log benchmarks},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/tiagomonteiro0715/wellbench}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
