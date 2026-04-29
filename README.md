# wellbench

Benchmark data: https://huggingface.co/datasets/monteirot/wellbench/tree/main

**Physics-based synthetic well-log benchmark generator for pore-pressure
prediction research.** Five regions calibrated against real-world wells via
Optuna optimisation (Jensen–Shannon divergence + Wasserstein distance against
real distributions), a deterministic physics generator, an optional CTGAN
baseline, and a CLI that reproduces a 15-dataset benchmark.

Use it to:

- Generate reproducible, physically plausible well-log datasets with one line.
- Stress-test pore-pressure / petrophysics models against ground truth you
  control (you set the seed, the depth axis, and the region parameters).
- Compare physics-based and GAN-based synthesis under a shared schema.
- Run a 15-CSV "benchmark suite" out of the box for paper-ready comparisons.

---

## Install

```bash
pip install wellbench
pip install wellbench[ctgan]   # adds the optional CTGAN baseline (torch + ctgan)
pip install wellbench[docs]    # to rebuild the Sphinx docs locally
```

Python ≥ 3.12 is required. Core deps are `numpy`, `pandas`, and `scipy`.
The CTGAN extra pulls in `torch` and `ctgan` lazily — they are only imported
when you actually use the GAN generator, so the base install stays light.

---

## Quickstart

```python
from wellbench import SyntheticWellLogGenerator, REGION_1

gen = SyntheticWellLogGenerator(REGION_1)
df  = gen.generate(seed=42)
print(df.head())
#    DEPTH         GR         DT      RHOB        RT        HP        OB    DT_NCT       PPP
# 0   500.0  140.2155  138.1432   1.6418   12.7382   645.94   926.85   137.43   615.33
# 1   500.5  142.9120  136.0091   1.6712   13.4711   646.57   928.39   137.39   612.10
# ...
```

A 60-second tour of every public entry point lives in [`examples.py`](examples.py):

```bash
python examples.py                  # run every example
python examples.py basic ctgan      # pick specific examples
```

---

## Command-line interface

The packaged `wellbench` console script reproduces the canonical 15-dataset
benchmark — 5 regions × 3 seeds — and writes one CSV per (region, seed) pair:

```bash
wellbench                           # all 15 datasets -> ./benchmark/
wellbench -r 2 -s 99 200            # region 2, seeds 99 and 200 (2 CSVs)
wellbench -r 1 2 3                  # only the pore-pressure regions
wellbench -o my_data                # custom output directory
wellbench --help                    # full reference
```

Output filenames are deterministic: `region_<N>_seed_<S>.csv`. With the
default seeds (`[42, 123, 7777]`) you get:

```
benchmark/
├── region_1_seed_42.csv
├── region_1_seed_123.csv
├── region_1_seed_7777.csv
├── region_2_seed_42.csv
...
└── region_5_seed_7777.csv      (15 files, 9 columns each for PP regions)
```

---

## Writing CSVs from Python

### Single well

```python
from wellbench import SyntheticWellLogGenerator, REGION_1

gen = SyntheticWellLogGenerator(REGION_1)
df  = gen.generate(seed=42)
df.to_csv("missa_keswal_seed42.csv", index=False)
```

### One CSV per seed, one folder per region

```python
from pathlib import Path
from wellbench import ALL_REGIONS, SyntheticWellLogGenerator

out = Path("synthetic_wells"); out.mkdir(exist_ok=True)
seeds = [1, 2, 3, 4, 5]

for i, region in enumerate(ALL_REGIONS, start=1):
    region_dir = out / f"region_{i}"
    region_dir.mkdir(exist_ok=True)
    gen = SyntheticWellLogGenerator(region)
    for seed in seeds:
        gen.generate(seed=seed).to_csv(
            region_dir / f"seed_{seed}.csv", index=False
        )
```

### Reproduce the full 15-dataset benchmark programmatically

```python
from wellbench import generate_benchmark

paths = generate_benchmark(output_dir="benchmark")
# returns the list of 15 written CSV paths
```

### Match a real well's depth axis

When you want one synthetic row per real measurement (e.g. for side-by-side
log plots), pass an explicit `depth` array:

```python
import pandas as pd
from wellbench import SyntheticWellLogGenerator, REGION_1

real  = pd.read_csv("real_well.csv", usecols=["DEPTH"])
synth = SyntheticWellLogGenerator(REGION_1).generate(
    seed=42, depth=real["DEPTH"].to_numpy(),
)
synth.to_csv("synthetic_aligned.csv", index=False)
```

### Custom depth ranges or sampling rates

```python
import numpy as np
from wellbench import SyntheticWellLogGenerator, REGION_4

depth = np.linspace(120, 700, 5_000)         # 5 000 samples in 120-700 m
df = SyntheticWellLogGenerator(REGION_4).generate(seed=7, depth=depth)
```

---

## Cleaning real or synthetic data

`clean_well_data` applies the same physical-bounds + outlier rules to any
DataFrame that has a `DEPTH` column plus log columns:

```python
import pandas as pd
from wellbench import clean_well_data

raw     = pd.read_csv("real_well.csv")
cleaned = clean_well_data(
    raw,
    outlier_std=5,        # drop values further than 5σ from the mean
    label="real_A",       # tag for the printed summary
    verbose=True,
)
cleaned.to_csv("real_well_clean.csv", index=False)
```

It will:

1. Drop the `SPHI` column if present.
2. Replace sentinel values (`-999`, `-999.25`) with `NaN`.
3. Replace out-of-physical-range values with `NaN`.
4. Replace > `outlier_std`-σ outliers with `NaN`.
5. Drop rows where every log column is `NaN`.

---

## CTGAN baseline (optional)

Five pre-trained CTGAN models — one per region — ship inside the wheel and
are loaded lazily. Install the extra:

```bash
pip install wellbench[ctgan]
```

…then sample with the same `.generate(seed, depth=…)` interface as the
physics generator:

```python
from wellbench import load_ctgan_generator

gen = load_ctgan_generator(region_index=1)        # ctgan_r1.pkl
df  = gen.generate(seed=42)
df.to_csv("ctgan_region1_seed42.csv", index=False)
```

You can also point it at your own checkpoint:

```python
from wellbench import CTGANSyntheticWellLogGenerator, REGION_1

gen = CTGANSyntheticWellLogGenerator(
    params=REGION_1,
    model_path="my_models/ctgan_custom.pkl",
)
df = gen.generate(seed=0, depth=my_depth_array)
```

CTGAN samples are i.i.d. tabular rows; `wellbench` orders them along the
depth axis you supply, applies `_CTGAN_COLUMN_RENAMES` so the output schema
matches the physics generator, and clips to `PHYSICAL_BOUNDS`.

---

## Defining your own region

A region is just a `dict` of physical parameters. The simplest recipe is to
copy a built-in and tweak:

```python
from wellbench import REGION_1, SyntheticWellLogGenerator

my_region = {
    **REGION_1,
    "name":         "My field",
    "depth_range": (1_000, 3_000),
    "depth_step":   0.25,
    "depth_unit":   "ft",
    "phi0":         0.42,           # surface porosity
    "compaction_c": 0.0006,         # Athy's exponential decay
    # …override anything else
}
df = SyntheticWellLogGenerator(my_region).generate(seed=42)
```

Required keys cover porosity (`phi0`, `compaction_c`, `phi_layer_amp`, …),
Wyllie/Archie/RHOB/GR transforms, and — if `has_pore_pressure=True` — the
hydrostatic / overburden / Eaton parameters. See
[`src/regions.py`](src/regions.py) for five fully worked examples.

---

## Plotting recipe

```python
import matplotlib.pyplot as plt
from wellbench import SyntheticWellLogGenerator, REGION_1

df = SyntheticWellLogGenerator(REGION_1).generate(seed=42)

fig, axes = plt.subplots(1, 4, figsize=(12, 8), sharey=True)
for ax, col in zip(axes, ["GR", "DT", "RHOB", "RT"]):
    ax.plot(df[col], df["DEPTH"])
    ax.set_xlabel(col); ax.invert_yaxis()
axes[3].set_xscale("log")            # RT is plotted on a log axis by convention
axes[0].set_ylabel("Depth")
fig.tight_layout()
fig.savefig("logs.png", dpi=150)
```

---

## Regions

| #  | Region                                  | Location                                | Pore pressure |
|----|-----------------------------------------|-----------------------------------------|---------------|
| 1  | Missa Keswal (Zone 1)                   | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 2  | PINDORI-1 (Zone 2)                      | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 3  | JOYAMAIR-4 / MINWAL-2 (Zone 3)          | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 4  | IODP Expedition 323, Hole U1343E        | Bering Sea                              | no            |
| 5  | Volve oil field                         | North Sea (Norway/UK)                   | no            |

Convenience constants: `ALL_REGIONS` (list of all five in order) and
`BENCHMARK_SEEDS` (`[42, 123, 7777]`).

---

## Output schema

| Column     | Always present | PP regions only | Units / range                           |
|------------|:--------------:|:---------------:|-----------------------------------------|
| `DEPTH`    | ✅             |                 | depends on region (`ft` or `m`)         |
| `GR`       | ✅             |                 | API, `[0, 200]`                          |
| `DT`       | ✅             |                 | µs/ft, `[30, 180]`                      |
| `RHOB`     | ✅             |                 | g/cc, `[1.2, 2.9]`                      |
| `RT`       | ✅             |                 | Ω·m, `[0.01, 10 000]`                   |
| `HP`       |                | ✅              | psi, hydrostatic pressure               |
| `OB`       |                | ✅              | psi, overburden                         |
| `DT_NCT`   |                | ✅              | µs/ft, normal compaction trend          |
| `PPP`      |                | ✅              | psi, pore pressure (Eaton)              |

All outputs are clipped to `wellbench.PHYSICAL_BOUNDS` so consumers can rely
on a fixed physical range. Inspect the bounds at runtime:

```python
from wellbench import PHYSICAL_BOUNDS
for col, (lo, hi) in PHYSICAL_BOUNDS.items():
    print(f"{col:<7} [{lo}, {hi}]")
```

---

## Physics models

- **Porosity** — exponential compaction (Athy's law) + layered sinusoids + Gaussian noise.
- **Sonic (DT)** — Wyllie time-average equation.
- **Density (RHOB)** — bulk density mixing law with a small lithology trend.
- **Resistivity (RT)** — Archie's equation.
- **Gamma ray (GR)** — shale-volume linear mixing.
- **Pore pressure (PPP)** — Eaton's method on a normal compaction trend (regions 1-3 only).

---

## Reproducibility

- Every generator method takes a `seed`. The same `(region, seed, depth)`
  triple is guaranteed to produce identical output across runs and platforms.
- The CTGAN sampler also seeds NumPy and PyTorch (CPU and CUDA) before each
  `.generate()` call.
- Region parameters are frozen dictionaries; if you want to record exactly
  which calibration produced a CSV, just dump the region dict alongside it.

---

## Documentation

Full Sphinx docs (autodoc + Napoleon) live under [`docs/`](docs/). Build them
locally with:

```bash
pip install wellbench[docs]
cd docs
make html             # POSIX
.\make.bat html       # Windows
# open _build/html/index.html
```

---

## Public API at a glance

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

    # Benchmark
    generate_benchmark,

    # Region presets
    ALL_REGIONS, BENCHMARK_SEEDS,
    REGION_1, REGION_2, REGION_3, REGION_4, REGION_5,
)
```

---

## Citing

If `wellbench` helps your research, please cite the underlying
physics-based synthetic-data study (see [`docs/`](docs/) for the current
reference) and link back to this package.

## License

MIT — see [`LICENSE`](LICENSE).
