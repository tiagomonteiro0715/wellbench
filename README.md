<!--
  README.md for wellbench
  Optimised to render correctly on both GitHub and PyPI.
  Image and link URLs are absolute (raw.githubusercontent.com / huggingface.co)
  so the long_description renders identically on https://pypi.org/project/wellbench/.
-->

<p align="center">
  <!-- Replace with the path to your logo if/when one exists; remove the block otherwise. -->
  <!-- <img src="https://raw.githubusercontent.com/monteirot/wellbench/main/docs/_static/logo.png" width="220" alt="wellbench"> -->
</p>

<h1 align="center">wellbench</h1>

<p align="center">
  <em>Physics-based synthetic well-log benchmark generator for pore-pressure prediction research.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/wellbench/"><img alt="PyPI" src="https://img.shields.io/pypi/v/wellbench.svg"></a>
  <a href="https://pypi.org/project/wellbench/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/wellbench.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://huggingface.co/datasets/monteirot/wellbench"><img alt="HF Dataset" src="https://img.shields.io/badge/🤗%20dataset-monteirot%2Fwellbench-yellow"></a>
  <a href="https://huggingface.co/api/datasets/monteirot/wellbench/croissant"><img alt="Croissant" src="https://img.shields.io/badge/metadata-Croissant-orange"></a>
  <!-- Add when available:
  <a href="https://github.com/monteirot/wellbench/actions/workflows/test.yml"><img alt="CI" src="https://github.com/monteirot/wellbench/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://wellbench.readthedocs.io"><img alt="Docs" src="https://readthedocs.org/projects/wellbench/badge/?version=latest"></a>
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg"></a>
  <a href="https://arxiv.org/abs/2XXX.XXXXX"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2XXX.XXXXX-b31b1b.svg"></a>
  -->
</p>

`wellbench` generates reproducible, physically plausible well-log datasets calibrated against real wells from five sedimentary settings. Five regional parameter sets are tuned via Optuna against real data using Jensen–Shannon divergence and Wasserstein distance; a deterministic physics generator and an optional CTGAN baseline both expose the same `.generate(seed=…, depth=…)` interface; and a CLI reproduces a 15-dataset benchmark (5 regions × 3 seeds) for paper-ready comparisons.

Use it to:

- Generate reproducible, physically plausible well-log datasets with one line.
- Stress-test pore-pressure / petrophysics models against ground truth you control (you set the seed, the depth axis, and the region parameters).
- Compare physics-based and GAN-based synthesis under a shared schema.
- Run the 15-CSV benchmark suite out of the box.

---

## Table of contents

- [Why wellbench](#why-wellbench)
- [Install](#install)
- [Quickstart](#quickstart)
- [Hugging Face dataset (Croissant)](#hugging-face-dataset-croissant)
- [Generators](#generators)
- [Command-line interface](#command-line-interface)
- [Writing CSVs from Python](#writing-csvs-from-python)
- [Cleaning real or synthetic data](#cleaning-real-or-synthetic-data)
- [CTGAN baseline (optional)](#ctgan-baseline-optional)
- [Defining your own region](#defining-your-own-region)
- [Plotting recipe](#plotting-recipe)
- [Regions](#regions)
- [Output schema](#output-schema)
- [Physics models](#physics-models)
- [Calibration & evaluation](#calibration--evaluation)
- [Reproducibility](#reproducibility)
- [Project structure](#project-structure)
- [Public API](#public-api)
- [Documentation](#documentation)
- [Comparison & positioning](#comparison--positioning)
- [Intended use, limitations & ethics](#intended-use-limitations--ethics)
- [Contributing](#contributing)
- [Citing](#citing)
- [Maintenance plan](#maintenance-plan)
- [License](#license)

---

## Why wellbench

Pore-pressure prediction is a safety-critical task in drilling, but the public corpus of labelled well logs is small, fragmented, and frequently encumbered by commercial confidentiality. Existing tabular synthesisers (CTGAN, TVAE, TabDDPM, …) treat well logs as i.i.d. rows and ignore the petrophysical relationships (Athy compaction, Wyllie/Archie equations, Eaton's method) that any plausible log must satisfy. `wellbench` fills this gap with:

1. A **deterministic physics generator** whose outputs are valid by construction — they obey the same closed-form transforms a petrophysicist would apply.
2. **Five regional calibrations** against real wells (Eastern Potwar Basin Pakistan, Bering Sea, Volve / North Sea), tuned via Optuna with JS-divergence and Wasserstein objectives.
3. A **side-by-side CTGAN baseline** under an identical `.generate()` interface and an identical output schema, so physics-based and learned synthesis can be compared on equal terms.
4. A **15-dataset benchmark** (5 regions × 3 seeds) with a reproducible CLI and a Croissant-described Hugging Face dataset, ready for use in NeurIPS-style evaluations.

---

## Install

```bash
pip install wellbench
pip install wellbench[ctgan]    # adds the optional CTGAN baseline (torch + ctgan)
pip install wellbench[docs]     # to rebuild the Sphinx docs locally
```

Python ≥ 3.12 is required. Core dependencies are `numpy`, `pandas`, and `scipy`. The CTGAN extra pulls in `torch` and `ctgan` lazily — they are only imported when you actually use the GAN generator, so the base install stays light.

Tested on Linux, macOS, and Windows on CPython 3.12 and 3.13.

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

## Hugging Face dataset (Croissant)

The pre-generated 15-dataset benchmark — calibration evidence and per-region splits — is hosted at:

> 🤗 **<https://huggingface.co/datasets/monteirot/wellbench>**

```python
import mlcroissant as mlc
import pandas as pd

ds = mlc.Dataset("https://huggingface.co/api/datasets/monteirot/wellbench/croissant")
df = pd.DataFrame(ds.records(record_set="physics-samples"))

train = df[df.well == "MISSA-KESWAL-01"]
val   = df[df.well == "MISSA-KESWAL-02"]
test  = df[df.well == "MISSA-KESWAL-03"]
```

You can also load via the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("monteirot/wellbench")
```

To validate the Croissant metadata locally:

```bash
pip install "mlcroissant[parquet]"
mlcroissant validate \
  --jsonld https://huggingface.co/api/datasets/monteirot/wellbench/croissant
```

The dataset card on the Hub mirrors this README's *Regions*, *Output schema*, *Calibration*, and *Intended use* sections, and exposes the same data through the Hugging Face Dataset Viewer and the Croissant 1.0 endpoint required by the NeurIPS Datasets & Benchmarks Track.

---

## Generators

| Name                              | Family             | Modality                | Stochastic | DP-capable | Reference / source                                       |
|-----------------------------------|--------------------|-------------------------|:----------:|:----------:|----------------------------------------------------------|
| `SyntheticWellLogGenerator`       | Physics-based      | Single-table well log   |     ✅     |     ❌     | Athy, Wyllie, Archie, Eaton — see [Physics models](#physics-models) |
| `CTGANSyntheticWellLogGenerator`  | GAN (tabular)      | Single-table well log   |     ✅     |     ❌     | Xu et al. 2019 — *Modeling Tabular Data using CGAN*       |

Both expose the same surface: `gen.generate(seed: int, depth: np.ndarray | None = None) -> pandas.DataFrame`. CTGAN samples are i.i.d. tabular rows; `wellbench` orders them along the depth axis you supply, applies `_CTGAN_COLUMN_RENAMES` so the output schema matches the physics generator, and clips to `PHYSICAL_BOUNDS`.

---

## Command-line interface

The packaged `wellbench` console script reproduces the canonical 15-dataset benchmark — 5 regions × 3 seeds — and writes one CSV per (region, seed) pair:

```bash
wellbench                           # all 15 datasets -> ./benchmark/
wellbench -r 2 -s 99 200            # region 2, seeds 99 and 200 (2 CSVs)
wellbench -r 1 2 3                  # only the pore-pressure regions
wellbench -o my_data                # custom output directory
wellbench --help                    # full reference
```

Output filenames are deterministic: `region_<N>_seed_<S>.csv`. With the default seeds (`[42, 123, 7777]`) you get:

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

When you want one synthetic row per real measurement (e.g. for side-by-side log plots), pass an explicit `depth` array:

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

`clean_well_data` applies the same physical-bounds + outlier rules to any DataFrame that has a `DEPTH` column plus log columns:

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

Five pre-trained CTGAN models — one per region — ship inside the wheel and are loaded lazily. Install the extra:

```bash
pip install wellbench[ctgan]
```

…then sample with the same `.generate(seed, depth=…)` interface as the physics generator:

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

---

## Defining your own region

A region is just a `dict` of physical parameters. The simplest recipe is to copy a built-in and tweak:

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

Required keys cover porosity (`phi0`, `compaction_c`, `phi_layer_amp`, …), Wyllie/Archie/RHOB/GR transforms, and — if `has_pore_pressure=True` — the hydrostatic / overburden / Eaton parameters. See [`src/regions.py`](src/regions.py) for five fully worked examples.

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

Convenience constants: `ALL_REGIONS` (list of all five in order) and `BENCHMARK_SEEDS` (`[42, 123, 7777]`).

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

All outputs are clipped to `wellbench.PHYSICAL_BOUNDS` so consumers can rely on a fixed physical range. Inspect the bounds at runtime:

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
- **Pore pressure (PPP)** — Eaton's method on a normal compaction trend (regions 1–3 only).

---

## Calibration & evaluation

Each of the five regions is calibrated against real-world wells from the corresponding basin using **Optuna** with two complementary fidelity objectives:

- **Jensen–Shannon divergence** between the marginal distribution of each log column (`GR`, `DT`, `RHOB`, `log RT`, …) on synthetic vs. real data — a bounded, symmetric measure of marginal fidelity.
- **Wasserstein-1 distance** on the same marginals — captures distance-on-support beyond what JS-divergence sees, particularly for heavy-tailed columns like `RT`.

The optimiser searches over the region's free physical parameters (`phi0`, `compaction_c`, `phi_layer_amp`, the Eaton exponent for PP regions, etc.) and selects the configuration that minimises a weighted sum of the two objectives across all log columns. Results, including per-region distributional plots, are reported in the paper and reproduced in [`benchmarks/`](benchmarks/) and on the Hugging Face dataset card.

The same metrics serve as a **utility / fidelity report** consumers can run on their own outputs:

```python
from wellbench.metrics import js_wasserstein_report

report = js_wasserstein_report(real_df, synthetic_df, columns=["GR", "DT", "RHOB", "RT"])
print(report)
```

---

## Reproducibility

- **Determinism.** Every generator method takes a `seed`. The same `(region, seed, depth)` triple is guaranteed to produce identical output across runs and platforms. The CTGAN sampler also seeds NumPy and PyTorch (CPU and CUDA) before each `.generate()` call.
- **Frozen region parameters.** Region parameters are frozen dictionaries; if you want to record exactly which calibration produced a CSV, just dump the region dict alongside it.
- **Tagged release.** The exact commit reviewed for the paper is tagged `v<release>`; `pip install wellbench==<release>` reproduces those numbers.
- **One-command paper reproduction.**
  ```bash
  git clone https://github.com/monteirot/wellbench && cd wellbench
  git checkout v<release>
  pip install -e ".[ctgan]"
  wellbench -o benchmark/                # 15 CSVs ≈ <runtime> on CPU
  python benchmarks/reproduce_paper.py   # tables & figures
  ```
- **Hardware.** All paper figures were produced on a single CPU laptop; no GPU is needed for the physics generator. CTGAN training (not required for inference, since checkpoints ship with the wheel) ran on a single NVIDIA GPU; expected wall-clock time per region is documented in `benchmarks/CTGAN_TRAINING.md`.

---

## Project structure

```
wellbench/
├── src/
│   └── wellbench/
│       ├── __init__.py
│       ├── cli.py
│       ├── physics.py            # SyntheticWellLogGenerator, PHYSICAL_BOUNDS
│       ├── ctgan.py              # CTGANSyntheticWellLogGenerator, loaders
│       ├── regions.py            # REGION_1 … REGION_5
│       ├── cleaning.py           # clean_well_data
│       └── models/               # bundled CTGAN checkpoints (ctgan_r1.pkl, …)
├── benchmarks/
│   └── reproduce_paper.py
├── docs/                         # Sphinx + autodoc + Napoleon
├── tests/
├── examples.py
├── pyproject.toml
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

    # Benchmark
    generate_benchmark,

    # Region presets
    ALL_REGIONS, BENCHMARK_SEEDS,
    REGION_1, REGION_2, REGION_3, REGION_4, REGION_5,
)
```

---

## Documentation

Full Sphinx docs (autodoc + Napoleon) live under [`docs/`](docs/). Build them locally with:

```bash
pip install wellbench[docs]
cd docs
make html             # POSIX
.\make.bat html       # Windows
# open _build/html/index.html
```

---

## Comparison & positioning

`wellbench` does not compete with general tabular synthesisers — it complements them. The table below shows where it sits.

| Library                      | Tabular | Physics-aware | Pretrained models | Domain          | License |
|------------------------------|:-------:|:-------------:|:-----------------:|-----------------|---------|
| [SDV](https://github.com/sdv-dev/SDV) | ✅ | ❌ | ❌ | General | BSL/Apache (varies) |
| [Synthcity](https://github.com/vanderschaarlab/synthcity) | ✅ | ❌ | ❌ | General / health | Apache-2.0 |
| [YData-Synthetic](https://github.com/ydataai/ydata-synthetic) | ✅ | ❌ | ❌ | General | MIT |
| **wellbench**                | ✅      | ✅ (Athy / Wyllie / Archie / Eaton) | ✅ (5 CTGANs) | Petrophysics / well logs | MIT |

If you need a general-purpose tabular synthesiser, reach for SDV or Synthcity. If you need *plausible well logs* with controllable depth axes and a pore-pressure label that respects Eaton's method, `wellbench` is the appropriate tool.

---

## Intended use, limitations & ethics

**Intended use.** Methods research — pore-pressure prediction model development, robustness evaluation, data augmentation for small real-well training sets, teaching, and software testing. The Hugging Face benchmark is intended for reproducible comparisons across papers.

**Out-of-scope.**
- *Operational drilling decisions.* The synthetic logs are calibrated to capture *distributional* properties of real basins, not to forecast pressures at any specific surveyed location. Do not use `wellbench` outputs to plan a real well.
- *Substituting for proprietary well data.* Synthetic data here approximates marginal and bulk distributions; subtle stratigraphic features needed for prospect evaluation are deliberately not modelled.
- *Privacy proxy.* Although the physics generator does not memorise individual real samples, the bundled CTGAN checkpoints were trained on cleaned real wells and have not been audited for membership inference. Treat them as research artefacts, not as anonymised data.

**Limitations.**
- Calibration is to five basins only; extrapolation to lithologies or tectonic settings outside those is the user's responsibility.
- The CTGAN baseline generates i.i.d. rows; depth ordering is imposed post-hoc. Vertical correlation structure is therefore weaker than in real logs and weaker than in the physics generator.
- Pore-pressure outputs use Eaton's method on a normal compaction trend; under-compaction is captured, but other overpressure mechanisms (kerogen maturation, tectonic loading) are not.
- Region calibrations were optimised on JS-divergence and Wasserstein-1 of marginals; joint distributions and cross-column residuals are not jointly optimised.

**Bias propagation.** Because the physics generator is parametric and the CTGAN baseline is fitted on cleaned subsets of public/semi-public wells, any sampling bias in those source wells (depth ranges over-represented in the public record, regional skew toward producing fields, etc.) propagates into synthetic outputs. Users running fairness-style evaluations should compare against the seed wells, not against the synthetic data alone.

---

## Contributing

Contributions are welcome — issue reports, new regions, new generators, improved calibration, additional metrics. Please:

1. Open an issue describing the change before sending a large PR.
2. Run `pre-commit run --all-files` and `pytest`.
3. Add a `CHANGELOG.md` entry.
4. By contributing you agree to license your contribution under the project's MIT licence.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## Citing

If `wellbench` helps your research, please cite both the paper and the software release:

```bibtex
@inproceedings{wellbench2026,
  title     = {wellbench: A Physics-Calibrated Synthetic Benchmark for Pore-Pressure Prediction},
  author    = {Monteiro, T. and ...},
  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://arxiv.org/abs/2XXX.XXXXX}
}

@software{wellbench_software,
  author  = {Monteiro, T. and ...},
  title   = {wellbench: Physics-based synthetic well-log benchmark generator},
  year    = {2026},
  version = {<release>},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/monteirot/wellbench}
}
```

For the Hugging Face dataset specifically, please also cite the Croissant record at <https://huggingface.co/datasets/monteirot/wellbench>.

---

## Maintenance plan

- Maintained by the original authors at the lab of record.
- Versioned with [SemVer](https://semver.org/). Breaking API changes are gated by a major version bump and announced in `CHANGELOG.md`.
- Long-term hosting on PyPI, GitHub, and Hugging Face. A Zenodo DOI is minted per release for archival citation.
- Security disclosures: please email the maintainers privately rather than opening a public issue.

---

## License

MIT — see [`LICENSE`](LICENSE).
