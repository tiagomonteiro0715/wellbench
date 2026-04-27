# wellbench

Physics-based synthetic well-log benchmark generator for pore-pressure
prediction research. Five regions calibrated against real-world wells (Optuna
optimisation against Jensen–Shannon divergence and Wasserstein distance), a
deterministic physics generator, an optional CTGAN baseline, and a CLI that
reproduces a 15-dataset benchmark.

## Install

```bash
pip install wellbench
pip install wellbench[ctgan]   # adds the optional CTGAN baseline
pip install wellbench[docs]    # to rebuild the Sphinx docs locally
```

## Quickstart

```python
from wellbench import SyntheticWellLogGenerator, REGION_1

gen = SyntheticWellLogGenerator(REGION_1)
df = gen.generate(seed=42)
print(df.head())
```

CLI:

```bash
wellbench                       # all 15 datasets (5 regions × 3 seeds)
wellbench -r 2 -s 99 200        # region 2, seeds 99 and 200
wellbench -o my_data            # custom output directory
```

A single, runnable tour of every public entry point lives in
[`examples.py`](examples.py):

```bash
python examples.py              # run every example
python examples.py basic ctgan  # pick specific examples
```

## Regions

| #  | Region                                  | Location                                | Pore pressure |
|----|-----------------------------------------|-----------------------------------------|---------------|
| 1  | Missa Keswal (Zone 1)                   | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 2  | PINDORI-1 (Zone 2)                      | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 3  | JOYAMAIR-4 / MINWAL-2 (Zone 3)          | Eastern Potwar Basin, Punjab, Pakistan  | yes           |
| 4  | IODP Expedition 323, Hole U1343E        | Bering Sea                              | no            |
| 5  | Volve oil field                         | North Sea (Norway/UK)                   | no            |

## Physics models

- **Porosity** — exponential compaction (Athy's law) + layered sinusoids + noise
- **Sonic (DT)** — Wyllie time-average equation
- **Density (RHOB)** — bulk density mixing law
- **Resistivity (RT)** — Archie's equation
- **Gamma ray (GR)** — shale-volume linear mixing
- **Pore pressure (PPP)** — Eaton's method on a normal compaction trend

All outputs are clipped to `wellbench.PHYSICAL_BOUNDS` so consumers can rely
on a fixed physical range.

## Documentation

Full docs (Sphinx + autodoc) live under [`docs/`](docs/). Build them with:

```bash
pip install wellbench[docs]
cd docs
make html             # POSIX
.\make.bat html       # Windows
# open _build/html/index.html
```

## License

MIT — see [`LICENSE`](LICENSE).
