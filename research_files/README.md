# Well-Log Synthesis Research Code

Reproducible local version of a research codebase originally developed in
Google Colab. The project benchmarks two ways of generating synthetic
well-log data — a **physics-based forward model** and a **CTGAN-based
generator** — and evaluates them with TRTR (Train Real, Test Real) and
TSTR (Train Synthetic, Test Real) machine-learning protocols.

## Repository layout

```
.
├── notebooks/          # Original Jupyter notebooks (unmodified Colab exports)
│   ├── final_optimal_Physics_based_generation.ipynb
│   ├── final_optimal_GAN_based_generation.ipynb
│   ├── final_POFM_TRTR_ML_models_study.ipynb
│   ├── final_POFM_TSTR_classic_ML_models_study.ipynb
│   ├── final_CTGAN_TRTR_ML_models_study.ipynb
│   └── final_CTGAN_TSTR_classic_ML_models_study.ipynb
├── scripts/            # Same code as .py files, patched to run locally
│   ├── _local_setup.py                # shared bootstrap (data dir, mem tracking)
│   ├── final_optimal_physics_based_generation.py
│   ├── final_optimal_gan_based_generation.py
│   ├── final_pofm_trtr_ml_models_study.py
│   ├── final_pofm_tstr_classic_ml_models_study.py
│   ├── final_ctgan_trtr_ml_models_study.py
│   └── final_ctgan_tstr_classic_ml_models_study.py
├── data/               # Auto-populated when scripts download well-log files
├── checkpoints/        # Auto-populated with Optuna / training artifacts
├── pyproject.toml      # Project metadata + dependency declaration
├── uv.lock             # Pinned, fully-resolved dependency graph (210 pkgs)
└── README.md
```

## Setting up the environment

The project uses [uv](https://docs.astral.sh/uv/) for package management.
A `uv.lock` file is committed, so collaborators only need one command to
get an identical environment.

### 1. Install uv (one-time)

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Sync the environment

From the project root:

```bash
uv sync
```

That single command will:

1. Read `pyproject.toml` + `uv.lock`,
2. Create a project-local virtual environment under `.venv/`,
3. Install every pinned package (CPU PyTorch by default).

No `pip`, no manual venv creation, no version drift between machines.

### 3. Run the scripts

```bash
# Physics-based generation
uv run python scripts/final_optimal_physics_based_generation.py

# CTGAN-based generation
uv run python scripts/final_optimal_gan_based_generation.py

# Train-Real / Test-Real ML benchmarks
uv run python scripts/final_pofm_trtr_ml_models_study.py
uv run python scripts/final_ctgan_trtr_ml_models_study.py

# Train-Synthetic / Test-Real ML benchmarks
uv run python scripts/final_pofm_tstr_classic_ml_models_study.py
uv run python scripts/final_ctgan_tstr_classic_ml_models_study.py
```

`uv run` automatically activates `.venv/` for the duration of the command —
you do not need to `activate` anything yourself.

### 4. Run the original notebooks (optional)

```bash
uv run jupyter lab notebooks/
```

## What changed vs. the Colab version

The notebooks under `notebooks/` are kept verbatim. The `scripts/` copies
were patched so they run on a normal Windows / macOS / Linux machine:

| Colab construct                              | Local replacement                                     |
| -------------------------------------------- | ----------------------------------------------------- |
| `!pip install ...`                           | Removed — handled by `uv sync`.                       |
| `import resource` (Linux-only)               | Cross-platform `psutil`-based `rss_mb()` helper.      |
| `from google.colab import drive` (hard fail) | Wrapped in `try/except`; falls back to local folder.  |
| `/content/...` data paths                    | Files now live under `data/` (created automatically). |
| `/content/drive/MyDrive/...` checkpoints     | Default to `checkpoints/<run>/` locally.              |

All other research logic — model definitions, Optuna search spaces,
metric calculations, plotting — is unchanged.

## Notes

* **GPU**: the lock file pins the CPU build of PyTorch, which works
  everywhere. If you want CUDA, edit the `torch` requirement in
  `pyproject.toml` and re-run `uv lock` followed by `uv sync`.
* **Data download**: the scripts call `gdown` on Google Drive file IDs.
  The first run will populate `data/`; subsequent runs reuse the cached
  files.
* **Checkpoints**: long-running Optuna studies save to
  `checkpoints/<benchmark>/`. Delete the folder if you want a clean run.
