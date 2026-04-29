Regions
=======

The five bundled region parameter dictionaries are calibrated against
real-world data via Optuna hyperparameter optimisation (Jensen–Shannon
divergence and Wasserstein distance against the real distributions).

================  ========================================  ============
Symbol            Field                                     Pore pressure
================  ========================================  ============
``REGION_1``      Missa Keswal (Eastern Potwar Basin, PK)   yes
``REGION_2``      PINDORI-1 (Eastern Potwar Basin, PK)      yes
``REGION_3``      JOYAMAIR-4 / MINWAL-2 (E. Potwar, PK)     yes
``REGION_4``      IODP Expedition 323, Hole U1343E          no
``REGION_5``      Volve oil field (North Sea)               no
================  ========================================  ============

The convenience list :data:`wellbench.ALL_REGIONS` contains all five in
order. The default seed list :data:`wellbench.BENCHMARK_SEEDS` is used by
:func:`wellbench.generate_benchmark` to produce the canonical
3-seeds-per-region benchmark.

Physics models
--------------

* **Porosity** — exponential compaction (Athy's law) plus layered
  sinusoidal variations and Gaussian noise.
* **Sonic (DT)** — Wyllie time-average equation.
* **Density (RHOB)** — bulk density mixing law with a small lithology
  trend.
* **Resistivity (RT)** — Archie's equation.
* **Gamma ray (GR)** — shale-volume linear mixing.
* **Pore pressure (PPP)** — Eaton's method on a normal compaction
  trend (regions 1-3 only).

All outputs are clipped to :data:`wellbench.PHYSICAL_BOUNDS` so consumers
can rely on a fixed physical range.

Region reference
----------------

.. autodata:: wellbench.REGION_1
.. autodata:: wellbench.REGION_2
.. autodata:: wellbench.REGION_3
.. autodata:: wellbench.REGION_4
.. autodata:: wellbench.REGION_5
.. autodata:: wellbench.ALL_REGIONS
.. autodata:: wellbench.BENCHMARK_SEEDS
