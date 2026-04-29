Quickstart
==========

The full runnable tour lives in ``examples.py`` at the repository root. Each
subsection below corresponds to one ``example_*`` function in that file —
running the file with no arguments runs every example::

   python examples.py
   python examples.py basic ctgan      # pick specific examples

Basic generation
----------------

.. code-block:: python

   from wellbench import SyntheticWellLogGenerator, REGION_1

   gen = SyntheticWellLogGenerator(REGION_1)
   df = gen.generate(seed=42)
   # columns: DEPTH, GR, DT, RHOB, RT  (+ HP, OB, DT_NCT, PPP for PP regions)

Custom depth axis
-----------------

Pass an explicit ``depth`` array — useful when you want one synthetic row
per real-well measurement:

.. code-block:: python

   import numpy as np
   from wellbench import SyntheticWellLogGenerator, REGION_4

   depth = np.linspace(120, 700, 2000)        # metres
   df = SyntheticWellLogGenerator(REGION_4).generate(seed=7, depth=depth)

Pore-pressure regions
---------------------

Regions 1-3 are calibrated for pore-pressure prediction; their
``has_pore_pressure`` flag is ``True`` and ``generate()`` adds the ``HP``,
``OB``, ``DT_NCT``, and ``PPP`` columns. Regions 4-5 produce only the
basic log suite.

Cleaning real or synthetic data
-------------------------------

:func:`wellbench.clean_well_data` applies the same physical bounds and
outlier rules to any DataFrame:

.. code-block:: python

   from wellbench import clean_well_data
   cleaned = clean_well_data(df, label="well_A", outlier_std=5)

Full benchmark
--------------

.. code-block:: python

   from wellbench import generate_benchmark
   paths = generate_benchmark(output_dir="benchmark")  # 15 CSV files

CTGAN baseline (optional)
-------------------------

Requires the ``[ctgan]`` extra (pulls in ``torch`` and ``ctgan``):

.. code-block:: python

   from wellbench import load_ctgan_generator

   gen = load_ctgan_generator(region_index=1)   # ctgan_r1.pkl
   df = gen.generate(seed=42)

Aligning to a real well
-----------------------

A common recipe — emit one synthetic row per real measurement:

.. code-block:: python

   import pandas as pd
   from wellbench import SyntheticWellLogGenerator, REGION_1

   real = pd.read_csv("real_well.csv", usecols=["DEPTH"])
   synth = SyntheticWellLogGenerator(REGION_1).generate(
       seed=42, depth=real["DEPTH"].to_numpy(),
   )
