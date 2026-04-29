wellbench
=========

**wellbench** is a physics-based synthetic well-log benchmark generator for
pore-pressure prediction research. It packages five regional parameter sets
(calibrated against real-world wells with Optuna), a deterministic physics
generator, an optional CTGAN baseline, and a CLI that reproduces a 15-dataset
benchmark.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   regions
   api

Installation
------------

.. code-block:: bash

   pip install wellbench

   # with the CTGAN baseline
   pip install wellbench[ctgan]

   # docs (only needed to rebuild this site)
   pip install wellbench[docs]

At a glance
-----------

.. code-block:: python

   from wellbench import SyntheticWellLogGenerator, REGION_1

   gen = SyntheticWellLogGenerator(REGION_1)
   df = gen.generate(seed=42)
   print(df.head())

Reproducing the benchmark from the command line:

.. code-block:: bash

   wellbench                       # all 15 datasets
   wellbench -r 2 -s 99 200        # region 2, seeds 99 and 200
   wellbench -o my_data            # custom output directory

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
