.. GB-BI documentation master file, created by
   sphinx-quickstart on Thu Jun  6 12:05:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started
================

.. include:: _toctree.rst

**Graph-Based Bayesian Illumination** (GB-BI) is an open-source software library that aims to make state-of-the-art, quality-diversity optimisation techniques infused with Bayesian optimisation easily accessible to scientific experts in medicinal chemistry and cheminformatics. We provide a modular codebase, novel benchmarks, and extensive documentation. The main novelty of Bayesian illumination compared to a previous quality diversity method for small molecule optimisation is the use of surrogate fitness and acquisition function calculations to inform the selection of a single molecule to be compared in direct evolutionary competition with the current occupant of the niche. 

After installing the software and running the tests, a basic usage example of Bayesian Illumination (i.e. the rediscovery of Troglitazone) can be called upon in the following manner:

.. code-block:: bash

    python illuminate.py

This command will call the config file in `configs` and makes use of Hydra for command line overwrites. Hydra is a framework for elegantly configuring complex applications. It allows you to compose your configuration dynamically, enabling you to overwrite any part of the configuration through the command line.

Installing
==========

Download the source code from Github to your local machine and create the environment from the `bayesian_illumination.yml` file:

.. code-block:: bash

    conda env create -f bayesian_illumination.yml

Activate the new environment:

.. code-block:: bash

    conda activate bayesian-illumination

Verify that the new environment was installed correctly:

.. code-block:: bash

    conda env list

Running the Tests
=================

To run the unit tests:

.. code-block:: bash

    pytest ./tests
