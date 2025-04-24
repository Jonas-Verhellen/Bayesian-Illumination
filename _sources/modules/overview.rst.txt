Core Functionality
===================

**Graph-Based Bayesian Illumination (GB-BI)** is an open-source software library that aims to make state-of-the-art, quality-diversity optimisation techniques infused with Bayesian optimisation easily accessible. We provide a modular codebase, novel benchmarks, and extensive documentation. In this section of the documentation, we discuss the core functionality of GB-BI in terms of fitness functions, molecular representations, acquisition functions, physicochemical descriptors, and structural filters. For practical considerations and specific configuration file settings, please see the tutorials.

Fitness Functions
------------------

GB-BI provides five classes of fitness functions out-of-the-box: fingerprint-based rediscovery, descriptor-based rediscovery, and SAS-modulated docking scores. These fitness functions can and have been used as benchmark tools to probe the efficiency of generative models but also have direct practical applications. Additional fitness functions can easily be added to the codebase.

.. list-table::
   :header-rows: 1

   * - **Task**
     - **Description**
   * - **Fingerprint Rediscovery**
     - A lightweight task focused on molecule rediscovery where the fitness of a molecule is the Tanimoto similarity to the target molecule, based on their respective extended-connectivity fingerprints. Implementation based on Gaucamol, but applicable to generic targets.
   * - **Descriptor Rediscovery**
     - An alternative molecule rediscovery task, with intermediate computational expense, where the fitness of a generated molecule is defined as the conformer-aggregated similarity to the target molecule. Conformer similarity is based on either USRCAT or Zernike descriptors.
   * - **Guacamol Benchmarks**
     - These tasks optimise molecules to score highly on the GuacaMol task provided by the TDC oracle: molecular properties, molecular similarity, drug rediscovery, isomers, MPOs, median molecule and a few others.
   * - **Organic Photovoltaics**
     - These tasks focus on the design of small organic donor molecules with optimal power conversion efficiency, based on single point GFN2-xTB calculations distilled through an autoML model provided by the Tartarus benchmarking suite.
   * - **SAS-Modulated Docking Scores**
     - A computationally intensive task, utilizing docking methods which evaluate the theoretical affinity between a small molecule and a target protein. To avoid pure exploitation of the docking method, the scores are modulated by the synthetic accessibility of the small molecule.


Representations
--------------------

GB-BI supports several molecular representations that are based on bit vectors or strings. These representations are used for the surrogate models using the Tanimoto kernel from GAUCHE. The string-based representations are turned into a bag-of-characters before being used in the kernel. Note that several of these vector representations are currently not natively supported by GAUCHE.

.. list-table::
   :header-rows: 1

   * - **Representation**
     - **Description**
   * - **ECFP**
     - Extended-Connectivity Fingerprints (ECFP) are circular topological fingerprints that represent the presence of particular substructures.
   * - **FCFP**
     - Functional-Class Fingerprints (FCFP) are circular topological fingerprints that represent the presence of particular pharmacophoric properties.
   * - **RDFP**
     - RDKit-specific fingerprints (RDFP) are inspired by public descriptions of the Daylight fingerprints, but differ significantly in practical implementation.
   * - **APFP**
     - Atom pair fingerprints (APFP) encode all unique triplets of atomic number, number of heavy atom neighbours, aromaticity, and chirality in a vector format.
   * - **TTFP**
     - Topological torsion fingerprints (TTFP) encode the long-range relationships captured in atom pair fingerprints through information on the torsion angles.
   * - **SMILES**
     - The simplified molecular-input line-entry system (SMILES) is a widely used line notation for describing a small molecule in terms of short ASCII strings.
   * - **SELFIES**
     - Self-referencing embedded strings (SELFIES) are an alternative line notation for a small molecule, designed to be used in arbitrary machine learning models.


Acquisition Functions
-------------------------

Acquisition functions are heuristics employed to evaluate the potential of candidate molecules based on their predicted fitness value and the associated uncertainty of a surrogate fitness model (i.e. the Gaussian process). A large literature exists on the topic of acquisition functions and their design. GB-BI supports several of the most well-known and often used acquisition functions.

.. list-table::
   :header-rows: 1

   * - **Acquisition Function**
     - **Description**
   * - **Mean**
     - The posterior mean (mean) is simply the direct fitness value as predicted by the surrogate fitness model.
   * - **UCB**
     - The upper confidence bound (UCB) balances exploration and exploitation based on a confidence boundary derived from the surrogate fitness model.
   * - **EI**
     - The expected improvement (EI) considers both the probability of improving on the current solutions and the magnitude of the predicted improvement.
   * - **logEI**
     - A numerically stable variant of the logarithm of the expected improvement (logEI), which was recently introduced to alleviate the vanishing gradient problems.

Physicochemical Archive
-------------------------

Users choose their own features of interest and define relevant ranges of variation to construct a feature space. If, for instance, a user wants to find medicinally relevant molecules in chemical space, they could construct a feature space based on physicochemical properties like lipophilicity and molecular mass. The chosen ranges in which to explore these features can be used to specify  a desired subset of chemical space in which to generate new molecules. GB-BI supports all descriptors from a selection of common RDKit modules.

.. list-table::
   :header-rows: 1

   * - **Module**
     - **Description**
   * - **AllChem**
     - Includes a variety of functions for molecular operations and calculations.
   * - **Crippen**
     - Contains methods for calculating logP and molar refractivity.
   * - **Lipinski**
     - Implements rules and functions related to Lipinski's rule of five for druglikeness.
   * - **Descriptors**
     - Provides a comprehensive set of molecular descriptors.
   * - **rdMolDescriptors**
     - Contains methods for calculating complicated molecular descriptors.

Structural Filters
----------------------

To rule out unwanted and potentially toxic molecules, we use functional group knowledge from the ChEMBL database and a combination of ADME property calculations. We remove undesirable compounds before they enter the evaluation step of the algorithm. Removing these compounds at an early stage makes the algorithm more efficient, increases the predictive value of the final outcome, and significantly decreases overall processing time. Specifically, we filter out molecules that contain macrocycles, fail at Veber's rule, or raise structural alerts.

.. list-table::
   :header-rows: 1

   * - **Rule Set**
     - **Number of Alerts**
     - **Description**
   * - **BMS**
     - 180
     - Alerts derived from Bristol-Myers Squibb, encompassing a broad range of concerns.
   * - **Dundee**
     - 105
     - Alerts identified by researchers at the University of Dundee.
   * - **Glaxo**
     - 55
     - Alerts from GlaxoSmithKline, focusing on known problematic groups.
   * - **Inpharmatica**
     - 91
     - Alerts from Inpharmatica Ltd, emphasizing computational toxicology findings.
   * - **LINT**
     - 57
     - Alerts from the LINT project, targeting specific structural liabilities.
   * - **MLSMR**
     - 116
     - Alerts from the Molecular Libraries Screening Center Network (MLSCN) repository.
   * - **PAINS**
     - 479
     - Pan-Assay INterference compoundS (PAINS) alerts known to interfere in assays.
   * - **SureChEMBL**
     - 166
     - Alerts derived from SureChEMBL, focusing on patent-related structural issues.
