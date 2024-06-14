.. GB-BI documentation master file, created by
   sphinx-quickstart on Thu Jun  6 12:05:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Benchmarks
==========
.. include:: _toctree.rst

In recent years, several open-source benchmarking suites have emerged for the de novo design of small molecules. Among the most prominent are GuacaMol, the Therapeutics Data Commons, and Tartarus. Drawing upon each of these benchmarking suites, we conduct three distinct numerical experiments that replicate various aspects of molecular design challenges in life and materials science. Specifically, we make use of a GuacaMol rediscovery task focused on molecular similarity, apply a Tartarus task aimed at designing efficient organic photovoltaics, and introduce an enhanced version of a Therapeutics Data Commons docking task. This task evaluates the theoretical binding affinity between a small molecule and a target protein while considering the synthetic accessibility of the candidate molecules.

Molecular Similarity
---------------------

GuacaMol features several molecule rediscovery tasks wherein the fitness of a generated molecule is evaluated based on the Tanimoto similarity between the generated molecule and the target molecule, determined by their respective extended-connectivity fingerprints. We opted for the most challenging of these tasks, rediscovering Troglitazone (an antidiabetic compound), to assess and quantify the optimisation efficacy of a comprehensive list of generative models for small molecules: a deep generative molecular model, two types of genetic algorithms, a quality-diversity method, and Bayesian Illumination.To enhance the discriminative capability of this task, we impose a maximum limit of 2500 fitness function evaluations.

.. list-table:: 
   :header-rows: 1

   * - Algorithm
     - Maximum Score
     - Mean Score
   * - **GB-BI**
     - **1.00 ± 0.00**
     - **0.80 ± 0.02**
   * - GB-EPI
     - 0.96 ± 0.07
     - 0.80 ± 0.06
   * - GB-GA
     - 0.78 ± 0.10
     - 0.58 ± 0.04
   * - REINVENT
     - 0.71 ± 0.05
     - 0.42 ± 0.03
   * - JANUS
     - 0.66 ± 0.08
     - 0.60 ± 0.04


Power Conversion Efficiency
----------------------------

The Tartarus benchmarking suite contains several tasks focused on the development of efficient organic solar cells, more specifically for the design of small organic donor molecules with optimal power conversion efficiency when used in bulk heterojunction devices. These tasks provide optimisation requirements based on single point GFN2-xTB calculations for LUMO energies, the HOMO-LUMO gap, and the molecular dipole moment of candidate molecules. The Tartarus benchmarking suite provides tasks for the maximisation of HOMO-LUMO gap and molecular dipole moment values, the minimisation of LUMO energies, and the maximisation of a combined score for power conversion efficiency defined as the molecular dipole moment plus the HOMO-LUMO gap minus the LUMO energy. For ease of evaluation, we make use of an autoML deep learning model, delivered with Tartarus, which was trained to predict the scoring functions of these tasks on a subset of approximately 25,000 molecules sampled from the Harvard Clean Energy Project Database.

.. list-table:: 
   :widths: 35 35 35 35 35
   :header-rows: 1

   * - Algorithm
     - Humo-Lumo 
     - Lumo Energy
     - Dipole Moment 
     - PCE 
   * - **GB-BI**
     - **2.76 ± 0.00**
     - **-9.44 ± 0.01**
     - **8.22 ± 0.21**
     - **18.18 ± 0.13**
   * - GB-EPI
     - 2.76 ± 0.00
     - -9.40 ± 0.01
     - 8.04 ± 0.10
     - 18.17 ± 0.10
   * - GB-GA
     - 2.73 ± 0.00
     - -9.29 ± 0.05
     - 7.68 ± 0.45
     - 17.46 ± 0.16
   * - JANUS
     - 2.75 ± 0.00
     - -9.42 ± 0.02
     - 7.74 ± 0.38
     - 18.11 ± 0.21
   * - REINVENT
     - 2.59 ± 0.03
     - -9.18 ± 0.04
     - 6.73 ± 0.11
     - 16.91 ± 0.36

Results for independent optimisation runs on the four tasks related to optimal power conversion efficiency of organic photovoltaics are shown in the table above. The results for GB-GA, JANUS, and REINVENT are calculated based on the data supplied with Tartarus codebase and paper. According to the settings used in these, we impose a maximum limit of 2500 fitness function evaluations but no structural filters on the optimisation runs of GB-BI and GB-EPI. Note the performance of JANUS on these specific tasks, when compared to GB-GA, is due to its explicit use of molecular fragments from the initial population of molecules with which it is supplied. The other algorithms follow the trend set in previous benchmarks of this paper, as GB-BI matches or outperforms GB-EPI and GB-GA on all four tasks. It is readily observed that REINVENT, a representative deep learning model, displays the usual and substantial sample efficiency issues in generating optimised molecules.

Protein Binding Affinity 
----------------------------

The Therapeutics Data Commons (TDC) provides docking molecule generation benchmarks which evaluate the theoretical binding affinity between small molecules and target proteins. Docking is widely used for virtual screening of compounds, as molecules with higher theoretical binding affinities are statistically more likely to have a higher bioactivity. To increase real-life relevance, we apply stringent structural and ADME filters to candidate molecules and modulate the docking results with a synthetic accessibility score (SAS), as suggested in the documentation, for the proposed small molecule. We select three different targets from the TDC benchmarking suite: a dopamine receptor (DRD3) implicated in schizophrenia and essential tremor syndrome, a tyrosine-protein kinase (ABL1) implicated in chronic myelogenous leukemia, and the epidermal growth factor receptor (EGFR) which has been strongly associated with a number of cancers, including lung cancer, glioblastoma, and epithelial tumors of the head and neck.

The results of the protein binding tasks for three independent runs of GB-BI, GB-EPI, and GB-GA are presented in the table above. GB-BI consistently outperforms both GB-EPI and GB-GA across all three tasks, achieving superior results in terms of both the minimum obtained docking score and the mean docking score for the 100 best compounds at the end of optimization for each algorithm. To evaluate the quality-diversity effectiveness of GB-BI and GB-EPI, we calculate the QD-score (the sum of all fitness values of the molecules present in the archive at the last generation) and the percentage of archive niches occupied by a molecule, known as archive coverage. Since GB-GA is not a quality-diversity algorithm, we do not calculate the QD-score or archive coverage for it. The QD-score is a widely used quality-diversity measure that assesses an algorithm's ability to populate its archive with diverse yet high-performing solutions.
