import torch
import numpy as np
import pandas as pd
from math import sqrt
from typing import List, Tuple, Type

from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from argenomic.base import Molecule

from scipy.stats import norm
from botorch.acquisition.analytic import _log_ei_helper

class Posterior_Mean:
    """
    A strategy class for the posterior mean of a list of molecules.
    """
    def __init__(self, archive, config) -> None:
        self.config = config
        self.archive = archive
        return None

    def __call__(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitness 
        return molecules

class Upper_Confidence_Bound:
    """
    A strategy class for the upper confidence bound of a list of molecules.
    """
    def __init__(self, archive, config) -> None:
        self.config = config
        self.beta = config.beta
        self.archive = archive
        return None

    def __call__(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitness + sqrt(self.beta) * molecule.predicted_uncertainty
        return molecules

class Expected_Improvement:
    """
    A strategy class for the expected improvement of a list of molecules.
    """
    def __init__(self, archive, config) -> None:
        self.config = config
        self.archive = archive
        return None

    def __call__(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules] 
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value =  molecule.predicted_uncertainty*(norm.pdf(Z) + Z*norm.cdf(Z))
        return molecules

class Log_Expected_Improvement:
    """
    A strategy class for the numerically stable logarithm of the expected improvement of a list of molecules.
    """
    def __init__(self, archive, config) -> None:
        self.config = config
        self.archive = archive
        return None

    def __call__(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules] 
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value = _log_ei_helper(torch.tensor(Z)).detach().item() + np.log(molecule.predicted_uncertainty)
        return molecules
    
