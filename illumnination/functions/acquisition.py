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

from abc import ABC, abstractmethod
from itertools import groupby

class BO_Acquisition(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.archive = None
        return None
    
    def __call__(self, molecules):
        molecules = self.calculate_acquisition_value(molecules)
        molecules = self.molecule_selection(molecules)
        return molecules
    
    def set_archive(self, archive):
        self.archive = archive
        return None

    @staticmethod
    def molecule_selection(molecules: List[Molecule]) -> List[Molecule]:
        molecules.sort(key = lambda molecule: molecule.niche_index)
        grouped_molecules = {index: list(molecule_group) for index, molecule_group in groupby(molecules, key = lambda molecule: molecule.niche_index)}
        molecules = [max(molecule_group, key = lambda molecule: molecule.acquisition_value) for molecule_group in grouped_molecules.values()]
        return molecules
    
    @abstractmethod
    def calculate_acquisition_value(self, molecules) -> None:
        raise NotImplementedError

class Posterior_Mean(BO_Acquisition):
    """
    A strategy class for the posterior mean of a list of molecules.
    """
    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitness 
        return molecules

class Upper_Confidence_Bound(BO_Acquisition):
    """
    A strategy class for the upper confidence bound of a list of molecules.
    """
    def __init__(self, config):
        super().__init__(config)
        self.beta = config.beta
        return None

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitness + sqrt(self.beta) * molecule.predicted_uncertainty
        return molecules

class Expected_Improvement(BO_Acquisition):
    """
    A strategy class for the expected improvement of a list of molecules.
    """

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules] 
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value =  molecule.predicted_uncertainty*(norm.pdf(Z) + Z*norm.cdf(Z))
        return molecules

class Log_Expected_Improvement(BO_Acquisition):
    """
    A strategy class for the numerically stable logarithm of the expected improvement of a list of molecules.
    """

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules.
        """
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules] 
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value = _log_ei_helper(torch.tensor(Z)).detach().item() + np.log(molecule.predicted_uncertainty)
        return molecules
    
