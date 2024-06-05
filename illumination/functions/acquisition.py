import torch
import numpy as np

from math import sqrt
from typing import List

from rdkit import rdBase
from scipy.stats import norm
from botorch.acquisition.analytic import _log_ei_helper

from itertools import groupby
from abc import ABC, abstractmethod
from illumination.base import Molecule

rdBase.DisableLog("rdApp.error")


class BO_Acquisition(ABC):
    """
    An abstract base class implementing the Bayesian Optimization acquisition strategy for selecting molecules.
    This class is responsible for calculating acquisition values for molecules and selecting the best ones based on these values.

    Attributes:
    - config: Configuration object containing settings for the acquisition strategy.
    - archive: An archive object containing elite molecules and associated data.

    Methods:
    - __init__: Initializes the BO_Acquisition object with the given configuration.
    - __call__: Calculates acquisition values for molecules and selects the best ones.
    - set_archive: Sets the archive containing elite molecules.
    - molecule_selection: Selects the best molecules based on their acquisition values and niche indices.
    - calculate_acquisition_value: Abstract method for calculating acquisition values for molecules (must be implemented in subclasses).
    """
    def __init__(self, config) -> None:
        """
        Initializes the BO_Acquisition object with the given configuration.

        Args:
            config: Configuration object containing settings for the acquisition strategy.
        """
        self.config = config
        self.archive = None
        return None

    def __call__(self, molecules):
        """
        Calculates acquisition values for molecules and selects the best ones.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            List[Molecule]: A list of selected Molecule objects based on their acquisition values.
        """
        molecules = self.calculate_acquisition_value(molecules)
        molecules = self.molecule_selection(molecules)
        return molecules

    def set_archive(self, archive):
        """
        Sets the archive containing elite molecules.

        Args:
            archive: An archive object containing elite molecules and associated data.
        """
        self.archive = archive
        return None

    @staticmethod
    def molecule_selection(molecules: List[Molecule]) -> List[Molecule]:
        """
        Selects the best molecules based on their acquisition values and niche indices.

        Args:
            molecules: A list of Molecule objects to select from.

        Returns:
            List[Molecule]: A list of Molecule objects that have the highest acquisition values in their respective niches.
        """
        molecules.sort(key=lambda molecule: molecule.niche_index)
        grouped_molecules = {index: list(molecule_group) for index, molecule_group in groupby(molecules, key=lambda molecule: molecule.niche_index)}
        molecules = [max(molecule_group, key=lambda molecule: molecule.acquisition_value) for molecule_group in grouped_molecules.values()]
        return molecules

    @abstractmethod
    def calculate_acquisition_value(self, molecules) -> None:
        """
        Abstract method for calculating acquisition values for molecules.
        This method must be implemented in subclasses.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            None
        """
        raise NotImplementedError


class Posterior_Mean(BO_Acquisition):
    """
    A strategy class for the posterior mean of a list of molecules.
    """

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules based on their predicted fitness.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            None
        """
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitness
        return molecules


class Upper_Confidence_Bound(BO_Acquisition):
    """
    A strategy class for the upper confidence bound of a list of molecules.
    """

    def __init__(self, config):
        """
        Initializes the Upper_Confidence_Bound object with the given configuration.

        Args:
            config: Configuration object containing settings for the upper confidence bound strategy.
        """
        super().__init__(config)
        self.beta = config.beta
        return None

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules based on their predicted fitness and uncertainty.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            None
        """
        for molecule in molecules:
            molecule.acquisition_value = molecule.predicted_fitnessn + sqrt(self.beta) * molecule.predicted_uncertainty
        return molecules


class Expected_Improvement(BO_Acquisition):
    """
    A strategy class for the expected improvement of a list of molecules.
    """

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules based on their expected improvement.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            None
        """
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules]
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value = molecule.predicted_uncertainty * (norm.pdf(Z) + Z * norm.cdf(Z))
        return molecules


class Log_Expected_Improvement(BO_Acquisition):
    """
    A strategy class for the numerically stable logarithm of the expected improvement of a list of molecules.
    """

    def calculate_acquisition_value(self, molecules) -> None:
        """
        Updates the acquisition value for a list of molecules based on the logarithm of their expected improvement.

        Args:
            molecules: A list of Molecule objects to evaluate.

        Returns:
            None
        """
        current_fitnesses = [self.archive.elites[molecule.niche_index].fitness for molecule in molecules]
        for molecule, current_fitness in zip(molecules, current_fitnesses):
            Z = (molecule.predicted_fitness - current_fitness) / molecule.predicted_uncertainty
            molecule.acquisition_value = _log_ei_helper(torch.tensor(Z)).detach().item() + np.log(molecule.predicted_uncertainty)
        return molecules
