import os
import csv
import hydra
import random

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict
from illumination.base import Molecule, Elite

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Lipinski

rdBase.DisableLog("rdApp.error")

class Controller:
    """
    A utility class for printing, logging, and controlling the status of the algorithm.

    The class provides methods for generating and displaying generation-specific statistics,
    as well as storing basic statistics and printing the archive to CSV files.

    Attributes:
        archive: An archive object containing elite molecules and associated data.
        surrogate: A surrogate model object used in the optimization process.
        generation: The current generation of the optimization algorithm.
        fitness_calls: The number of fitness function calls made so far.
        memory_of_molecules: A list to store molecules across generations.
        max_generations: The maximum number of generations for the optimization process.
        max_fitness_calls: The maximum number of fitness function calls allowed.
        remaining_fitness_calls: The remaining number of fitness function calls.

    Methods:
        __init__(config): Initializes a Controller object with the given configuration.
        set_archive(archive): Sets the archive object for the controller.
        active(): Checks if the optimization process is still active.
        update(): Updates the controller state and archives statistics.
        add_fitness_calls(fitness_calls): Adds to the total number of fitness function calls.
        write_statistics(statistics, metrics): Prints archive statistics to the console.
        store_statistics(statistics, metrics): Appends archive statistics to a CSV file.
        calculate_statistics(archive_data): Calculates various statistics from archive data.
        calculate_surrogate_metrics(molecules): Calculates surrogate model metrics.
        get_archive_data(): Retrieves elite molecule attributes and creates a DataFrame.
        store_molecules(): Stores all molecules in memory to a CSV file.
    """

    def __init__(self, config) -> None:
        """
        Initializes a Controller object with the given configuration.

        Args:
            config: Configuration object containing settings for the controller.
        """
        self.archive = None
        self.surrogate = None
        self.generation = 0
        self.fitness_calls = 0
        self.memory_of_molecules = []
        self.max_generations = config.max_generations
        self.max_fitness_calls = config.max_fitness_calls
        self.remaining_fitness_calls = self.max_fitness_calls

    def set_archive(self, archive) -> None:
        """
        Sets the archive object for the controller.

        Args:
            archive: An archive object containing elite molecules and associated data.
        """
        self.archive = archive
        return None

    def active(self) -> bool:
        """
        Checks if the optimization process is still active.

        Returns:
            bool: True if the process is active, False otherwise.
        """
        return self.generation < self.max_generations and self.fitness_calls < self.max_fitness_calls

    def update(self) -> None:
        """
        Updates the controller state and archives statistics.

        This includes generating and printing statistics for the current generation,
        writing archive data to a CSV file, and appending basic statistics to another CSV file.
        """
        archive_data = self.get_archive_data()
        molecules = self.archive.incoming_molecules
        archive_statistics = self.calculate_statistics(archive_data)
        surrogate_metrics = self.calculate_surrogate_metrics(molecules)
        self.write_statistics(archive_statistics, surrogate_metrics)
        self.store_statistics(archive_statistics, surrogate_metrics)
        pd.DataFrame(data=archive_data).to_csv("archive_{}.csv".format(self.generation), index=False)
        self.generation = self.generation + 1
        return None

    def add_fitness_calls(self, fitness_calls: int) -> None:
        """
        Adds to the total number of fitness function calls.

        Args:
            fitness_calls: The number of fitness calls to add.
        """
        self.fitness_calls = self.fitness_calls + fitness_calls
        self.remaining_fitness_calls = self.max_fitness_calls - self.fitness_calls
        return None

    def write_statistics(self, statistics: pd.DataFrame, metrics: pd.DataFrame) -> None:
        """
        Prints statistics about the archive to the console.

        Args:
            statistics: A DataFrame containing various archive and quality-diversity metrics.
            metrics: A DataFrame containing surrogate model metrics.
        """
        print("Generation: {}, Size: {:.2f}%, QD Score: {:.2f}".format(self.generation, statistics["coverage"] * 100, statistics["quality_diversity_score"]))
        print("Fitness Max: {:.5f}, Fitness Mean: {:.5f}, Function Calls: {:.0f}".format(statistics["max_fitness"], statistics["mean_fitness"], self.fitness_calls))
        print("Surrogate model overview | Max Error: {:.2f}, MSE: {:.4f}, MAE: {:.4f}".format(metrics["max_err"], metrics["mse"], metrics["mae"]))
        return None

    def store_statistics(self, statistics: pd.DataFrame, metrics: pd.DataFrame) -> None:
        """
        Appends basic archive statistics to a CSV file saved to disk.

        Args:
            statistics: A DataFrame containing various archive and quality-diversity metrics.
            metrics: A DataFrame containing surrogate model metrics.
        """
        if os.path.isfile("statistics.csv"):
            with open("statistics.csv", "a") as file:
                csv.writer(file).writerow(
                    [self.generation]
                    + [
                        statistics["max_fitness"],
                        statistics["mean_fitness"],
                        statistics["quality_diversity_score"],
                        statistics["coverage"] * 100,
                    ]
                    + [self.fitness_calls]
                    + [metrics["max_err"], metrics["mse"], metrics["mae"]]
                )
                file.close()
        else:
            with open("statistics.csv", "w") as file:

                csv.writer(file).writerow(
                    ["generation"]  + ["maximum fitness"]
                    + ["mean fitness"] + ["quality diversity score"]
                    + ["coverage"]  + ["function calls"]
                    + ["max_err"] + ["mse"] + ["mae"]
                )

                csv.writer(file).writerow(
                    [self.generation]
                    + [
                        statistics["max_fitness"],
                        statistics["mean_fitness"],
                        statistics["quality_diversity_score"],
                        statistics["coverage"] * 100,
                    ]
                    + [self.fitness_calls]
                    + [metrics["max_err"], metrics["mse"], metrics["mae"]]
                )

                file.close()
        return None

    def calculate_statistics(self, archive_data) -> Dict:
        """
        Calculates various statistics from the provided archive data.

        Args:
            archive_data: A DataFrame containing data about the elite molecules of the archive.

        Returns:
            Dict: A dictionary containing the calculated statistics.
        """
        coverage = len(archive_data["smiles"]) / self.archive.archive_size
        quality_diversity_score = np.sum(archive_data["fitness"])
        max_fitness, mean_fitness = np.max(archive_data["fitness"]), np.mean(archive_data["fitness"])
        return {
            "coverage": coverage,
            "max_fitness": max_fitness,
            "mean_fitness": mean_fitness,
            "quality_diversity_score": quality_diversity_score,
        }

    def calculate_surrogate_metrics(self, molecules: List[Molecule]) -> Dict:
        """
        Calculates metrics for the surrogate model based on the provided molecules.

        Args:
            molecules (List[Molecule]): A list of Molecule objects.

        Returns:
            Dict: A dictionary containing surrogate model metrics.
        """
        self.memory_of_molecules = self.memory_of_molecules + self.archive.incoming_molecules
        if self.generation == 0:
            max_err, mae, mse = np.nan, np.nan, np.nan
        else:
            fitnesses = np.array([molecule.fitness for molecule in molecules])
            predicted_fitnesses = np.array([molecule.predicted_fitness for molecule in molecules])
            max_err, mae, mse = max_error(fitnesses, predicted_fitnesses), mean_absolute_error(fitnesses, predicted_fitnesses), mean_squared_error(fitnesses, predicted_fitnesses)
        return {"max_err": max_err, "mae": mae, "mse": mse}

    def get_archive_data(self) -> None:
        """
        Retrieves elite molecule attributes and creates a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing attributes of elite molecules.
        """
        elite_molecules = [elite.molecule for elite in self.archive.elites if elite.molecule]
        elite_attributes = [{attr: getattr(molecule, attr) for attr in dir(molecule) if not callable(getattr(molecule, attr)) and not attr.startswith("__")} for molecule in elite_molecules]
        return pd.DataFrame(elite_attributes)

    def store_molecules(self) -> None:
        """
        Stores all molecules in memory to a CSV file.
        """
        molecule_df = pd.DataFrame([{attr: getattr(molecule, attr) for attr in dir(molecule) if not callable(getattr(molecule, attr)) and not attr.startswith("__")} for molecule in self.memory_of_molecules])
        molecule_df.to_csv("molecules.csv", index=False)
        return None


class Archive:
    """
    A composite class containing the current elite molecules in a CVT tree structure.

    This class allows for processing new molecules, sampling existing elite molecules,
    and storing the current state of the archive on disk. The CVT centers are either loaded from
    or saved to cache disk storage.

    Attributes:
        archive_size: The size of the archive.
        archive_accuracy: The accuracy setting for the archive.
        archive_dimensions: The dimensionality of the archive.
        cache_string: The string for cache file naming.
        cvt_location: The location path for the CVT cache file.
        cvt_centers: The centers of the CVT clusters.
        cvt: A KDTree structure for CVT centers.
        elites: A list of Elite objects representing the archive.
        incoming_molecules: A list to store incoming molecules.

    Methods:
        __init__(config, archive_dimensions): Initializes the Archive with the given configuration and dimensions.
        cvt_index(descriptor): Returns the CVT index for the niche nearest to the given descriptor.
        update_niche_index(molecule): Calculates and stores the niche index of a molecule in the molecule object.
        add_to_archive(molecules): Adds molecules to the archive, keeping only the most fit molecule per niche.
        sample(size): Returns a list of elite molecules of the requested size, weighted by fitness.
        sample_pairs(size): Returns a list of pairs of elite molecules of the requested size, weighted by fitness.
    """
    def __init__(self, config, archive_dimensions: int) -> None:
        """
        Initializes the Archive with the given configuration and dimensions.

        Args:
            config: Configuration object containing settings for the archive.
            archive_dimensions: The dimensionality of the archive.
        """
        self.archive_size = config.size
        self.archive_accuracy = config.accuracy
        self.archive_dimensions = archive_dimensions
        self.cache_string = "cache_{}_{}.csv".format(self.archive_dimensions, self.archive_accuracy)
        self.cvt_location = hydra.utils.to_absolute_path("data/cvt/" + self.cache_string)
        if os.path.isfile(self.cvt_location):
            self.cvt_centers = np.loadtxt(self.cvt_location)
        else:
            kmeans = KMeans(n_clusters=self.archive_size)
            kmeans = kmeans.fit(np.random.rand(config.accuracy, self.archive_dimensions))
            self.cvt_centers = kmeans.cluster_centers_
            np.savetxt(self.cvt_location, self.cvt_centers)
        self.cvt = KDTree(self.cvt_centers, metric="euclidean")
        self.elites = [Elite(index) for index, _ in enumerate(self.cvt_centers, start=0)]
        self.incoming_molecules = []
        return None

    def cvt_index(self, descriptor: List[float]) -> int:
        """
        Returns the CVT index for the niche nearest to the given descriptor.

        Args:
            descriptor: A list of descriptor values for the molecule.

        Returns:
            int: The CVT index for the nearest niche.
        """
        return self.cvt.query([descriptor], k=1)[1][0][0]

    def update_niche_index(self, molecule: Molecule) -> Molecule:
        """
        Calculates and stores the niche index of a molecule in the molecule object.

        Args:
            molecule: The molecule for which to calculate the niche index.

        Returns:
            Molecule: The updated molecule with the niche index set.
        """
        molecule.niche_index = self.cvt_index(molecule.descriptor)
        return molecule

    def add_to_archive(self, molecules: List[Molecule]) -> None:
        """
        Adds molecules to the archive, keeping only the most fit molecule per niche.

        Args:
            molecules: A list of molecules to be added to the archive.
        """
        for molecule in molecules:
            self.elites[self.cvt_index(molecule.descriptor)].update(molecule)
        self.incoming_molecules = molecules
        return None

    def sample(self, size: int) -> List[Chem.Mol]:
        """
        Returns a list of elite molecules of the requested size, weighted by fitness.

        Args:
            size: The number of elite molecules to sample.

        Returns:
            List[Chem.Mol]: A list of sampled elite molecules.
        """
        pairs = [(elite.molecule, elite.molecule.fitness) for elite in self.elites if elite.molecule]
        molecules, weights = map(list, zip(*pairs))
        return random.choices(molecules, k=size, weights=weights)

    def sample_pairs(self, size: int) -> List[Tuple[Chem.Mol, Chem.Mol]]:
        """
        Returns a list of pairs of elite molecules of the requested size, weighted by fitness.

        Args:
            size: The number of pairs of elite molecules to sample.

        Returns:
            List[Tuple[Chem.Mol, Chem.Mol]]: A list of sampled pairs of elite molecules.
        """
        pairs = [(elite.molecule, elite.molecule.fitness) for elite in self.elites if elite.molecule]
        molecules, weights = map(list, zip(*pairs))
        sample_molecules = random.choices(molecules, k=size, weights=weights)
        sample_pairs = np.random.choice(list(filter(None, sample_molecules)), size=(size, 2), replace=True)
        sample_pairs = [tuple(sample_pair) for sample_pair in sample_pairs]
        return sample_pairs


class Arbiter:
    """
    A catalog class containing different drug-like filters for small molecules.

    This class includes the option to run structural filters from ChEMBL.

    Attributes:
        cache_smiles: A list to store SMILES strings of molecules for duplication checks.
        rules_dict: A DataFrame containing filter rules loaded from a CSV file.
        rules_list: A list of SMARTS strings for the filter rules.
        tolerance_list: A list of tolerance values for the filter rules.
        pattern_list: A list of RDKit molecule patterns for the filter rules.

    Methods:
        __init__(arbiter_config): Initializes the Arbiter with the given configuration.
        __call__(molecules): Applies the chosen filters to a list of molecules and removes duplicates.
        unique_molecules(molecules): Checks if a molecule in a list of molecules is duplicated.
        molecule_filter(molecular_graph): Checks if a given molecular structure passes through the chosen filters.
        toxicity(molecular_graph): Checks if a given molecule fails the structural filters.
        hologenicity(molecular_graph): Checks if a given molecule fails the hologenicity filters.
        ring_infraction(molecular_graph): Checks if a given molecule fails the ring infraction filters.
        veber_infraction(molecular_graph): Checks if a given molecule fails the veber infraction filters.
    """
    def __init__(self, config) -> None:
        """
        Initializes the Arbiter with the given configuration.

        Args:
            config: Configuration object containing settings for the filters.
        """
        self.cache_smiles = []
        self.rules_dict = pd.read_csv(hydra.utils.to_absolute_path("data/smarts/alert_collection.csv"))
        self.rules_dict = self.rules_dict[self.rules_dict.rule_set_name.isin(config.rules)]
        self.rules_list = self.rules_dict["smarts"].values.tolist()
        self.tolerance_list = pd.to_numeric(self.rules_dict["max"]).values.tolist()
        self.pattern_list = [Chem.MolFromSmarts(smarts) for smarts in self.rules_list]

    def __call__(self, molecules: List[Molecule]):
        """
        Applies the chosen filters (hologenicity, veber infractions, ChEMBL structural alerts, etc.)
        to a list of molecules and removes duplicates.

        Args:
            molecules: A list of molecules to be filtered.

        Returns:
            List[Molecule]: A list of filtered molecules.
        """

        filtered_molecules = []
        molecules = self.unique_molecules(molecules)
        for molecule in molecules:
            molecular_graph = Chem.MolFromSmiles(molecule.smiles)
            if self.molecule_filter(molecular_graph):
                filtered_molecules.append(molecule)
        return filtered_molecules

    def unique_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Checks if a molecule in a list of molecules is duplicated, either in this batch or before.

        Args:
            molecules: A list of molecules to check for duplicates.

        Returns:
            List[Molecule]: A list of unique molecules.
        """
        unique_molecules = []
        for molecule in molecules:
            if molecule.smiles not in self.cache_smiles:
                unique_molecules.append(molecule)
                self.cache_smiles.append(molecule.smiles)
        return unique_molecules

    def molecule_filter(self, molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecular structure passes through the chosen filters (hologenicity,
        veber infractions, ChEMBL structural alerts, etc.).

        Args:
            molecular_graph: The molecular graph to be checked.

        Returns:
            bool: True if the molecule passes all filters, False otherwise.
        """
        toxicity = self.toxicity(molecular_graph)
        hologenicity = self.hologenicity(molecular_graph)
        veber_infraction = self.veber_infraction(molecular_graph)
        validity = not (toxicity or hologenicity or veber_infraction)
        if molecular_graph.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
            ring_infraction = self.ring_infraction(molecular_graph)
            validity = validity and not (ring_infraction)
        return validity

    def toxicity(self, molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the structural filters.

        Args:
            molecular_graph: The molecular graph to be checked.

        Returns:
            bool: True if the molecule fails the structural filters, False otherwise.
        """
        for pattern, tolerance in zip(self.pattern_list, self.tolerance_list):
            if len(molecular_graph.GetSubstructMatches(pattern)) > tolerance:
                return True
        return False

    @staticmethod
    def hologenicity(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the hologenicity filters.

        Args:
            molecular_graph: The molecular graph to be checked.

        Returns:
            bool: True if the molecule fails the hologenicity filters, False otherwise.
        """
        fluorine_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts("[F]"))) > 6
        bromide_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts("[Br]"))) > 3
        chlorine_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts("[Cl]"))) > 3
        return chlorine_saturation or bromide_saturation or fluorine_saturation

    @staticmethod
    def ring_infraction(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the ring infraction filters.

        Args:
            molecular_graph: The molecular graph to be checked.

        Returns:
            bool: True if the molecule fails the ring infraction filters, False otherwise.
        """
        ring_allene = molecular_graph.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
        macro_cycle = max([len(j) for j in molecular_graph.GetRingInfo().AtomRings()]) > 6
        double_bond_in_small_ring = molecular_graph.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))
        return ring_allene or macro_cycle or double_bond_in_small_ring

    @staticmethod
    def veber_infraction(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the veber infraction filters.

        Args:
            molecular_graph: The molecular graph to be checked.

        Returns:
            bool: True if the molecule fails the veber infraction filters, False otherwise.
        """
        rotatable_bond_saturation = Lipinski.NumRotatableBonds(molecular_graph) > 10
        hydrogen_bond_saturation = Lipinski.NumHAcceptors(molecular_graph) + Lipinski.NumHDonors(molecular_graph) > 10
        return rotatable_bond_saturation or hydrogen_bond_saturation
