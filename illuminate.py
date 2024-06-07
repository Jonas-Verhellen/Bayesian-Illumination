import hydra
import logging
from typing import List
from omegaconf import OmegaConf

from illumination.base import Molecule
from illumination.operations import Generator
from illumination.infrastructure import Arbiter, Archive, Controller
from illumination.mechanism import Fitness, Descriptor, Surrogate, Acquisition


class Illuminate:
    def __init__(self, config) -> None:
        """
        Initialize the Illuminate class with the given configuration. The Illuminate
        class implements a graph-based Bayesian illumination algorithm for optimizing
        small molecules. The algorithm begins by initializing the population from a
        given database. Subsequent populations are formed by mutations and crossovers.
        Molecules are filtered based on structural criteria and physicochemical descriptors
        are calculated for the remaining ones. Those molecules are assigned to niches
        based on their descriptors. Surrogate models predict the fitness of molecules,
        and acquisition functions guide the selection of promising molecules. Selected
        molecules are compared in direct evolutionary competition with current
        niche occupants. The process continues until a predetermined fitness function
        budget is exhausted or a maximum number generations is reached.

        Args:
            config: Configuration object containing settings for all components.
        """
        self.arbiter = Arbiter(config.arbiter)
        self.fitness = Fitness(config.fitness)
        self.generator = Generator(config.generator)
        self.descriptor = Descriptor(config.descriptor)

        self.surrogate = Surrogate(config.surrogate)
        self.acquisition = Acquisition(config.acquisition)
        self.controller = Controller(config.controller)

        self.archive = Archive(config.archive, self.descriptor.dimensionality)

        self.generator.set_archive(self.archive)
        self.controller.set_archive(self.archive)
        self.acquisition.set_archive(self.archive)
        return None

    def __call__(self) -> None:
        """
        Executes the Bayesian Illumination optimization process.

        This function initializes the population and iteratively generates, processes,
        and evaluates molecules until the controller deactivates when the maximum amount
        of fitness calls or generations is reached. It then stores the final archive
        of molecules on disk.
        """
        self.initial_population()
        while self.controller.active():
            molecules = self.generator()
            molecules = self.process_molecules(molecules)
            self.archive.add_to_archive(molecules)
            self.surrogate.add_to_prior_data(molecules)
            self.controller.update()
        self.controller.store_molecules()
        return None

    def process_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Process a list of molecules by fitlering out unwanted or invalid structures,
        calcualting phsyichcemical descriptors, applying the acquisition rules based on
        the surrogate model and calculating the actual fitness for the remaining molecules.

        Args:
            molecules: List of molecules to be processed.

        Returns:
            List of processed molecules.
        """
        molecules = self.arbiter(molecules)
        molecules = self.calculate_descriptors(molecules)
        molecules = self.apply_acquisition(molecules)
        molecules = self.calculate_fitnesses(molecules)
        return molecules

    def calculate_descriptors(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Calculate descriptors for a list of molecules and update their niche index.
        Removes the molcules that all outside the physicochemical ranges of the archive
        as specified in the configuration file.

        Args:
            molecules: List of molecules.

        Returns:
            List of molecules with valid descriptors and updated niche indices.
        """
        molecules = [self.descriptor(molecule) for molecule in molecules]
        molecules = [molecule for molecule in molecules if all(1.0 > property > 0.0 for property in molecule.descriptor)]
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        return molecules

    def calculate_fitnesses(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Calculate fitnesses for a list of molecules. Splits the incoming list
        in the case that the maximum amount of fitness calls would be exceeded.

        Args:
            molecules: List of molecules.

        Returns:
            List of molecules with calculated fitnesses.
        """

        if self.controller.remaining_fitness_calls >= len(molecules):
            molecules = [self.fitness(molecule) for molecule in molecules]
        else:
            molecules = molecules[: self.controller.remaining_fitness_calls]
            molecules = [self.fitness(molecule) for molecule in molecules]
        self.controller.add_fitness_calls(len(molecules))
        return molecules

    def apply_acquisition(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Apply the surrogate function to a list of molecules and filter the
        molecules based on their acquisition function values.

        Args:
            molecules: List of molecules.

        Returns:
            List of molecules after acquisition function application.
        """
        molecules = self.surrogate(molecules)
        molecules = self.acquisition(molecules)
        return molecules

    def initial_population(self) -> None:
        """
        Generate and process the initial population of molecules.

        This function loads initial molecules from a database, processes them through
        the arbiter, applies the descriptor and fitness calculations, adds them to
        the archive and uses them es a prior for the surrogate model. Finally,
        it updates the controller state.
        """
        molecules = self.generator.load_from_database()
        molecules = self.arbiter(molecules)
        molecules = self.calculate_descriptors(molecules)
        molecules = self.calculate_fitnesses(molecules)
        self.archive.add_to_archive(molecules)
        self.surrogate.add_to_prior_data(molecules)
        self.controller.update()
        return None


@hydra.main(config_path="configuration", config_name="config.yaml")
def launch(config) -> None:
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(config))
    current_instance = Illuminate(config)
    current_instance()


if __name__ == "__main__":
    launch()
