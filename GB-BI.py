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
        molecules = self.arbiter(molecules)
        molecules = self.calculate_descriptors(molecules)
        molecules = self.apply_acquisition(molecules)
        molecules = self.calculate_fitnesses(molecules)
        return molecules

    def calculate_descriptors(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = [self.descriptor(molecule) for molecule in molecules]
        molecules = [molecule for molecule in molecules if all(1.0 > property > 0.0 for property in molecule.descriptor)]
        molecules = [self.archive.update_niche_index(molecule) for molecule in molecules]
        return molecules

    def calculate_fitnesses(self, molecules: List[Molecule]) -> List[Molecule]:
        if self.controller.remaining_fitness_calls >= len(molecules):
            molecules = [self.fitness(molecule) for molecule in molecules]
        else:
            molecules = molecules[: self.controller.remaining_fitness_calls]
            molecules = [self.fitness(molecule) for molecule in molecules]
        self.controller.add_fitness_calls(len(molecules))
        return molecules

    def apply_acquisition(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = self.surrogate(molecules)
        molecules = self.acquisition(molecules)
        return molecules

    def initial_population(self) -> None:
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
