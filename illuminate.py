import hydra
import torch
import numpy as np
import pandas as pd
from typing import List

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from argenomic.base import Molecule
from argenomic.operations import Mutator, Crossover
from argenomic.infrastructure import Arbiter, Archive, Printer
from argenomic.mechanism import Fitness, Descriptor, Surrogate, Acquisition

from itertools import groupby

class Illuminate:
    def __init__(self, config) -> None:
        self.data_file = config.data_file
        self.batch_size = config.batch_size
        self.initial_size = config.initial_size

        self.arbiter = Arbiter(config.arbiter)
        self.fitness = Fitness(config.fitness)
        self.mutator = Mutator(config.mutator)
        self.crossover = Crossover()

        self.descriptor = Descriptor(config.descriptor)
        self.archive = Archive(config.archive, config.descriptor)
        self.surrogate = Surrogate(config.surrogate)

        self.printer = Printer(self.archive)
        self.acquisition = Acquisition(self.archive, config.acquisition)

        self.generations = 0
        self.fitness_calls = 0
        self.max_generations = config.max_generations
        self.max_fitness_calls = config.max_fitness_calls
        return None

    def __call__(self) -> None:
        self.self_similiarity()
        self.initial_population()
        while self.generations <= self.max_generations and self.fitness_calls <= self.max_fitness_calls:
            molecules = self.generate_molecules()
            molecules = self.process_molecules(molecules)
            self.archive.add_to_archive(molecules)
            self.printer(self.generations, self.fitness_calls)
            self.surrogate.update_model(molecules)
            self.generations = self.generations + 1
        return None

    def initial_population(self) -> None:
        molecules = self.arbiter(self.load_from_database())
        molecules = self.calculate_descriptors(molecules)
        molecules = self.calculate_fingerprints(molecules)
        molecules = self.calculate_fitnesses(molecules)
        self.archive.add_to_archive(molecules)
        self.printer(self.generations, self.fitness_calls)
        self.surrogate.intitialise_model(molecules)
        return None
    
    def self_similiarity(self) -> None:
        pedigree = ("database", "no reaction", "target")
        self_similiarity = self.fitness(Molecule(Chem.CanonSmiles(Chem.MolToSmiles(self.fitness.target)), pedigree)).fitness 
        print(f"Self similarity: {self_similiarity}")
        return None

    def load_from_database(self) -> List[Molecule]:
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.data_file))
        smiles_list = dataframe['smiles'].sample(n=self.initial_size).tolist()
        pedigree = ("database", "no reaction", "database")   
        molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list]
        return molecules

    def generate_molecules(self) -> List[Molecule]:
        molecules = []
        molecule_samples = self.archive.sample(self.batch_size)
        molecule_sample_pairs = self.archive.sample_pairs(self.batch_size)
        for molecule in molecule_samples:
            molecules.extend(self.mutator(molecule)) 
        for molecule_pair in molecule_sample_pairs:
            molecules.extend(self.crossover(molecule_pair)) 
        return molecules

    def process_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = self.arbiter(molecules)
        molecules = self.calculate_descriptors(molecules)
        molecules = self.calculate_fingerprints(molecules)
        molecules = self.apply_acquisition_function(molecules)
        molecules = self.calculate_fitnesses(molecules)
        return molecules
    
    def calculate_fingerprints(self, molecules: List[Molecule]) -> List[Molecule]:
        fp_generator = GetMorganGenerator(radius=3, fpSize=2048)
        for molecule in molecules:
            molecular_graph = Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles))
            molecule.fingerprint = np.array(fp_generator.GetFingerprint(molecular_graph))
        return molecules
    
    def calculate_descriptors(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = [self.descriptor(molecule) for molecule in molecules]
        molecules = [molecule for molecule in molecules if all(1.0 > property > 0.0 for property in molecule.descriptor)]
        return molecules
    
    def apply_acquisition_function(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = self.surrogate(molecules)
        molecules = self.acquisition(molecules)
        molecules = self.molecule_selection(molecules)
        return molecules
                
    @staticmethod
    def molecule_selection(molecules: List[Molecule]) -> List[Molecule]:
        molecules.sort(key = lambda molecule: molecule.niche_index)
        grouped_molecules = {index: list(molecule_group) for index, molecule_group in groupby(molecules, key = lambda molecule: molecule.niche_index)}
        molecules = [max(molecule_group, key = lambda molecule: molecule.acquisition_value) for molecule_group in grouped_molecules.values()]
        return molecules

    def calculate_fitnesses(self, molecules: List[Molecule]) -> List[Molecule]:
        remaining_fitness_calls = self.max_fitness_calls - self.fitness_calls
        if remaining_fitness_calls >= len(molecules):
            molecules = [self.fitness(molecule) for molecule in molecules]
            self.fitness_calls += len(molecules)        
        else:
            molecules = molecules[:remaining_fitness_calls]
            molecules = [self.fitness(molecule) for molecule in molecules]
            self.fitness_calls += len(molecules) 
        return molecules
    
@hydra.main(config_path="configuration", config_name="config.yaml")
def launch(config) -> None:
    print(config)
    current_instance = Illuminate(config)
    current_instance()
    current_instance.client.close()

if __name__ == "__main__":
    launch()
