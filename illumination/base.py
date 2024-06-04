from typing import List, Tuple
from dataclasses import dataclass


class Elite:
    def __init__(self, index):
        self.index = index
        self.molecule = None
        self.fitness = 0.0

    def update(self, molecule):
        if self.molecule is None or (molecule.fitness - self.fitness) > 0.0:
            self.molecule = molecule
            self.fitness = molecule.fitness
        return None


@dataclass()
class Molecule:
    smiles: str
    pedigree: Tuple[str, str, str]
    fitness: float = None
    niche_index: int = None
    descriptor: List[float] = None
    predicted_fitness: int = 0
    predicted_uncertainty: int = 0
    acquisition_value: float = None
