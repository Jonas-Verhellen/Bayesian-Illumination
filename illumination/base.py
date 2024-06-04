from typing import List, Tuple
from dataclasses import dataclass

@dataclass()
class Molecule:
    """
    The Molecule dataclass represents a molecule with various attributes used in optimization.

    Attributes:
        smiles: The SMILES string representation of the molecule.
        pedigree: The lineage or origin of the molecule.
        fitness: The fitness score of the molecule.
        niche_index: The index of the niche the molecule belongs to.
        descriptor: The list of physicochemical descriptors of the molecule.
        predicted_fitness: The predicted fitness score from the surrogate model.
        predicted_uncertainty: The uncertainty associated with the predicted fitness.
        acquisition_value: The value assigned by the acquisition function.
    """
    smiles: str
    pedigree: Tuple[str, str, str]
    fitness: float = None
    niche_index: int = None
    descriptor: List[float] = None
    predicted_fitness: int = 0
    predicted_uncertainty: int = 0
    acquisition_value: float = None

class Elite:
    """
    The Elite class represents a phenotypic elite molecule in the Bayesian Illumination algorithm.
    An elite molecule is one that has the highest fitness within its niche. The class manages
    the storage and update of this elite molecule.

    Attributes:
        index: The index of the niche that this elite belongs to.
        molecule: The molecule with the highest fitness in this niche.
        fitness: The fitness score of the elite molecule.
    """
    def __init__(self, index: int) -> None:
        """
        Initializes the Elite object with the given niche index.

        Args:
            index: The index of the niche.
        """
        self.index = index
        self.molecule = None
        self.fitness = 0.0

    def update(self, molecule: Molecule) -> None:
        """
        Updates the elite molecule if the new molecule has a higher fitness.

        Args:
            molecule: The new molecule to be considered as the potential elite.
        """
        if self.molecule is None or (molecule.fitness - self.fitness) > 0.0:
            self.molecule = molecule
            self.fitness = molecule.fitness
        return None
