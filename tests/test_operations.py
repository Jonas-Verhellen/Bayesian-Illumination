import pytest
from unittest.mock import patch
import pandas as pd
from illumination.operations import Generator, Mutator, Crossover

# Sample configuration for testing
class Config:
    def __init__(self, mutation_data=None, batch_size=10, initial_data=None, initial_size=5):
        self.mutation_data = mutation_data
        self.batch_size = batch_size
        self.initial_data = initial_data
        self.initial_size = initial_size

@pytest.fixture
def generator_config():
    return Config(
        mutation_data="./data/mutation_collection.tsv",
        batch_size=10,
        initial_data="./data/ChEMBL_approved_650.smi",
        initial_size=2
    )

@pytest.fixture
def mutator_config():
    return Config(mutation_data="./data/mutation_collection.tsv")

# Sample Molecule class
class Molecule:
    def __init__(self, smiles, pedigree):
        self.smiles = smiles
        self.pedigree = pedigree

# Test Generator class
def test_generator_init(generator_config):
    generator = Generator(generator_config)
    assert generator.batch_size == 10
    assert generator.initial_data == "./data/ChEMBL_approved_650.smi"
    assert generator.initial_size == 2

def test_mutator_init(mutator_config):
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({
            "smarts": ["[C:1]>>[C:1]", "[O:1]>>[O:1]"],
            "probability": [0.5, 0.5]
        })
        mutator = Mutator(mutator_config.mutation_data)
        assert len(mutator.mutation_data) == 2

def test_mutator_call(mutator_config):
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({
            "smarts": ["[C:1]>>[C:1]", "[O:1]>>[O:1]"],
            "probability": [0.5, 0.5]
        })
        mutator = Mutator(mutator_config.mutation_data)
        molecule = Molecule("CCO", ("database", "no reaction", "database"))
        mutated_molecules = mutator(molecule)
        assert isinstance(mutated_molecules, list)

def test_crossover_init():
    crossover = Crossover()
    assert crossover is not None

def test_crossover_call():
    crossover = Crossover()
    molecule1 = Molecule("CCO", ("database", "no reaction", "database"))
    molecule2 = Molecule("CCC", ("database", "no reaction", "database"))
    new_molecules = crossover((molecule1, molecule2))
    assert isinstance(new_molecules, list)

if __name__ == "__main__":
    pytest.main()
