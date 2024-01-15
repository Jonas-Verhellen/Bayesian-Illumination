import pytest
import omegaconf
from rdkit import Chem
from argenomic.mechanism import Descriptor, Fitness
from argenomic.base import Molecule

@pytest.fixture
def default_descriptor():
    '''
    Returns a descriptor instance, set-up to calculate normalised ExactMolWt and MolLogP.
    '''
    configuration_file = omegaconf.OmegaConf.load("./tests/test_config.yaml")
    return Descriptor(configuration_file.descriptor)

@pytest.fixture
def default_fitness():
    configuration_file = omegaconf.OmegaConf.load("./tests/test_config.yaml")
    return Fitness(configuration_file.fitness)

@pytest.fixture
def default_molecules():
    smiles_list = ["Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCC(=O)O", "CC1=CC(Cl)=CC(C(=O)N[C@@H]2C[C@@H]3CCCC[C@@H]32)=C1C"]
    pedigree = ("database", "no reaction", "no parent")   
    molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list]
    return molecules

def test_default_descriptor(default_descriptor, default_molecules):
    for molecule in default_molecules:
        molecule = default_descriptor(molecule)
        for descriptor in molecule.descriptors:
            assert 0.00 <= descriptor
            assert descriptor <= 1.00
    

def test_default_descriptor(default_fitness, default_molecules):
    for molecule in default_molecules:
        molecule = default_fitness(molecule)
        assert 0.00 <= molecule.fitness
        assert molecule.fitness <= 1.00
