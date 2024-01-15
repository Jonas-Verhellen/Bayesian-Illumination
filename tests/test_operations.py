import pytest
import omegaconf
from rdkit import Chem
from argenomic.operations import Mutator, Crossover
from argenomic.base import Molecule
 
@pytest.fixture
def default_mutator():
    '''
    Returns an instance of a mutator.
    '''
    configuration_file = omegaconf.OmegaConf.load("./tests/test_config.yaml")
    return Mutator(configuration_file.mutator)

@pytest.fixture
def default_crossover():
    '''
    Returns an instance of a crossover.
    '''
    return Crossover()

@pytest.fixture
def default_molecules():
    smiles_list = ["Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCC(=O)O", "CC1=CC(Cl)=CC(C(=O)N[C@@H]2C[C@@H]3CCCC[C@@H]32)=C1C"]
    pedigree = ("database", "no reaction", "no parent")   
    molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list]
    return molecules

@pytest.mark.xfail
def test_default_mutator(default_mutator, default_molecules):
    '''
    Tests the action of the mutator. May fail occasionally due to stochasticity.
    The result of this test is reported separtely.
    '''
    molecules = default_molecules
    for molecule in molecules:
        assert len(default_mutator(molecule)) > 0

def test_default_crossover(default_crossover, default_molecules):
    '''
    Tests the action of the crossover.
    '''
    molecules = default_molecules
    assert len(default_crossover(molecules)) > 0
