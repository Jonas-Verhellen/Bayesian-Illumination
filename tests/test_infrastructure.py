import pytest
import omegaconf
import numpy as np
from rdkit import Chem
from argenomic.infrastructure import Archive, Arbiter
from argenomic.base import Molecule

@pytest.fixture
def default_archive():
    configuration_file = omegaconf.OmegaConf.load("./tests/test_config.yaml")
    '''
    Returns an archive instance of a 150 niches, spanned by ExactMolWt and MolLogP.
    '''
    return Archive(configuration_file.archive, configuration_file.descriptor)

@pytest.fixture
def default_arbiter():
    configuration_file = omegaconf.OmegaConf.load("./tests/test_config.yaml")
    '''
    Returns an arbiter instance, initialised with GSK structural alerts.
    '''
    return Arbiter(configuration_file.arbiter)

@pytest.fixture
def default_molecules():
    smiles_list = ["Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCC(=O)O", "CC1=CC(Cl)=CC(C(=O)N[C@@H]2C[C@@H]3CCCC[C@@H]32)=C1C"]
    pedigree = ("database", "no reaction", "no parent")   
    molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree, fitness=0.2 , descriptor=[np.random.rand(), np.random.rand()]) for smiles in smiles_list]
    return molecules

def test_default_archive(default_archive, default_molecules):
    default_archive.add_to_archive(default_molecules)
    assert len(default_archive.sample(2)) == 2
    assert len(default_archive.sample_pairs(5)) == 5

def test_default_arbiter(default_arbiter, default_molecules):
    assert len(default_arbiter.pattern_list) == 55
    assert len(default_arbiter(default_molecules)) == 2
