import pytest
import pandas as pd
from rdkit import Chem
from unittest.mock import Mock, patch

from illumination.infrastructure import Arbiter

# Sample configuration for testing
class Config:
    rules = ['structural_alert', 'hologenicity', 'veber']

@pytest.fixture
def config():
    return Config()

# Fixture for setting up the Arbiter object
@pytest.fixture
def arbiter(config):
    with patch('illumination.infrastructure.hydra.utils.to_absolute_path', return_value='data/smarts/alert_collection.csv'):
        with patch('illumination.infrastructure.pd.read_csv') as mock_read_csv:
            mock_data = {
                'rule_set_name': ['structural_alert', 'hologenicity', 'veber'],
                'smarts': ['[C]', '[F]', '[O]'],
                'max': [1, 6, 10]
            }
            mock_read_csv.return_value = pd.DataFrame(mock_data)
            return Arbiter(config)

# Mock data for the molecules
@pytest.fixture
def mock_molecules():
    molecule1 = Mock(smiles='CCO')
    molecule2 = Mock(smiles='CCF')
    return [molecule1, molecule2]

# Test __init__ method
def test_init(arbiter):
    assert len(arbiter.rules_list) == 3
    assert arbiter.rules_list == ['[C]', '[F]', '[O]']
    assert arbiter.tolerance_list == [1, 6, 10]
    assert len(arbiter.pattern_list) == 3

# Test unique_molecules method
def test_unique_molecules(arbiter, mock_molecules):
    unique_molecules = arbiter.unique_molecules(mock_molecules)
    assert len(unique_molecules) == 2
    assert arbiter.cache_smiles == ['CCO', 'CCF']

# Test molecule_filter method
def test_molecule_filter(arbiter):
    molecular_graph = Chem.MolFromSmiles('CCO')
    with patch.object(arbiter, 'toxicity', return_value=False):
        with patch.object(arbiter, 'hologenicity', return_value=False):
            with patch.object(arbiter, 'ring_infraction', return_value=False):
                with patch.object(arbiter, 'veber_infraction', return_value=False):
                    assert arbiter.molecule_filter(molecular_graph) is True

# Test toxicity method
def test_toxicity(arbiter):
    molecular_graph = Chem.MolFromSmiles('CCO')
    assert arbiter.toxicity(molecular_graph) is True

# Test hologenicity method
def test_hologenicity(arbiter):
    molecular_graph = Chem.MolFromSmiles('CCCCCCC')
    assert arbiter.hologenicity(molecular_graph) is False

# Test ring_infraction method
def test_ring_infraction(arbiter):
    molecular_graph = Chem.MolFromSmiles('C1CCC1')
    assert arbiter.ring_infraction(molecular_graph) is False

# Test veber_infraction method
def test_veber_infraction(arbiter):
    molecular_graph = Chem.MolFromSmiles('CCO')
    assert arbiter.veber_infraction(molecular_graph) is False

# Test __call__ method
def test_call(arbiter, mock_molecules):
    with patch.object(arbiter, 'unique_molecules', return_value=mock_molecules):
        with patch.object(arbiter, 'molecule_filter', return_value=True):
            filtered_molecules = arbiter(mock_molecules)
            assert len(filtered_molecules) == 2

if __name__ == "__main__":
    pytest.main()
