import pytest
import numpy as np
from unittest.mock import Mock, patch

from illumination.infrastructure import Archive

# Sample configuration for testing
class Config:
    size = 150
    accuracy = 25000

@pytest.fixture
def config():
    return Config()

# Fixture for setting up the Archive object
@pytest.fixture
def archive(config):
    with patch('illumination.infrastructure.hydra.utils.to_absolute_path', return_value='./data/cache_2_25000.csv'):
        with patch('illumination.infrastructure.KMeans'):
            return Archive(config, 2)

# Mock data for the CVT centers
@pytest.fixture
def mock_cvt_centers():
    return np.array([[0.1, 0.2], [0.9, 0.8]])

# Mock data for the molecules
@pytest.fixture
def mock_molecules():
    molecule1 = Mock(descriptor=[0.1, 0.2], fitness=0.9, niche_index=None)
    molecule2 = Mock(descriptor=[0.9, 0.8], fitness=0.8, niche_index=None)
    return [molecule1, molecule2]

# Test __init__ method
def test_init(archive, mock_cvt_centers):
    assert archive.archive_size == 150
    assert archive.archive_accuracy == 25000
    assert archive.archive_dimensions == 2
    assert archive.cvt_centers.shape == (150, 2)
    assert isinstance(archive.elites, list)
    assert len(archive.elites) == 150

# Test cvt_index method
def test_cvt_index(archive, mock_cvt_centers):
    with patch.object(archive, 'cvt', Mock()):
        archive.cvt.query = Mock(return_value=([0.1], [[0]]))
        index = archive.cvt_index([0.1, 0.2])
        assert index == 0

# Test update_niche_index method
def test_update_niche_index(archive, mock_molecules):
    with patch.object(archive, 'cvt_index', return_value=1):
        molecule = archive.update_niche_index(mock_molecules[0])
        assert molecule.niche_index == 1

# Test add_to_archive method
def test_add_to_archive(archive, mock_molecules):
    archive.elites[0] = Mock()
    archive.elites[1] = Mock()
    with patch.object(archive, 'cvt_index', side_effect=[0, 1]):
        archive.add_to_archive(mock_molecules)
        archive.elites[0].update.assert_called_once_with(mock_molecules[0])
        archive.elites[1].update.assert_called_once_with(mock_molecules[1])
        assert archive.incoming_molecules == mock_molecules

# Test sample method
def test_sample(archive, mock_molecules):
    for idx, elite in enumerate(archive.elites[:2]):
        elite.molecule = mock_molecules[idx]
    samples = archive.sample(2)
    assert len(samples) == 2
    assert all(isinstance(sample, Mock) for sample in samples)

# Test sample_pairs method
def test_sample_pairs(archive, mock_molecules):
    for idx, elite in enumerate(archive.elites[:2]):
        elite.molecule = mock_molecules[idx]
    pairs = archive.sample_pairs(2)
    assert len(pairs) == 2
    assert all(isinstance(pair, tuple) for pair in pairs)
    assert all(isinstance(molecule, Mock) for pair in pairs for molecule in pair)

if __name__ == "__main__":
    pytest.main()
