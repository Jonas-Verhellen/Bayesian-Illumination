import pytest
from unittest.mock import Mock, patch
from illumination.mechanism import Descriptor, Surrogate, Fitness, Acquisition

# Sample configuration for testing
class Config:
    def __init__(self, type_, ranges=None, properties=None):
        self.type = type_
        self.ranges = ranges if ranges else []
        self.properties = properties if properties else []

@pytest.fixture
def descriptor_config():
    return Config(
        type_="Descriptor",
        ranges=[[0, 1], [0, 10], [0, 100]],
        properties=[
            "Descriptors.MolWt",
            "Descriptors.NumHDonors",
            "Descriptors.NumHAcceptors"
        ]
    )

@pytest.fixture
def surrogate_config():
    return Config(type_="String")

@pytest.fixture
def fitness_config():
    return Config(type_="Fingerprint")

@pytest.fixture
def acquisition_config():
    return Config(type_="Mean")

# Test Descriptor class
def test_descriptor_init(descriptor_config):
    descriptor = Descriptor(descriptor_config)
    assert len(descriptor.properties) == 3
    assert descriptor.dimensionality == 3

def test_descriptor_call(descriptor_config):
    descriptor = Descriptor(descriptor_config)
    molecule = Mock(smiles='CCO')
    updated_molecule = descriptor(molecule)
    assert hasattr(updated_molecule, 'descriptor')
    assert len(updated_molecule.descriptor) == 3

# Test Surrogate class
def test_surrogate_init(surrogate_config):
    with patch('illumination.mechanism.String_Surrogate') as MockSurrogate:
        surrogate = Surrogate.__new__(Surrogate, surrogate_config) # noqa
        MockSurrogate.assert_called_once_with(surrogate_config)

def test_surrogate_invalid_type():
    config = Config(type_="InvalidType")
    with pytest.raises(ValueError):
        Surrogate.__new__(Surrogate, config)

# Test Fitness class
def test_fitness_init(fitness_config):
    with patch('illumination.mechanism.Fingerprint_Fitness') as MockFitness:
        fitness = Fitness.__new__(Fitness, fitness_config) # noqa
        MockFitness.assert_called_once_with(fitness_config)

def test_fitness_invalid_type():
    config = Config(type_="InvalidType")
    with pytest.raises(ValueError):
        Fitness.__new__(Fitness, config)

# Test Acquisition class
def test_acquisition_init(acquisition_config):
    with patch('illumination.mechanism.Posterior_Mean') as MockAcquisition:
        acquisition = Acquisition.__new__(Acquisition, acquisition_config) # noqa
        MockAcquisition.assert_called_once_with(acquisition_config)

def test_acquisition_invalid_type():
    config = Config(type_="InvalidType")
    with pytest.raises(ValueError):
        Acquisition.__new__(Acquisition, config)

if __name__ == "__main__":
    pytest.main()
