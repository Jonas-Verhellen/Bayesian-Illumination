import pytest
from illumination.functions.surrogate import Fingerprint_Surrogate, String_Surrogate

# Mock configuration class for testing
class Config:
    def __init__(self, representation="ECFP4", max_ngram=5):
        self.representation = representation
        self.max_ngram = max_ngram

# Sample Molecule class
class Molecule:
    def __init__(self, smiles, fitness):
        self.smiles = smiles
        self.fitness = fitness
        self.predicted_fitness = None
        self.predicted_uncertainty = None


# Test Fingerprint_Surrogate class
def test_fingerprint_surrogate():
    config = Config()
    surrogate = Fingerprint_Surrogate(config)
    molecules = [Molecule("CCO", 0.5), Molecule("CCC", 0.8)]
    surrogate.add_to_prior_data(molecules)
    encodings = surrogate.calculate_encodings(molecules)
    assert len(encodings) == len(molecules)

# Test String_Surrogate class
def test_string_surrogate():
    config = Config(representation="Smiles", max_ngram=2)
    surrogate = String_Surrogate(config)
    molecules = [Molecule("CCO", 0.5), Molecule("CCC", 0.8)]
    surrogate.add_to_prior_data(molecules)
    encodings = surrogate.calculate_encodings(molecules)
    assert len(encodings) == len(molecules)

if __name__ == "__main__":
    pytest.main()
