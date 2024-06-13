import pytest
import torch
from unittest.mock import Mock
import numpy as np
from scipy.stats import norm
from illumination.functions.acquisition import Posterior_Mean, Upper_Confidence_Bound, Expected_Improvement, Log_Expected_Improvement
from botorch.acquisition.analytic import _log_ei_helper

# Sample configuration for testing
class Config:
    def __init__(self, beta=None):
        self.beta = beta

# Sample Molecule class
class Molecule:
    def __init__(self, smiles, niche_index, predicted_fitness, predicted_uncertainty):
        self.smiles = smiles
        self.niche_index = niche_index
        self.predicted_fitness = predicted_fitness
        self.predicted_uncertainty = predicted_uncertainty
        self.acquisition_value = None
        self.fitness = None

# Test Posterior_Mean class
def test_posterior_mean():
    config = Config()
    acquisition = Posterior_Mean(config)
    molecules = [Molecule("CCO", 0, 1.0, 0.1), Molecule("CCC", 1, 2.0, 0.2)]
    acquisition.calculate_acquisition_value(molecules)
    assert molecules[0].acquisition_value == 1.0
    assert molecules[1].acquisition_value == 2.0

# Test Upper_Confidence_Bound class
def test_upper_confidence_bound():
    config = Config(beta=2.0)
    acquisition = Upper_Confidence_Bound(config)
    molecules = [Molecule("CCO", 0, 1.0, 0.1), Molecule("CCC", 1, 2.0, 0.2)]
    acquisition.calculate_acquisition_value(molecules)
    assert molecules[0].acquisition_value == pytest.approx(1.1414, rel=1e-3)
    assert molecules[1].acquisition_value == pytest.approx(2.2828, rel=1e-3)

# Test Expected_Improvement class
def test_expected_improvement():
    archive = Mock()
    archive.elites = {0: Mock(fitness=0.5), 1: Mock(fitness=1.5)}
    config = Config()
    acquisition = Expected_Improvement(config)
    acquisition.set_archive(archive)
    molecules = [Molecule("CCO", 0, 1.0, 0.1), Molecule("CCC", 1, 2.0, 0.2)]
    acquisition.calculate_acquisition_value(molecules)
    Z_0 = (1.0 - 0.5) / 0.1
    expected_value_0 = 0.1 * (norm.pdf(Z_0) + Z_0 * norm.cdf(Z_0))
    Z_1 = (2.0 - 1.5) / 0.2
    expected_value_1 = 0.2 * (norm.pdf(Z_1) + Z_1 * norm.cdf(Z_1))
    assert molecules[0].acquisition_value == pytest.approx(expected_value_0, rel=1e-3)
    assert molecules[1].acquisition_value == pytest.approx(expected_value_1, rel=1e-3)

# Test Log_Expected_Improvement class
def test_log_expected_improvement():
    archive = Mock()
    archive.elites = {0: Mock(fitness=0.5), 1: Mock(fitness=1.5)}
    config = Config()
    acquisition = Log_Expected_Improvement(config)
    acquisition.set_archive(archive)
    molecules = [Molecule("CCO", 0, 1.0, 0.1), Molecule("CCC", 1, 2.0, 0.2)]
    acquisition.calculate_acquisition_value(molecules)
    Z_0 = (1.0 - 0.5) / 0.1
    log_ei_0 = _log_ei_helper(torch.tensor(Z_0)).item() + np.log(0.1)
    Z_1 = (2.0 - 1.5) / 0.2
    log_ei_1 = _log_ei_helper(torch.tensor(Z_1)).item() + np.log(0.2)
    assert molecules[0].acquisition_value == pytest.approx(log_ei_0, rel=1e-3)
    assert molecules[1].acquisition_value == pytest.approx(log_ei_1, rel=1e-3)

if __name__ == "__main__":
    pytest.main()
