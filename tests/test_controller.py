import os
import pandas as pd
import pytest
from unittest.mock import Mock
from illumination.infrastructure import Controller

# Sample configuration for testing
class Config:
    max_generations = 10
    max_fitness_calls = 100

# Fixture for setting up the Controller object
@pytest.fixture
def controller():
    config = Config()
    return Controller(config)

# Mock data for the archive and surrogate
@pytest.fixture
def mock_archive():
    archive = Mock()
    archive.archive_size = 100
    archive.elites = [Mock(molecule=Mock(fitness=0.9, smiles="CCO"))]
    archive.incoming_molecules = [Mock(fitness=0.8, predicted_fitness=0.75)]
    return archive

# Test __init__ method
def test_init(controller):
    assert controller.generation == 0
    assert controller.fitness_calls == 0
    assert controller.max_generations == 10
    assert controller.max_fitness_calls == 100
    assert controller.remaining_fitness_calls == 100

# Test set_archive method
def test_set_archive(controller, mock_archive):
    controller.set_archive(mock_archive)
    assert controller.archive == mock_archive

# Test active method
def test_active(controller):
    assert controller.active() == True # noqa
    controller.generation = 10
    assert controller.active() == False # noqa
    controller.generation = 0
    controller.fitness_calls = 100
    assert controller.active() == False # noqa

# Test add_fitness_calls method
def test_add_fitness_calls(controller):
    controller.add_fitness_calls(10)
    assert controller.fitness_calls == 10
    assert controller.remaining_fitness_calls == 90

# Mock write_statistics method to avoid actual print during testing
def test_write_statistics(controller):
    statistics = pd.DataFrame({"coverage": [0.5], "quality_diversity_score": [0.9], "max_fitness": [ 0.9], "mean_fitness": [0.85]})
    metrics = pd.DataFrame({"max_err": [0.1], "mse": [0.01], "mae": [0.02]})
    controller.write_statistics(statistics.iloc[0], metrics.iloc[0])
    # No assert needed, just making sure it runs without error

# Test store_statistics method
def test_store_statistics(controller, tmpdir):
    statistics = pd.DataFrame({"coverage": [0.5], "quality_diversity_score": [0.9], "max_fitness": [0.9], "mean_fitness": [0.85]})
    metrics = pd.DataFrame({"max_err": [0.1], "mse": [0.01], "mae": [0.02]})
    temp_path = "./statistics.csv"

    # Mocking os.path.isfile to return False initially
    os.path.isfile = Mock(return_value=False)
    with open(temp_path, 'w') as f:
        pass

    controller.store_statistics(statistics.iloc[0], metrics.iloc[0])

    # Check if the file has been created and contains the expected header and row
    with open(temp_path, 'r') as f:
        lines = f.readlines()
        assert "generation,maximum fitness,mean fitness,quality diversity score,coverage,function calls,max_err,mse,mae" in lines[0]
        assert "0,0.9,0.85,0.9,50.0,0,0.1,0.01,0.02" in lines[1]

# Test calculate_statistics method
def test_calculate_statistics(controller, mock_archive):
    controller.set_archive(mock_archive)
    archive_data = {
        "smiles": ["CCO"],
        "fitness": [0.9]
    }
    stats = controller.calculate_statistics(archive_data)
    assert stats["coverage"] == 0.01
    assert stats["max_fitness"] == 0.9
    assert stats["mean_fitness"] == 0.9
    assert stats["quality_diversity_score"] == 0.9

# Test get_archive_data method
def test_get_archive_data(controller, mock_archive):
    controller.set_archive(mock_archive)
    archive_data = controller.get_archive_data()
    assert "fitness" in archive_data.columns
    assert "smiles" in archive_data.columns
    assert archive_data["fitness"].iloc[0] == 0.9
    assert archive_data["smiles"].iloc[0] == "CCO"

# Test store_molecules method
def test_store_molecules(controller, mock_archive):
    controller.set_archive(mock_archive)
    controller.memory_of_molecules = mock_archive.incoming_molecules
    temp_path = "./molecules.csv"

    controller.store_molecules()

    # Check if the file has been created and contains the expected data
    molecule_df = pd.read_csv(temp_path)
    assert "fitness" in molecule_df.columns
    assert molecule_df["fitness"].iloc[0] == 0.8
    assert "predicted_fitness" in molecule_df.columns
    assert molecule_df["predicted_fitness"].iloc[0] == 0.75

# Test update method
def test_update(controller, mock_archive, tmpdir):
    controller.set_archive(mock_archive)
    temp_path = "./archive_0.csv"
    controller.update()

    # Check if the archive file has been created and contains the expected data
    archive_df = pd.read_csv(temp_path)
    assert "fitness" in archive_df.columns
    assert "smiles" in archive_df.columns
    assert archive_df["fitness"].iloc[0] == 0.9
    assert archive_df["smiles"].iloc[0] == "CCO"

    # Ensure generation is incremented
    assert controller.generation == 1
