from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from abc import ABC, abstractmethod

from rdkit.Chem import rdFingerprintGenerator

from botorch import fit_gpytorch_model
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from rdkit import Chem

import torch
import numpy as np
import selfies as sf


from sklearn.feature_extraction.text import CountVectorizer


class GP_Surrogate(ABC):
    """
    A strategy class representing a surrogate model for predicting fitness values of molecules.
    The surrogate model is based on Gaussian Processes (GP) regression using the Tanimoto kernel.

    Attributes:
        - model: The Gaussian Process (GP) regression model.
        - mll: The Exact Marginal Log-Likelihood (mll) associated with the GP model.
        - state_dict: The state dictionary of the GP model for model persistence.
        - representations: Numpy array storing fingerprint representations of molecules.
        - fitnesses: Numpy array storing fitness values of molecules.

    Methods:
        - __init__: Initializes the Surrogate object with default or provided parameters.
        - __call__: Evaluates the surrogate model on a set of molecules, updating their predicted fitness and uncertainty.
        - update_model: Abstract method that needs to be implemented to update the surrogate model with new molecules and their fitness values.
        - intitialise_model: Abstract method that needs to be implemented to initializes the surrogate model with an initial set of molecules and their fitness values.
        - calculate_representations: Abstract method that needs to be implemented to calculate represenations of molecules for the surrogate model.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Surrogate object with default or provided parameters.

        Parameters:
        - config: An object specifying the configuration for the surrogate model.
        """
        self.config = config
        self.model = None
        self.mll = None
        self.state_dict = None
        self.encodings = None
        self.fitnesses = None
        return None

    def __call__(self, molecules):
        """
        Evaluates the surrogate model on a set of molecules, updating their predicted fitness and uncertainty.

        Parameters:
        - molecules: A list of molecules to be evaluated.

        Returns:
        A list of molecules with updated predicted fitness and uncertainty.
        """
        encodings = self.calculate_encodings(molecules)
        self.update_model()
        molecules = self.inference(molecules, encodings)
        return molecules

    def inference(self, molecules, encodings):
        self.mll.eval()
        self.model.eval()
        predictions = self.model(torch.tensor(encodings))
        for molecule, prediction_mean, prediction_variance in zip(molecules, predictions.mean, predictions.variance):
            molecule.predicted_fitness = prediction_mean.detach().item()
            molecule.predicted_uncertainty = prediction_variance.detach().item()
        return molecules

    def update_model(self):
        self.model = TanimotoGP(torch.tensor(self.encodings), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)
        fit_gpytorch_model(self.mll)
        self.state_dict = self.model.state_dict()
        return None

    @abstractmethod
    def add_to_prior_data(self, molecules):
        raise NotImplementedError

    @abstractmethod
    def calculate_encodings(self, molecules):
        raise NotImplementedError


class TanimotoGP(ExactGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, likelihood=GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class Fingerprint_Surrogate(GP_Surrogate):
    def __init__(self, config):
        super().__init__(config)
        self.representation = self.config.representation
        match self.representation:
            case "ECFP4":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            case "ECFP6":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
            case "FCFP4":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=2,
                    fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
                )
            case "FCFP6":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=3,
                    fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
                )
            case "RDFP":
                self.generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
            case "APFP":
                self.generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
            case "TTFP":
                self.generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
            case _:
                raise ValueError(f"{self.representation} is not a supported fingerprint type.")

    def add_to_prior_data(self, molecules):
        """
        Updates the prior data for the surrogate model with new molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        if self.encodings is not None and self.fitnesses is not None:
            self.encodings = np.append(self.encodings, self.calculate_encodings(molecules), axis=0)
            self.fitnesses = np.append(
                self.fitnesses,
                np.array([molecule.fitness for molecule in molecules]),
                axis=None,
            )
        else:
            self.encodings = self.calculate_encodings(molecules)
            self.fitnesses = np.array([molecule.fitness for molecule in molecules])
        return None

    def calculate_encodings(self, molecules):
        molecular_graphs = [Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)) for molecule in molecules]
        return np.array([self.generator.GetFingerprintAsNumPy(molecular_graph) for molecular_graph in molecular_graphs]).astype(np.float64)


class String_Surrogate(GP_Surrogate):
    def __init__(self, config):
        super().__init__(config)
        self.smiles = []
        self.representation = self.config.representation
        self.cv = CountVectorizer(ngram_range=(1, self.config.max_ngram), analyzer="char", lowercase=False)

    def calculate_encodings(self, molecules):
        smiles = [molecule.smiles for molecule in molecules]
        combined_smiles = self.smiles + smiles
        if self.representation == "Smiles":
            bag_of_characters = self.cv.fit_transform(combined_smiles)
        elif self.representation == "Selfies":
            bag_of_characters = self.cv.fit_transform([sf.encoder(smiles) for smiles in combined_smiles])
        else:
            raise ValueError(f"{self.representation} is not a supported type of molecular string.")
        self.encodings = bag_of_characters[: len(self.smiles)].toarray().astype(np.float64)
        return bag_of_characters[-len(smiles) :].toarray().astype(np.float64)

    def add_to_prior_data(self, molecules):
        """
        Updates the prior data for the surrogate model with new molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        if self.smiles is not None and self.fitnesses is not None:
            self.smiles = self.smiles + [molecule.smiles for molecule in molecules]
            self.fitnesses = np.append(
                self.fitnesses,
                np.array([molecule.fitness for molecule in molecules]),
                axis=None,
            )
        else:
            self.smiles = [molecule.smiles for molecule in molecules]
            self.fitnesses = np.array([molecule.fitness for molecule in molecules])
        return None
