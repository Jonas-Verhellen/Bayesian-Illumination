from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import RQKernel
from abc import ABC, abstractmethod
   
from rdkit.Chem import rdFingerprintGenerator

from botorch import fit_gpytorch_model
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from rdkit import Chem

import torch
import numpy as np

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
        self.representations = None
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
        self.mll.eval()
        self.model.eval()
        representations = self.calculate_representations(molecules) 
        predictions = self.model(torch.tensor(representations))
        for molecule, prediction_mean, prediction_variance  in zip(molecules, predictions.mean, predictions.variance):
            molecule.predicted_fitness = prediction_mean.detach().item()
            molecule.predicted_uncertainty = prediction_variance.detach().item()
        return molecules

    @abstractmethod
    def update_model(self, molecules):
        raise NotImplementedError
    
    @abstractmethod
    def intitialise_model(self, molecules):
        raise NotImplementedError

    @abstractmethod
    def calculate_representations(self, molecules):
        raise NotImplementedError

class Smiles_Surrogate(GP_Surrogate):
    def __init__(self, config):
        super().__init__(config)
        self.config.max_ngram
        self.cv = CountVectorizer(ngram_range=(1, self.config.max_ngram), analyzer="char", lowercase=False)

    def update_model(self, molecules):
        """
        Updates the surrogate model with new molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        self.fitnesses = np.append(self.fitnesses,  np.array([molecule.fitness for molecule in molecules]), axis=None)
        self.representations = np.append(self.representations, self.calculate_representations(molecules) , axis=0)
        bag_of_characters  = self.cv.fit_transform(self.representations).toarray()
        self.model = TanimotoGP(torch.tensor(bag_of_characters), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)
        fit_gpytorch_model(self.mll)
        self.state_dict = self.model.state_dict()
        return None
    
    def intitialise_model(self, molecules):
        """
        Initializes the surrogate model with an initial set of molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        self.fitnesses = np.array([molecule.fitness for molecule in molecules])
        self.representations = self.calculate_representations(molecules)
        bag_of_characters  = self.cv.fit_transform(self.representations).toarray()
        self.model = TanimotoGP(torch.tensor(bag_of_characters), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll)
        self.state_dict = self.model.state_dict()
        return None
    
    def calculate_representations(self, molecules):
        return [molecule.smiles for molecule in molecules]
        
class Fingerprint_Surrogate(GP_Surrogate):
    def __init__(self, config):
        super().__init__(config)
        match self.config.representation:
            case "ECFP4":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
            case "ECFP6":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
            case "FCFP4":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
            case "FCFP6":
                self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
            case "RDFP":
                self.generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
            case "APFP":
                self.generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
            case "TTFP":
                self.generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
            case _:
                raise ValueError(f"{self.config.representation} is not a supported fingerprint type.")

    def update_model(self, molecules):
        """
        Updates the surrogate model with new molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        self.fitnesses = np.append(self.fitnesses, np.array([molecule.fitness for molecule in molecules]), axis=None)
        self.representations = np.append(self.representations, self.calculate_representations(molecules) , axis=0)
        self.model = TanimotoGP(torch.tensor(self.representations), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)
        fit_gpytorch_model(self.mll)
        self.state_dict = self.model.state_dict()
        return None
    
    def intitialise_model(self, molecules):
        """
        Initializes the surrogate model with an initial set of molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        self.representations = self.calculate_representations(molecules)
        self.fitnesses = np.array([molecule.fitness for molecule in molecules])
        self.model = TanimotoGP(torch.tensor(self.representations), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll) 
        self.state_dict = self.model.state_dict()
        return None

    def calculate_representations(self, molecules):
        molecular_graphs = [Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)) for molecule in molecules]
        return np.array([self.generator.GetFingerprintAsNumPy(molecular_graph) for molecular_graph in molecular_graphs]).astype(np.float64)
    
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
