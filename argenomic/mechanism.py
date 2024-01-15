import sys
import torch
import numpy as np
from typing import List, Tuple, Type

from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

from argenomic.fitness_functions import Fingerprint_Fitness, USR_Fitness, USRCAT_Fitness, Zernike_Fitness
from argenomic.acquisition_functions import Posterior_Mean, Upper_Confidence_Bound, Expected_Improvement, Log_Expected_Improvement

from botorch import fit_gpytorch_model
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

class Descriptor:
    """
    A strategy class for calculating the descriptor vector of a molecule.
    """
    def __init__(self, config) -> None:
        self.properties = []
        self.ranges = config.ranges   
        self.property_names = config.properties
        for name in self.property_names:
            module, function = name.split(".")
            module = getattr(sys.modules[__name__], module)
            self.properties.append(getattr(module, function))
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the descriptor vector of a molecule.
        """
        descriptor = []
        molecular_graph = Chem.MolFromSmiles(molecule.smiles)
        for property, range in zip(self.properties, self.ranges):
            descriptor.append(self.rescale(property(molecular_graph), range))
        molecule.descriptor = descriptor
        return molecule

    @staticmethod
    def rescale(feature: List[float], range: List[float]) -> List[float]:
        """
        Rescales the feature to the unit range.
        """
        rescaled_feature = (feature - range[0])/(range[1] - range[0])
        return rescaled_feature

from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import RQKernel
    
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
    
class Surrogate:
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
        - update_model: Updates the surrogate model with new molecules and their fitness values.
        - intitialise_model: Initializes the surrogate model with an initial set of molecules and their fitness values.
    """
    def __init__(self, config) -> None:
        """
        Initializes the Surrogate object with default or provided parameters.

        Parameters:
        - config: An object specifying the configuration for the surrogate model.
        """
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
        representations = np.array([molecule.fingerprint for molecule in molecules]).astype(np.float64)
        predictions = self.model(torch.tensor(representations))
        for molecule, prediction_mean, prediction_variance  in zip(molecules, predictions.mean, predictions.variance):
            molecule.predicted_fitness = prediction_mean.detach().item()
            molecule.predicted_uncertainty = prediction_variance.detach().item()
        return molecules
        
    def update_model(self, molecules):
        """
        Updates the surrogate model with new molecules and their fitness values.

        Parameters:
        - molecules: A list of molecules with associated fitness values.

        Returns:
        None
        """
        self.fitnesses = np.append(self.fitnesses,  np.array([molecule.fitness for molecule in molecules]), axis=None)
        self.representations = np.append(self.representations, np.array([molecule.fingerprint for molecule in molecules]).astype(np.float64), axis=0)
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
        self.fitnesses = np.array([molecule.fitness for molecule in molecules])
        self.representations = np.array([molecule.fingerprint for molecule in molecules]).astype(np.float64)
        self.model = TanimotoGP(torch.tensor(self.representations), torch.tensor(self.fitnesses).flatten())
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll)
        self.state_dict = self.model.state_dict()
        return None

class Fitness:
    """
    A factory class for creating instances of various fitness functions based on the provided configuration.
    The supported fitness function types are "Fingerprint", "USR", "USRCAT", and "Zernike".

    Methods:
    __new__: Static method for creating and returning an instance of a specific fitness function based on the configuration.
    """
    @staticmethod
    def __new__(self, config):
        """
        Static method for creating and returning an instance of a specific fitness function based on the configuration.

        Parameters:
        - config: An object specifying the configuration for the fitness function, including the type.

        Returns:
        An instance of a fitness function based on the specified type in the configuration.
        """
        if config.type == 'Fingerprint':
            return Fingerprint_Fitness(config)
        elif config.type == 'USR':
            return USR_Fitness(config)
        elif config.type == 'USRCAT':
            return USRCAT_Fitness(config)
        elif config.type == 'Zernike':
            return Zernike_Fitness(config)
        else:
            raise ValueError(f"{config.type} is not a supported fitness function type.")

class Acquisition:
    """
    A factory class for creating instances of various acquisition functions based on the provided configuration. 
    Requires an Archive instance to be passed with the configuration file. The supported acquisition function types 
    are "Mean", "UCB", "EI", and "logEI".

    Methods:
    __new__: Static method for creating and returning an instance of a specific acquisition function based on the configuration.
    """
    @staticmethod
    def __new__(self, archive, config):
        """
        Static method for creating and returning an instance of a specific acquisition function based on the configuration.

        Parameters:
        - archive: An object providing access to the archive of elite molecules.
        - config: An object specifying the configuration for the acquisition function, including the type.

        Returns:
        An instance of an acquisition function based on the specified type in the configuration.
        """
        if config.type == 'Mean':
            return Posterior_Mean(archive, config)
        elif config.type == 'UCB':
            return Upper_Confidence_Bound(archive, config)
        elif config.type == 'EI':
            return Expected_Improvement(archive, config)
        elif config.type == 'logEI':
            return Log_Expected_Improvement(archive, config)
        else:
            raise ValueError(f"{config.type} is not a supported acquisition function type.")