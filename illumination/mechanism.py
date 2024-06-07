import sys
from typing import List

from illumination.functions.surrogate import String_Surrogate, Fingerprint_Surrogate
from illumination.functions.fitness import Fingerprint_Fitness, Gaucamol_Fitness, USRCAT_Fitness, Zernike_Fitness, OVC_Fitness
from illumination.functions.acquisition import Posterior_Mean, Upper_Confidence_Bound, Expected_Improvement, Log_Expected_Improvement

from rdkit import Chem
from rdkit import rdBase

from rdkit.Chem import AllChem # noqa
from rdkit.Chem import Crippen # noqa
from rdkit.Chem import Lipinski # noqa
from rdkit.Chem import Descriptors # noqa
from rdkit.Chem import rdMolDescriptors # noqa

rdBase.DisableLog("rdApp.error")


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
        self.dimensionality = len(self.property_names)
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
        rescaled_feature = (feature - range[0]) / (range[1] - range[0])
        return rescaled_feature


class Surrogate:
    """
    A factory class for creating instances of various surrogate functions based on the provided configuration.
    The supported surrogate function types are "Smiles" and "Fingerprint". Note that Fingerprint acquisition functions
    need further specifications by passing on the config file.

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

        match config.type:
            case "String":
                return String_Surrogate(config)
            case "Fingerprint":
                return Fingerprint_Surrogate(config)
            case _:
                raise ValueError(f"{config.type} is not a supported surrogate function type.")


class Fitness:
    """
    A factory class for creating instances of various fitness functions based on the provided configuration.
    The supported fitness function types are "Fingerprint", "USRCAT", and "Zernike".

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
        print(config.type)
        match config.type:
            case "Fingerprint":
                return Fingerprint_Fitness(config)
            case "Guacamol":
                return Gaucamol_Fitness(config)
            case "USRCAT":
                return USRCAT_Fitness(config)
            case "Zernike":
                return Zernike_Fitness(config)
            case "OVC":
                return OVC_Fitness(config)
            case _:
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
    def __new__(self, config):
        """
        Static method for creating and returning an instance of a specific acquisition function based on the configuration.

        Parameters:
        - config: An object specifying the configuration for the acquisition function, including the type.

        Returns:
        An instance of an acquisition function based on the specified type in the configuration.
        """

        match config.type:
            case "Mean":
                return Posterior_Mean(config)
            case "UCB":
                return Upper_Confidence_Bound(config)
            case "EI":
                return Expected_Improvement(config)
            case "logEI":
                return Log_Expected_Improvement(config)
            case _:
                raise ValueError(f"{config.type} is not a supported acquisition function type.")
