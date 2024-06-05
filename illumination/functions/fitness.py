import os
import math
import hydra
import torch

import numpy as np
from scipy.spatial import distance

from tdc import Oracle
from numba import complex128, float64
from numba.experimental import jitclass

from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT, GetUSR

rdBase.DisableLog("rdApp.error")


class Fingerprint_Fitness:
    """
    A strategy class for calculating the fitness of a molecule based on its fingerprint similarity to a target molecule.

    Attributes:
        fingerprint_type: The type of fingerprint representation used (e.g., ECFP4, ECFP6, FCFP4, FCFP6).
        target: The target molecule as an RDKit Mol object.
        target_fingerprint: The fingerprint of the target molecule.

    Methods:
        __init__: Initializes the Fingerprint_Fitness object with the specified configuration.
        __call__: Updates the fitness value of a molecule based on its fingerprint similarity to the target molecule.
        get_fingerprint: Retrieves the fingerprint of a molecular graph based on the specified fingerprint type.
        get_ECFP4: Generates the ECFP4 fingerprint for a molecular graph.
        get_ECFP6: Generates the ECFP6 fingerprint for a molecular graph.
        get_FCFP4: Generates the FCFP4 fingerprint for a molecular graph.
        get_FCFP6: Generates the FCFP6 fingerprint for a molecular graph.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Fingerprint_Fitness object with the specified configuration.

        Args:
            config: An object specifying the configuration for the fitness calculation, including the target molecule and fingerprint type.
        """
        self.fingerprint_type = config.representation
        self.target = Chem.MolFromSmiles(config.target)
        self.target_fingerprint = self.get_fingerprint(self.target, self.fingerprint_type)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule based on its fingerprint similarity to the target molecule.

        Args:
            molecule: The molecule whose fitness value is to be updated.

        Returns:
            molecule: The molecule with an updated fitness value.
        """
        molecular_graph = Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles))
        molecule_fingerprint = self.get_fingerprint(molecular_graph, self.fingerprint_type)
        fitness = TanimotoSimilarity(self.target_fingerprint, molecule_fingerprint)
        molecule.fitness = fitness
        return molecule

    def get_fingerprint(self, molecular_graph: Chem.Mol, fingerprint_type: str):
        """
        Retrieves the fingerprint of a molecular graph based on the specified fingerprint type.

        Args:
            molecular_graph: The RDKit Mol object representing the molecule.
            fingerprint_type: The type of fingerprint representation to use.

        Returns:
            The fingerprint of the molecular graph.
        """
        method_name = "get_" + fingerprint_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception("{} is not a supported fingerprint type.".format(fingerprint_type))
        return method(molecular_graph)

    def get_ECFP4(self, molecular_graph: Chem.Mol):
        """
        Generates the ECFP4 fingerprint for a molecular graph.

        Args:
            molecular_graph: The RDKit Mol object representing the molecule.

        Returns:
            The ECFP4 fingerprint of the molecular graph.
        """
        return AllChem.GetMorganFingerprint(molecular_graph, 2)

    def get_ECFP6(self, molecular_graph: Chem.Mol):
        """
        Generates the ECFP6 fingerprint for a molecular graph.

        Args:
            molecular_graph: The RDKit Mol object representing the molecule.

        Returns:
            The ECFP6 fingerprint of the molecular graph.
        """
        return AllChem.GetMorganFingerprint(molecular_graph, 3)

    def get_FCFP4(self, molecular_graph: Chem.Mol):
        """
        Generates the FCFP4 fingerprint for a molecular graph.

        Args:
            molecular_graph: The RDKit Mol object representing the molecule.

        Returns:
            The FCFP4 fingerprint of the molecular graph.
        """
        return AllChem.GetMorganFingerprint(molecular_graph, 2, useFeatures=True)

    def get_FCFP6(self, molecular_graph: Chem.Mol):
        """
        Generates the FCFP6 fingerprint for a molecular graph.

        Args:
            molecular_graph: The RDKit Mol object representing the molecule.

        Returns:
            The FCFP6 fingerprint of the molecular graph.
        """
        return AllChem.GetMorganFingerprint(molecular_graph, 3, useFeatures=True)


class Gaucamol_Fitness:
    """
    A strategy class for calculating the fitness of a molecule, based on the Gaucamol benchmarks.

    Attributes:
        target: The target molecule.
        oracle: The oracle used for fitness evaluation.

    Methods:
        __init__: Initializes the Gaucamol_Fitness object with the target molecule and the oracle.
        __call__: Updates the fitness value of a molecule.
    """


    def __init__(self, config) -> None:
        """
        Initializes the Gaucamol_Fitness object with the target molecule and the oracle.

        Args:
            config: An object specifying the configuration for the fitness calculation.
        """
        self.target = config.target
        self.oracle = Oracle(name=config.target)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.

        Args:
            molecule: The molecule for which the fitness value needs to be updated.

        Returns:
            None
        """
        molecule.fitness = self.oracle(molecule.smiles)
        return molecule


class USR_Fitness:
    """
    A strategy class for calculating the fitness of a molecule,based on its USR descriptors.

    Methods:
        __init__: Initializes the USR_Fitness object with the target molecule and configuration parameters.
        __call__: Updates the fitness value of a molecule.

    Attributes:
        n_conformers: The number of conformers to generate for each molecule.
        param: The parameters for embedding molecules.
        target: The target molecule.
        target_configuration: The embedded configuration of the target molecule.
        target_usrcat: The USR descriptor of the target molecule.
    """
    def __init__(self, config) -> None:
        """
        Initializes the USR_Fitness object with the target molecule and configuration parameters.

        Args:
            config: An object specifying the configuration for the fitness calculation.
        """
        self.n_conformers = config.conformers
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed = 0xFB0D
        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        self.target_configuration = AllChem.EmbedMolecule(self.target, self.param)
        self.target_usrcat = GetUSR(self.target)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.

        Args:
            molecule: The molecule for which the fitness value needs to be updated.

        Returns:
            None
        """
        usrcat_scores = np.zeros(self.n_conformers)
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        conf_ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        for conf_id in conf_ids:
            usrcat = GetUSR(molecular_graph, confId=int(conf_id))
            score = GetUSRScore(usrcat, self.target_usrcat)
            usrcat_scores[conf_id] = score
        molecule.fitness = np.max(usrcat_scores)
        return molecule


class USRCAT_Fitness:
    """
    A strategy class for calculating the fitness of a molecule.

    Methods:
        __init__: Initializes the USRCAT_Fitness object with the target molecule and configuration parameters.
        __call__: Updates the fitness value of a molecule.

    Attributes:
        n_conformers: The number of conformers to generate for each molecule.
        param: The parameters for embedding molecules.
        target: The target molecule.
        target_configuration: The embedded configuration of the target molecule.
        target_usrcat: The USRCAT descriptor of the target molecule.
    """
    def __init__(self, config) -> None:
        """
        Initializes the USRCAT_Fitness object with the target molecule and configuration parameters.

        Args:
            config: An object specifying the configuration for the fitness calculation.
        """
        self.n_conformers = config.conformers
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed = 0xFB0D
        self.param.numThreads = config.numThreads
        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        self.target_configuration = AllChem.EmbedMultipleConfs(self.target, 1, self.param)
        self.target_usrcat = GetUSRCAT(self.target)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.

        Args:
            molecule: The molecule for which the fitness value needs to be updated.

        Returns:
            None
        """
        usrcat_scores = np.zeros(self.n_conformers)
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        conf_ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        for conf_id in conf_ids:
            usrcat = GetUSRCAT(molecular_graph, confId=int(conf_id))
            score = GetUSRScore(usrcat, self.target_usrcat)
            usrcat_scores[conf_id] = score
        molecule.fitness = np.max(usrcat_scores)
        return molecule


class Zernike_Fitness:
    """
    A strategy class for calculating the fitness of a molecule using Zernike moments.

    Methods:
        __init__: Initializes the Zernike_Fitness object with the target molecule and configuration parameters.
        __call__: Updates the fitness value of a molecule.

    Attributes:
        N: The expansion order of the Zernike moments.
        prefactor: The prefactor for calculating Zernike moments.
        param: The parameters for embedding molecules.
        n_conformers: The number of conformers to generate for each molecule.
        Yljm: Coefficients used for computing Zernike moments.
        Qklnu: Coefficients used for computing Zernike moments.
        engine: An instance of the Zernike_JIT class for computing Zernike moments.
        target: The target molecule.
        target_zernike: The Zernike moments of the target molecule.
    """
    def __init__(self, config) -> None:
        """
        Initializes the Zernike_Fitness object with the target molecule and configuration parameters.

        Args:
            config: An object specifying the configuration for the fitness calculation.
        """
        self.N = config.expansion
        self.prefactor = 3 / (4 * math.pi)
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed = 0xFB0D
        self.n_conformers = config.conformers
        self.param.numThreads = config.numThreads
        self.Yljm = np.load(hydra.utils.to_absolute_path("data/coefficients/Yljm.npy"))
        self.Qklnu = np.load(hydra.utils.to_absolute_path("data/coefficients/Qklnu.npy"))
        self.engine = Zernike_JIT(self.Qklnu, self.Yljm)

        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        AllChem.EmbedMultipleConfs(self.target, 1, self.param)
        self.target_zernike = self.get_zernike(self.target, conf_id=0)

    def coordinate_extractor(self, molecule, confid=-1):
        """
        Extracts atomic coordinates from a molecule.

        Args:
            molecule: The molecule from which to extract coordinates.
            confid (int, optional): The conformer ID. Defaults to -1.

        Returns:
            numpy.ndarray: The atomic coordinates.
        """
        x_points, y_points, z_points = [], [], []
        PDBBlock = Chem.MolToPDBBlock(molecule, confId=confid).split("\n")
        for line in PDBBlock:
            split = line.split()
            if split[0] == "HETATM":
                x_points.append(float(split[5]))
                y_points.append(float(split[6]))
                z_points.append(float(split[7]))
            if split[0] == "END":
                break
        x_points, y_points, z_points = np.array(x_points), np.array(y_points), np.array(z_points)
        coordinates = np.column_stack((x_points, y_points, z_points))
        return coordinates

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.

        Args:
            molecule: The molecule for which the fitness value needs to be updated.

        Returns:
            None
        """
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        conf_ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        zernikes = [self.get_zernike(molecular_graph, conf_id) for conf_id in conf_ids]
        scores = [1.0 / (1 + distance.canberra(zernike, self.target_zernike)) for zernike in zernikes]
        molecule.fitness = np.max(scores)
        return molecule

    def get_zernike(self, molecular_graph, conf_id):
        """
        Calculates Zernike moments for a molecule.

        Args:
            molecular_graph: The molecule.
            conf_id: The conformer ID.

        Returns:
            numpy.ndarray: The Zernike moments.
        """
        AllChem.ComputeGasteigerCharges(molecular_graph)
        coordinates = self.coordinate_extractor(molecular_graph, conf_id)
        charges = [molecular_graph.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(molecular_graph.GetNumAtoms())]
        zernike = self.invariants(coordinates, charges, self.N)
        return zernike

    def invariants(self, coordinates, features, N):
        """
        Computes Zernike invariants for a set of coordinates and features.

        Args:
            coordinates: Atomic coordinates.
            features: Atomic features.
            N: The expansion order.

        Returns:
            numpy.ndarray: The Zernike invariants.
        """
        features_plus = [max(feature, 0.0) for feature in features]
        features_negative = [-1.0 * min(feature, 0.0) for feature in features]
        invariants_plus = self.unsigned_invariants(coordinates, features_plus, N)
        invariants_negative = self.unsigned_invariants(coordinates, features_negative, N)
        return invariants_plus - invariants_negative

    def unsigned_invariants(self, coordinates, features, N):
        """
        Computes unsigned Zernike invariants.

        Args:
            coordinates: Atomic coordinates.
            features: Atomic features.
            N: The expansion order.

        Returns:
            numpy.ndarray: The unsigned Zernike invariants.
        """
        x_points, y_points, z_points = map(list, list(zip(*coordinates)))
        x_points = x_points - np.mean(x_points)
        y_points = y_points - np.mean(y_points)
        z_points = z_points - np.mean(z_points)
        features = np.array(features)
        geometric_moments = self.engine.geometric_moments(features, x_points, y_points, z_points, N)
        invariants = self.engine.zernike_invariants(geometric_moments, N)
        return np.real(invariants)


spec = [("prefactor", float64), ("Qklnu", complex128[:, :, :]),    ("Yljm", complex128[:, :, :]),]
@jitclass(spec)
class Zernike_JIT:
    """
    A class for computing Zernike moments and invariants using just-in-time (JIT) compilation.

    Methods:
        __init__: Initializes the Zernike_JIT object with the precomputed coefficients.
        geometric_moments: Computes geometric moments for a set of features and coordinates.
        choose: Computes the binomial coefficient.
        zernike_invariants: Computes Zernike invariants from geometric moments.

    Attributes:
        prefactor: The prefactor for calculating Zernike moments.
        Qklnu: Coefficients used for computing Zernike moments.
        Yljm: Coefficients used for computing Zernike moments.
    """
    def __init__(self, Qklnu, Yljm):
        """
        Initializes the Zernike_JIT object with the precomputed coefficients.

        Parameters:
            Qklnu: Coefficients used for computing Zernike moments.
            Yljm: Coefficients used for computing Zernike moments.
        """
        self.prefactor = 3 / (4 * math.pi)
        self.Qklnu = Qklnu
        self.Yljm = Yljm

    def geometric_moments(self, features, x_points, y_points, z_points, N):
        """
        Computes geometric moments for a set of features and coordinates.

        Parameters:
            features: The features for each atom.
            x_points: The x-coordinates of the atoms.
            y_points: The y-coordinates of the atoms.
            z_points: The z-coordinates of the atoms.
            N: The expansion order of the Zernike moments.

        Returns:
            numpy.ndarray: The geometric moments.
        """
        geometric_moments = np.zeros((N + 1, N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                for k in range(N + 1):
                    if N >= (i + j + k):
                        for f, x, y, z in zip(features, x_points, y_points, z_points):
                            geometric_moments[i, j, k] += f * (np.power(x, i) * np.power(y, j) * np.power(z, k))
        return geometric_moments

    def choose(self, n, k):
        """
        Computes the binomial coefficient.

        Parameters:
            n: The total number of items.
            k: The number of items to choose.

        Returns:
            int: The binomial coefficient.
        """
        if 0 <= k <= n:
            ntok = 1.0
            ktok = 1.0
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1.0
            return ntok // ktok
        else:
            return 0.0

    def zernike_invariants(self, geometric_moments, N):
        """
        Computes Zernike invariants from geometric moments.

        Parameters:
            geometric_moments: The geometric moments.
            N: The expansion order of the Zernike moments.

        Returns:
            numpy.ndarray: The Zernike invariants.
        """
        invariants = []
        V = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        W = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        X = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        Y = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        Z = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)

        for a in range(N + 1):
            for b in range(N + 1):
                for c in range(N + 1):
                    if N >= 2 * a + b + c:
                        for alpha in range(a + c + 1):
                            V[a, b, c] += np.power(1j, alpha) * self.choose(a + c, alpha) * geometric_moments[2 * a + c - alpha, alpha, b]

        for a in range(N + 1):
            for b in range(N + 1):
                for c in range(N + 1):
                    if N >= 2 * a + b + c:
                        for alpha in range(a + 1):
                            W[a, b, c] += np.power(-1.0, alpha) * np.power(2.0, a - alpha) * self.choose(a, alpha) * V[a - alpha, b, c + 2 * alpha]

        for a in range(N + 1):
            for b in range(N + 1):
                for c in range(N + 1):
                    if N >= 2 * a + b + c:
                        for alpha in range(a + 1):
                            X[a, b, c] += self.choose(a, alpha) * W[a - alpha, b + 2 * alpha, c]

        for a in range(N + 1):
            for b in range(a + 1):
                for c in range(math.floor((N - a) / 2) + 1):
                    for d in range(math.floor((a - b) / 2) + 1):
                        Y[a, c, b] += self.Yljm[a, d, b] * X[c + d, a - b - 2 * d, b]

        for a in range(N + 1):
            for b in range(a + 1):
                for c in range(b + 1):
                    if (a - b) % 2 == 0:
                        d = int((a - b) / 2)
                        for e in range(d + 1):
                            Z[a, b, c] += self.prefactor * self.Qklnu[d, b, e] * np.conj(Y[b, e, c])

        for a in range(N + 1):
            for b in range(a + 1):
                if (a - b) % 2 == 0:
                    sigma_vector = []
                    for c in range(-b, b + 1):
                        if c < 0:
                            sigma_vector.append(((-1) ** abs(c)) * np.conj(Z[a, b, abs(c)]))
                        else:
                            sigma_vector.append(Z[a, b, c])
                    norm = np.sqrt(np.sum(np.array([np.conj(sigma) * sigma for sigma in sigma_vector])))
                    invariants.append(norm)
        invariants = np.array(invariants)
        return invariants


class OVC_Fitness:
    """
    A strategy class for calculating the fitness of a molecule using various models for
    organic photovoltaics based on autoML models provided by Tartarus.

    Attributes:
        target: The type of fitness model to be used.
        model_homo_lumo: The model for predicting HOMO-LUMO energy gap.
        model_lumo_val: The model for predicting LUMO energy value.
        model_dipole: The model for predicting dipole moment.
        model_combined: The combined model for multiple properties.

    Methods:
        __init__: Initializes the OVC_Fitness object with the specified target model.
        __call__: Updates the fitness value of a molecule using the specified model.

    """
    def __init__(self, config) -> None:
        """
        Initializes the OVC_Fitness object with the specified target model.

        Args:
            config: An object specifying the configuration for fitness calculation.
        """
        self.target = config.target
        self.model_homo_lumo = OVC_Model(hydra.utils.to_absolute_path("data/trained_model/homo_lumo"))
        self.model_lumo_val = OVC_Model(hydra.utils.to_absolute_path("data/trained_model/lumo_val"))
        self.model_dipole = OVC_Model(hydra.utils.to_absolute_path("data/trained_model/dipole"))
        self.model_combined = OVC_Model(hydra.utils.to_absolute_path("data/trained_model/function"))
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.

        Args:
            molecule: The molecule for which the fitness value needs to be updated.

        Returns:
            None
        """
        model = getattr(self, "model_" + self.target)
        if model is None:
            raise Exception("{} is not a supported model type.".format(self.target))
        fitness = model.forward(molecule.smiles)
        molecule.fitness = np.abs(fitness[0])
        return molecule


class OVC_Model(object):
    """
    A class representing a machine learning model for predicting molecular properties.

    Attributes:
        use_ensemble: Flag indicating whether to use ensemble prediction.
        model_list: List of models for predicting the specified OVC target.

    Methods:
        __init__: Initializes the OVC_Model object for the specified target model.
        forward: Performs forward pass of the specified target model.
        get_fingerprint: Generates molecular fingerprint from SMILES for featurisation.

    """

    def __init__(self, model_list_dir, use_ensemble=False):
        """
        Initializes the OVC_Model object with the specified model directory.

        Parameters:
            model_list_dir: The directory containing model files.
            use_ensemble: Flag indicating whether to use ensemble prediction.
        """
        super(OVC_Model, self).__init__()
        self.use_ensemble = use_ensemble

        model_state_dicts = os.listdir(model_list_dir)
        self.model_list = []
        for model_state_dict in model_state_dicts:
            self.model_list.append(torch.load(os.path.join(model_list_dir, model_state_dict), map_location=torch.device("cpu")))
            if use_ensemble is False:
                break

    def forward(self, x):
        """
        Performs forward pass of the model.

        Parameters:
            x: Input data to the model.

        Returns:
            The model prediction.
        """
        if isinstance(x, str):
            x = self.get_fingerprint(smile=x, nBits=2048, ecfp_degree=2)

        x = torch.tensor(x).to(dtype=torch.float32)
        predictions = []
        for model in self.model_list:
            predictions.append(model(x).detach().cpu().numpy())

        predictions = np.array(predictions)

        mean = np.mean(predictions, axis=0)
        var = np.var(predictions, axis=0)

        if self.use_ensemble:
            return mean, var
        else:
            return mean

    def get_fingerprint(self, smile, nBits, ecfp_degree=2):
        """
        Generates molecular fingerprint from SMILES.

        Parameters:
            smile (str): The SMILES representation of the molecule.
            nBits (int): The number of bits for fingerprint.
            ecfp_degree (int): The degree of ECFP fingerprint.

        Returns:
            numpy.ndarray: The molecular fingerprint.
        """
        m1 = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(m1, ecfp_degree, nBits=nBits)
        x = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, x)
        return x
