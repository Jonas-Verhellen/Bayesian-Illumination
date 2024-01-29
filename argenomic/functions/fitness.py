import math
import hydra
import numpy as np
import time

from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from typing import List, Tuple, Type

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT, GetUSR

from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem

from numba import complex128, float64
from numba.experimental import jitclass
from scipy.spatial import distance

class Fingerprint_Fitness:
    """
    A strategy class for calculating the fitness of a molecule.
    """
    def __init__(self, config) -> None:
        self.fingerprint_type = config.representation
        self.target = Chem.MolFromSmiles(config.target)
        self.target_fingerprint = self.get_fingerprint(self.target, self.fingerprint_type)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.
        """
        molecular_graph = Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles))
        molecule_fingerprint = self.get_fingerprint(molecular_graph, self.fingerprint_type)
        fitness = TanimotoSimilarity(self.target_fingerprint, molecule_fingerprint)
        molecule.fitness = fitness
        return molecule

    def get_fingerprint(self, molecular_graph: Chem.Mol, fingerprint_type: str):
        method_name = 'get_' + fingerprint_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception('{} is not a supported fingerprint type.'.format(fingerprint_type))
        return method(molecular_graph)

    def get_ECFP4(self, molecular_graph: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecular_graph, 2)

    def get_ECFP6(self, molecular_graph: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecular_graph, 3)

    def get_FCFP4(self, molecular_graph: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecular_graph, 2, useFeatures=True)

    def get_FCFP6(self, molecular_graph: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecular_graph, 3, useFeatures=True)
    

class USR_Fitness:
    """
    A strategy class for calculating the fitness of a molecule.
    """
    def __init__(self, config) -> None:
        self.n_conformers = config.conformers
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed=0xfb0d
        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        self.target_configuration = AllChem.EmbedMolecule(self.target, self.param)
        self.target_usrcat = GetUSR(self.target)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.
        """
        usrcat_scores = np.zeros(self.n_conformers)
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        for conf_id in range(self.n_conformers):
            usrcat = GetUSR(molecular_graph, confId=int(conf_id))
            score = GetUSRScore(usrcat, self.target_usrcat)
            usrcat_scores[conf_id] = score
        molecule.fitness = np.max(usrcat_scores)
        return molecule

class USRCAT_Fitness:
    """
    A strategy class for calculating the fitness of a molecule.
    """
    def __init__(self, config) -> None:
        self.n_conformers = config.conformers
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed=0xfb0d
        self.param.numThreads = config.numThreads
        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        self.target_configuration = AllChem.EmbedMultipleConfs(self.target, 1, self.param)
        self.target_usrcat = GetUSRCAT(self.target)
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.
        """
        usrcat_scores = np.zeros(self.n_conformers)
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        for conf_id in range(self.n_conformers):
            usrcat = GetUSRCAT(molecular_graph, confId=int(conf_id))
            score = GetUSRScore(usrcat, self.target_usrcat)
            usrcat_scores[conf_id] = score
        molecule.fitness = np.max(usrcat_scores)
        return molecule
    
class Zernike_Fitness:
    """
    A strategy class for calculating the fitness of a molecule.
    """
    def __init__(self, config) -> None:
        self.N = config.expansion
        self.prefactor = 3/(4*math.pi)
        self.param = rdDistGeom.ETKDGv3()
        self.param.randomSeed=0xfb0d
        self.n_conformers = config.conformers
        self.param.numThreads = config.numThreads
        self.Yljm = np.load(hydra.utils.to_absolute_path("data/coefficients/Yljm.npy"))
        self.Qklnu = np.load(hydra.utils.to_absolute_path("data/coefficients/Qklnu.npy")) 
        self.engine = Zernike_JIT(self.Qklnu, self.Yljm)

        self.target = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(config.target)))
        AllChem.EmbedMultipleConfs(self.target, 1, self.param)
        self.target_zernike = self.get_zernike(self.target, conf_id=0)

    def coordinate_extractor(self, molecule, confid=-1):
        x_points, y_points, z_points = [], [], []
        PDBBlock = Chem.MolToPDBBlock(molecule, confId=confid).split('\n')
        for line in PDBBlock:
            split = line.split()
            if split[0] == 'HETATM':
                x_points.append(float(split[5]))
                y_points.append(float(split[6])) 
                z_points.append(float(split[7]))
            if split[0] == 'END':
                break
        x_points, y_points, z_points = np.array(x_points), np.array(y_points), np.array(z_points)
        coordinates = np.column_stack((x_points, y_points, z_points))
        return coordinates

    def __call__(self, molecule) -> None:
        """
        Updates the fitness value of a molecule.
        """
        molecular_graph = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(molecule.smiles)))
        conf_ids = AllChem.EmbedMultipleConfs(molecular_graph, self.n_conformers, self.param)
        zernikes = [self.get_zernike(molecular_graph, conf_id) for conf_id in conf_ids]
        scores = [1. / (1 + distance.canberra(zernike, self.target_zernike)) for zernike in zernikes]
        molecule.fitness = np.max(scores)
        return molecule
    
    def get_zernike(self, molecular_graph, conf_id):
        AllChem.ComputeGasteigerCharges(molecular_graph)
        coordinates = self.coordinate_extractor(molecular_graph, conf_id)
        charges = [molecular_graph.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(molecular_graph.GetNumAtoms())]
        zernike = self.invariants(coordinates, charges, self.N)
        return zernike
        
    def invariants(self, coordinates, features, N):
        features_plus = [max(feature, 0.0)for feature in features]
        features_negative = [-1.0*min(feature, 0.0) for feature in features]
        invariants_plus = self.unsigned_invariants(coordinates, features_plus, N)
        invariants_negative  = self.unsigned_invariants(coordinates, features_negative, N)
        return invariants_plus - invariants_negative

    def unsigned_invariants(self, coordinates, features, N):
        x_points, y_points, z_points = map(list, list(zip(*coordinates)))
        lengths = [len(features), len(x_points), len(y_points), len(z_points)]
        x_points = x_points - np.mean(x_points)
        y_points = y_points - np.mean(y_points)
        z_points = z_points - np.mean(z_points)
        features = np.array(features)
        geometric_moments = self.engine.geometric_moments(features, x_points, y_points, z_points, N)
        invariants = self.engine.zernike_invariants(geometric_moments, N)
        return np.real(invariants)
        
spec = [('prefactor', float64), ('Qklnu', complex128[:,:,:]), ('Yljm', complex128[:,:,:])]
@jitclass(spec)
class Zernike_JIT:
    def __init__(self, Qklnu, Yljm):
        self.prefactor = 3/(4*math.pi)
        self.Qklnu = Qklnu
        self.Yljm = Yljm
        
    def geometric_moments(self, features, x_points, y_points, z_points, N):
        geometric_moments = np.zeros((N + 1, N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1): 
                for k in range(N+1):
                    if N >= (i+j+k):
                        for f, x, y, z in zip(features, x_points, y_points, z_points):
                            geometric_moments[i, j, k] += f*(np.power(x, i) * np.power(y, j) * np.power(z, k))
        return geometric_moments
    
    def choose(self, n, k):
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
        invariants = []
        V = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        W = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        X = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        Y = np.zeros((N + 1, N + 1, N + 1), dtype=complex128)
        Z = np.zeros((N + 1, N + 1, N + 1), dtype=complex128) 
        
        for a in range(N + 1):
            for b in range(N + 1): 
                for c in range(N + 1):
                    if N >= 2*a+b+c:
                    	for alpha in range(a+c+1):
                            V[a,b,c] += np.power(1j, alpha)*self.choose(a+c, alpha)*geometric_moments[2*a+c-alpha, alpha, b]
                    	
        for a in range(N + 1):
            for b in range(N + 1): 
                for c in range(N + 1):
                    if N >= 2*a+b+c:
                    	for alpha in range(a+1):
                    		W[a,b,c] += np.power(-1.0, alpha)*np.power(2.0, a-alpha)*self.choose(a, alpha)*V[a-alpha, b, c+2*alpha]
                                
        for a in range(N + 1):
            for b in range(N + 1): 
                for c in range(N + 1):
                    if N >= 2*a+b+c:
                    	for alpha in range(a+1):
                    		X[a,b,c] += self.choose(a, alpha)*W[a-alpha, b+2*alpha, c]
        
        for l in range(N + 1):
            for m in range(l + 1): 
                for nu in range(math.floor((N-l)/2)+1):
                        for j in range(math.floor((l-m)/2)+1):
                            Y[l,nu,m] += self.Yljm[l,j,m]*X[nu+j,l-m-2*j,m]
       
        for n in range(N + 1):
            for l in range(n + 1): 
                for m in range(l+1):
                    if (n-l)%2 == 0:
                        k = int((n-l)/2)
                        for nu in range(k+1):
                            Z[n,l,m] += self.prefactor*self.Qklnu[k,l,nu]*np.conj(Y[l,nu,m])

        for n in range(N+1):
            for l in range(n+1):
                if (n-l)%2 == 0:
                    sigma_vector = []
                    for m in range(-l,l+1):
                        if m < 0:
                            sigma_vector.append(((-1)**abs(m))*np.conj(Z[n,l,abs(m)])) 
                        else:             
                            sigma_vector.append(Z[n,l,m])
                    norm = np.sqrt(np.sum(np.array([np.conj(c)*c for c in sigma_vector])))
                    invariants.append(norm)
        invariants = np.array(invariants)           
        return invariants                             