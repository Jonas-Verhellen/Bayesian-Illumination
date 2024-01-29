import hydra
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import AllChem
from rdkit.Chem import rdMMPA

from argenomic.base import Molecule


class Generator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning. 
    """
    def __init__(self, config) -> None:
        self.archive = None
        self.crossover = Crossover()
        self.mutator = Mutator(config.mutation_data)
        self.batch_size = config.batch_size
        self.initial_data = config.initial_data
        self.initial_size = config.initial_size

    def set_archive(self, archive):
        self.archive = archive
        return None

    def __call__(self) -> List[Molecule]:
        molecules = []
        molecule_samples = self.archive.sample(self.batch_size)
        molecule_sample_pairs = self.archive.sample_pairs(self.batch_size)
        for molecule in molecule_samples:
            molecules.extend(self.mutator(molecule)) 
        for molecule_pair in molecule_sample_pairs:
            molecules.extend(self.crossover(molecule_pair)) 
        return molecules

    def load_from_database(self) -> List[Molecule]:
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.initial_data))
        smiles_list = dataframe['smiles'].sample(n=self.initial_size).tolist()
        pedigree = ("database", "no reaction", "database")   
        molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list]
        return molecules

class Mutator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning. 
    """
    def __init__(self, mutation_data) -> None:
        self.mutation_data = pd.read_csv(hydra.utils.to_absolute_path(mutation_data), sep='\t')

    def __call__(self, molecule) -> List[Molecule]:
        sampled_mutation = self.mutation_data.sample(n=1, weights='probability').iloc[0]
        reaction = AllChem.ReactionFromSmarts(sampled_mutation['smarts'])
        pedigree = ("mutation", sampled_mutation['smarts'], molecule.smiles)   
        try:
            molecular_graphs = [products[0] for products in reaction.RunReactants([Chem.MolFromSmiles(molecule.smiles)])]
            smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
            molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
        except:
            molecules = []
        return molecules

class Crossover:
    """
    A strategy class implementing a parent-centric crossover of small molecules.
    """
    def __init__(self):
        pass

    def __call__(self, molecule_pair):
        pedigree = ("crossover", molecule_pair[0].smiles, molecule_pair[1].smiles)
        smiles_list = self.merge(molecule_pair)
        molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
        return molecules

    def merge(self, molecule_pair):
        molecular_graphs = []
        graph_cores, graph_sidechains = self.fragment(molecule_pair)
        random.shuffle(graph_sidechains)
        reaction = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        for core, sidechain in zip(graph_cores, graph_sidechains):
            molecular_graphs.append(reaction.RunReactants((core, sidechain))[0][0])
        smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
        return smiles_list

    def fragment(self, molecule_pair):
        graph_cores = []
        graph_sidechains = []
        for molecule in molecule_pair:
            graph_frags = rdMMPA.FragmentMol(Chem.MolFromSmiles(molecule.smiles), maxCuts=1, resultsAsMols=False)
            if len(graph_frags) > 0:
                _, graph_frags = map(list, zip(*graph_frags))
                for frag_pair in graph_frags:
                    core, sidechain = frag_pair.split(".")
                    graph_cores.append(Chem.MolFromSmiles(core.replace("[*:1]", "[1*]")))
                    graph_sidechains.append(Chem.MolFromSmiles(sidechain.replace("[*:1]", "[1*]")))
        return graph_cores, graph_sidechains
