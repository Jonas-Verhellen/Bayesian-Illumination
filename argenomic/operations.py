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
   
class Mutator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning. 
    """
    def __init__(self, config_mutator) -> None:
        self.mutation_data = pd.read_csv(hydra.utils.to_absolute_path(config_mutator.data_file), sep='\t')

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
