import hydra
import random
import pandas as pd
from typing import List

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import rdMMPA

from illumination.base import Molecule

rdBase.DisableLog("rdApp.error")


class Generator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning.

    Attributes:
        archive: The archive of elite molecules used for generating new molecules.
        crossover: An instance of the Crossover class for generating molecule pairs.
        mutator: An instance of the Mutator class for mutating molecules.
        batch_size: The number of molecules to sample and mutate/crossover per batch.
        initial_data: The path to the initial data file containing molecule information.
        initial_size: The initial number of molecules to load from the database.

    Methods:
        __init__(config): Initializes the Generator with the given configuration.
        set_archive(archive): Sets the archive for the Generator.
        __call__(): Generates a batch of new molecules by mutating and crossing over sampled molecules.
        load_from_database(): Loads a set of molecules from the initial data database.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Generator with the given configuration.

        Args:
            config: Configuration object containing settings for the Generator.
        """
        self.archive = None
        self.crossover = Crossover()
        self.mutator = Mutator(config.mutation_data, )
        self.batch_size = config.batch_size
        self.initial_data = config.initial_data
        self.initial_size = config.initial_size
        self.sampling_method = config.sampling_method
        self.use_crossover = config.use_crossover
        self.use_scanning = config.use_scanning

    def set_archive(self, archive):
        """
        Sets the archive for the Generator.

        Args:
            archive: The archive of elite molecules.
        """
        self.archive = archive
        return None

    def __call__(self) -> List[Molecule]:
        """
        Generates a batch of new molecules by mutating and crossing over sampled molecules.

        Returns:
            List[Molecule]: A list of newly generated molecules.
        """
        molecules = []
        molecule_samples = self.archive.sample(self.batch_size, self.sampling_method)
        for molecule in molecule_samples:
            molecules.extend(self.mutator(molecule, self.use_scanning))
        if self.use_crossover:
            molecule_sample_pairs = self.archive.sample_pairs(self.batch_size, self.sampling_method)
            for molecule_pair in molecule_sample_pairs:
                molecules.extend(self.crossover(molecule_pair))
        return molecules

    def load_from_database(self) -> List[Molecule]:
        """
        Loads a set of molecules from the initial data database.

        Returns:
            List[Molecule]: A list of molecules loaded from the database.
        """
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.initial_data))
        smiles_list = dataframe["smiles"].sample(n=self.initial_size).tolist()
        pedigree = ("database", "no reaction", "database")
        molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list]
        return molecules


class Mutator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning.

    Attributes:
        mutation_data: A dataframe containing mutation SMARTS patterns and their associated probabilities.

    Methods:
        __init__(mutation_data): Initializes the Mutator with the given mutation data.
        __call__(molecule): Applies a mutation to a given molecule and returns the resulting molecules.
    """
    def __init__(self, mutation_data: str) -> None:
        """
        Initializes the Mutator with the given mutation data.

        Args:
            mutation_data (str): The path to the mutation data file containing SMARTS patterns and probabilities.
        """
        self.mutation_data = pd.read_csv(hydra.utils.to_absolute_path(mutation_data), sep="\t")

    def __call__(self, molecule: Molecule, use_scanning: bool) -> List[Molecule]:
        """
        Applies a mutation to a given molecule and returns the resulting molecules.

        Args:
            molecule: The molecule to be mutated.

        Returns:
            List[Molecule]: A list of new molecules resulting from the mutation as applied
            by positional analogue scanning or single molecule sampled from that list.
        """
        sampled_mutation = self.mutation_data.sample(n=1, weights="probability").iloc[0]
        reaction = AllChem.ReactionFromSmarts(sampled_mutation["smarts"])
        pedigree = ("mutation", sampled_mutation["smarts"], molecule.smiles)
        try:
            molecular_graphs = [products[0] for products in reaction.RunReactants([Chem.MolFromSmiles(molecule.smiles)])]
            smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
            molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
            if not use_scanning:
                molecules = [random.choice(molecules)]
        except Exception:
            molecules = []
        return molecules


class Crossover:
    """
    A strategy class implementing a parent-centric crossover of small molecules.

    Methods:
        __init__(): Initializes the Crossover object.
        __call__(molecule_pair): Performs a crossover on a pair of molecules.
        merge(molecule_pair): Merges the fragments of a molecule pair.
        fragment(molecule_pair): Fragments a molecule pair into cores and sidechains.
    """

    def __init__(self):
        """
        Initializes the Crossover object.
        """
        pass

    def __call__(self, molecule_pair):
        """
        Performs a crossover on a pair of molecules.

        Args:
            molecule_pair: A pair of molecules to be crossed over.

        Returns:
            List[Molecule]: A list of new molecules resulting from the crossover.
        """
        pedigree = ("crossover", molecule_pair[0].smiles, molecule_pair[1].smiles)
        smiles_list = self.merge(molecule_pair)
        molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
        return molecules

    def merge(self, molecule_pair):
        """
        Merges the fragments of a molecule pair.

        Args:
            molecule_pair: A pair of molecules to be merged.

        Returns:
            List[str]: A list of SMILES strings representing the merged molecules.
        """
        molecular_graphs = []
        graph_cores, graph_sidechains = self.fragment(molecule_pair)
        random.shuffle(graph_sidechains)
        reaction = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        for core, sidechain in zip(graph_cores, graph_sidechains):
            molecular_graphs.append(reaction.RunReactants((core, sidechain))[0][0])
        smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
        return smiles_list

    def fragment(self, molecule_pair):
        """
        Fragments a molecule pair into cores and sidechains.

        Args:
            molecule_pair: A pair of molecules to be fragmented.

        Returns:
            Tuple[List[Chem.Mol], List[Chem.Mol]]: Two lists containing the cores and sidechains of the fragmented molecules.
        """
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

# import random
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdMMPA

# class Crossover:
#     """
#     Performs parent-centric crossover on a pair of molecules by swapping fragments.
#     Limits the number of fragment merges to avoid combinatorial explosion.
#     """

#     def __init__(self, max_fragments_per_molecule=3):
#         """
#         Args:
#             max_fragments_per_molecule (int): Max number of fragments to use from each parent molecule.
#         """
#         self.max_fragments = max_fragments_per_molecule

#     def __call__(self, molecule_pair):
#         """
#         Perform crossover on a pair of molecules.

#         Args:
#             molecule_pair (Tuple[Molecule, Molecule]): Pair of molecules to crossover.

#         Returns:
#             List[Molecule]: List of new molecules generated by crossover.
#         """
#         smiles_set = set()
#         crossover_smiles = self.merge_fragments(molecule_pair)
#         pedigree = ("crossover", molecule_pair[0].smiles, molecule_pair[1].smiles)
        
#         new_molecules = []
#         for smi in crossover_smiles:
#             mol = Chem.MolFromSmiles(smi)
#             if mol and smi not in smiles_set:
#                 smiles_set.add(smi)
#                 new_molecules.append(Molecule(smi, pedigree))
        
#         return new_molecules

#     def merge_fragments(self, molecule_pair):
#         """
#         Fragment each molecule, shuffle sidechains, pair fragments 1-to-1,
#         and merge them with a simple reaction SMARTS.

#         Args:
#             molecule_pair (Tuple[Molecule, Molecule]): Pair of molecules to fragment and merge.

#         Returns:
#             List[str]: SMILES strings of merged molecules.
#         """
#         cores, sidechains = self.fragment_molecules(molecule_pair)
        
#         # Limit number of fragments from each parent to avoid explosion
#         limited_cores = cores[:self.max_fragments]
#         limited_sidechains = sidechains[:self.max_fragments]
        
#         # Randomize sidechains to create diversity
#         random.shuffle(limited_sidechains)
        
#         # Reaction to merge core and sidechain fragments by connecting attachment points
#         merge_reaction_smarts = "[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]"
#         merge_reaction = AllChem.ReactionFromSmarts(merge_reaction_smarts)
        
#         merged_smiles = []
        
#         # Pair fragments one-to-one instead of all-vs-all
#         for core_frag, sidechain_frag in zip(limited_cores, limited_sidechains):
#             products = merge_reaction.RunReactants((core_frag, sidechain_frag))
#             for product_tuple in products:
#                 product_mol = product_tuple[0]
#                 if product_mol is not None:
#                     merged_smiles.append(Chem.MolToSmiles(product_mol))
        
#         # Deduplicate SMILES
#         return list(set(merged_smiles))

#     def fragment_molecules(self, molecule_pair):
#         """
#         Fragment each molecule into core and sidechain fragments.

#         Args:
#             molecule_pair (Tuple[Molecule, Molecule]): Pair of molecules to fragment.

#         Returns:
#             Tuple[List[Chem.Mol], List[Chem.Mol]]: Lists of core and sidechain fragments.
#         """
#         cores = []
#         sidechains = []

#         for molecule in molecule_pair:
#             mol = Chem.MolFromSmiles(molecule.smiles)
#             if mol is None:
#                 continue
            
#             # Fragment molecule using MMPA (one cut max)
#             fragments = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
#             if not fragments:
#                 continue
            
#             # fragments is a list of tuples (core_smiles, sidechain_smiles)
#             for _, fragment_smiles in fragments:
#                 # Fragment smiles is "core.sidechain"
#                 try:
#                     core_smiles, sidechain_smiles = fragment_smiles.split(".")
#                 except ValueError:
#                     # Sometimes the split might fail if no dot - skip
#                     continue
                
#                 # Replace attachment points [*:1] with [1*] for reaction
#                 core_smiles = core_smiles.replace("[*:1]", "[1*]")
#                 sidechain_smiles = sidechain_smiles.replace("[*:1]", "[1*]")
                
#                 core_mol = Chem.MolFromSmiles(core_smiles)
#                 sidechain_mol = Chem.MolFromSmiles(sidechain_smiles)
                
#                 if core_mol is not None and sidechain_mol is not None:
#                     cores.append(core_mol)
#                     sidechains.append(sidechain_mol)

#         return cores, sidechains

