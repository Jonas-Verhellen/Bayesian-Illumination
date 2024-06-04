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
        self.mutator = Mutator(config.mutation_data)
        self.batch_size = config.batch_size
        self.initial_data = config.initial_data
        self.initial_size = config.initial_size

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
        molecule_samples = self.archive.sample(self.batch_size)
        molecule_sample_pairs = self.archive.sample_pairs(self.batch_size)
        for molecule in molecule_samples:
            molecules.extend(self.mutator(molecule))
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

    def __call__(self, molecule: Molecule) -> List[Molecule]:
        """
        Applies a mutation to a given molecule and returns the resulting molecules.

        Args:
            molecule: The molecule to be mutated.

        Returns:
            List[Molecule]: A list of new molecules resulting from the mutation as applied
            by positional analogue scanning.
        """
        sampled_mutation = self.mutation_data.sample(n=1, weights="probability").iloc[0]
        reaction = AllChem.ReactionFromSmarts(sampled_mutation["smarts"])
        pedigree = ("mutation", sampled_mutation["smarts"], molecule.smiles)
        try:
            molecular_graphs = [products[0] for products in reaction.RunReactants([Chem.MolFromSmiles(molecule.smiles)])]
            smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
            molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
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
