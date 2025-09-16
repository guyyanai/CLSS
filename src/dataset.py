"""
This module defines the dataset for the CLSS model.

It includes the `CLSSDataset` class, which handles loading, processing, and caching
of protein sequence and structure data for contrastive learning.
"""

import os
import pickle
from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL


class CLSSDataset(Dataset):
    def __init__(
        self,
        tokenizer_checkpoint: str,
        structures_dir: str,
        ecod_ids: List[str],
        file_suffix: str = ".pdb",
        pickle_file: Optional[str] = None,
    ):
        """Initializes the CLSSDataset.

        This involves loading tokenized sequences and structure embeddings from a pickle file if available,
        otherwise it processes PDB files to generate them.

        Args:
            tokenizer_checkpoint (str): The Hugging Face Hub checkpoint for the EsmTokenizer.
            structures_dir (str): The root directory containing the PDB files, organized by ECOD ID.
            ecod_ids (List[str]): A list of ECOD domain IDs to include in the dataset.
            file_suffix (str, optional): The file extension for the structure files. Defaults to ".pdb".
            pickle_file (str, optional): Path to a pickle file for caching processed data.
                                         If the file exists, data is loaded from it. Otherwise,
                                         processed data is saved to this file. Defaults to None.
        """

        if pickle_file and os.path.exists(pickle_file):
            self.tokenized_sequences, self.structure_embeddings = self.load_from_pickle(pickle_file)
            return

        self.tokenizer = self.load_tokenizer(tokenizer_checkpoint)

        print(f"Loading {len(ecod_ids)} sequences & structures...")
        sequences, structures = self.find_and_parse_pdbs(
            ecod_ids, structures_dir, file_suffix
        )

        self.tokenized_sequences = self.tokenize_sequences(sequences)
        self.structure_embeddings = self.embed_structures(structures)

        if pickle_file is not None:
            self.write_to_pickle(pickle_file)

    def load_from_pickle(self, pickle_file: str) -> Tuple[dict, torch.Tensor]:
        """Loads the tokenized sequences and structure embeddings from a pickle file.

        Args:
            pickle_file (str): The path to the pickle file.

        Returns:
            Tuple[dict, torch.Tensor]: A tuple containing the tokenized sequences (as a dictionary from the tokenizer)
                                       and the structure embeddings as a tensor.
        """
        print(f"Loading dataset from pickle file: {pickle_file}")

        with open(pickle_file, "rb") as f:
            tokenized_sequences, structure_embeddings = pickle.load(f)

        return tokenized_sequences, structure_embeddings

    def write_to_pickle(self, pickle_file: str):
        """Saves the tokenized sequences and structure embeddings to a pickle file.

        Args:
            pickle_file (str): The path to the pickle file.
        """
        print(f"Saving dataset to pickle file: {pickle_file}")
        
        with open(pickle_file, "wb") as f:
            pickle.dump([self.tokenized_sequences, self.structure_embeddings], f)

    def load_tokenizer(self, checkpoint: str) -> EsmTokenizer:
        """Loads the ESM tokenizer from a specified checkpoint.

        Args:
            checkpoint (str): The Hugging Face Hub checkpoint for the EsmTokenizer.

        Returns:
            EsmTokenizer: The loaded tokenizer.
        """
        return EsmTokenizer.from_pretrained(checkpoint)

    def tokenize_sequences(self, sequences: List[str]) -> dict:
        """Tokenizes a list of protein sequences.

        Args:
            sequences (List[str]): A list of protein sequences.

        Returns:
            dict: A dictionary containing tokenized 'input_ids' and 'attention_mask' tensors.
        """
        # Tokenize all sequences in parallel
        return self.tokenizer(sequences, padding=True, return_tensors="pt")

    def find_and_parse_pdbs(
        self, ecod_ids: List[str], structures_dir: str, file_suffix: str
    ) -> Tuple[List[str], List[ESMProtein]]:
        """Finds and parses PDB files for a list of ECOD IDs.

        Args:
            ecod_ids (List[str]): A list of ECOD domain IDs.
            structures_dir (str): The root directory containing the PDB files.
            file_suffix (str): The file extension of the structure files.

        Returns:
            Tuple[List[str], List[ESMProtein]]: A tuple containing a list of protein sequences and a list
                                               of ESMProtein objects.
        """

        sequences = []
        structures = []

        for domain_id in tqdm(ecod_ids, desc="Finding and parsing PDBs"):
            sequence, structure = self.parse_single_domain(
                domain_id, structures_dir, file_suffix
            )
            sequences.append(sequence)
            structures.append(structure)

        return sequences, structures

    def parse_single_domain(
        self, domain_id: str, structures_dir: str, file_suffix: str
    ) -> Tuple[str, ESMProtein]:
        """Parses a single PDB file to extract its sequence and structure.

        Args:
            domain_id (str): The ECOD domain ID.
            structures_dir (str): The root directory for PDB files.
            file_suffix (str): The file extension of the structure file.

        Raises:
            Exception: If the PDB file for the given domain ID is not found.

        Returns:
            Tuple[str, ESMProtein]: A tuple containing the protein sequence and an ESMProtein object.
        """
        pdb_file_path = os.path.join(
            structures_dir, domain_id[2:7], domain_id, f"{domain_id}{file_suffix}"
        )

        if not os.path.exists(pdb_file_path):
            raise Exception(f"Failed to find pdb file of domain: {domain_id}")

        domain_chain = ProteinChain.from_pdb(pdb_file_path)
        load_domain = ESMProtein.from_protein_chain(domain_chain)

        sequence = load_domain.sequence
        structure = ESMProtein(coordinates=load_domain.coordinates)  # type: ignore

        return sequence, structure

    def embed_structures(self, structures: List[ESMProtein]) -> torch.Tensor:
        """Generates structure embeddings for a list of ESMProtein objects using ESM-3.

        Args:
            structures (List[ESMProtein]): A list of ESMProtein objects.

        Returns:
            torch.Tensor: A tensor of structure embeddings.
        """
        model = ESM3.from_pretrained(ESM3_OPEN_SMALL)

        structure_embeddings = []

        print("Computing structure embeddings...")

        for structure in tqdm(structures):
            encoded_structure = model.encode(structure)
            with torch.no_grad():
                structure_output = model.forward_and_sample(
                    encoded_structure, SamplingConfig(return_mean_embedding=True)  # type: ignore
                )

            structure_embeddings.append(structure_output.mean_embedding.detach().cpu())  # type: ignore

        del model

        return torch.stack(structure_embeddings)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.tokenized_sequences["input_ids"])  # type: ignore

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the input IDs,
                                                             attention mask, and structure embedding
                                                             for the specified sample.
        """
        # Extract the tokenized tensors
        input_ids = self.tokenized_sequences["input_ids"][idx]
        attention_mask = self.tokenized_sequences["attention_mask"][idx]
        structure_embedding = self.structure_embeddings[idx]

        return input_ids, attention_mask, structure_embedding
