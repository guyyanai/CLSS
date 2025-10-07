import os
from typing import List, Optional
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
import torch
import numpy as np
from utils import cache_to_pickle

from args import CLIArgs


def load_domain_dataset(
    dataset_path: str,
    id_column: str,
    label_column: str,
    fasta_path_column: Optional[str] = None,
    pdb_path_column: Optional[str] = None,
    hex_color_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the domain dataset from a CSV file and validate required columns.

    Args:
        dataset_path: Path to the CSV dataset file
        id_column: Name of the domain ID column
        label_column: Name of the label column
        fasta_path_column: Name of the FASTA path column
        pdb_path_column: Name of the PDB path column
        hex_color_column: Name of the column with hex color codes for points
        
    Returns:
        pd.DataFrame: Loaded domain dataset

    Raises:
        ValueError: If required columns are missing in the dataset
    """
    domain_dataframe = pd.read_csv(dataset_path, dtype={id_column: str})

    if id_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{id_column}' not found in dataset")

    if label_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")

    if fasta_path_column is not None and fasta_path_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{fasta_path_column}' not found in dataset")

    if pdb_path_column is not None and pdb_path_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{pdb_path_column}' not found in dataset")

    if hex_color_column is not None and hex_color_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{hex_color_column}' not found in dataset")

    return domain_dataframe


def load_sequence_from_fasta(fasta_path: str) -> str:
    """
    Load a sequence from a FASTA file.

    Args:
        fasta_path: Path to the FASTA file

    Returns:
        str: The loaded sequence

    Raises:
        FileNotFoundError: If the FASTA file does not exist
        ValueError: If the FASTA file is empty or improperly formatted
    """
    # Check file existence and non-empty
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File not found: {fasta_path}")
    if os.path.getsize(fasta_path) == 0:
        raise ValueError("The FASTA file is empty.")

    # Try reading the first record
    records = SeqIO.parse(fasta_path, "fasta")
    try:
        record = next(records)
    except StopIteration:
        raise ValueError("No FASTA sequences found in the file.")

    # Validate the content
    if not record.id or not record.seq:
        raise ValueError("Invalid FASTA format: header or sequence missing.")
    if len(record.seq) == 0:
        raise ValueError("Sequence is empty.")

    return str(record.seq)


@cache_to_pickle(path_param_name="cache_path")
def load_sequences(
    domain_dataframe: pd.DataFrame,
    fasta_path_column: str,
    cache_path: Optional[str] = None,
) -> List[str | None]:
    """
    Load sequences from FASTA files specified in the dataframe.

    Args:
        domain_dataframe: DataFrame containing domain data with FASTA paths
        fasta_path_column: Name of the column containing FASTA file paths

    Returns:
        List[str]: List of loaded sequences
    """
    sequences: List[str | None] = []
    for index, row in tqdm(
        domain_dataframe.iterrows(),
        total=len(domain_dataframe),
        desc="Loading sequences",
    ):
        fasta_path = row[fasta_path_column]
        if fasta_path is None:
            sequences.append(None)
            continue

        try:
            sequence = load_sequence_from_fasta(fasta_path)
            sequences.append(sequence)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading FASTA file for row {index}: {e}")
            sequences.append(None)
            continue

    return sequences


def load_structure_from_pdb(pdb_path: str) -> torch.Tensor:
    """
    Load a structure from a PDB file.

    Args:
        pdb_path: Path to the PDB file
    Returns:
        torch.Tensor: The loaded structure coordinates
    Raises:
        FileNotFoundError: If the PDB file does not exist
        ValueError: If the PDB file is improperly formatted or has no coordinates
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    try:
        chain = ProteinChain.from_pdb(pdb_path)
        loaded_protein = ESMProtein.from_protein_chain(chain)
        coordinates = loaded_protein.coordinates

        if coordinates is None:
            raise ValueError("No coordinates found in PDB file.")

        return coordinates
    except Exception as e:
        raise ValueError(f"Error loading PDB file: {e}") from e


@cache_to_pickle(path_param_name="cache_path")
def load_structures(
    domain_dataframe: pd.DataFrame,
    pdb_path_column: str,
    cache_path: Optional[str] = None,
) -> List[torch.Tensor | None]:
    """
    Load structures from PDB files specified in the dataframe.

    Args:
        domain_dataframe: DataFrame containing domain data with PDB paths
        pdb_path_column: Name of the column containing PDB file paths

    Returns:
        List[torch.Tensor]: List of loaded structures
    """
    structures: List[torch.Tensor | None] = []

    for index, row in tqdm(
        domain_dataframe.iterrows(),
        total=len(domain_dataframe),
        desc="Loading structures",
    ):
        pdb_path = row[pdb_path_column]
        if pdb_path is None:
            structures.append(None)
            continue

        try:
            structure = load_structure_from_pdb(pdb_path)
            structures.append(structure)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading PDB file for row {index}: {e}")
            structures.append(None)
            continue

    return structures


def create_embedded_dataframe(
    domain_dataframe: pd.DataFrame,
    sequence_embeddings: List[np.ndarray | None],
    structure_embeddings: List[np.ndarray | None],
    id_column: str,
    label_column: str,
    hex_color_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a new DataFrame with embedded sequences and structures.

    Args:
        domain_dataframe: Original DataFrame containing domain data
        sequence_embeddings: List of sequence embeddings
        structure_embeddings: List of structure embeddings
        id_column: Name of the domain ID column
        label_column: Name of the label column
        hex_color_column: Name of the column with hex color codes for points (optional)

    Returns:
        pd.DataFrame: New DataFrame with embedded sequences and structures
    """
    domain_ids = []
    modalities = []
    embeddings = []
    labels = []
    colors = []

    for (_, row), seq_emb, struct_emb in zip(
        domain_dataframe.iterrows(), sequence_embeddings, structure_embeddings
    ):
        if seq_emb is not None:
            domain_ids.append(row[id_column])
            modalities.append("sequence")
            embeddings.append(seq_emb)
            labels.append(row[label_column])
            if hex_color_column is not None:
                colors.append(row[hex_color_column])

        if struct_emb is not None:
            domain_ids.append(row[id_column])
            modalities.append("structure")
            embeddings.append(struct_emb)
            labels.append(row[label_column])
            if hex_color_column is not None:
                colors.append(row[hex_color_column])

    embedded_dataframe = pd.DataFrame(
        {
            id_column: domain_ids,
            "modality": modalities,
            "embedding": embeddings,
            label_column: labels,
        }
    )

    if hex_color_column is not None:
        embedded_dataframe[hex_color_column] = colors

    return embedded_dataframe


def create_map_dataframe(
    embedded_dataframe: pd.DataFrame, reduced_embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Create a DataFrame suitable for visualization with reduced embeddings.

    Args:
        embedded_dataframe: DataFrame containing embedded sequences and structures
        reduced_embeddings: 2D array of reduced embeddings

    Returns:
        pd.DataFrame: DataFrame suitable for visualization
    """
    map_df = embedded_dataframe.copy()
    map_df = map_df.drop(columns=["embedding"])
    map_df["x"] = reduced_embeddings[:, 0]
    map_df["y"] = reduced_embeddings[:, 1]
    return map_df
