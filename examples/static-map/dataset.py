import os
import logging
from typing import List, Optional
from functools import lru_cache
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
import torch
import numpy as np

from utils import cache_to_pickle
from constants import MODALITY_COLUMN, MODALITY_SEQUENCE, MODALITY_STRUCTURE, EMBEDDING_COLUMN, X_COLUMN, Y_COLUMN

logger = logging.getLogger(__name__)


def load_domain_dataset(
    dataset_path: str,
    id_column: str,
    label_column: str,
    optional_columns: List[Optional[str]],
) -> pd.DataFrame:
    """
    Load the domain dataset from a CSV file and validate required columns.

    Args:
        dataset_path: Path to the CSV dataset file
        id_column: Name of the domain ID column
        label_column: Name of the label column
        optional_columns: List of optional column names to validate

    Returns:
        pd.DataFrame: Loaded domain dataset

    Raises:
        ValueError: If required columns are missing in the dataset
    """
    domain_dataframe = pd.read_csv(dataset_path, dtype={id_column: str})
    domain_dataframe = domain_dataframe.where(pd.notna(domain_dataframe), None)

    if id_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{id_column}' not found in dataset")

    if label_column not in domain_dataframe.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")

    for col in optional_columns:
        if col is not None and col not in domain_dataframe.columns:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    return domain_dataframe


@lru_cache(maxsize=128)
def _load_fasta_file(fasta_path: str) -> dict:
    """
    Load entire FASTA file into memory and cache the results.
    
    This function is cached to avoid repeated file I/O when accessing
    the same FASTA file multiple times.

    Args:
        fasta_path: Path to the FASTA file

    Returns:
        dict: Dictionary mapping record IDs to SeqRecord objects

    Raises:
        FileNotFoundError: If the FASTA file does not exist
        ValueError: If the FASTA file is empty or improperly formatted
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File not found: {fasta_path}")
    if os.path.getsize(fasta_path) == 0:
        raise ValueError("The FASTA file is empty.")

    try:
        records = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            raise ValueError("No FASTA sequences found in the file.")
        return records
    except Exception as e:
        raise ValueError(f"Error parsing FASTA file: {e}") from e


def load_sequence_from_fasta(fasta_path: str, record_id: Optional[str] = None) -> str:
    """
    Load a sequence from a FASTA file using cached file data.

    This function uses a cached loader to efficiently handle repeated
    access to the same FASTA file.

    Args:
        fasta_path: Path to the FASTA file
        record_id: Optional ID of the specific record to load (if None, load the first record)

    Returns:
        str: The loaded sequence

    Raises:
        FileNotFoundError: If the FASTA file does not exist
        ValueError: If the FASTA file is empty or improperly formatted, or record_id not found
    """
    # Load the FASTA file (will be cached after first call)
    records = _load_fasta_file(fasta_path)
    
    # Get the requested record
    if record_id is None:
        # Return first record
        record = next(iter(records.values()))
    else:
        record = records.get(record_id)
        if record is None:
            raise ValueError(f"Record ID '{record_id}' not found in FASTA file.")
    
    # Validate the content
    if not record.id or not record.seq:
        raise ValueError("Invalid FASTA format: header or sequence missing.")
    if len(record.seq) == 0:
        raise ValueError("Sequence is empty.")

    return str(record.seq)


@lru_cache(maxsize=128)
def load_protein_from_pdb_file(pdb_path: str) -> ESMProtein:
    """
    Load a protein structure from a PDB file.
    Args:
        pdb_path: Path to the PDB file
    Returns:
        ESMProtein: The loaded protein structure
    Raises:
        FileNotFoundError: If the PDB file does not exist
        ValueError: If the PDB file is improperly formatted
    """

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    try:
        chain = ProteinChain.from_pdb(pdb_path)
        loaded_protein = ESMProtein.from_protein_chain(chain)
        return loaded_protein
    except Exception as e:
        raise ValueError(f"Error loading PDB file: {e}") from e


def load_sequence_from_pdb(pdb_path: str) -> str:
    """
    Load a sequence from a PDB file.

    Args:
        pdb_path: Path to the PDB file
    Returns:
        str: The loaded sequence
    Raises:
        FileNotFoundError: If the PDB file does not exist
        ValueError: If the PDB file is improperly formatted or has no sequence
    """

    loaded_protein = load_protein_from_pdb_file(pdb_path)
    try:
        sequence = loaded_protein.sequence

        if sequence is None:
            raise ValueError("No sequence found in PDB file.")

        return sequence
    except Exception as e:
        raise ValueError(f"Error loading PDB file: {e}") from e


@cache_to_pickle(path_param_name="cache_path")
def load_sequences(
    domain_dataframe: pd.DataFrame,
    domain_id_column: str,
    use_pdb_sequences: bool = False,
    pdb_path_column: Optional[str] = None,
    fasta_path_column: Optional[str] = None,
    use_record_id: bool = False,
    cache_path: Optional[str] = None,
) -> List[str | None]:
    """
    Load sequences from FASTA/PDB files specified in the dataframe.

    Args:
        domain_dataframe: DataFrame containing domain data
        domain_id_column: Name of the domain ID column
        use_pdb_sequences: Whether to use PDB sequences instead of FASTA
        pdb_path_column: Name of the column containing PDB file paths
        fasta_path_column: Name of the column containing FASTA file paths
        use_record_id: Whether to use the domain ID as the record ID in FASTA files
        cache_path: Optional path to cache the loaded sequences

    Returns:
        List[str | None]: List of loaded sequences
    """
    sequences: List[str | None] = []
    for index, row in tqdm(
        domain_dataframe.iterrows(),
        total=len(domain_dataframe),
        desc="Loading sequences",
    ):
        should_use_pdb = (
            use_pdb_sequences and pdb_path_column and row[pdb_path_column] is not None
        )
        file_path_column = pdb_path_column if should_use_pdb else fasta_path_column

        file_path = str(row[file_path_column])
        if file_path is None:
            sequences.append(None)
            continue

        fasta_record_id = str(row[domain_id_column]) if use_record_id else None
        try:
            sequence = (
                load_sequence_from_pdb(file_path)
                if should_use_pdb
                else load_sequence_from_fasta(file_path, fasta_record_id)
            )
            sequences.append(sequence)
        except (FileNotFoundError, ValueError) as e:
            logger.error(
                f"Error loading {'PDB' if should_use_pdb else 'FASTA'} file for row {index}: {e}"
            )
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
    loaded_protein = load_protein_from_pdb_file(pdb_path)

    try:
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
        cache_path: Optional path to cache the loaded structures

    Returns:
        List[torch.Tensor]: List of loaded structures
    """
    structures: List[torch.Tensor | None] = []

    for index, row in tqdm(
        domain_dataframe.iterrows(),
        total=len(domain_dataframe),
        desc="Loading structures",
    ):
        pdb_path = str(row[pdb_path_column])
        if pdb_path is None:
            structures.append(None)
            continue

        try:
            structure = load_structure_from_pdb(pdb_path)
            structures.append(structure)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading PDB file for row {index}: {e}")
            structures.append(None)
            continue

    return structures


def create_embedded_dataframe(
    domain_dataframe: pd.DataFrame,
    sequence_embeddings: List[np.ndarray | None],
    structure_embeddings: List[np.ndarray | None],
    id_column: str,
) -> pd.DataFrame:
    """
    Create a new DataFrame with embedded sequences and structures.

    Args:
        domain_dataframe: Original DataFrame containing domain data
        sequence_embeddings: List of sequence embeddings
        structure_embeddings: List of structure embeddings
        id_column: Name of the domain ID column

    Returns:
        pd.DataFrame: New DataFrame with embedded sequences and structures
    """
    domain_ids = []
    modalities = []
    embeddings = []

    for (_, row), seq_emb, struct_emb in zip(
        domain_dataframe.iterrows(), sequence_embeddings, structure_embeddings
    ):
        if seq_emb is not None:
            domain_ids.append(row[id_column])
            modalities.append(MODALITY_SEQUENCE)
            embeddings.append(seq_emb)

        if struct_emb is not None:
            domain_ids.append(row[id_column])
            modalities.append(MODALITY_STRUCTURE)
            embeddings.append(struct_emb)

    embedded_dataframe = pd.DataFrame(
        {
            id_column: domain_ids,
            MODALITY_COLUMN: modalities,
            EMBEDDING_COLUMN: embeddings,
        }
    )

    # Merge embedded dataframe with original dataframe to retain additional columns
    embedded_dataframe = embedded_dataframe.merge(
        domain_dataframe, on=id_column, how="left"
    )

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
    map_df = map_df.drop(columns=[EMBEDDING_COLUMN])
    map_df[X_COLUMN] = reduced_embeddings[:, 0]
    map_df[Y_COLUMN] = reduced_embeddings[:, 1]
    return map_df
