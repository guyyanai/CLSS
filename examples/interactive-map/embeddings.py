from typing import Optional, List, Tuple
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from utils import cache_to_pickle
from clss import CLSSModel


def load_clss(model_repo: str, model_filename: str) -> CLSSModel:
    """
    Load a pre-trained CLSS model.
    Args:
        model_repo: HuggingFace model repository name
        model_filename: Model checkpoint filename
    Returns:
        CLSSModel: The loaded CLSS model
    """

    print(f"Loading CLSS model from {model_repo}/{model_filename}")
    model = CLSSModel.from_pretrained(model_repo, model_filename)
    model.load_esm3()
    model = model.eval()
    return model

@cache_to_pickle(path_param_name="cache_path")
def embed_sequences(
    model: CLSSModel,
    sequences: List[str | None],
    cache_path: Optional[str] = None,
) -> List[np.ndarray | None]:
    """
    Embed a list of sequences using the CLSS model.
    Args:
        model (CLSSModel): The pre-trained CLSS model
        sequences (List[str | None]): List of protein sequences
        cache_path (Optional[str]): Path to the cache file (if any)
    
    Returns:
        List[np.ndarray | None]: List of sequence embeddings
    """
    sequence_embeddings: List[np.ndarray | None] = []

    for sequence in tqdm(
        sequences,
        total=len(sequences),
        desc="Embedding sequences",
    ):
        if sequence is None:
            sequence_embeddings.append(None)
        else:
            with torch.no_grad():
                sequence_embedding = model.embed_sequences([sequence])
                sequence_embeddings.append(sequence_embedding[0].cpu().numpy())

    return sequence_embeddings

@cache_to_pickle(path_param_name="cache_path")
def embed_structures(
    model: CLSSModel,
    structures: List[torch.Tensor | None],
    cache_path: Optional[str] = None,
) -> List[np.ndarray | None]:
    """
    Embed a list of structures using the CLSS model.
    Args:
        model (CLSSModel): The pre-trained CLSS model
        structures (List[torch.Tensor | None]): List of protein structures
        cache_path (Optional[str]): Path to the cache file (if any)

    Returns:
        List[np.ndarray | None]: List of structure embeddings
    """
    structure_embeddings: List[np.ndarray | None] = []

    for structure in tqdm(
        structures,
        total=len(structures),
        desc="Embedding structures",
    ):
        if structure is None:
            structure_embeddings.append(None)
        else:
            with torch.no_grad():
                structure_embedding = model.embed_structures([structure])
                structure_embeddings.append(structure_embedding[0].cpu().numpy())

    return structure_embeddings


def embed_dataframe(
    model: CLSSModel,
    domain_dataframe: pd.DataFrame,
    sequence_column: Optional[str],
    structure_column: Optional[str],
    sequence_emb_cache_path: Optional[str] = None,
    structure_emb_cache_path: Optional[str] = None,
) -> Tuple[List[np.ndarray | None], List[np.ndarray | None]]:
    """
    Embed sequences in the dataframe using the CLSS model.
    Args:
        model (CLSSModel): The pre-trained CLSS model
        domain_dataframe (pd.DataFrame): DataFrame containing domain data with sequences
        sequence_column (Optional[str]): Name of the column containing sequences
        structure_column (Optional[str]): Name of the column containing structures
        sequence_emb_cache_path (Optional[str]): Path to the sequence cache file (if any)
        structure_emb_cache_path (Optional[str]): Path to the structure cache file (if any)
    Returns:
        Tuple[List[np.ndarray | None], List[np.ndarray | None]]: List of sequence embeddings and list of structure embeddings
    """

    sequences = domain_dataframe[sequence_column].tolist() if sequence_column else [None] * len(domain_dataframe)
    structures = domain_dataframe[structure_column].tolist() if structure_column else [None] * len(domain_dataframe)

    sequence_embeddings = embed_sequences(model, sequences, cache_path=sequence_emb_cache_path)
    structure_embeddings = embed_structures(model, structures, cache_path=structure_emb_cache_path)

    return sequence_embeddings, structure_embeddings
