from typing import Optional, List, Tuple
import logging
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from utils import cache_to_pickle
from clss import CLSSModel
from constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


def load_clss(model_repo: str, model_filename: str) -> CLSSModel:
    """
    Load a pre-trained CLSS model.
    Args:
        model_repo: HuggingFace model repository name
        model_filename: Model checkpoint filename
    Returns:
        CLSSModel: The loaded CLSS model
    """

    logger.info(f"Loading CLSS model from {model_repo}/{model_filename}")
    model = CLSSModel.from_pretrained(model_repo, model_filename)
    model.load_esm3()
    model = model.eval()
    return model

@cache_to_pickle(path_param_name="cache_path")
def embed_sequences(
    model: CLSSModel,
    sequences: List[str | None],
    cache_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[np.ndarray | None]:
    """
    Embed a list of sequences using the CLSS model with batching for efficiency.
    
    Args:
        model (CLSSModel): The pre-trained CLSS model
        sequences (List[str | None]): List of protein sequences
        cache_path (Optional[str]): Path to the cache file (if any)
        batch_size (int): Number of sequences to process in each batch
    
    Returns:
        List[np.ndarray | None]: List of sequence embeddings
    """
    sequence_embeddings: List[np.ndarray | None] = [None] * len(sequences)
    
    # Separate valid sequences from None values
    valid_indices = [i for i, seq in enumerate(sequences) if seq is not None]
    valid_sequences = [sequences[i] for i in valid_indices]
    
    if not valid_sequences:
        return sequence_embeddings
    
    # Process in batches
    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(valid_sequences), batch_size),
            desc="Embedding sequences",
            total=(len(valid_sequences) + batch_size - 1) // batch_size,
        ):
            batch_end = min(batch_start + batch_size, len(valid_sequences))
            batch = valid_sequences[batch_start:batch_end]
            
            # Embed entire batch at once
            batch_embeddings = model.embed_sequences(batch)
            
            # Store results at correct indices
            for i, embedding in enumerate(batch_embeddings):
                original_idx = valid_indices[batch_start + i]
                sequence_embeddings[original_idx] = embedding.cpu().numpy()

    return sequence_embeddings

@cache_to_pickle(path_param_name="cache_path")
def embed_structures(
    model: CLSSModel,
    structures: List[torch.Tensor | None],
    cache_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[np.ndarray | None]:
    """
    Embed a list of structures using the CLSS model with batching for efficiency.
    
    Args:
        model (CLSSModel): The pre-trained CLSS model
        structures (List[torch.Tensor | None]): List of protein structures
        cache_path (Optional[str]): Path to the cache file (if any)
        batch_size (int): Number of structures to process in each batch

    Returns:
        List[np.ndarray | None]: List of structure embeddings
    """
    structure_embeddings: List[np.ndarray | None] = [None] * len(structures)
    
    # Separate valid structures from None values
    valid_indices = [i for i, struct in enumerate(structures) if struct is not None]
    valid_structures = [structures[i] for i in valid_indices]
    
    if not valid_structures:
        return structure_embeddings
    
    # Process in batches
    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(valid_structures), batch_size),
            desc="Embedding structures",
            total=(len(valid_structures) + batch_size - 1) // batch_size,
        ):
            batch_end = min(batch_start + batch_size, len(valid_structures))
            batch = valid_structures[batch_start:batch_end]
            
            # Embed entire batch at once
            batch_embeddings = model.embed_structures(batch)
            
            # Store results at correct indices
            for i, embedding in enumerate(batch_embeddings):
                original_idx = valid_indices[batch_start + i]
                structure_embeddings[original_idx] = embedding.cpu().numpy()

    return structure_embeddings


def embed_dataframe(
    model: CLSSModel,
    domain_dataframe: pd.DataFrame,
    sequence_column: Optional[str],
    structure_column: Optional[str],
    sequence_emb_cache_path: Optional[str] = None,
    structure_emb_cache_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
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
        batch_size (int): Number of items to process in each batch
        
    Returns:
        Tuple[List[np.ndarray | None], List[np.ndarray | None]]: List of sequence embeddings and list of structure embeddings
    """

    sequences = domain_dataframe[sequence_column].tolist() if sequence_column else [None] * len(domain_dataframe)
    structures = domain_dataframe[structure_column].tolist() if structure_column else [None] * len(domain_dataframe)

    sequence_embeddings = embed_sequences(model, sequences, cache_path=sequence_emb_cache_path, batch_size=batch_size)
    structure_embeddings = embed_structures(model, structures, cache_path=structure_emb_cache_path, batch_size=batch_size)

    return sequence_embeddings, structure_embeddings
