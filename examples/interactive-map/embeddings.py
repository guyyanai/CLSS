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
def embed_dataframe(
    model: CLSSModel,
    domain_dataframe: pd.DataFrame,
    sequence_column: Optional[str],
    structure_column: Optional[str],
    cache_path: Optional[str] = None,
) -> Tuple[List[np.ndarray | None], List[np.ndarray | None]]:
    """
    Embed sequences in the dataframe using the CLSS model.
    Args:
        model (CLSSModel): The pre-trained CLSS model
        domain_dataframe (pd.DataFrame): DataFrame containing domain data with sequences
        sequence_column (Optional[str]): Name of the column containing sequences
        structure_column (Optional[str]): Name of the column containing structures
        cache_path (Optional[str]): Path to the cache file (if any)
    Returns:
        Tuple[List[np.ndarray | None], List[np.ndarray | None]]: List of sequence embeddings and list of structure embeddings
    """

    sequence_embeddings: List[np.ndarray | None] = []
    structure_embeddings: List[np.ndarray | None] = []

    for _, row in tqdm(
        domain_dataframe.iterrows(),
        total=len(domain_dataframe),
        desc="Embedding sequences",
    ):
        sequence = row[sequence_column] if sequence_column else None
        structure = row[structure_column] if structure_column else None

        if sequence is None:
            sequence_embeddings.append(None)
        else:
            with torch.no_grad():
                sequence_embedding = model.embed_sequences([sequence])
                sequence_embeddings.append(sequence_embedding[0].cpu().numpy())

        if structure is None:
            structure_embeddings.append(None)
        else:
            with torch.no_grad():
                structure_embedding = model.embed_structures([structure])
                structure_embeddings.append(structure_embedding[0].cpu().numpy())

    return sequence_embeddings, structure_embeddings
