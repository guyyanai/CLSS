from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

from utils import cache_to_pickle


def execute_tsne(
    embeddings: np.ndarray, perplexity: int, max_iterations: int, random_state: int = 0
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        max_iter=max_iterations,
        perplexity=perplexity,
        verbose=1,
    )

    print("Running t-SNE...")
    tsne_results = tsne.fit_transform(embeddings)
    print("Finished t-SNE!")

    return tsne_results


@cache_to_pickle(path_param_name="cache_path")
def apply_dim_reduction(
    embedded_dataframe: pd.DataFrame,
    perplexity: int,
    max_iterations: int,
    random_state: int,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Apply dimensionality reduction to the embeddings in the DataFrame.

    Args:
        embedded_dataframe: DataFrame containing embeddings
        perplexity: Perplexity parameter for t-SNE
        max_iterations: Maximum iterations for t-SNE
        random_state: Random state for reproducibility
        cache_path: Optional path to cache the results

    Returns:
        np.ndarray: 2D array of reduced embeddings
    """
    all_embeddings = np.array(embedded_dataframe["embedding"].tolist())
    reduced_embeddings = execute_tsne(
        all_embeddings, perplexity, max_iterations, random_state
    )

    return reduced_embeddings
