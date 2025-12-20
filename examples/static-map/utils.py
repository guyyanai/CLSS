import warnings
import os
import pickle
import functools
import logging
from typing import Callable, Optional, NamedTuple

from constants import (
    CACHE_SEQUENCES_FILE,
    CACHE_STRUCTURES_FILE,
    CACHE_SEQUENCE_EMBEDDINGS_FILE,
    CACHE_STRUCTURE_EMBEDDINGS_FILE,
    CACHE_REDUCED_EMBEDDINGS_FILE,
)

logger = logging.getLogger(__name__)


class CachePaths(NamedTuple):
    """Structured container for cache file paths."""
    sequences: Optional[str]
    structures: Optional[str]
    sequence_embeddings: Optional[str]
    structure_embeddings: Optional[str]
    reduced_embeddings: Optional[str]


def disable_warnings():
    """
    Disable specific warnings from external libraries.
    Note: Only disables warnings from biotite and esm, not all warnings.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")
    warnings.filterwarnings("ignore", category=FutureWarning, module="esm")


def create_cache_paths(base_cache_path: Optional[str] = None) -> CachePaths:
    """
    Create and return structured paths for caching various data components.

    Args:
        base_cache_path: Base directory for cache files.

    Returns:
        CachePaths: Named tuple with paths for caching various data components.
    """
    if base_cache_path is None:
        return CachePaths(None, None, None, None, None)
    
    os.makedirs(base_cache_path, exist_ok=True)
    return CachePaths(
        sequences=os.path.join(base_cache_path, CACHE_SEQUENCES_FILE),
        structures=os.path.join(base_cache_path, CACHE_STRUCTURES_FILE),
        sequence_embeddings=os.path.join(base_cache_path, CACHE_SEQUENCE_EMBEDDINGS_FILE),
        structure_embeddings=os.path.join(base_cache_path, CACHE_STRUCTURE_EMBEDDINGS_FILE),
        reduced_embeddings=os.path.join(base_cache_path, CACHE_REDUCED_EMBEDDINGS_FILE),
    )


def cache_to_pickle(path_param_name: str) -> Callable:
    """
    Decorator to cache the result of a function to a pickle file. If the pickle file exists, load the result from it instead of executing the function.
    Args:
        path_param_name: Name of the parameter in the decorated function that specifies the path to the pickle file.
    Returns:
        Callable: Decorated function with caching behavior.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract path from arguments
            file_path = kwargs.get(path_param_name, None)

            if not file_path:
                logger.info("No cache path provided. Executing function without caching.")
                return func(*args, **kwargs)

            # Check if cache exists
            if os.path.exists(file_path):
                logger.info(f"Loading result from cache: {file_path}")
                with open(file_path, "rb") as f:
                    return pickle.load(f)

            # Execute function and cache result if no cache is found
            logger.info(f"Cache not found. Executing function and saving to: {file_path}")
            result = func(*args, **kwargs)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save result to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
