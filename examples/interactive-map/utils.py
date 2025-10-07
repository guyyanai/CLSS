import warnings
import os
import pickle
import functools
from typing import Callable, Tuple, Optional


def disable_warnings():
    """
    Disable warnings from the Biotite library.
    """

    warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")
    warnings.filterwarnings("ignore", category=FutureWarning, module="esm")


def create_cache_paths(base_cache_path: Optional[str] = None) -> Tuple:
    """
    Create and return paths for caching various data components.

    Args:
        base_cache_path: Base directory for cache files.

    Returns:
        Tuple: Paths for caching various data components.
    """
    cache_paths = (None, None, None, None)

    if base_cache_path is not None:
        os.makedirs(base_cache_path, exist_ok=True)
        return (
            os.path.join(base_cache_path, "sequences.pkl"),
            os.path.join(base_cache_path, "structures.pkl"),
            os.path.join(base_cache_path, "embeddings.pkl"),
            os.path.join(base_cache_path, "reduced_embeddings.pkl"),
        )

    return cache_paths


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
                print("No cache path provided. Executing function without caching.")
                return func(*args, **kwargs)

            # Check if cache exists
            if os.path.exists(file_path):
                print(f"Loading result from cache: {file_path}")
                with open(file_path, "rb") as f:
                    return pickle.load(f)

            # Execute function and cache result if no cache is found
            print(f"Cache not found. Executing function and saving to: {file_path}")
            result = func(*args, **kwargs)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save result to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
