import os
import pickle
import functools

def cache_to_pickle(path_param_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract path from arguments
            file_path = kwargs.get(path_param_name, None)
            
            if not file_path:
                raise ValueError(f"Path argument '{path_param_name}' is required.")
            
            # Check if cache exists
            if os.path.exists(file_path):
                print(f"Loading result from cache: {file_path}")
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            
            # Execute function and cache result if no cache is found
            print(f"Cache not found. Executing function and saving to: {file_path}")
            result = func(*args, **kwargs)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save result to pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        
        return wrapper
    
    return decorator
