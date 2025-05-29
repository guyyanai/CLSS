import os
import warnings
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import pickle
import torch
from biotools.tools.esm_inference import (
    esm2_infer_sequences,
    esm3_infer_structures,
)
from biotools.tools.domain_loading import load_domain_from_pdb
from biotools.tools.model_loading import load_clss, load_esm3
from biotools.tools.sequences import (
    sample_subsequences,
)
from utils import cache_to_pickle

ecod_redundancy = 'F40'
# ecod_redundancy = 'F100'

# Constants
structures_path = f'data/{ecod_redundancy}structures/'
dataset_path = f'data/{ecod_redundancy}/domains.csv'
checkpoint = f'checkpoints/F100_b180_g8_h32_r10_AF2_1M.lckpt'
output_dir = 'outputs'
cache_dir = 'cache'

def get_cache_paths() -> Tuple[str, str]:
    loaded_domains_cache_path = os.path.join(cache_dir, f'{ecod_redundancy}_loaded_domains.pkl')
    embeddings_cache_path = os.path.join(cache_dir, f'{ecod_redundancy}_embeddings.pkl')

    return loaded_domains_cache_path, embeddings_cache_path

def get_output_path() -> str:
    return os.path.join(output_dir, f'{ecod_redundancy}_domain_embeddings_dict.pkl')

@cache_to_pickle(path_param_name='cache_path')
def load_domains(domain_ids: List[str], cache_path: str) -> Tuple[List[str], List[str], List[torch.Tensor], List[str]]:
    sequences = []
    structures = []
    bad_ids = []

    for domain_id in tqdm(domain_ids, desc='Loading domains'):
        try:
            sequence, structure = load_domain_from_pdb(domain_id, structures_path, 'ECOD-UID', False, '.pdbnum.pdb')

            if sequence is None or structure is None:
                raise Exception(f'sequence or structures is None')
        except Exception as e:
            print(f'An error occurred while loading domain: {domain_id}, error: {e}')
            bad_ids.append(domain_id)
            continue

        sequences.append(sequence)
        structures.append(structure)
    
    subsequences = [sample_subsequences(sequence, 1, 20, 60)[0] if len(sequence) > 20 else sequence for sequence in sequences]

    return sequences, subsequences, structures, bad_ids

@cache_to_pickle(path_param_name='cache_path')
def compute_embeddings(sequences: List[str], subsequences: List[str], structures: List[torch.Tensor], cache_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    esm2, esm2_tokenizer, esm2_projection_head, esm3_projection_head = load_clss(checkpoint)
    esm3 = load_esm3()
    sequence_embeddings = esm2_infer_sequences(sequences, esm2, esm2_tokenizer, esm2_projection_head, True) # type: ignore
    subsequences_embeddings = esm2_infer_sequences(subsequences, esm2, esm2_tokenizer, esm2_projection_head, True) # type: ignore
    structure_embeddings = esm3_infer_structures(structures, esm3, esm3_projection_head, True) # type: ignore
    
    return sequence_embeddings, subsequences_embeddings, structure_embeddings

def write_embedding_dict(dataset: pd.DataFrame, sequence_embeddings: torch.Tensor, subsequences_embeddings: torch.Tensor, structure_embeddings: torch.Tensor, output_path: str):
    dict_runs = {}

    for index, domain in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        try:
            dict_runs[domain.domain_id] = {
                "embeddings": torch.stack((sequence_embeddings[index], subsequences_embeddings[index], structure_embeddings[index])).cpu().detach().numpy(), # type: ignore
                "xfold": domain.xfold
            }
        except Exception as e:
            print(f'An error occurred while setting dict for domain: {domain.domain_id}, index: {index}, exception: {e}')

    with open(output_path, 'wb') as f:
        pickle.dump(dict_runs, f)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    loaded_domains_cache_path, embeddings_cache_path = get_cache_paths()
    dict_output_path = get_output_path()

    dataset = pd.read_csv(dataset_path, dtype={ 'domain_id': str })

    sequences, subsequences, structures, bad_ids = load_domains(dataset.domain_id.tolist(), cache_path=loaded_domains_cache_path)

    dataset = dataset[~dataset.domain_id.isin(bad_ids)].reset_index()

    sequence_embeddings, subsequences_embeddings, structure_embeddings = compute_embeddings(sequences, subsequences, structures, cache_path=embeddings_cache_path)
    
    write_embedding_dict(dataset, sequence_embeddings, subsequences_embeddings, structure_embeddings, dict_output_path)
