import os
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.manifold import TSNE
from typing import List, Tuple

ecod_redundancy = "F40"
output_dir = os.path.join("outputs", ecod_redundancy)

input_dict_path = os.path.join(
    output_dir, f"domain_embeddings_dict.pkl"
)
output_pkl_path = os.path.join(output_dir, f"tsne_results.pkl")


def load_embeddings_dict(dict_path: str) -> Tuple[List[str], List[int], np.ndarray]:
    embeddings_dict = None

    print("Loading pickle file...")
    with open(dict_path, "rb") as f:
        embeddings_dict = pickle.load(f)

    clss_embeds_seqs = []
    clss_embeds_seq_stretches = []
    clss_embeds_strs = []
    xfolds = []
    domain_ids = []

    print("Iterating through loaded dictionary")
    for [key, value] in tqdm(embeddings_dict.items()):
        clss_embeds_seqs.append(value["embeddings"][0])
        clss_embeds_seq_stretches.append(value["embeddings"][1])
        clss_embeds_strs.append(value["embeddings"][2])
        xfolds.append(value["xfold"])
        domain_ids.append(key)

    clss_embeds_seq_np = np.array(clss_embeds_seqs)
    clss_embeds_seq_stretch_np = np.array(clss_embeds_seq_stretches)
    clss_embeds_str_np = np.array(clss_embeds_strs)
    clss_embeds_all_np = np.concatenate(
        [clss_embeds_seq_np, clss_embeds_seq_stretch_np, clss_embeds_str_np]
    )

    return domain_ids, xfolds, clss_embeds_all_np


def execute_tsne(embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=0, n_iter=1000, perplexity=30, verbose=1)

    print("Running t-SNE...")
    tsne_results = tsne.fit_transform(embeddings)
    print("Finished t-SNE!")

    return tsne_results


def write_output(
    domain_ids: List[str], xfolds: List[int], tsne_results: np.ndarray, output_path: str
):
    data_to_save = {
        "domain_id": domain_ids,
        "xfold": xfolds,
        "tsne_results": tsne_results,
    }

    print("Dumping output to pickle file...")
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)


if __name__ == "__main__":
    domain_ids, xfolds, embeddings = load_embeddings_dict(input_dict_path)

    tsne_results = execute_tsne(embeddings)

    write_output(domain_ids, xfolds, tsne_results, output_pkl_path)
