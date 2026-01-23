import argparse
import glob
import os
import pickle
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from clss import CLSSModel
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CLSS embeddings and FAISS index."
    )

    # Inputs
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input FASTA file or directory containing PDB files.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["sequence", "structure", "both"],
        default="sequence",
        help="Modality to embed.",
    )

    # Outputs
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to save cached inputs and embeddings.",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        required=True,
        help="Path to save FAISS index (.faiss).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save metadata CSV (.csv).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=cpu_count() or 1,
        help="Number of worker processes for data loading.",
    )

    return parser.parse_args()


# --- Generic Caching Utility ---


def load_cached_or_compute(
    cache_path: Optional[str],
    compute_fn: Callable[[], Any],
    use_torch_serialization: bool = False,
) -> Any:
    """
    Generic utility to handle check-load-compute-save cycle.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached result from {cache_path}...")
        if use_torch_serialization:
            return torch.load(cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = compute_fn()

    if cache_path:
        print(f"Saving result cache to {cache_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        if use_torch_serialization:
            torch.save(result, cache_path)
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

    return result


# --- Data Loading Functions ---


def load_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """Parses a FASTA file and returns ids and sequences."""
    ids = []
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(str(record.id))
        sequences.append(str(record.seq))
    return ids, sequences


def init_worker() -> None:
    """Initializer for multiprocessing workers."""
    # Only filter warnings in workers if strictly necessary, scoping to likely safe ones
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _process_pdb_file(pdb_path: str) -> Tuple[bool, Any]:
    """Helper function to process a single PDB file."""
    try:
        # Load chain
        chain = ProteinChain.from_pdb(pdb_path)
        protein = ESMProtein.from_protein_chain(chain)

        # ID from filename (without extension)
        file_id = os.path.splitext(os.path.basename(pdb_path))[0]

        # Extract data
        sequence = chain.sequence
        structure = protein.coordinates

        if sequence is None or structure is None or structure.shape[0] == 0:
            raise ValueError("Missing sequence or structure data.")

        return (True, (file_id, sequence, structure))
    except Exception as e:
        return (False, (pdb_path, str(e)))


def load_pdbs(
    pdb_dir: str, processes: int = 1
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """
    Parses PDB files in a directory.
    Returns ids, sequences (extracted from structure), and coordinate tensors.
    """
    ids = []
    sequences = []
    structures = []

    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    if not pdb_files:
        print(f"Warning: No .pdb files found directly in {pdb_dir}.")

    print(f"Found {len(pdb_files)} PDB files.")

    failed_domains = 0

    if processes > 1:
        print(f"Loading PDBs with {processes} processes...")
        counts = {"SUCCESS": 0, "FAILED": 0}
        results = []
        with Pool(processes=processes, initializer=init_worker) as pool:
            with tqdm(total=len(pdb_files), desc="Loading PDBs") as pbar:
                for res in pool.imap_unordered(_process_pdb_file, pdb_files):
                    results.append(res)
                    success = res[0]
                    if success:
                        counts["SUCCESS"] += 1
                    else:
                        counts["FAILED"] += 1
                    pbar.update(1)
                    pbar.set_postfix(success=counts["SUCCESS"], failed=counts["FAILED"])
    else:
        results = [_process_pdb_file(p) for p in tqdm(pdb_files, desc="Loading PDBs")]

    for success, data in results:
        if success:
            file_id, seq_val, struct_val = data
            ids.append(file_id)
            sequences.append(seq_val)
            structures.append(struct_val)
        else:
            pdb_path, error_msg = data
            print(f"Skipping {pdb_path}: {error_msg}")
            failed_domains += 1

    print(f"Loaded {len(ids)} PDBs. Failed: {failed_domains}")
    return ids, sequences, structures


# --- Model & Embedding Functions ---


def load_model(device: str, load_esm3: bool = False) -> CLSSModel:
    print(f"Loading CLSS Model on {device}...")
    model = CLSSModel.from_pretrained()
    model.to(device)
    model.eval()

    if load_esm3:
        print("Loading ESM3 structure encoder (this may take a moment)...")
        model.load_esm3()

    return model

@torch.no_grad()
def embed_sequences(sequences: List[str], model: CLSSModel) -> torch.Tensor:
    sequence_embs = []
    for sequence in tqdm(sequences, desc="Embedding Sequences"):
        sequence_emb = model.embed_sequences([sequence])
        sequence_embs.append(sequence_emb.cpu())

        del sequence_emb
        torch.cuda.empty_cache()

    return torch.cat(sequence_embs)

@torch.no_grad()
def embed_structures(structures: List[torch.Tensor], model: CLSSModel) -> torch.Tensor:
    structure_embs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for structure in tqdm(structures, desc="Embedding Structures"):
        structure = structure.to(device)
        structure_emb = model.embed_structures([structure])
        structure_embs.append(structure_emb.cpu())

        del structure
        del structure_emb
        torch.cuda.empty_cache()

    return torch.cat(structure_embs)


# --- Core Logic Blocks ---


def process_fasta(
    args: argparse.Namespace, model: CLSSModel
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    input_name = os.path.splitext(os.path.basename(args.input))[0]

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    ids, sequences = load_fasta(args.input)
    print(f"Loaded {len(ids)} sequences.")

    # Compute Embeddings
    seq_cache_path = os.path.join(args.cache_dir, f"{input_name}_seq_embeddings.pkl")

    def compute_embs():
        return embed_sequences(sequences, model)

    embs = load_cached_or_compute(
        seq_cache_path, compute_embs, use_torch_serialization=True
    )

    return ids, ["sequence"] * len(ids), [embs]


def process_directory(
    args: argparse.Namespace, model: CLSSModel
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    input_name = os.path.basename(os.path.normpath(args.input))

    # Load PDBs with caching
    pdb_cache_path = os.path.join(args.cache_dir, f"{input_name}_pdbs.pkl")

    def compute_pdbs():
        return load_pdbs(args.input, processes=args.processes)

    ids, sequences, structures = load_cached_or_compute(pdb_cache_path, compute_pdbs)
    print(f"Loaded {len(ids)} valid PDBs.")

    all_embs = []
    all_types = []

    # Sequence Embeddings
    if args.modality in ["sequence", "both"]:
        seq_cache_path = os.path.join(
            args.cache_dir, f"{input_name}_seq_embeddings.pkl"
        )

        def compute_seq_embs():
            return embed_sequences(sequences, model)

        seq_embs = load_cached_or_compute(
            seq_cache_path, compute_seq_embs, use_torch_serialization=True
        )
        all_embs.append(seq_embs)
        all_types.extend(["sequence"] * len(ids))

    # Structure Embeddings
    if args.modality in ["structure", "both"]:
        struct_cache_path = os.path.join(
            args.cache_dir, f"{input_name}_struct_embeddings.pkl"
        )

        def compute_struct_embs():
            return embed_structures(structures, model)

        struct_embs = load_cached_or_compute(
            struct_cache_path, compute_struct_embs, use_torch_serialization=True
        )
        all_embs.append(struct_embs)
        all_types.extend(["structure"] * len(ids))

    # We need to duplicate IDs if both modalities are present
    final_ids = []
    if args.modality == "both":
        # IDs for sequence, then IDs for structure
        final_ids = ids + ids
    else:
        final_ids = ids

    return final_ids, all_types, all_embs


def save_outputs(
    ids: List[str],
    modalities: List[str],
    embedding_tensors: List[torch.Tensor],
    output_index_path: str,
    output_csv_path: str,
) -> None:
    if not embedding_tensors:
        print("No embeddings were generated. Exiting.")
        return

    final_embeddings = torch.cat(embedding_tensors).detach().numpy().astype(np.float32)
    num_vectors, dim = final_embeddings.shape

    print(f"Total vectors: {num_vectors}, Dimension: {dim}")
    if len(ids) != num_vectors:
        raise ValueError(
            f"Mismatch between IDs ({len(ids)}) and embeddings ({num_vectors})! Data integrity compromised."
        )

    # FAISS
    print("Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(dim)
    index.add(final_embeddings)

    print(f"Saving FAISS index to {output_index_path}...")
    faiss.write_index(index, output_index_path)

    # Metadata
    print(f"Saving metadata to {output_csv_path}...")
    df = pd.DataFrame(
        {"faiss_id": range(num_vectors), "domain_id": ids, "modality": modalities}
    )
    df.to_csv(output_csv_path, index=False)
    print("Process complete successfully.")


def main() -> None:
    # Setup
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input path {args.input} does not exist.")

    is_fasta = os.path.isfile(args.input)
    is_dir = os.path.isdir(args.input)

    if args.modality in ["structure", "both"]:
        if is_fasta:
            raise ValueError(
                "Cannot embed structures from a FASTA file. Please provide a directory of PDBs."
            )
        if not is_dir:
            raise ValueError("Structure embedding requires a directory of PDB files.")

    # Create directories
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load Model
    need_esm3 = args.modality in ["structure", "both"]
    model = load_model(args.device, load_esm3=need_esm3)

    # Process
    if is_fasta:
        ids, modalities, embs_list = process_fasta(args, model)
    elif is_dir:
        ids, modalities, embs_list = process_directory(args, model)
    else:
        raise ValueError("Input must be a FASTA file or a directory.")

    # Save
    save_outputs(ids, modalities, embs_list, args.output_index, args.output_csv)


if __name__ == "__main__":
    main()
