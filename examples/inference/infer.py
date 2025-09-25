import os
import warnings
from typing import List, Tuple
import argparse
import torch
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from clss import CLSSModel

def setup_inference_args():
    """Set up command line arguments for inference."""
    parser = argparse.ArgumentParser(description="CLSS Inference")
    parser.add_argument(
        "--pdb-dir",
        type=str,
        default="sample-pdbs",
        help="Directory containing PDB files"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="guyyanai/CLSS",
        help="Hugging Face model repository"
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="h32_r10.lckpt",
        help="Model checkpoint filename"
    )
    return parser.parse_args()


def process_pdb_files(pdb_dir: str) -> Tuple[List[str], List[torch.Tensor]]:
    """Process PDB files and extract sequences and structures."""
    sequences = []
    structures = []

    for pdb_file in os.listdir(pdb_dir):
        if not pdb_file.endswith(".pdb"):
            continue

        pdb_file_path = os.path.join(pdb_dir, pdb_file)
        print(f"Processing {pdb_file_path}")

        try:
            # Load the protein chain and extract sequence and structure
            chain = ProteinChain.from_pdb(pdb_file_path)
            loaded_protein = ESMProtein.from_protein_chain(chain)
            sequences.append(loaded_protein.sequence)
            structures.append(loaded_protein.coordinates)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue

    return sequences, structures


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    """Main inference function that can be called as a console script."""
    args = setup_inference_args()
    
    # Load pre-trained model
    print(f"Loading model from {args.model_repo}/{args.model_filename}")
    model = CLSSModel.from_pretrained(args.model_repo, args.model_filename)
    
    # Process PDB files
    print(f"Processing PDB files from {args.pdb_dir}")
    sequences, structures = process_pdb_files(args.pdb_dir)
    
    if not sequences:
        print("No valid PDB files found!")
        return
    
    # Embed sequences
    print("Computing sequence embeddings...")
    sequence_embeddings = model.embed_sequences(sequences)
    no_adapter_sequence_embeddings = model.embed_sequences(sequences, apply_adapter=False)
    per_residue_sequence_embeddings = model.embed_sequence_residues(sequences)

    # Load ESM3 and embed structures
    print("Loading ESM3 and computing structure embeddings...")
    model.load_esm3()
    structure_embeddings = model.embed_structures(structures)
    no_adapter_structure_embeddings = model.embed_structures(structures, apply_adapter=False)
    per_residue_structure_embeddings = model.embed_structure_residues(structures)

    # Print shapes
    print("Sequence embeddings shape:", sequence_embeddings.shape)
    print("No adapter sequence embeddings shape:", no_adapter_sequence_embeddings.shape)
    print("Per-residue sequence embeddings shape:", len(per_residue_sequence_embeddings), per_residue_sequence_embeddings[0].shape)
    print("Structure embeddings shape:", structure_embeddings.shape)
    print("No adapter structure embeddings shape:", no_adapter_structure_embeddings.shape)
    print("Per-residue structure embeddings shape:", len(per_residue_structure_embeddings), per_residue_structure_embeddings[0].shape)
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
