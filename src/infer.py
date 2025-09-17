import os
from huggingface_hub import hf_hub_download
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from model import CLSSModel
import torch

# Path to folder with pdb files
pdb_files_dir = "pdbs"

# Download the pre-trained model from Hugging Face
local_path = hf_hub_download(
    repo_id="guyyanai/CLSS",
    filename="h32_r10.lckpt",
    repo_type="model"
)

# Load the model
model = CLSSModel.load_from_checkpoint(local_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Process each pdb file in the directory
sequences = []
structures = []

for pdb_file in os.listdir(pdb_files_dir):
    if not pdb_file.endswith(".pdb"):
        continue

    pdb_file_path = os.path.join(pdb_files_dir, pdb_file)
    print(f"Processing {pdb_file_path}")

    # Load the protein chain and extract sequence and structure
    chain = ProteinChain.from_pdb(pdb_file_path)
    loaded_protein = ESMProtein.from_protein_chain(chain)
    sequences.append(loaded_protein.sequence)
    structures.append(loaded_protein.coordinates)

# Embed sequences
sequence_embeddings = model.embed_sequences(sequences)
no_adapter_sequence_embeddings = model.embed_sequences(sequences, apply_adapter=False)
per_residue_sequence_embeddings = model.embed_sequence_residues(sequences)

# Make sure the ESM3 model is loaded
model.load_esm3()

# Embed structures
structure_embeddings = model.embed_structures(structures)
no_adapter_structure_embeddings = model.embed_structures(structures, apply_adapter=False)
per_residue_structure_embeddings = model.embed_structure_residues(structures)

print("Sequence embeddings shape:", sequence_embeddings.shape)
print("No adapter sequence embeddings shape:", no_adapter_sequence_embeddings.shape)
print("Per-residue sequence embeddings shape:", len(per_residue_sequence_embeddings), per_residue_sequence_embeddings[0].shape)
print("Structure embeddings shape:", structure_embeddings.shape)
print("No adapter structure embeddings shape:", no_adapter_structure_embeddings.shape)
print("Per-residue structure embeddings shape:", len(per_residue_structure_embeddings), per_residue_structure_embeddings[0].shape)
