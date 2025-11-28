from args import parse_args
from dataset import (
    load_domain_dataset,
    load_sequences,
    load_structures,
    create_embedded_dataframe,
    create_map_dataframe,
)
from embeddings import load_clss, embed_dataframe
from dim_reducer import apply_dim_reduction
from mapper import create_and_export_visualization
from utils import create_cache_paths, disable_warnings

# Constants for DataFrame columns
SEQUENCE_COLUMN = "sequence"
STRUCTURE_COLUMN = "structure"

# Parse command-line arguments
args = parse_args()

# Disable Biotite and ESM warnings
disable_warnings()

# Create cache paths
(
    sequences_cache_path,
    structures_cache_path,
    sequence_embeddings_cache_path,
    structure_embeddings_cache_path,
    reduced_embeddings_cache_path,
) = create_cache_paths(args.cache_path)

domain_dataframe = load_domain_dataset(
    args.dataset_path,
    args.id_column,
    args.label_column,
    [
        args.fasta_path_column,
        args.pdb_path_column,
        args.hex_color_column,
        args.line_width_column,
        args.line_color_column,
        args.alpha_column,
    ],
)
print(f"Loaded domain dataset with {len(domain_dataframe)} entries")

if args.fasta_path_column or args.use_pdb_sequences:
    print(
        f"Loading sequences from {'FASTA' if not args.use_pdb_sequences else 'PDB'} files..."
    )
    sequences = load_sequences(
        domain_dataframe=domain_dataframe,
        domain_id_column=args.id_column,
        use_pdb_sequences=args.use_pdb_sequences,
        pdb_path_column=args.pdb_path_column,
        fasta_path_column=args.fasta_path_column,
        use_record_id=args.use_record_id,
        cache_path=sequences_cache_path,
    )
    domain_dataframe[SEQUENCE_COLUMN] = sequences
    print(
        f"Loaded {len([s for s in sequences if s])} sequences from {'FASTA' if not args.use_pdb_sequences else 'PDB'} files."
    )

if args.pdb_path_column and not args.exclude_structures:
    print("Loading structures from PDB files...")
    structures = load_structures(
        domain_dataframe=domain_dataframe,
        pdb_path_column=args.pdb_path_column,
        cache_path=structures_cache_path,
    )
    domain_dataframe[STRUCTURE_COLUMN] = structures
    print(
        f"Loaded {len([s for s in structures if s is not None])} structures from PDB files."
    )

# Load CLSS model
clss_model = load_clss(args.model_repo, args.model_filename)

# Embed sequences and structures
sequence_embeddings, structure_embeddings = embed_dataframe(
    clss_model,
    domain_dataframe,
    sequence_column=(
        SEQUENCE_COLUMN
        if args.fasta_path_column or (args.use_pdb_sequences and args.pdb_path_column)
        else None
    ),
    structure_column=(
        STRUCTURE_COLUMN
        if args.pdb_path_column and not args.exclude_structures
        else None
    ),
    sequence_emb_cache_path=sequence_embeddings_cache_path,
    structure_emb_cache_path=structure_embeddings_cache_path,
)

embedded_dataframe = create_embedded_dataframe(
    domain_dataframe,
    sequence_embeddings,
    structure_embeddings,
    id_column=args.id_column,
    label_column=args.label_column,
    hex_color_column=args.hex_color_column,
    line_width_column=args.line_width_column,
    line_color_column=args.line_color_column,
    alpha_column=args.alpha_column,
)

print(
    f"Created embedded dataframe with {len(embedded_dataframe)} entries, with {len(embedded_dataframe[embedded_dataframe['modality'] == 'sequence'])} sequence embeddings and {len(embedded_dataframe[embedded_dataframe['modality'] == 'structure'])} structure embeddings."
)

reduced_embeddings = apply_dim_reduction(
    embedded_dataframe,
    perplexity=args.tsne_perplexity,
    max_iterations=args.tsne_max_iterations,
    random_state=args.tsne_random_state,
    cache_path=reduced_embeddings_cache_path,
)
print(f"Reduced embeddings to 2D with shape {reduced_embeddings.shape}.")

map_dataframe = create_map_dataframe(embedded_dataframe, reduced_embeddings)
print(f"Created map dataframe with {len(map_dataframe)} entries.")

# Create and export interactive visualization
create_and_export_visualization(
    map_dataframe=map_dataframe,
    id_column=args.id_column,
    label_column=args.label_column,
    output_path=args.html_output_path,
    hex_color_column=args.hex_color_column,
    line_width_column=args.line_width_column,
    line_color_column=args.line_color_column,
    alpha_column=args.alpha_column,
    title="CLSS Protein Domain Interactive Map",
)
