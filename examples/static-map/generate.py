"""
Static map generator for CLSS model embeddings.

This script generates publication-quality static visualizations of protein domain embeddings
using the CLSS model. It supports both sequence and structure embeddings with dimensionality
reduction via t-SNE.

Usage:
    python generate.py --dataset-path data.csv --id-column domain_id --label-column label \\
                       --fasta-path-column fasta_path --output-path output.png

For more options, run:
    python generate.py --help
"""

import logging

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
from plotter import create_static_scatter_plot
from utils import create_cache_paths, disable_warnings
from constants import SEQUENCE_COLUMN, STRUCTURE_COLUMN, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )


def main():
    """Main entry point for static map generation."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging (could add --verbose flag to args if needed)
    setup_logging(verbose=False)
    logger = logging.getLogger(__name__)
    
    # Disable Biotite and ESM warnings
    disable_warnings()
    
    # Create cache paths (now returns CachePaths NamedTuple)
    cache_paths = create_cache_paths(args.cache_path)
    
    # Load domain dataset
    domain_dataframe = load_domain_dataset(
        args.dataset_path,
        args.id_column,
        args.label_column,
        [
            args.fasta_path_column,
            args.pdb_path_column,
            args.hex_color_column,
            args.marker_size_column,
            args.marker_shape_column,
            args.alpha_column,
        ],
    )
    logger.info(f"Loaded domain dataset with {len(domain_dataframe)} entries")
    
    # Load sequences if needed
    if args.fasta_path_column or args.use_pdb_sequences:
        logger.info(
            f"Loading sequences from {'FASTA' if not args.use_pdb_sequences else 'PDB'} files..."
        )
        sequences = load_sequences(
            domain_dataframe=domain_dataframe,
            domain_id_column=args.id_column,
            use_pdb_sequences=args.use_pdb_sequences,
            pdb_path_column=args.pdb_path_column,
            fasta_path_column=args.fasta_path_column,
            use_record_id=args.use_record_id,
            cache_path=cache_paths.sequences,
        )
        domain_dataframe[SEQUENCE_COLUMN] = sequences
        logger.info(
            f"Loaded {len([s for s in sequences if s])} sequences from {'FASTA' if not args.use_pdb_sequences else 'PDB'} files."
        )
    
    # Load structures if needed
    if args.pdb_path_column and not args.exclude_structures:
        logger.info("Loading structures from PDB files...")
        structures = load_structures(
            domain_dataframe=domain_dataframe,
            pdb_path_column=args.pdb_path_column,
            cache_path=cache_paths.structures,
        )
        domain_dataframe[STRUCTURE_COLUMN] = structures
        logger.info(
            f"Loaded {len([s for s in structures if s is not None])} structures from PDB files."
        )
    
    # Load CLSS model
    clss_model = load_clss(args.model_repo, args.model_filename)
    
    # Embed sequences and structures (now uses batching internally)
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
        sequence_emb_cache_path=cache_paths.sequence_embeddings,
        structure_emb_cache_path=cache_paths.structure_embeddings,
    )
    
    # Create embedded dataframe
    embedded_dataframe = create_embedded_dataframe(
        domain_dataframe,
        sequence_embeddings,
        structure_embeddings,
        id_column=args.id_column,
    )
    
    logger.info(
        f"Created embedded dataframe with {len(embedded_dataframe)} entries: "
        f"{len(embedded_dataframe[embedded_dataframe['modality'] == 'sequence'])} sequence embeddings, "
        f"{len(embedded_dataframe[embedded_dataframe['modality'] == 'structure'])} structure embeddings."
    )
    
    # Apply dimensionality reduction
    reduced_embeddings = apply_dim_reduction(
        embedded_dataframe,
        perplexity=args.tsne_perplexity,
        max_iterations=args.tsne_max_iterations,
        random_state=args.tsne_random_state,
        cache_path=cache_paths.reduced_embeddings,
    )
    logger.info(f"Reduced embeddings to 2D with shape {reduced_embeddings.shape}.")
    
    # Create map dataframe
    map_dataframe = create_map_dataframe(embedded_dataframe, reduced_embeddings)
    logger.info(f"Created map dataframe with {len(map_dataframe)} entries.")
    
    # Generate static scatter plot using PlotConfig
    logger.info("Generating static scatter plot...")
    plot_config = args.to_plot_config()
    create_static_scatter_plot(
        map_dataframe=map_dataframe,
        id_column=args.id_column,
        label_column=args.label_column,
        output_path=args.output_path,
        plot_config=plot_config,
        title="CLSS Protein Domain Static Map",
    )
    
    logger.info("✅ Static map generation complete!")


if __name__ == "__main__":
    main()
