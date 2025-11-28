"""
Modern CLI argument parsing with structured configuration.
Provides type-safe access to command line arguments with autocomplete support.
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIArgs:
    """
    Structured container for CLI arguments with type hints for autocomplete.
    
    Attributes:
        model_repo: HuggingFace model repository name (required)
        model_filename: Model checkpoint filename (required)
        dataset_path: Path to the dataset file (required)
        id_column: Name of the domain ID column in the dataset (required)
        label_column: Name of the label column (dictates coloring) in the dataset (required)
        tsne_perplexity: Perplexity parameter for t-SNE (required)
        tsne_max_iterations: Maximum iterations for t-SNE (required)
        tsne_random_state: Random state for t-SNE reproducibility (required)
        html_output_path: Path to the HTML output file (required)
        exclude_structures: Whether to exclude structure data if PDB column is provided (default: False)
        use_pdb_sequences: Whether to use sequences extracted from PDB files when PDB column is provided (default: False)
        use_record_id: Whether to use the domain ID as the record ID in FASTA files (default: False)
        fasta_path_column: Name of the FASTA path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)
        pdb_path_column: Name of the PDB path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)
        hex_color_column: Name of the column with hex color codes for points (optional)
        line_width_column: Name of the column for marker line widths (optional)
        line_color_column: Name of the column for marker line colors (optional)
        alpha_column: Name of the column for marker opacity/transparency values (optional)
        cache_path: Path to the cache directory (optional)

    Raises:
        ValueError: If required arguments are missing or invalid
    """
    model_repo: str
    model_filename: str
    dataset_path: str
    id_column: str
    label_column: str
    tsne_perplexity: int
    tsne_max_iterations: int
    tsne_random_state: int
    html_output_path: str
    exclude_structures: bool = False
    use_pdb_sequences: bool = False
    use_record_id: bool = False
    fasta_path_column: Optional[str] = None
    pdb_path_column: Optional[str] = None
    hex_color_column: Optional[str] = None
    line_width_column: Optional[str] = None
    line_color_column: Optional[str] = None
    alpha_column: Optional[str] = None
    cache_path: Optional[str] = None

    def __post_init__(self):
        """Validate arguments after initialization."""
        if not self.fasta_path_column and not self.pdb_path_column:
            raise ValueError("At least one of fasta_path_column or pdb_path_column must be provided")

        if self.use_pdb_sequences and not self.pdb_path_column:
            raise ValueError("use_pdb_sequences is set to True but pdb_path_column is not provided")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Interactive map application for CLSS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-repo",
        type=str,
        default="guyyanai/CLSS",
        help="HuggingFace model repository name"
    )
    
    parser.add_argument(
        "--model-filename", 
        type=str,
        default="h32_r10.lckpt",
        help="Model checkpoint filename"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset file (required)"
    )

    parser.add_argument(
        "--id-column",
        type=str,
        required=True,
        help="Name of the domain ID column in the dataset (required)"
    )

    parser.add_argument(
        "--label-column",
        type=str,
        required=True,
        help="Name of the label column (dictates coloring) in the dataset (required)"
    )

    parser.add_argument(
        "--fasta-path-column",
        type=str,
        default=None,
        help="Name of the FASTA path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)"
    )

    parser.add_argument(
        "--pdb-path-column",
        type=str,
        default=None,
        help="Name of the PDB path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)"
    )

    parser.add_argument(
        "--hex-color-column",
        type=str,
        default=None,
        help="Name of the column with hex color codes for points (optional)"
    )

    parser.add_argument(
        "--line-width-column",
        type=str,
        default=None,
        help="Name of the column for marker line widths (optional)"
    )

    parser.add_argument(
        "--line-color-column",
        type=str,
        default=None,
        help="Name of the column for marker line colors (optional)"
    )

    parser.add_argument(
        "--alpha-column",
        type=str,
        default=None,
        help="Name of the column for marker opacity/transparency values (optional)"
    )

    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="Perplexity parameter for t-SNE"
    )

    parser.add_argument(
        "--tsne-max-iterations",
        type=int,
        default=1000,
        help="Maximum iterations for t-SNE"
    )

    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=0,
        help="Random state for t-SNE reproducibility"
    )

    parser.add_argument(
        "--html-output-path",
        type=str,
        required=True,
        help="Path to the HTML output file"
    )

    parser.add_argument(
        "--exclude-structures",
        action="store_true",
        help="Whether to exclude structure data if PDB column is provided (default: False)"
    )

    parser.add_argument(
        "--use-pdb-sequences",
        action="store_true",
        help="Whether to use sequences extracted from PDB files when PDB column is provided (default: False)"
    )

    parser.add_argument(
        "--use-record-id",
        action="store_true",
        help="Whether to use the domain ID as the record ID in FASTA files (default: False)"
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to the cache directory"
    )

    return parser


def parse_args() -> CLIArgs:
    """
    Parse command line arguments and return structured configuration.
        
    Returns:
        CLIArgs instance with parsed values
        
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    parser = create_argument_parser()
    parsed = parser.parse_args()
    
    return CLIArgs(
        model_repo=parsed.model_repo,
        model_filename=parsed.model_filename,
        dataset_path=parsed.dataset_path,
        id_column=parsed.id_column,
        label_column=parsed.label_column,
        tsne_perplexity=parsed.tsne_perplexity,
        tsne_max_iterations=parsed.tsne_max_iterations,
        tsne_random_state=parsed.tsne_random_state,
        html_output_path=parsed.html_output_path,
        exclude_structures=parsed.exclude_structures,
        use_pdb_sequences=parsed.use_pdb_sequences,
        use_record_id=parsed.use_record_id,
        fasta_path_column=parsed.fasta_path_column,
        pdb_path_column=parsed.pdb_path_column,
        hex_color_column=parsed.hex_color_column,
        line_width_column=parsed.line_width_column,
        line_color_column=parsed.line_color_column,
        alpha_column=parsed.alpha_column,
        cache_path=parsed.cache_path,
    )
