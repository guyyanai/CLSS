"""
CLI argument parsing for static map generation.
Provides type-safe access to command line arguments with autocomplete support.
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PlotConfig:
    """
    Configuration for plot styling and layout.
    Extracted from CLIArgs to reduce parameter passing complexity.
    """
    marker_size: float = 50
    marker_size_column: Optional[str] = None
    marker_shape_column: Optional[str] = None
    alpha: float = 0.7
    alpha_column: Optional[str] = None
    hex_color_column: Optional[str] = None
    legend_title: Optional[str] = None
    legend_position: str = "right"
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    edge_color: Optional[str] = "black"
    edge_width: float = 0.5
    custom_legend_csv: Optional[str] = None
    figsize: Tuple[float, float] = (12, 10)
    dpi: int = 300
    output_format: str = "png"


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
        output_path: Path to the output image file (required)
        output_format: Output format (png, pdf, svg) (default: png)
        dpi: DPI for output image (default: 300)
        figsize_width: Figure width in inches (default: 12)
        figsize_height: Figure height in inches (default: 10)
        exclude_structures: Whether to exclude structure data if PDB column is provided (default: False)
        use_pdb_sequences: Whether to use sequences extracted from PDB files when PDB column is provided (default: False)
        use_record_id: Whether to use the domain ID as the record ID in FASTA files (default: False)
        fasta_path_column: Name of the FASTA path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)
        pdb_path_column: Name of the PDB path column in the dataset (at least one of fasta_path_column or pdb_path_column is required)
        hex_color_column: Name of the column with hex color codes for points (optional)
        marker_size: Base marker size for scatter plot (default: 50)
        marker_size_column: Name of the column for marker sizes (optional)
        marker_shape_column: Name of the column for marker shapes (optional)
        alpha: Default alpha/transparency value (default: 0.7)
        alpha_column: Name of the column for marker opacity/transparency values (optional)
        legend_title: Title for the legend (optional)
        legend_position: Legend position outside the plot (right, left, top, bottom) (default: right)
        x_min: Minimum x-axis limit (optional, auto if not specified)
        x_max: Maximum x-axis limit (optional, auto if not specified)
        y_min: Minimum y-axis limit (optional, auto if not specified)
        y_max: Maximum y-axis limit (optional, auto if not specified)
        edge_color: Color of marker edges/borders (optional, use 'none' for no border, default: black)
        edge_width: Width of marker edges/borders in points (default: 0.5)
        custom_legend_csv: Path to CSV file with custom legend (class,color columns) for use with hex_color_column (optional)
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
    output_path: str
    output_format: str = "png"
    dpi: int = 300
    figsize_width: float = 12
    figsize_height: float = 10
    exclude_structures: bool = False
    use_pdb_sequences: bool = False
    use_record_id: bool = False
    fasta_path_column: Optional[str] = None
    pdb_path_column: Optional[str] = None
    hex_color_column: Optional[str] = None
    marker_size: float = 50
    marker_size_column: Optional[str] = None
    marker_shape_column: Optional[str] = None
    alpha: float = 0.7
    alpha_column: Optional[str] = None
    legend_title: Optional[str] = None
    legend_position: str = "right"
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    edge_color: Optional[str] = "black"
    edge_width: float = 0.5
    custom_legend_csv: Optional[str] = None
    cache_path: Optional[str] = None

    def __post_init__(self):
        """Validate arguments after initialization."""
        if not self.fasta_path_column and not self.pdb_path_column:
            raise ValueError("At least one of fasta_path_column or pdb_path_column must be provided")

        if self.use_pdb_sequences and not self.pdb_path_column:
            raise ValueError("use_pdb_sequences is set to True but pdb_path_column is not provided")
        
        if self.output_format not in ["png", "pdf", "svg"]:
            raise ValueError(f"Invalid output format: {self.output_format}. Must be one of: png, pdf, svg")
    
    def to_plot_config(self) -> PlotConfig:
        """
        Create a PlotConfig instance from CLI arguments.
        
        Returns:
            PlotConfig: Configuration object for plotting
        """
        return PlotConfig(
            marker_size=self.marker_size,
            marker_size_column=self.marker_size_column,
            marker_shape_column=self.marker_shape_column,
            alpha=self.alpha,
            alpha_column=self.alpha_column,
            hex_color_column=self.hex_color_column,
            legend_title=self.legend_title,
            legend_position=self.legend_position,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            edge_color=self.edge_color,
            edge_width=self.edge_width,
            custom_legend_csv=self.custom_legend_csv,
            figsize=(self.figsize_width, self.figsize_height),
            dpi=self.dpi,
            output_format=self.output_format,
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Static map generator for CLSS model embeddings",
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
        "--marker-size",
        type=float,
        default=50,
        help="Base marker size for scatter plot"
    )

    parser.add_argument(
        "--marker-size-column",
        type=str,
        default=None,
        help="Name of the column for marker sizes (optional)"
    )

    parser.add_argument(
        "--marker-shape-column",
        type=str,
        default=None,
        help="Name of the column for marker shapes (optional). Supported shapes: o (circle), s (square), ^ (triangle up), v (triangle down), D (diamond), * (star), + (plus), x (x), etc."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Default alpha/transparency value (0-1)"
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
        "--output-path",
        type=str,
        required=True,
        help="Path to the output image file (required)"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for the image file"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output image (higher = better quality, larger file)"
    )

    parser.add_argument(
        "--figsize-width",
        type=float,
        default=12,
        help="Figure width in inches"
    )

    parser.add_argument(
        "--figsize-height",
        type=float,
        default=10,
        help="Figure height in inches"
    )

    parser.add_argument(
        "--legend-title",
        type=str,
        default=None,
        help="Title for the legend (optional)"
    )

    parser.add_argument(
        "--legend-position",
        type=str,
        choices=["right", "left", "top", "bottom"],
        default="right",
        help="Legend position outside the plot area"
    )

    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum x-axis limit (optional, auto-calculated if not specified)"
    )

    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum x-axis limit (optional, auto-calculated if not specified)"
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum y-axis limit (optional, auto-calculated if not specified)"
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum y-axis limit (optional, auto-calculated if not specified)"
    )

    parser.add_argument(
        "--edge-color",
        type=str,
        default="black",
        help="Color of marker edges/borders (e.g., 'black', 'white', '#FF0000', or 'none' for no border)"
    )

    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.5,
        help="Width of marker edges/borders in points"
    )

    parser.add_argument(
        "--custom-legend-csv",
        type=str,
        default=None,
        help="Path to CSV file with custom legend (must have 'class' and 'color' columns). Use with --hex-color-column to show a legend."
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
        output_path=parsed.output_path,
        output_format=parsed.output_format,
        dpi=parsed.dpi,
        figsize_width=parsed.figsize_width,
        figsize_height=parsed.figsize_height,
        exclude_structures=parsed.exclude_structures,
        use_pdb_sequences=parsed.use_pdb_sequences,
        use_record_id=parsed.use_record_id,
        fasta_path_column=parsed.fasta_path_column,
        pdb_path_column=parsed.pdb_path_column,
        hex_color_column=parsed.hex_color_column,
        marker_size=parsed.marker_size,
        marker_size_column=parsed.marker_size_column,
        marker_shape_column=parsed.marker_shape_column,
        alpha=parsed.alpha,
        alpha_column=parsed.alpha_column,
        legend_title=parsed.legend_title,
        legend_position=parsed.legend_position,
        x_min=parsed.x_min,
        x_max=parsed.x_max,
        y_min=parsed.y_min,
        y_max=parsed.y_max,
        edge_color=parsed.edge_color,
        edge_width=parsed.edge_width,
        custom_legend_csv=parsed.custom_legend_csv,
        cache_path=parsed.cache_path,
    )
