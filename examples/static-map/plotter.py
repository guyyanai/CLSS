"""
Static visualization module for protein domain embeddings.
Creates publication-quality scatter plots using Matplotlib and Seaborn.
"""

import os
import logging
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from args import PlotConfig
from constants import X_COLUMN, Y_COLUMN, MODALITY_COLUMN

logger = logging.getLogger(__name__)


# Valid Matplotlib marker shapes
MATPLOTLIB_MARKERS = {
    'o': 'circle',
    's': 'square', 
    '^': 'triangle_up',
    'v': 'triangle_down',
    '<': 'triangle_left',
    '>': 'triangle_right',
    'D': 'diamond',
    'd': 'thin_diamond',
    'p': 'pentagon',
    'h': 'hexagon1',
    'H': 'hexagon2',
    '8': 'octagon',
    '*': 'star',
    '+': 'plus',
    'x': 'x',
    'P': 'plus_filled',
    'X': 'x_filled',
    '|': 'vline',
    '_': 'hline',
}


def create_static_scatter_plot(
    map_dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    output_path: str,
    plot_config: PlotConfig,
    title: str = "CLSS Protein Domain Static Map",
) -> None:
    """
    Create a static scatter plot and save to file.

    Args:
        map_dataframe: DataFrame with x, y coordinates and domain information
        id_column: Name of the domain ID column
        label_column: Name of the label column (for color coding)
        output_path: Path to save the output image
        plot_config: PlotConfig dataclass with all plotting parameters
        title: Plot title
    """
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=plot_config.figsize, dpi=plot_config.dpi)
    
    # Determine if we need to plot by groups or all at once
    if plot_config.hex_color_column and plot_config.hex_color_column in map_dataframe.columns:
        # Plot with custom colors
        _plot_with_custom_colors(
            ax, map_dataframe, id_column, label_column, plot_config
        )
    else:
        # Plot with automatic color mapping by label
        _plot_with_label_colors(
            ax, map_dataframe, id_column, label_column, plot_config
        )
    
    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Set axis limits if specified
    if plot_config.x_min is not None or plot_config.x_max is not None:
        ax.set_xlim(left=plot_config.x_min, right=plot_config.x_max)
    if plot_config.y_min is not None or plot_config.y_max is not None:
        ax.set_ylim(bottom=plot_config.y_min, top=plot_config.y_max)
    
    # Remove grid for cleaner look
    ax.grid(False)
    
    # Adjust the plot to make room for the outside legend
    if plot_config.legend_position == 'right':
        plt.subplots_adjust(right=0.75)
    elif plot_config.legend_position == 'left':
        plt.subplots_adjust(left=0.25)
    elif plot_config.legend_position == 'top':
        plt.subplots_adjust(top=0.85)
    elif plot_config.legend_position == 'bottom':
        plt.subplots_adjust(bottom=0.2)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format=plot_config.output_format, dpi=plot_config.dpi, bbox_inches='tight')
    logger.info(f"Static map saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def _plot_with_custom_colors(
    ax,
    dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    plot_config: PlotConfig,
) -> None:
    """
    Plot scatter with custom colors from the dataframe.
    """
    # Determine if we have custom markers
    has_custom_markers = plot_config.marker_shape_column and plot_config.marker_shape_column in dataframe.columns
    
    if has_custom_markers:
        # Plot each marker shape separately
        unique_shapes = dataframe[plot_config.marker_shape_column].unique()
        for shape in unique_shapes:
            if pd.isna(shape):
                shape = 'o'  # Default to circle
            
            shape_data = dataframe[dataframe[plot_config.marker_shape_column] == shape]
            
            # Get sizes
            sizes = _get_marker_sizes(shape_data, plot_config.marker_size, plot_config.marker_size_column)
            
            # Get alphas
            alphas = _get_alphas(shape_data, plot_config.alpha, plot_config.alpha_column)
            
            # Vectorized plotting for custom colors
            # If hex_color_column is present, use it directly as color array
            colors = shape_data[plot_config.hex_color_column] if plot_config.hex_color_column in shape_data.columns else '#1f77b4'
            
            # Handle missing colors if column exists
            if isinstance(colors, pd.Series):
                colors = colors.fillna('#1f77b4')
            
            ax.scatter(
                shape_data[X_COLUMN], 
                shape_data[Y_COLUMN],
                c=colors,
                s=sizes,
                marker=shape if shape in MATPLOTLIB_MARKERS else 'o',
                alpha=alphas if isinstance(alphas, float) else alphas.fillna(plot_config.alpha),
                edgecolors=plot_config.edge_color,
                linewidths=plot_config.edge_width,
            )
    else:
        # No custom markers, plot all at once
        sizes = _get_marker_sizes(dataframe, plot_config.marker_size, plot_config.marker_size_column)
        alphas = _get_alphas(dataframe, plot_config.alpha, plot_config.alpha_column)
        
        # Vectorized plotting for custom colors
        colors = dataframe[plot_config.hex_color_column] if plot_config.hex_color_column in dataframe.columns else '#1f77b4'
        
        # Handle missing colors if column exists
        if isinstance(colors, pd.Series):
            colors = colors.fillna('#1f77b4')
            
        ax.scatter(
            dataframe[X_COLUMN], 
            dataframe[Y_COLUMN],
            c=colors,
            s=sizes,
            marker='o',
            alpha=alphas if isinstance(alphas, float) else alphas.fillna(plot_config.alpha),
            edgecolors=plot_config.edge_color,
            linewidths=plot_config.edge_width,
        )
    
    # Add custom legend if provided
    if plot_config.custom_legend_csv:
        _add_custom_legend(ax, plot_config.custom_legend_csv, plot_config.legend_title, plot_config.legend_position)


def _plot_with_label_colors(
    ax,
    dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    plot_config: PlotConfig,
) -> None:
    """
    Plot scatter with automatic color mapping by label.
    """
    # Get unique labels and create a color palette
    unique_labels = dataframe[label_column].unique()
    n_colors = len(unique_labels)
    
    # Use a good color palette from seaborn
    if n_colors <= 10:
        palette = sns.color_palette("tab10", n_colors)
    elif n_colors <= 20:
        palette = sns.color_palette("tab20", n_colors)
    else:
        palette = sns.color_palette("husl", n_colors)
    
    color_map = dict(zip(unique_labels, palette))
    
    # Determine if we have custom markers
    has_custom_markers = plot_config.marker_shape_column and plot_config.marker_shape_column in dataframe.columns
    
    # Plot by label (for legend)
    for label in unique_labels:
        label_data = dataframe[dataframe[label_column] == label]
        
        if has_custom_markers:
            # Plot by shape within this label
            unique_shapes = label_data[plot_config.marker_shape_column].unique()
            for shape in unique_shapes:
                if pd.isna(shape):
                    shape = 'o'
                
                shape_data = label_data[label_data[plot_config.marker_shape_column] == shape]
                sizes = _get_marker_sizes(shape_data, plot_config.marker_size, plot_config.marker_size_column)
                alphas = _get_alphas(shape_data, plot_config.alpha, plot_config.alpha_column)
                
                ax.scatter(
                    shape_data[X_COLUMN], shape_data[Y_COLUMN],
                    c=[color_map[label]],
                    s=sizes,
                    marker=shape if shape in MATPLOTLIB_MARKERS else 'o',
                    alpha=alphas if isinstance(alphas, float) else alphas.iloc[0] if len(alphas) > 0 else plot_config.alpha,
                    label=label if shape == unique_shapes[0] else None,  # Only label once per label
                    edgecolors=plot_config.edge_color,
                    linewidths=plot_config.edge_width,
                )
        else:
            # No custom markers
            sizes = _get_marker_sizes(label_data, plot_config.marker_size, plot_config.marker_size_column)
            alphas = _get_alphas(label_data, plot_config.alpha, plot_config.alpha_column)
            
            ax.scatter(
                label_data[X_COLUMN], label_data[Y_COLUMN],
                c=[color_map[label]],
                s=sizes,
                marker='o',
                alpha=alphas if isinstance(alphas, float) else alphas.iloc[0] if len(alphas) > 0 else plot_config.alpha,
                label=label,
                edgecolors=plot_config.edge_color,
                linewidths=plot_config.edge_width,
            )
    
    # Add legend outside the plot area
    legend_kwargs = {
        'title': plot_config.legend_title if plot_config.legend_title else label_column,
        'frameon': True,
        'fancybox': True,
        'shadow': True,
        'fontsize': 9,
    }
    
    # Position legend outside based on user choice
    if plot_config.legend_position == 'right':
        legend_kwargs['loc'] = 'center left'
        legend_kwargs['bbox_to_anchor'] = (1.05, 0.5)
    elif plot_config.legend_position == 'left':
        legend_kwargs['loc'] = 'center right'
        legend_kwargs['bbox_to_anchor'] = (-0.05, 0.5)
    elif plot_config.legend_position == 'top':
        legend_kwargs['loc'] = 'lower center'
        legend_kwargs['bbox_to_anchor'] = (0.5, 1.05)
        legend_kwargs['ncol'] = min(3, len(ax.get_legend_handles_labels()[0]))  # Multiple columns for top
    elif plot_config.legend_position == 'bottom':
        legend_kwargs['loc'] = 'upper center'
        legend_kwargs['bbox_to_anchor'] = (0.5, -0.05)
        legend_kwargs['ncol'] = min(3, len(ax.get_legend_handles_labels()[0]))  # Multiple columns for bottom
    
    legend = ax.legend(**legend_kwargs)
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight('bold')


def _add_custom_legend(
    ax,
    custom_legend_csv: str,
    legend_title: Optional[str],
    legend_position: str,
) -> None:
    """Add custom legend from CSV file."""
    try:
        # Read the legend CSV
        legend_df = pd.read_csv(custom_legend_csv)
        
        # Validate required columns
        if 'class' not in legend_df.columns or 'color' not in legend_df.columns:
            logger.warning("Custom legend CSV must have 'class' and 'color' columns. Skipping legend.")
            return
        
        # Create legend patches
        legend_patches = []
        for _, row in legend_df.iterrows():
            patch = mpatches.Patch(
                facecolor=row['color'],
                edgecolor='black',
                label=row['class']
            )
            legend_patches.append(patch)
        
        # Add legend with positioning
        legend_kwargs = {
            'handles': legend_patches,
            'title': legend_title if legend_title else 'Legend',
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'fontsize': 9,
        }
        
        # Position legend outside based on user choice
        if legend_position == 'right':
            legend_kwargs['loc'] = 'center left'
            legend_kwargs['bbox_to_anchor'] = (1.05, 0.5)
        elif legend_position == 'left':
            legend_kwargs['loc'] = 'center right'
            legend_kwargs['bbox_to_anchor'] = (-0.05, 0.5)
        elif legend_position == 'top':
            legend_kwargs['loc'] = 'lower center'
            legend_kwargs['bbox_to_anchor'] = (0.5, 1.05)
            legend_kwargs['ncol'] = min(3, len(legend_patches))  # Multiple columns for top
        elif legend_position == 'bottom':
            legend_kwargs['loc'] = 'upper center'
            legend_kwargs['bbox_to_anchor'] = (0.5, -0.05)
            legend_kwargs['ncol'] = min(3, len(legend_patches))  # Multiple columns for bottom
        
        legend = ax.legend(**legend_kwargs)
        legend.get_title().set_fontsize(10)
        legend.get_title().set_fontweight('bold')
        
    except FileNotFoundError:
        logger.warning(f"Custom legend CSV file not found: {custom_legend_csv}. Skipping legend.")
    except Exception as e:
        logger.warning(f"Error reading custom legend CSV: {e}. Skipping legend.")


def _get_marker_sizes(
    dataframe: pd.DataFrame,
    base_size: float,
    size_column: Optional[str]
) -> pd.Series | float:
    """Get marker sizes from column or use base size."""
    if size_column and size_column in dataframe.columns:
        return dataframe[size_column].fillna(base_size)
    return base_size


def _get_alphas(
    dataframe: pd.DataFrame,
    base_alpha: float,
    alpha_column: Optional[str]
) -> pd.Series | float:
    """Get alpha values from column or use base alpha."""
    if alpha_column and alpha_column in dataframe.columns:
        return dataframe[alpha_column].fillna(base_alpha)
    return base_alpha


def add_modality_markers(
    ax,
    map_dataframe: pd.DataFrame,
    legend_position: str = "upper right"
) -> None:
    """
    Add markers to distinguish between sequence and structure modalities.
    
    Args:
        ax: Matplotlib axis
        map_dataframe: DataFrame with modality information
        legend_position: Position for the modality legend
    """
    if MODALITY_COLUMN not in map_dataframe.columns:
        return
    
    # Create custom legend for modalities
    modality_markers = {
        'sequence': 'o',
        'structure': 's',
    }
    
    legend_elements = []
    for modality, marker in modality_markers.items():
        if modality in map_dataframe[MODALITY_COLUMN].unique():
            legend_elements.append(
                mpatches.Patch(
                    facecolor='gray', 
                    edgecolor='black',
                    label=modality.capitalize()
                )
            )
    
    if legend_elements:
        # Add a second legend for modalities
        modality_legend = ax.legend(
            handles=legend_elements,
            loc=legend_position,
            title="Modality",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
        )
        ax.add_artist(modality_legend)
