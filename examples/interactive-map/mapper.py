"""
Interactive visualization module for protein domain embeddings.
Creates Plotly scatter plots and exports them to HTML files.
"""

import os
import json
from typing import Optional, Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models import HighlightStyle, PairingMetadata
from pairing_utils import validate_pairings, build_trace_point_mapping, build_pairings_data
from js_templates import HIGHLIGHT_SCRIPT_TEMPLATE

# Valid Plotly marker shapes
VALID_MARKER_SHAPES = {
    'circle', 'circle-open', 'circle-dot', 'circle-open-dot',
    'square', 'square-open', 'square-dot', 'square-open-dot',
    'diamond', 'diamond-open', 'diamond-dot', 'diamond-open-dot',
    'cross', 'cross-open', 'cross-dot', 'cross-open-dot',
    'x', 'x-open', 'x-dot', 'x-open-dot',
    'triangle-up', 'triangle-up-open', 'triangle-up-dot', 'triangle-up-open-dot',
    'triangle-down', 'triangle-down-open', 'triangle-down-dot', 'triangle-down-open-dot',
    'triangle-left', 'triangle-left-open', 'triangle-left-dot', 'triangle-left-open-dot',
    'triangle-right', 'triangle-right-open', 'triangle-right-dot', 'triangle-right-open-dot',
    'triangle-ne', 'triangle-ne-open', 'triangle-ne-dot', 'triangle-ne-open-dot',
    'triangle-se', 'triangle-se-open', 'triangle-se-dot', 'triangle-se-open-dot',
    'triangle-sw', 'triangle-sw-open', 'triangle-sw-dot', 'triangle-sw-open-dot',
    'triangle-nw', 'triangle-nw-open', 'triangle-nw-dot', 'triangle-nw-open-dot',
    'pentagon', 'pentagon-open', 'pentagon-dot', 'pentagon-open-dot',
    'hexagon', 'hexagon-open', 'hexagon-dot', 'hexagon-open-dot',
    'hexagon2', 'hexagon2-open', 'hexagon2-dot', 'hexagon2-open-dot',
    'octagon', 'octagon-open', 'octagon-dot', 'octagon-open-dot',
    'star', 'star-open', 'star-dot', 'star-open-dot',
    'hexagram', 'hexagram-open', 'hexagram-dot', 'hexagram-open-dot',
    'star-triangle-up', 'star-triangle-up-open', 'star-triangle-up-dot', 'star-triangle-up-open-dot',
    'star-triangle-down', 'star-triangle-down-open', 'star-triangle-down-dot', 'star-triangle-down-open-dot',
    'star-square', 'star-square-open', 'star-square-dot', 'star-square-open-dot',
    'star-diamond', 'star-diamond-open', 'star-diamond-dot', 'star-diamond-open-dot',
    'diamond-tall', 'diamond-tall-open', 'diamond-tall-dot', 'diamond-tall-open-dot',
    'diamond-wide', 'diamond-wide-open', 'diamond-wide-dot', 'diamond-wide-open-dot',
    'hourglass', 'hourglass-open', 'bowtie', 'bowtie-open',
    'circle-cross', 'circle-cross-open', 'circle-x', 'circle-x-open',
    'square-cross', 'square-cross-open', 'square-x', 'square-x-open',
    'diamond-cross', 'diamond-cross-open', 'diamond-x', 'diamond-x-open',
    'cross-thin', 'cross-thin-open', 'x-thin', 'x-thin-open',
    'asterisk', 'asterisk-open', 'hash', 'hash-open', 'hash-dot', 'hash-open-dot',
    'y-up', 'y-up-open', 'y-down', 'y-down-open',
    'y-left', 'y-left-open', 'y-right', 'y-right-open',
    'line-ew', 'line-ew-open', 'line-ns', 'line-ns-open',
    'line-ne', 'line-ne-open', 'line-nw', 'line-nw-open'
}


def create_interactive_scatter_plot(
    map_dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    hex_color_column: Optional[str] = None,
    line_width_column: Optional[str] = None,
    line_color_column: Optional[str] = None,
    alpha_column: Optional[str] = None,
    hover_columns: Optional[List[str]] = None,
    marker_shape_column: Optional[str] = None,
    title: str = "Protein Domain Interactive Map",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """
    Create an interactive scatter plot from the map dataframe.

    Args:
        map_dataframe: DataFrame with x, y coordinates and domain information
        id_column: Name of the domain ID column
        label_column: Name of the label column (for color coding)
        hex_color_column: Optional column with custom hex colors
        line_width_column: Optional column for marker line widths
        line_color_column: Optional column for marker line colors
        alpha_column: Optional column for marker opacity/transparency values (0-1)
        hover_columns: Optional list of additional columns to include in hover info
        marker_shape_column: Optional column with marker shape values (must contain valid Plotly marker shapes)
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
    """

    # Create hover text with domain information
    hover_data = {
        id_column: True,
        label_column: True,
        "modality": True,
    }

    for col in hover_columns or []:
        hover_data[col] = True

    # Handle marker shape configuration
    symbol_column = None
    symbol_map = None
    
    if marker_shape_column:
        # Validate marker shape values
        unique_shapes = map_dataframe[marker_shape_column].dropna().unique()
        invalid_shapes = [s for s in unique_shapes if s not in VALID_MARKER_SHAPES]
        if invalid_shapes:
            raise ValueError(
                f"Invalid marker shapes found in column '{marker_shape_column}': {invalid_shapes}. "
                f"Valid shapes are: {sorted(VALID_MARKER_SHAPES)}"
            )
        
        # Use the marker shape column for symbols
        symbol_column = marker_shape_column
        # Create identity mapping for valid shapes
        symbol_map = {shape: shape for shape in unique_shapes}
        
        # Add to hover data if not already present
        if marker_shape_column not in hover_data:
            hover_data[marker_shape_column] = True
    # If no marker_shape_column, use circles for all points (don't use modality)

    label_to_color = None

    # Create the base scatter plot
    if hex_color_column and hex_color_column in map_dataframe.columns:
        # Use custom colors but with label-based legend
        # Create a mapping from labels to colors
        label_to_color = {}
        for _, row in map_dataframe.iterrows():
            label = row[label_column]
            color = row[hex_color_column]
            if label not in label_to_color:
                label_to_color[label] = color

    # Add dataframe index as a temporary column for efficient alpha mapping
    df_with_index = map_dataframe.copy()
    df_with_index['_plot_index'] = df_with_index.index

    # Create the plot using label column for grouping
    fig = px.scatter(
        df_with_index,
        x="x",
        y="y",
        color=label_column,
        symbol=symbol_column,  # Use marker_shape_column if provided, else None (all circles)
        symbol_map=symbol_map,  # Use shape mapping if provided, else None
        hover_data=hover_data,
        custom_data=['_plot_index'],  # Store index for efficient lookup
        title=title,
        labels={
            "x": "t-SNE Dimension 1",
            "y": "t-SNE Dimension 2",
            "modality": "Modality",
        },
        width=width,
        height=height,
        color_discrete_map=label_to_color,
    )

    # Customize the plot appearance
    marker_config = {"size": 8, "line": {"width": 0}}

    # Set line width - either from column or default
    if line_width_column and line_width_column in map_dataframe.columns:
        marker_config["line"]["width"] = map_dataframe[line_width_column].tolist()

    # Set line color - either from column or default
    if line_color_column and line_color_column in map_dataframe.columns:
        marker_config["line"]["color"] = map_dataframe[line_color_column].tolist()

    fig.update_traces(marker=marker_config, selector=dict(mode="markers"))

    # Set opacity/alpha per marker - must be done after update_traces
    # because px.scatter creates multiple traces (one per label group)
    if alpha_column and alpha_column in map_dataframe.columns:
        # Get the alpha values for all points in original dataframe order
        alpha_values = map_dataframe[alpha_column].tolist()
        
        # px.scatter creates one trace per unique value in the color column
        # We need to map the alpha values to each trace's data points
        for trace in fig.data:
            # Extract indices from customdata (first column is _plot_index)
            trace_indices = [int(cd[0]) for cd in trace.customdata]
            
            # Set opacity for this trace's markers using the indices
            trace.marker.opacity = [alpha_values[idx] for idx in trace_indices]

    # Update layout for better interactivity and full screen
    fig.update_layout(
        font=dict(size=12),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=20, r=150, t=50, b=20),
        hovermode="closest",
        template="plotly_white",
        # Full screen configuration
        width=width,
        height=height,
        autosize=True if width is None and height is None else False,
        # Enable scrollwheel zoom
        dragmode="pan",
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    return fig


def export_to_html(
    fig: go.Figure,
    output_path: str,
    include_plotlyjs: str = "cdn",
    config: Optional[Dict[str, Any]] = None,
    pairings_index_map: Optional[Dict[int, List[int]]] = None,
    highlight_style: Optional[HighlightStyle] = None,
) -> Optional[PairingMetadata]:
    """
    Export the Plotly figure to an HTML file.

    Args:
        fig: Plotly figure to export
        output_path: Path to save the HTML file
        include_plotlyjs: How to include Plotly.js ('cdn', 'inline', 'directory', etc.)
        config: Optional configuration dictionary for the plot
        pairings_index_map: Optional mapping from source DataFrame indices to target indices for click highlighting
        highlight_style: Optional custom highlight styling (defaults to HighlightStyle())
    
    Returns:
        PairingMetadata if pairings were provided, None otherwise
    """

    # Default configuration for better user experience with scrollwheel zoom
    default_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [],
        "scrollZoom": True,  # Enable scrollwheel zoom
        "doubleClick": "reset+autosize",  # Double-click to reset and autosize
        "showTips": True,
        "responsive": True,  # Make plot responsive
        "toImageButtonOptions": {
            "format": "png",
            "filename": "protein_domain_map",
            "height": 1080,
            "width": 1920,
            "scale": 2,
        },
    }

    if config:
        default_config.update(config)

    # Use default highlight style if not provided
    if highlight_style is None:
        highlight_style = HighlightStyle()

    # Prepare JavaScript for click-to-highlight functionality
    pairings_js = ""
    pairing_metadata = None
    
    if pairings_index_map:
        # Build trace/point mapping from figure
        mapping = build_trace_point_mapping(fig)
        
        # Convert pairings to trace/point coordinates
        pairings_data = build_pairings_data(pairings_index_map, mapping)
        
        # Generate JavaScript with injected data
        pairings_js = HIGHLIGHT_SCRIPT_TEMPLATE.format(
            pairings_json=json.dumps(pairings_data),
            highlight_style_json=json.dumps(highlight_style.to_dict())
        )
    
    # Export to HTML with full screen styling
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CLSS Protein Domain Interactive Map</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        #protein-domain-map {{
            width: 100vw;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <div id="protein-domain-map"></div>
    {{plot_div}}
    {pairings_js}
</body>
</html>
"""

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to HTML
    fig.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        config=default_config,
        div_id="protein-domain-map",
        full_html=False,
    )

    # Read the generated HTML and wrap it in our full-screen template
    with open(output_path, "r", encoding="utf-8") as f:
        plot_content = f.read()

    # Write the full-screen version
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template.replace("{plot_div}", plot_content))
    
    return pairing_metadata


def create_and_export_visualization(
    map_dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    output_path: str,
    hex_color_column: Optional[str] = None,
    line_width_column: Optional[str] = None,
    line_color_column: Optional[str] = None,
    alpha_column: Optional[str] = None,
    hover_columns: Optional[List[str]] = None,
    marker_shape_column: Optional[str] = None,
    title: str = "Protein Domain Interactive Map",
    width: Optional[int] = None,
    height: Optional[int] = None,
    pairings_index_map: Optional[Dict[int, List[int]]] = None,
    highlight_style: Optional[HighlightStyle] = None,
) -> None:
    """
    Complete workflow to create and export an interactive visualization.

    Args:
        map_dataframe: DataFrame with x, y coordinates and domain information
        id_column: Name of the domain ID column
        label_column: Name of the label column (for color coding)
        output_path: Path to save the HTML file
        hex_color_column: Optional column with custom hex colors
        line_width_column: Optional column for marker line widths
        line_color_column: Optional column for marker line colors
        alpha_column: Optional column for marker opacity/transparency values (0-1)
        hover_columns: Optional list of additional columns to include in hover info
        marker_shape_column: Optional column with marker shape values (must contain valid Plotly marker shapes)
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        pairings_index_map: Optional mapping from source DataFrame indices to target indices for click highlighting
        highlight_style: Optional custom highlight styling (defaults to HighlightStyle())
    """

    print(f"Creating interactive visualization...")

    # Validate pairings if provided
    if pairings_index_map:
        pairing_metadata = validate_pairings(pairings_index_map, map_dataframe)
        if pairing_metadata.has_issues():
            print(f"⚠️  Pairing validation warnings:")
            pairing_metadata.print_summary()

    # Create the figure
    fig = create_interactive_scatter_plot(
        map_dataframe=map_dataframe,
        id_column=id_column,
        label_column=label_column,
        hex_color_column=hex_color_column,
        line_width_column=line_width_column,
        line_color_column=line_color_column,
        alpha_column=alpha_column,
        hover_columns=hover_columns,
        marker_shape_column=marker_shape_column,
        title=title,
        width=width,
        height=height,
    )

    print(f"Exporting visualization to {output_path}...")

    # Export to HTML
    export_to_html(
        fig, 
        output_path, 
        pairings_index_map=pairings_index_map,
        highlight_style=highlight_style
    )

    # Print statistics
    total_points = len(map_dataframe)
    sequence_points = len(map_dataframe[map_dataframe["modality"] == "sequence"])
    structure_points = len(map_dataframe[map_dataframe["modality"] == "structure"])
    unique_labels = map_dataframe[label_column].nunique()

    print(f"✅ Visualization exported successfully!")
    print(f"   📊 Total points: {total_points}")
    print(f"   🧬 Sequence points: {sequence_points}")
    print(f"   🏗️  Structure points: {structure_points}")
    print(f"   🏷️  Unique labels: {unique_labels}")
    
    # Print pairing statistics if available
    if pairings_index_map:
        pairing_metadata = validate_pairings(pairings_index_map, map_dataframe)
        pairing_metadata.print_summary()
    
    print(f"   📁 File saved to: {output_path}")
