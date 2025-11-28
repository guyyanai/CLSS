"""
Interactive visualization module for protein domain embeddings.
Creates Plotly scatter plots and exports them to HTML files.
"""

import os
from typing import Optional, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_interactive_scatter_plot(
    map_dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    hex_color_column: Optional[str] = None,
    line_width_column: Optional[str] = None,
    line_color_column: Optional[str] = None,
    alpha_column: Optional[str] = None,
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

    # Add custom colors if available
    if hex_color_column and hex_color_column in map_dataframe.columns:
        hover_data[hex_color_column] = True

    # Define symbols for different modalities
    symbol_map = {"sequence": "circle", "structure": "square"}

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
        symbol="modality",
        symbol_map=symbol_map,
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
) -> None:
    """
    Export the Plotly figure to an HTML file.

    Args:
        fig: Plotly figure to export
        output_path: Path to save the HTML file
        include_plotlyjs: How to include Plotly.js ('cdn', 'inline', 'directory', etc.)
        config: Optional configuration dictionary for the plot
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
    with open(output_path, "r") as f:
        plot_content = f.read()

    # Write the full-screen version
    with open(output_path, "w") as f:
        f.write(html_template.replace("{plot_div}", plot_content))


def create_and_export_visualization(
    map_dataframe: pd.DataFrame,
    id_column: str,
    label_column: str,
    output_path: str,
    hex_color_column: Optional[str] = None,
    line_width_column: Optional[str] = None,
    line_color_column: Optional[str] = None,
    alpha_column: Optional[str] = None,
    title: str = "Protein Domain Interactive Map",
    width: Optional[int] = None,
    height: Optional[int] = None,
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
        alpha_column: Optional column
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
    """

    print(f"Creating interactive visualization...")

    # Create the figure
    fig = create_interactive_scatter_plot(
        map_dataframe=map_dataframe,
        id_column=id_column,
        label_column=label_column,
        hex_color_column=hex_color_column,
        line_width_column=line_width_column,
        line_color_column=line_color_column,
        alpha_column=alpha_column,
        title=title,
        width=width,
        height=height,
    )

    print(f"Exporting visualization to {output_path}...")

    # Export to HTML
    export_to_html(fig, output_path)

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
    print(f"   📁 File saved to: {output_path}")
