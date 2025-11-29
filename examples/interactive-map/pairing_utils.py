"""
Utilities for handling pairings in interactive visualizations.
"""

from typing import Dict, List, Set
import pandas as pd
import plotly.graph_objects as go
from models import PairingMetadata, TracePointMapping


def validate_pairings(
    pairings_index_map: Dict[int, List[int]],
    dataframe: pd.DataFrame
) -> PairingMetadata:
    """
    Validate pairings against the DataFrame and collect metadata.
    
    Args:
        pairings_index_map: Mapping from source indices to target indices
        dataframe: The visualization DataFrame with all points
    
    Returns:
        PairingMetadata with validation results
    """
    valid_indices = set(dataframe.index)
    orphaned_sources = []
    orphaned_targets = []
    all_target_indices: Set[int] = set()
    max_targets = 0
    
    for source_idx, target_indices in pairings_index_map.items():
        # Check if source exists in DataFrame
        if source_idx not in valid_indices:
            orphaned_sources.append(source_idx)
        
        # Track max targets per source
        max_targets = max(max_targets, len(target_indices))
        
        # Check each target
        for target_idx in target_indices:
            all_target_indices.add(target_idx)
            if target_idx not in valid_indices:
                orphaned_targets.append(target_idx)
    
    return PairingMetadata(
        total_sources=len(pairings_index_map),
        total_targets=len(all_target_indices),
        orphaned_sources=orphaned_sources,
        orphaned_targets=list(set(orphaned_targets)),  # Deduplicate
        max_targets_per_source=max_targets
    )


def build_trace_point_mapping(fig: go.Figure) -> TracePointMapping:
    """
    Build mapping from DataFrame indices to trace/point coordinates.
    
    Args:
        fig: Plotly figure with customdata containing _plot_index
    
    Returns:
        TracePointMapping object
    """
    index_to_trace_point = {}
    
    for trace_idx, trace in enumerate(fig.data):
        for point_idx, customdata in enumerate(trace.customdata):
            df_index = int(customdata[0])
            index_to_trace_point[df_index] = {
                "traceIndex": trace_idx,
                "pointIndex": point_idx
            }
    
    return TracePointMapping(index_to_trace_point=index_to_trace_point)


def build_pairings_data(
    pairings_index_map: Dict[int, List[int]],
    mapping: TracePointMapping
) -> Dict[str, List[Dict[str, int]]]:
    """
    Convert DataFrame index-based pairings to trace/point-based pairings.
    
    Args:
        pairings_index_map: Mapping from source DataFrame indices to target indices
        mapping: TracePointMapping for coordinate conversion
    
    Returns:
        Dictionary mapping "traceIdx_pointIdx" to list of target coordinates
    """
    pairings_data: Dict[str, List[Dict[str, int]]] = {}
    
    for source_idx, target_indices in pairings_index_map.items():
        source_coords = mapping.get_trace_point(source_idx)
        if not source_coords:
            continue
        
        source_key = f"{source_coords['traceIndex']}_{source_coords['pointIndex']}"
        pairings_data[source_key] = []
        
        for target_idx in target_indices:
            target_coords = mapping.get_trace_point(target_idx)
            if target_coords:
                pairings_data[source_key].append(target_coords)
    
    return pairings_data
