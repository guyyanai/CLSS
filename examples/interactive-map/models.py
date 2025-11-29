"""
Data models for interactive visualization.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class HighlightStyle:
    """Configuration for highlighted marker appearance."""
    line_width: float = 10.0
    line_color: str = "#00FFFF"  # Bright cyan
    size: float = 12.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "lineWidth": self.line_width,
            "lineColor": self.line_color,
            "size": self.size,
        }


@dataclass
class TracePointMapping:
    """Mapping from DataFrame index to trace/point coordinates."""
    index_to_trace_point: Dict[int, Dict[str, int]]
    
    def get_trace_point(self, df_index: int) -> Optional[Dict[str, int]]:
        """Get trace/point mapping for a DataFrame index."""
        return self.index_to_trace_point.get(df_index)
    
    def has_index(self, df_index: int) -> bool:
        """Check if a DataFrame index exists in the mapping."""
        return df_index in self.index_to_trace_point


@dataclass
class PairingMetadata:
    """Metadata about pairings for validation and statistics."""
    total_sources: int
    total_targets: int
    orphaned_sources: List[int]
    orphaned_targets: List[int]
    max_targets_per_source: int
    
    def has_issues(self) -> bool:
        """Check if there are any validation issues."""
        return len(self.orphaned_sources) > 0 or len(self.orphaned_targets) > 0
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print(f"   🔗 Pairing sources: {self.total_sources}")
        print(f"   🎯 Pairing targets: {self.total_targets}")
        print(f"   📈 Max targets/source: {self.max_targets_per_source}")
        if self.orphaned_sources:
            print(f"   ⚠️  Orphaned sources (not in plot): {len(self.orphaned_sources)}")
        if self.orphaned_targets:
            print(f"   ⚠️  Orphaned targets (not in plot): {len(self.orphaned_targets)}")
