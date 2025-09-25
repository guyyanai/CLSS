"""
Configuration and utilities for CLSS package.
"""

from typing import Any, Dict
from dataclasses import dataclass, asdict


@dataclass
class CLSSConfig:
    """Configuration dataclass for CLSS."""

    # Model configuration
    esm2_checkpoint: str = "facebook/esm2_t12_35M_UR50D"
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 180
    init_temperature: float = 0.5
    should_learn_temperature: bool = False
    random_sequence_stretches: bool = True
    random_stretch_min_size: int = 10
    use_global_loss: bool = False
    should_load_esm3: bool = False

    # Data configuration
    structures_dir: str = "/path/to/structures/pdbs/folder"
    dataset_limit: int = 1000000
    validation_frac: float = 0.2

    # Training configuration
    epochs: int = 80
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return asdict(self)
