"""
CLSS: Contrastive Learning for Sequence and Structure

A contrastive learning model that co-embeds protein sequences and structures
into a unified 32-dimensional space using a two-tower architecture.
"""

__version__ = "0.3.3"

# Configuration and utilities
from .config import CLSSConfig

# Core model and dataset classes
from .model import CLSSModel
from .utils import download_pretrained_model

# Version info
__author__ = "Guy Yanai, Gabriel Axel, Liam M. Longo, Nir Ben-Tal, Rachel Kolodny"
__email__ = "guy@shay.co.il"

# Primary API exports
__all__ = [
    # Configuration
    "CLSSConfig",
    # Core classes (for advanced usage)
    "CLSSModel",
    # Utilities
    "download_pretrained_model",
]
