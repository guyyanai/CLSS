"""
Constants for static map generation.
Centralizes hardcoded values for easier maintenance.
"""

# DataFrame column names
SEQUENCE_COLUMN = "sequence"
STRUCTURE_COLUMN = "structure"
MODALITY_COLUMN = "modality"
EMBEDDING_COLUMN = "embedding"
X_COLUMN = "x"
Y_COLUMN = "y"

# Modality types
MODALITY_SEQUENCE = "sequence"
MODALITY_STRUCTURE = "structure"

# Cache filenames
CACHE_SEQUENCES_FILE = "sequences.pkl"
CACHE_STRUCTURES_FILE = "structures.pkl"
CACHE_SEQUENCE_EMBEDDINGS_FILE = "sequence_embeddings.pkl"
CACHE_STRUCTURE_EMBEDDINGS_FILE = "structure_embeddings.pkl"
CACHE_REDUCED_EMBEDDINGS_FILE = "reduced_embeddings.pkl"

# Default batch size for embeddings
DEFAULT_BATCH_SIZE = 32

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
