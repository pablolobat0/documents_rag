import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

COLLECTION_NAME = "evaluation"

DATASET_NAME = "G4KMU/vectara_open_ragbench"
DATASET_SPLIT = "text_tables"
CACHE_PATH = Path(__file__).parent / "samples_cache.json"


class Mode(StrEnum):
    retrieval = "retrieval"
    generation = "generation"
    full = "full"


@dataclass(frozen=True)
class EvalSettings:
    """Evaluation-specific settings, read from EVAL_* env vars."""

    ollama_url: str = os.getenv("EVAL_OLLAMA_URL", "http://localhost:11434")
    qdrant_url: str = os.getenv("EVAL_QDRANT_URL", "http://localhost:6333")
    redis_url: str = os.getenv("EVAL_REDIS_URL", "redis://localhost:6379")
    model: str = os.getenv("EVAL_MODEL", "qwen3:1.7b")
    judge_model: str = os.getenv("EVAL_JUDGE_MODEL", "")
    embeddings_model: str = os.getenv("EVAL_EMBEDDINGS_MODEL", "all-minilm")
    retrieval_num_documents: int = int(os.getenv("EVAL_RETRIEVAL_NUM_DOCUMENTS", "10"))
    chunk_size: int = int(os.getenv("EVAL_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("EVAL_CHUNK_OVERLAP", "50"))
