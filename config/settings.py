import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # LLM Models
    model: str = os.getenv("MODEL", "qwen3:1.7b")
    summary_model: str = os.getenv("SUMMARY_MODEL", "llama3.2:3b")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
    image_captioning_model: str = os.getenv("IMAGE_CAPTIONING_MODEL", "gemma3:4b")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # Infrastructure URLs
    ollama_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")

    # Qdrant
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")

    # Image processing
    min_image_width: int = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
    min_image_height: int = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))

    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )


settings = Settings()
