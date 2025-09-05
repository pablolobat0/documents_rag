import os
from app.services.chat_service import ChatService
from app.services.vector_storage import VectorStorageService
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")


def get_chat_service():
    return ChatService()


def get_vector_storage_service():
    return VectorStorageService(
        url=QDRANT_URL, embeddings_url=OLLAMA_URL, embeddings_model=EMBEDDINGS_MODEL
    )
