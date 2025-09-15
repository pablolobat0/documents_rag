import os
from app.services.chat import ChatService
from app.services.vector_storage import VectorStorageService
from app.services.image_captioning import ImageCaptioningService
from src.agent import Agent
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
IMAGE_CAPTIONING_MODEL = os.getenv("IMAGE_CAPTIONING_MODEL", "llava")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

agent = Agent()
chat_service = ChatService(agent=agent)
image_captioning_service = ImageCaptioningService(
    model=IMAGE_CAPTIONING_MODEL, base_url=OLLAMA_URL
)


def get_chat_service():
    return chat_service


def get_vector_storage_service():
    return VectorStorageService(
        url=QDRANT_URL,
        embeddings_url=OLLAMA_URL,
        embeddings_model=EMBEDDINGS_MODEL,
        image_captioning_service=image_captioning_service,
        image_storage_service=image_storage_service,
    )
