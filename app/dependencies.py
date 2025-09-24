import os

from langchain_ollama import ChatOllama
from app.services.chat import ChatService
from app.services.metadata import MetadataService
from app.services.vector_storage import VectorStorageService
from app.services.image_captioning import ImageCaptioningService
from src.agent import Agent
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
IMAGE_CAPTIONING_MODEL = os.getenv("IMAGE_CAPTIONING_MODEL", "llava")
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

model = ChatOllama(model=IMAGE_CAPTIONING_MODEL, base_url=OLLAMA_URL)

agent = Agent()
chat_service = ChatService(agent=agent)
image_captioning_service = ImageCaptioningService(
    model=IMAGE_CAPTIONING_MODEL,
    base_url=OLLAMA_URL,
    min_width=MIN_IMAGE_WIDTH,
    min_height=MIN_IMAGE_HEIGHT
)
metadata_service = MetadataService(
    model=model
)


def get_chat_service():
    return chat_service


def get_vector_storage_service():
    return VectorStorageService(
        url=QDRANT_URL,
        embeddings_url=OLLAMA_URL,
        embeddings_model=EMBEDDINGS_MODEL,
        image_captioning_service=image_captioning_service,
        metadata_service=metadata_service
    )
