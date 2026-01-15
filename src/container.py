from functools import cached_property

from config.settings import settings
from src.application.use_cases.chat_with_documents import ChatWithDocumentsUseCase
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.infrastructure.agent.langgraph import LanggraphAgent
from src.infrastructure.llm.ollama import OllamaLLM
from src.infrastructure.processing.document_classifier import KeywordClassifier
from src.infrastructure.processing.image_captioner import OllamaImageCaptioner
from src.infrastructure.processing.pdf_processor import PypdfProcessor
from src.infrastructure.processing.text_splitter import LangchainTextSplitter
from src.infrastructure.storage.filesystem import FilesystemStorage
from src.infrastructure.storage.qdrant import QdrantVectorStore
from src.infrastructure.storage.redis import RedisCheckpoint


class Container:
    """Dependency injection container using cached_property for lazy singletons."""

    @cached_property
    def ollama(self) -> OllamaLLM:
        return OllamaLLM(
            base_url=settings.ollama_url,
            chat_model=settings.model,
            embeddings_model=settings.embeddings_model,
            summary_model=settings.summary_model,
            image_captioning_model=settings.image_captioning_model,
        )

    @cached_property
    def qdrant(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            ollama=self.ollama,
        )

    @cached_property
    def redis(self) -> RedisCheckpoint:
        return RedisCheckpoint(redis_url=settings.redis_url)

    @cached_property
    def filesystem(self) -> FilesystemStorage:
        return FilesystemStorage(storage_dir="document_metadata")

    @cached_property
    def image_captioner(self) -> OllamaImageCaptioner:
        return OllamaImageCaptioner(
            ollama=self.ollama,
            min_width=settings.min_image_width,
            min_height=settings.min_image_height,
        )

    @cached_property
    def pdf_processor(self) -> PypdfProcessor:
        return PypdfProcessor(image_captioner=self.image_captioner)

    @cached_property
    def document_classifier(self) -> KeywordClassifier:
        return KeywordClassifier()

    @cached_property
    def text_splitter(self) -> LangchainTextSplitter:
        return LangchainTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    @cached_property
    def agent(self) -> LanggraphAgent:
        return LanggraphAgent(
            ollama=self.ollama,
            qdrant=self.qdrant,
            redis=self.redis,
        )

    @cached_property
    def chat_use_case(self) -> ChatWithDocumentsUseCase:
        return ChatWithDocumentsUseCase(agent=self.agent)

    @cached_property
    def process_document_use_case(self) -> ProcessDocumentUseCase:
        return ProcessDocumentUseCase(
            llm=self.ollama,
            embeddings=self.ollama,
            vector_store=self.qdrant,
            pdf_processor=self.pdf_processor,
            metadata_storage=self.filesystem,
            text_splitter=self.text_splitter,
            document_classifier=self.document_classifier,
        )


_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance (singleton)."""
    global _container
    if _container is None:
        _container = Container()
    return _container
