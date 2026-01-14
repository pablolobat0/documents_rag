from functools import cached_property

from config.settings import settings
from src.application.use_cases.chat_with_documents import ChatWithDocumentsUseCase
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.services.document_classifier import DocumentClassifier
from src.domain.services.text_splitter import TextSplitter
from src.infrastructure.adapters.filesystem_adapter import FilesystemAdapter
from src.infrastructure.adapters.ollama_adapter import OllamaAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.adapters.redis_adapter import RedisAdapter
from src.infrastructure.rag.agent import Agent
from src.infrastructure.services.image_captioning import ImageCaptioningService
from src.infrastructure.services.pdf_processor import PdfProcessor


class Container:
    """Dependency injection container using cached_property for lazy singletons."""

    @cached_property
    def ollama_adapter(self) -> OllamaAdapter:
        return OllamaAdapter(
            base_url=settings.ollama_url,
            chat_model=settings.model,
            embeddings_model=settings.embeddings_model,
            summary_model=settings.summary_model,
            image_captioning_model=settings.image_captioning_model,
        )

    @cached_property
    def qdrant_adapter(self) -> QdrantAdapter:
        return QdrantAdapter(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            ollama_adapter=self.ollama_adapter,
        )

    @cached_property
    def redis_adapter(self) -> RedisAdapter:
        return RedisAdapter(redis_url=settings.redis_url)

    @cached_property
    def filesystem_adapter(self) -> FilesystemAdapter:
        return FilesystemAdapter(storage_dir="document_metadata")

    @cached_property
    def image_captioning_service(self) -> ImageCaptioningService:
        return ImageCaptioningService(
            ollama_adapter=self.ollama_adapter,
            min_width=settings.min_image_width,
            min_height=settings.min_image_height,
        )

    @cached_property
    def pdf_processor(self) -> PdfProcessor:
        return PdfProcessor(image_captioning_service=self.image_captioning_service)

    @cached_property
    def document_classifier(self) -> DocumentClassifier:
        return DocumentClassifier()

    @cached_property
    def text_splitter(self) -> TextSplitter:
        return TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    @cached_property
    def agent(self) -> Agent:
        return Agent(
            ollama_adapter=self.ollama_adapter,
            qdrant_adapter=self.qdrant_adapter,
            redis_adapter=self.redis_adapter,
        )

    @cached_property
    def chat_use_case(self) -> ChatWithDocumentsUseCase:
        return ChatWithDocumentsUseCase(agent=self.agent)

    @cached_property
    def process_document_use_case(self) -> ProcessDocumentUseCase:
        return ProcessDocumentUseCase(
            ollama_adapter=self.ollama_adapter,
            qdrant_adapter=self.qdrant_adapter,
            pdf_processor=self.pdf_processor,
            filesystem_adapter=self.filesystem_adapter,
            text_splitter=self.text_splitter,
            document_classifier=self.document_classifier,
        )


# Global container instance
_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance (singleton)."""
    global _container
    if _container is None:
        _container = Container()
    return _container
