import logging
from functools import cached_property

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.redis import RedisSaver

from config.settings import settings
from src.application.use_cases.chat_with_documents import ChatWithDocumentsUseCase
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.infrastructure.agent.langgraph import LanggraphAgent
from src.infrastructure.processing.document_classifier import KeywordClassifier
from src.infrastructure.processing.image_captioner import LangchainImageCaptioner
from src.infrastructure.processing.metadata_extractor import LLMMetadataExtractor
from src.infrastructure.processing.pdf_processor import PypdfProcessor
from src.infrastructure.processing.text_splitter import LangchainTextSplitter
from src.infrastructure.storage.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


class Container:
    """Dependency injection container using cached_property for lazy singletons."""

    def __init__(self):
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate required configuration values at initialization."""
        errors = []

        if not settings.ollama_url:
            errors.append("OLLAMA_URL is required")
        if not settings.qdrant_url:
            errors.append("QDRANT_URL is required")
        if not settings.redis_url:
            errors.append("REDIS_URL is required")
        if not settings.model:
            errors.append("MODEL is required")
        if not settings.embeddings_model:
            errors.append("EMBEDDINGS_MODEL is required")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        logger.debug("Configuration validated successfully")

    @cached_property
    def chat_model(self) -> ChatOllama:
        return ChatOllama(
            model=settings.model,
            base_url=settings.ollama_url,
        )

    @cached_property
    def summary_model(self) -> ChatOllama:
        model = settings.summary_model or settings.model
        return ChatOllama(
            model=model,
            base_url=settings.ollama_url,
        )

    @cached_property
    def image_captioning_model(self) -> ChatOllama:
        model = settings.image_captioning_model or settings.model
        return ChatOllama(
            model=model,
            base_url=settings.ollama_url,
        )

    @cached_property
    def embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=settings.embeddings_model, base_url=settings.ollama_url
        )

    @cached_property
    def qdrant(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            embeddings=self.embeddings,
        )

    @cached_property
    def image_captioner(self) -> LangchainImageCaptioner:
        return LangchainImageCaptioner(
            llm=self.image_captioning_model,
            min_width=settings.min_image_width,
            min_height=settings.min_image_height,
        )

    @cached_property
    def pdf_processor(self) -> PypdfProcessor:
        return PypdfProcessor(image_captioner=self.image_captioner)

    @cached_property
    def document_classifier(self) -> KeywordClassifier:
        return KeywordClassifier(llm=self.chat_model)

    @cached_property
    def metadata_extractor(self) -> LLMMetadataExtractor:
        return LLMMetadataExtractor(llm=self.chat_model)

    @cached_property
    def text_splitter(self) -> LangchainTextSplitter:
        return LangchainTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    @cached_property
    def checkpointer(self) -> RedisSaver:
        return RedisSaver(settings.redis_url)

    @cached_property
    def agent(self) -> LanggraphAgent:
        return LanggraphAgent(
            llm=self.chat_model,
            vector_store=self.qdrant,
            checkpointer=self.checkpointer,
        )

    @cached_property
    def chat_use_case(self) -> ChatWithDocumentsUseCase:
        return ChatWithDocumentsUseCase(agent=self.agent)

    @cached_property
    def process_document_use_case(self) -> ProcessDocumentUseCase:
        return ProcessDocumentUseCase(
            vector_store=self.qdrant,
            pdf_processor=self.pdf_processor,
            text_splitter=self.text_splitter,
            document_classifier=self.document_classifier,
            metadata_extractor=self.metadata_extractor,
        )


_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance (singleton)."""
    global _container
    if _container is None:
        _container = Container()
    return _container
