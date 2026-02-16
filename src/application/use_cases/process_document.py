import logging
from datetime import datetime

from pydantic import ValidationError

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.domain.entities.metadata import Metadata
from src.domain.ports.metadata_classifier_port import MetadataClassifierPort
from src.domain.ports.text_splitter_port import TextSplitterPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.document_classification import DocumentClassification
from src.domain.value_objects.page_content import PageContent
from src.infrastructure.processing.content_extractor_registry import (
    ContentExtractorRegistry,
)

logger = logging.getLogger(__name__)


class ProcessDocumentUseCase:
    """Use case for processing and storing documents."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        content_extractor_registry: ContentExtractorRegistry,
        text_splitter: TextSplitterPort,
        metadata_classifier: MetadataClassifierPort | None = None,
    ):
        self._vector_store = vector_store
        self._content_extractor_registry = content_extractor_registry
        self._text_splitter = text_splitter
        self._metadata_classifier = metadata_classifier

    def execute(self, request: ProcessDocumentRequest) -> ProcessDocumentResponse:
        """Process a document and store it in the vector database."""
        try:
            # Get appropriate extractor for content type
            extractor = self._content_extractor_registry.get_extractor(
                request.content_type
            )
            if extractor is None:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=Metadata(pages=0),
                    chunks_created=0,
                    message=f"Unsupported content type: {request.content_type}",
                )

            # Extract content with page/section information
            result = extractor.extract_content(request.content)

            if not result.page_contents:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=Metadata(pages=0),
                    chunks_created=0,
                    message="No content found in document",
                )

            now = datetime.now()
            metadata = Metadata(
                pages=result.total_pages,
                document_name=request.filename,
                file_size=len(request.content),
                file_type=request.content_type,
                created_at=now,
                processed_at=now,
                frontmatter=result.document_metadata,
            )

            # Extract classification from frontmatter or LLM
            classification = self._extract_classification(
                request.content_type, result.page_contents, metadata.frontmatter
            )

            # Build chunk metadata combining file info and frontmatter
            chunk_metadata = self._build_chunk_metadata(metadata, classification)

            # Split pages into chunks with metadata
            chunks = self._text_splitter.split_pages(
                result.page_contents, chunk_metadata
            )

            if not chunks:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=metadata,
                    chunks_created=0,
                    message="No chunks created from document",
                )

            # Store chunks with their metadata in vector store
            self._vector_store.upsert_chunks(chunks)

            return ProcessDocumentResponse(
                success=True,
                metadata=metadata,
                chunks_created=len(chunks),
                message="Document processed successfully",
            )

        except Exception as e:
            return ProcessDocumentResponse(
                success=False,
                metadata=Metadata(pages=0),
                chunks_created=0,
                message=f"Error processing document: {e!s}",
            )

    def _extract_classification(
        self,
        content_type: str,
        page_contents: list[PageContent],
        frontmatter: dict,
    ) -> DocumentClassification:
        """Extract document classification from frontmatter or LLM."""
        if content_type.startswith("text/markdown") and frontmatter:
            try:
                fm_data = {}
                if "type" in frontmatter:
                    fm_data["type"] = frontmatter["type"]
                if "tags" in frontmatter:
                    fm_data["tags"] = frontmatter["tags"]
                if fm_data:
                    return DocumentClassification(**fm_data)
            except ValidationError:
                logger.debug("Frontmatter classification validation failed")

        if self._metadata_classifier and page_contents:
            combined = "\n\n".join(p.content for p in page_contents[:3] if p.content)
            if combined.strip():
                return self._metadata_classifier.classify(combined)

        return DocumentClassification()

    def _build_chunk_metadata(
        self,
        metadata: Metadata,
        classification: DocumentClassification | None = None,
    ) -> dict:
        """Build a flat metadata dict for chunks from file info and frontmatter."""
        base = {
            "document_name": metadata.document_name,
            "file_type": metadata.file_type,
            "file_size": metadata.file_size,
            "total_pages": metadata.pages,
            "created_at": metadata.created_at,
            "processed_at": metadata.processed_at,
        }

        # Flatten frontmatter values into chunk metadata, skipping type/tags
        for key, value in metadata.frontmatter.items():
            if key in ("type", "tags"):
                continue
            if isinstance(value, list):
                base[key] = ", ".join(str(v) for v in value)
            else:
                base[key] = value

        # Add validated classification
        if classification:
            if classification.type:
                base["type"] = classification.type
            if classification.tags:
                base["tags"] = classification.tags

        return base
