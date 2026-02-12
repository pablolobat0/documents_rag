import logging
from datetime import datetime

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.domain.entities.metadata import Metadata
from src.domain.ports.text_splitter_port import TextSplitterPort
from src.domain.ports.vector_store_port import VectorStorePort
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
    ):
        self._vector_store = vector_store
        self._content_extractor_registry = content_extractor_registry
        self._text_splitter = text_splitter

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

            # Build chunk metadata combining file info and frontmatter
            chunk_metadata = self._build_chunk_metadata(metadata)

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

    def _build_chunk_metadata(self, metadata: Metadata) -> dict:
        """Build a flat metadata dict for chunks from file info and frontmatter."""
        base = {
            "document_name": metadata.document_name,
            "file_type": metadata.file_type,
            "file_size": metadata.file_size,
            "total_pages": metadata.pages,
            "created_at": metadata.created_at,
            "processed_at": metadata.processed_at,
        }

        # Flatten frontmatter values into chunk metadata
        for key, value in metadata.frontmatter.items():
            if isinstance(value, list):
                base[key] = ", ".join(str(v) for v in value)
            else:
                base[key] = value

        return base
