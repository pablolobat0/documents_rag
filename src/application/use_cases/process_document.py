import logging
from datetime import datetime

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt
from src.domain.ports.document_classifier_port import DocumentClassifierPort
from src.domain.ports.metadata_extractor_port import MetadataExtractorPort
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
        document_classifier: DocumentClassifierPort,
        metadata_extractor: MetadataExtractorPort,
    ):
        self._vector_store = vector_store
        self._content_extractor_registry = content_extractor_registry
        self._text_splitter = text_splitter
        self._document_classifier = document_classifier
        self._metadata_extractor = metadata_extractor

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
            page_contents, pages = extractor.extract_content(request.content)

            if not page_contents:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=Metadata(pages=0),
                    chunks_created=0,
                    message="No content found in document",
                )

            # Build base metadata
            base_metadata = Metadata(
                pages=pages,
                document_name=request.filename,
                file_size=len(request.content),
                file_type=request.content_type,
                created_at=datetime.now(),
                processed_at=datetime.now(),
            )

            # Classification and metadata extraction (using full text)
            full_text = "\n".join(pc.content for pc in page_contents)
            doc_type = self._document_classifier.classify(full_text)
            extracted_metadata = self._metadata_extractor.extract(
                full_text, doc_type, base_metadata
            )

            # Build metadata dict for chunks
            chunk_base_metadata = self._build_chunk_metadata(
                extracted_metadata, doc_type
            )

            # Split pages into chunks with metadata
            chunks = self._text_splitter.split_pages(page_contents, chunk_base_metadata)

            if not chunks:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=base_metadata,
                    chunks_created=0,
                    message="No chunks created from document",
                )

            # Store chunks with their metadata in vector store
            self._vector_store.upsert_chunks(chunks)

            return ProcessDocumentResponse(
                success=True,
                metadata=extracted_metadata,
                chunks_created=len(chunks),
                message="Document processed successfully",
            )

        except Exception as e:
            return ProcessDocumentResponse(
                success=False,
                metadata=Metadata(pages=0),
                chunks_created=0,
                message=f"Error processing document: {str(e)}",
            )

    def _build_chunk_metadata(self, metadata: Metadata, doc_type) -> dict:
        """Build a flat metadata dict for chunks from the extracted metadata."""
        base = {
            "document_name": metadata.document_name,
            "file_type": metadata.file_type,
            "file_size": metadata.file_size,
            "total_pages": metadata.pages,
            "document_type": doc_type.value,
            "created_at": metadata.created_at,
            "processed_at": metadata.processed_at,
        }

        # Add type-specific fields
        if isinstance(metadata, CurriculumVitae):
            base.update(
                {
                    "cv_name": metadata.name,
                    "cv_email": metadata.email,
                    "cv_phone": metadata.phone_number,
                    "cv_linkedin": metadata.linkedin_profile,
                    "cv_skills": metadata.skills,
                }
            )
        elif isinstance(metadata, Receipt):
            base.update(
                {
                    "receipt_merchant": metadata.merchant_name,
                    "receipt_date": metadata.transaction_date,
                    "receipt_total": metadata.total_amount,
                }
            )

        return base
