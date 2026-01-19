from datetime import datetime

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.domain.entities.metadata import Metadata
from src.domain.ports.document_classifier_port import DocumentClassifierPort
from src.domain.ports.llm_port import EmbeddingsPort
from src.domain.ports.metadata_extractor_port import MetadataExtractorPort
from src.domain.ports.metadata_storage_port import MetadataStoragePort
from src.domain.ports.pdf_processor_port import PdfProcessorPort
from src.domain.ports.text_splitter_port import TextSplitterPort
from src.domain.ports.vector_store_port import VectorStorePort


class ProcessDocumentUseCase:
    """Use case for processing and storing documents."""

    def __init__(
        self,
        embeddings: EmbeddingsPort,
        vector_store: VectorStorePort,
        pdf_processor: PdfProcessorPort,
        metadata_storage: MetadataStoragePort,
        text_splitter: TextSplitterPort,
        document_classifier: DocumentClassifierPort,
        metadata_extractor: MetadataExtractorPort,
    ):
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._pdf_processor = pdf_processor
        self._metadata_storage = metadata_storage
        self._text_splitter = text_splitter
        self._document_classifier = document_classifier
        self._metadata_extractor = metadata_extractor

    def execute(self, request: ProcessDocumentRequest) -> ProcessDocumentResponse:
        """Process a document and store it in the vector database."""
        try:
            if request.content_type == "application/pdf":
                documents, pages = self._pdf_processor.extract_content(request.content)
            else:
                content = request.content.decode("utf-8")
                documents = [content]
                pages = 1

            if not documents:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=Metadata(pages=0),
                    chunks_created=0,
                    message="No content found in document",
                )

            base_metadata = Metadata(
                pages=pages,
                document_name=request.filename,
                file_size=len(request.content),
                file_type=request.content_type,
                created_at=datetime.now(),
                processed_at=datetime.now(),
            )

            full_text = "\n".join(documents)

            doc_type = self._document_classifier.classify(full_text)
            extracted_metadata = self._metadata_extractor.extract(
                full_text, doc_type, base_metadata
            )

            chunks = self._text_splitter.split(full_text)

            if not chunks:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=base_metadata,
                    chunks_created=0,
                    message="No chunks created from document",
                )

            embeddings = self._embeddings.embed_documents(chunks)

            self._vector_store.upsert(chunks, embeddings)

            try:
                self._metadata_storage.save_metadata(extracted_metadata)
            except Exception as e:
                print(f"Warning: Failed to save metadata: {e}")

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
