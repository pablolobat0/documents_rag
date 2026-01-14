from datetime import datetime
from typing import Union

from langchain_core.language_models import BaseChatModel

from src.application.dto.upload_dto import ProcessDocumentRequest, ProcessDocumentResponse
from src.domain.entities.metadata import (
    CurriculumVitae,
    DocumentClassification,
    Metadata,
    Receipt,
)
from src.domain.services.document_classifier import DocumentClassifier, DocumentType
from src.domain.services.text_splitter import TextSplitter
from src.infrastructure.adapters.filesystem_adapter import FilesystemAdapter
from src.infrastructure.adapters.ollama_adapter import OllamaAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.services.pdf_processor import PdfProcessor


class ProcessDocumentUseCase:
    """Use case for processing and storing documents."""

    def __init__(
        self,
        ollama_adapter: OllamaAdapter,
        qdrant_adapter: QdrantAdapter,
        pdf_processor: PdfProcessor,
        filesystem_adapter: FilesystemAdapter,
        text_splitter: TextSplitter,
        document_classifier: DocumentClassifier,
    ):
        self.ollama_adapter = ollama_adapter
        self.qdrant_adapter = qdrant_adapter
        self.pdf_processor = pdf_processor
        self.filesystem_adapter = filesystem_adapter
        self.text_splitter = text_splitter
        self.document_classifier = document_classifier
        self.llm = ollama_adapter.get_chat_model()

    def execute(self, request: ProcessDocumentRequest) -> ProcessDocumentResponse:
        """Process a document and store it in the vector database."""
        try:
            # Extract content based on file type
            if request.content_type == "application/pdf":
                documents, pages = self.pdf_processor.extract_content(request.content)
            else:
                # Text file
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

            # Create base metadata
            base_metadata = Metadata(
                pages=pages,
                document_name=request.filename,
                file_size=len(request.content),
                file_type=request.content_type,
                created_at=datetime.now(),
                processed_at=datetime.now(),
            )

            # Classify document and extract metadata
            full_text = "\n".join(documents)
            extracted_metadata = self._extract_metadata(full_text, base_metadata)

            # Split into chunks
            chunks = self.text_splitter.split(full_text)

            if not chunks:
                return ProcessDocumentResponse(
                    success=False,
                    metadata=base_metadata,
                    chunks_created=0,
                    message="No chunks created from document",
                )

            # Generate embeddings
            embeddings = self.ollama_adapter.embed_documents(chunks)

            # Store in vector database
            self.qdrant_adapter.upsert(chunks, embeddings)

            # Save metadata to filesystem
            try:
                self.filesystem_adapter.save_metadata(extracted_metadata)
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

    def _extract_metadata(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Extract metadata using classification and LLM."""
        doc_type, cv_score, receipt_score = self.document_classifier.classify(document)

        if doc_type == DocumentType.UNKNOWN:
            # Use LLM for classification
            return self._classify_with_llm(document, base_metadata)
        elif doc_type == DocumentType.CV:
            return self._extract_cv_metadata(document, base_metadata)
        else:
            return self._extract_receipt_metadata(document, base_metadata)

    def _classify_with_llm(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Use LLM to classify document when keyword scoring is inconclusive."""
        structured_llm = self.llm.with_structured_output(DocumentClassification)

        classification_prompt = f"""
        Classify this document as either 'cv' (curriculum vitae/resume), 'receipt', or 'none'.

        Document text: {document}
        """

        try:
            result = structured_llm.invoke(classification_prompt)

            if result.document_type == "cv":
                return self._extract_cv_metadata(document, base_metadata)
            elif result.document_type == "receipt":
                return self._extract_receipt_metadata(document, base_metadata)
            else:
                return base_metadata

        except Exception:
            return base_metadata

    def _extract_cv_metadata(
        self, document: str, base_metadata: Metadata
    ) -> CurriculumVitae:
        """Extract CV metadata using LLM."""
        structured_llm = self.llm.with_structured_output(CurriculumVitae)

        cv_prompt = f"""
        Extract the following information from this CV/resume text:

        Text: {document}

        Extract the person's name, email, phone number, LinkedIn profile, skills, work experience, and education.
        Return the data according to the CurriculumVitae schema.
        """

        try:
            result = structured_llm.invoke(cv_prompt)
            result.document_name = base_metadata.document_name
            result.file_path = base_metadata.file_path
            result.file_size = base_metadata.file_size
            result.file_type = base_metadata.file_type
            result.created_at = base_metadata.created_at
            result.processed_at = base_metadata.processed_at
            result.pages = base_metadata.pages
            return result
        except Exception:
            return CurriculumVitae(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
                name=None,
                email=None,
                phone_number=None,
                linkedin_profile=None,
                skills=[],
                experience=[],
                education=[],
            )

    def _extract_receipt_metadata(
        self, document: str, base_metadata: Metadata
    ) -> Receipt:
        """Extract receipt metadata using LLM."""
        structured_llm = self.llm.with_structured_output(Receipt)

        receipt_prompt = f"""
        Extract the following information from this receipt text:

        Text: {document}

        Extract the merchant name, address, transaction date/time, total amount, and items purchased.
        Return the data according to the Receipt schema.
        """

        try:
            result = structured_llm.invoke(receipt_prompt)
            result.document_name = base_metadata.document_name
            result.file_path = base_metadata.file_path
            result.file_size = base_metadata.file_size
            result.file_type = base_metadata.file_type
            result.created_at = base_metadata.created_at
            result.processed_at = base_metadata.processed_at
            result.pages = base_metadata.pages
            return result
        except Exception:
            return Receipt(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
                merchant_name=None,
                merchant_address=None,
                transaction_date=None,
                transaction_time=None,
                total_amount=None,
                items=[],
            )
