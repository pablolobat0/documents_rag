from datetime import datetime
from typing import Union

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt
from src.domain.ports.document_classifier_port import (
    DocumentClassifierPort,
    DocumentType,
)
from src.domain.ports.llm_port import EmbeddingsPort, LLMPort
from src.domain.ports.metadata_storage_port import MetadataStoragePort
from src.domain.ports.pdf_processor_port import PdfProcessorPort
from src.domain.ports.text_splitter_port import TextSplitterPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.entities.classification import DocumentClassification
from src.domain.prompts.document_processing import DocumentPrompts
from src.domain.value_objects.chat_message import ChatMessage


class ProcessDocumentUseCase:
    """Use case for processing and storing documents."""

    def __init__(
        self,
        llm: LLMPort,
        embeddings: EmbeddingsPort,
        vector_store: VectorStorePort,
        pdf_processor: PdfProcessorPort,
        metadata_storage: MetadataStoragePort,
        text_splitter: TextSplitterPort,
        document_classifier: DocumentClassifierPort,
    ):
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._pdf_processor = pdf_processor
        self._metadata_storage = metadata_storage
        self._text_splitter = text_splitter
        self._document_classifier = document_classifier

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
            extracted_metadata = self._extract_metadata(full_text, base_metadata)

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

    def _extract_metadata(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Extract metadata using classification and LLM."""
        doc_type = self._document_classifier.classify(document)

        if doc_type == DocumentType.RECEIPT:
            return self._extract_receipt_metadata(document, base_metadata)
        elif doc_type == DocumentType.CV:
            return self._extract_cv_metadata(document, base_metadata)
        else:
            return self._classify_with_llm(document, base_metadata)

    def _classify_with_llm(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Use LLM to classify document when keyword scoring is inconclusive."""
        try:
            result = self._llm.invoke_structured(
                [
                    ChatMessage(
                        role="assistant",
                        content=DocumentPrompts.CLASSIFICATION_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=DocumentPrompts.format_classification_prompt(document),
                    ),
                ],
                DocumentClassification,
            )

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
        try:
            result = self._llm.invoke_structured(
                [
                    ChatMessage(
                        role="assistant",
                        content=DocumentPrompts.EXTRACT_CV_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=DocumentPrompts.format_extraction_prompt(document),
                    ),
                ],
                CurriculumVitae,
            )
            return CurriculumVitae(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
                name=getattr(result, "name", None),
                email=getattr(result, "email", None),
                phone_number=getattr(result, "phone_number", None),
                linkedin_profile=getattr(result, "linkedin_profile", None),
                skills=getattr(result, "skills", []),
                experience=getattr(result, "experience", []),
                education=getattr(result, "education", []),
            )
        except Exception:
            return CurriculumVitae(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
            )

    def _extract_receipt_metadata(
        self, document: str, base_metadata: Metadata
    ) -> Receipt:
        """Extract receipt metadata using LLM."""
        try:
            result = self._llm.invoke_structured(
                [
                    ChatMessage(
                        role="assistant",
                        content=DocumentPrompts.EXTRACT_RECEIPT_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=DocumentPrompts.format_extraction_prompt(document),
                    ),
                ],
                Receipt,
            )
            return Receipt(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
                merchant_name=getattr(result, "merchant_name", None),
                merchant_address=getattr(result, "merchant_address", None),
                transaction_date=getattr(result, "transaction_date", None),
                transaction_time=getattr(result, "transaction_time", None),
                total_amount=getattr(result, "total_amount", None),
                items=getattr(result, "items", []),
            )
        except Exception:
            return Receipt(
                pages=base_metadata.pages,
                document_name=base_metadata.document_name,
                file_path=base_metadata.file_path,
                file_size=base_metadata.file_size,
                file_type=base_metadata.file_type,
                created_at=base_metadata.created_at,
                processed_at=base_metadata.processed_at,
            )
