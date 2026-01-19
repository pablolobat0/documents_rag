from typing import Union

from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt
from src.domain.ports.document_classifier_port import DocumentType
from src.domain.ports.llm_port import LLMPort
from src.domain.prompts.document_processing import DocumentPrompts


class LLMMetadataExtractor:
    """LLM-based metadata extractor."""

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def extract(
        self, text: str, document_type: DocumentType, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Extract metadata from document based on its type."""
        if document_type == DocumentType.CV:
            return self._extract_cv_metadata(text, base_metadata)
        elif document_type == DocumentType.RECEIPT:
            return self._extract_receipt_metadata(text, base_metadata)
        else:
            return base_metadata

    def _extract_cv_metadata(self, text: str, base_metadata: Metadata) -> CurriculumVitae:
        """Extract CV metadata using LLM."""
        try:
            result = self._llm.with_structured_output(CurriculumVitae).invoke(
                [
                    ("system", DocumentPrompts.EXTRACT_CV_SYSTEM_PROMPT),
                    ("human", DocumentPrompts.format_extraction_prompt(text)),
                ]
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

    def _extract_receipt_metadata(self, text: str, base_metadata: Metadata) -> Receipt:
        """Extract receipt metadata using LLM."""
        try:
            result = self._llm.with_structured_output(Receipt).invoke(
                [
                    ("system", DocumentPrompts.EXTRACT_RECEIPT_SYSTEM_PROMPT),
                    ("human", DocumentPrompts.format_extraction_prompt(text)),
                ]
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
