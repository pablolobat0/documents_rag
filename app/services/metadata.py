from langchain_core.language_models import BaseChatModel
from app.schemas.metadata import (
    Metadata,
    CurriculumVitae,
    Receipt,
    DocumentClassification,
)
from app.services.metadata_storage import MetadataStorageService
from typing import Union, Optional
import re
import os
from datetime import datetime
from pathlib import Path

CV_THRESHOLD = 8.0
RECEIPT_THRESHOLD = 8.0


class MetadataService:
    def __init__(
        self,
        model: BaseChatModel,
        storage_service: MetadataStorageService | None = None,
    ) -> None:
        self.llm = model
        self.storage_service = storage_service or MetadataStorageService()

        # CV keywords with weights for scoring
        self.cv_keywords = {
            "curriculum vitae": 5,
            "resume": 5,
            "cv": 4,
            "experience": 3,
            "education": 3,
            "skills": 3,
            "work history": 3,
            "employment": 3,
            "qualifications": 2,
            "professional summary": 4,
            "career objective": 2,
            "job title": 2,
            "company": 1,
            "position": 2,
            "duration": 1,
            "degree": 3,
            "university": 3,
            "certification": 2,
            "languages": 1,
            "portfolio": 2,
            "projects": 1,
            "references": 2,
            "contact information": 2,
            "phone": 1,
            "email": 1,
            "linkedin": 2,
            "github": 1,
            "location": 1,
            "profile": 1,
            "career": 1,
        }

        # Receipt keywords with weights for scoring
        self.receipt_keywords = {
            "receipt": 5,
            "invoice": 4,
            "bill": 3,
            "payment": 3,
            "transaction": 2,
            "total": 4,
            "subtotal": 3,
            "tax": 2,
            "amount due": 3,
            "amount paid": 3,
            "cash": 3,
            "credit": 3,
            "debit": 3,
            "purchase": 2,
            "sale": 2,
            "order": 2,
            "customer": 1,
            "merchant": 2,
            "store": 2,
            "shop": 2,
            "restaurant": 2,
            "date": 1,
            "time": 1,
            "item": 2,
            "quantity": 2,
            "price": 3,
            "cost": 2,
            "discount": 2,
            "change": 2,
            "balance": 1,
            "paid": 2,
            "due": 1,
            "transaction id": 2,
            "order number": 2,
            "invoice number": 2,
            "payment method": 2,
            "card": 2,
            "cash register": 3,
            "pos": 3,
            "store number": 1,
            "location": 1,
            "address": 1,
            "phone": 1,
            "thank you": 2,
            "have a nice day": 1,
            "keep the receipt": 2,
        }

    def calculate_cv_score(self, text: str) -> float:
        """
        Calculates a CV score based on weighted keyword matches.
        """
        text_lower = text.lower()
        total_score = 0

        for keyword, weight in self.cv_keywords.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def calculate_receipt_score(self, text: str) -> float:
        """
        Calculates a receipt score based on weighted keyword matches.
        """
        text_lower = text.lower()
        total_score = 0

        for keyword, weight in self.receipt_keywords.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def classify_with_llm(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """
        Uses LLM with structured output to classify document when keyword scores are inconclusive.
        """
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

    def extract_metadata(
        self, document: str, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """
        Extracts metadata from a document using keyword scoring and LLM classification.

        Args:
            document: The text content of the document
            base_metadata: Base metadata object containing file information
        """
        cv_score = self.calculate_cv_score(document)
        receipt_score = self.calculate_receipt_score(document)

        # If neither score meets threshold, use LLM classification
        if cv_score < CV_THRESHOLD and receipt_score < RECEIPT_THRESHOLD:
            return self.classify_with_llm(document, base_metadata)

        # If both scores meet threshold, use the higher one
        if cv_score >= CV_THRESHOLD or receipt_score >= RECEIPT_THRESHOLD:
            if cv_score > receipt_score:
                return self._extract_cv_metadata(document, base_metadata)
            else:
                return self._extract_receipt_metadata(document, base_metadata)

        return base_metadata

    def _extract_cv_metadata(
        self, document: str, base_metadata: Metadata
    ) -> CurriculumVitae:
        """
        Extracts CV metadata using structured LLM output with .with_structured_output()
        """
        structured_llm = self.llm.with_structured_output(CurriculumVitae)

        cv_prompt = f"""
        Extract the following information from this CV/resume text:

        Text: {document}

        Extract the person's name, email, phone number, LinkedIn profile, skills, work experience, and education.
        Return the data according to the CurriculumVitae schema.
        """

        try:
            result = structured_llm.invoke(cv_prompt)
            # Copy file information from base metadata
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
        """
        Extracts receipt metadata using structured LLM output with .with_structured_output()
        """
        structured_llm = self.llm.with_structured_output(Receipt)

        receipt_prompt = f"""
        Extract the following information from this receipt text:

        Text: {document}

        Extract the merchant name, address, transaction date/time, total amount, and items purchased.
        Return the data according to the Receipt schema.
        """

        try:
            result = structured_llm.invoke(receipt_prompt)
            # Copy file information from base metadata
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
