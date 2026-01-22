import re

from src.domain.entities.classification import DocumentClassification
from src.domain.ports.document_classifier_port import DocumentType
from src.domain.ports.llm_port import LLMPort
from src.domain.prompts.document_processing import DocumentPrompts
from src.infrastructure.processing.classification_keywords import (
    CV_KEYWORDS,
    CV_THRESHOLD,
    RECEIPT_KEYWORDS,
    RECEIPT_THRESHOLD,
)


class KeywordClassifier:
    """Document classifier with keyword scoring and LLM fallback."""

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def _calculate_cv_score(self, text: str) -> float:
        """Calculate CV score based on weighted keyword matches."""
        text_lower = text.lower()
        total_score = 0.0

        for keyword, weight in CV_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def _calculate_receipt_score(self, text: str) -> float:
        """Calculate receipt score based on weighted keyword matches."""
        text_lower = text.lower()
        total_score = 0.0

        for keyword, weight in RECEIPT_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def classify(self, text: str) -> DocumentType:
        """Classify a document using keyword scoring with LLM fallback."""
        cv_score = self._calculate_cv_score(text)
        receipt_score = self._calculate_receipt_score(text)

        if cv_score >= CV_THRESHOLD and cv_score > receipt_score:
            return DocumentType.CV
        elif receipt_score >= RECEIPT_THRESHOLD and receipt_score > cv_score:
            return DocumentType.RECEIPT
        else:
            return self._classify_with_llm(text)

    def _classify_with_llm(self, text: str) -> DocumentType:
        """Use LLM to classify document when keyword scoring is inconclusive."""
        try:
            result = self._llm.with_structured_output(DocumentClassification).invoke(
                [
                    ("system", DocumentPrompts.CLASSIFICATION_SYSTEM_PROMPT),
                    ("human", DocumentPrompts.format_classification_prompt(text)),
                ]
            )

            if result.document_type == "cv":
                return DocumentType.CV
            elif result.document_type == "receipt":
                return DocumentType.RECEIPT
            else:
                return DocumentType.UNKNOWN
        except Exception:
            return DocumentType.UNKNOWN
