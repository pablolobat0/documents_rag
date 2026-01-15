from enum import Enum
from typing import Protocol


class DocumentType(Enum):
    CV = "cv"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class DocumentClassifierPort(Protocol):
    """Port for document classification operations."""

    def classify(self, text: str) -> tuple[DocumentType, float, float]:
        """
        Classify a document based on its content.

        Args:
            text: Document text content

        Returns:
            Tuple of (document_type, cv_score, receipt_score)
        """
        ...

    def needs_llm_classification(self, text: str) -> bool:
        """Returns True if classification is inconclusive and LLM should be used."""
        ...
