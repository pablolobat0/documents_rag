from enum import Enum
from typing import Protocol


class DocumentType(Enum):
    CV = "cv"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class DocumentClassifierPort(Protocol):
    """Port for document classification."""

    def classify(self, text: str) -> DocumentType:
        """Classify a document based on its content."""
        ...
