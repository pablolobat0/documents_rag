from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DocumentClassification:
    """Result of document classification."""

    document_type: Literal["receipt", "cv", "none"]
