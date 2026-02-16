from typing import Protocol

from src.domain.value_objects.document_classification import DocumentClassification


class MetadataClassifierPort(Protocol):
    def classify(self, content: str) -> DocumentClassification: ...
