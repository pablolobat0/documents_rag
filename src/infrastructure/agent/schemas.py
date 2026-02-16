from pydantic import BaseModel

from src.domain.value_objects.document_classification import DocumentTag, DocumentType


class DocumentRelevance(BaseModel):
    index: int
    is_useful: bool


class RankedDocuments(BaseModel):
    query: str
    documents: list[DocumentRelevance]


class SearchDocumentsInput(BaseModel):
    query: str
    type: DocumentType | None = None
    tags: list[DocumentTag] | None = None
