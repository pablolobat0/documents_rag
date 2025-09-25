from typing import List, Optional
from pydantic import BaseModel


class DocumentRelevance(BaseModel):
    index: int
    is_useful: bool


class RankedDocuments(BaseModel):
    query: str
    documents: List[DocumentRelevance]