from typing import Literal

from pydantic import BaseModel

DocumentType = Literal["book", "recipe", "project", "prompt", "concept"]
DocumentTag = Literal[
    "AI", "LLM", "investment", "attention", "rag", "transformers", "psychology"
]


class DocumentClassification(BaseModel):
    type: DocumentType | None = None
    tags: list[DocumentTag] | None = None
