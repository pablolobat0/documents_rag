from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequestSchema(BaseModel):
    session_id: str
    messages: list[ChatMessageSchema]


class ChatResponseSchema(BaseModel):
    content: str
    session_id: str
    timestamp: datetime
    sources_used: list[str] | None = None
    tool_calls: list[str] | None = None


class MetadataSchema(BaseModel):
    pages: int
    document_name: str | None = None
    file_path: str | None = None
    file_size: int | None = None
    file_type: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None
    frontmatter: dict = Field(default_factory=dict)


class DocumentResultSchema(BaseModel):
    filename: str
    success: bool
    chunks_created: int
    message: str = ""
    metadata: MetadataSchema


class BatchResponseSchema(BaseModel):
    total_documents: int
    successful: int
    failed: int
    results: list[DocumentResultSchema]
