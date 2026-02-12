from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant"] = Field(description="Who sent the message")
    content: str = Field(description="Message text")


class ChatRequestSchema(BaseModel):
    """Request body for the chat endpoint."""

    session_id: str = Field(
        description="Unique session identifier for conversation continuity",
        min_length=1,
    )
    messages: list[ChatMessageSchema] = Field(
        description="Full conversation history (user and assistant turns)",
        min_length=1,
    )


class ChatResponseSchema(BaseModel):
    """Response body from the chat endpoint."""

    content: str = Field(description="Assistant response text")
    session_id: str = Field(description="Session identifier echoed back")
    timestamp: datetime = Field(description="When the response was generated")
    sources_used: list[str] | None = Field(
        default=None, description="Document sources referenced in the answer"
    )
    tool_calls: list[str] | None = Field(
        default=None, description="Tools invoked during generation"
    )


class MetadataSchema(BaseModel):
    """Document metadata produced during processing."""

    pages: int = Field(description="Number of pages/sections extracted")
    document_name: str | None = None
    file_path: str | None = None
    file_size: int | None = None
    file_type: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None
    frontmatter: dict = Field(
        default_factory=dict,
        description="Frontmatter key-value pairs (Markdown files)",
    )


class DocumentResultSchema(BaseModel):
    """Processing result for a single document in a batch."""

    filename: str = Field(description="Original filename")
    success: bool = Field(description="Whether processing succeeded")
    chunks_created: int = Field(description="Number of vector-store chunks created")
    message: str = Field(default="", description="Error message on failure")
    metadata: MetadataSchema


class BatchResponseSchema(BaseModel):
    """Aggregated results for a batch document upload."""

    total_documents: int = Field(description="Number of documents submitted")
    successful: int = Field(description="Documents processed successfully")
    failed: int = Field(description="Documents that failed processing")
    results: list[DocumentResultSchema] = Field(description="Per-document results")
