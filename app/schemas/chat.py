from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatConversation(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str | None = None
    sources_used: list[str] | None = None
    tool_calls: list[str] | None = None
