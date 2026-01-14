from dataclasses import dataclass, field
from datetime import datetime

from src.domain.value_objects.chat_message import ChatMessage, SessionId


@dataclass
class ChatRequest:
    session_id: SessionId
    messages: list[ChatMessage]


@dataclass
class ChatResponse:
    content: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources_used: list[str] | None = None
    tool_calls: list[str] | None = None
