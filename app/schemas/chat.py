from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatConversation(BaseModel):
    session_id: str
    messages: list[ChatMessage]
