from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatConversation(BaseModel):
    messages: list[ChatMessage]
