from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class SessionId:
    value: str

    def __post_init__(self):
        if not self.value or len(self.value) < 1:
            raise ValueError("SessionId cannot be empty")

    def __str__(self) -> str:
        return self.value
