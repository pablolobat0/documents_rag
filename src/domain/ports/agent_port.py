from typing import Protocol

from src.domain.value_objects.chat_message import ChatMessage


class AgentPort(Protocol):
    """Port for RAG Agent operations."""

    def run(self, messages: list[ChatMessage], session_id: str) -> str:
        """
        Run the agent workflow with conversation messages.

        Args:
            messages: List of chat messages in the conversation
            session_id: Unique session identifier for conversation persistence

        Returns:
            Agent response content as string
        """
        ...
