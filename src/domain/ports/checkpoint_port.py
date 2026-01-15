from typing import Any, Protocol


class CheckpointPort(Protocol):
    """Port for conversation checkpoint storage."""

    def get_checkpointer(self) -> Any:
        """Get the checkpointer for LangGraph."""
        ...
