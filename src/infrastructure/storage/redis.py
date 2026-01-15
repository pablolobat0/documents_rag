from typing import Any

from langgraph.checkpoint.redis import RedisSaver


class RedisCheckpoint:
    """Redis checkpoint implementation. Implements CheckpointPort."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._checkpointer = RedisSaver(redis_url)

    def get_checkpointer(self) -> Any:
        """Get the Redis checkpointer for LangGraph."""
        return self._checkpointer
