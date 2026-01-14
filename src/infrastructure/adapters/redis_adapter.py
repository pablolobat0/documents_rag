from langgraph.checkpoint.redis import RedisSaver


class RedisAdapter:
    """Adapter for Redis checkpointing."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._checkpointer = RedisSaver(redis_url)

    def get_checkpointer(self) -> RedisSaver:
        """Get the Redis checkpointer for LangGraph."""
        return self._checkpointer
