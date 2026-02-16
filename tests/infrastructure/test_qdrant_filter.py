from qdrant_client.http import models

from src.infrastructure.storage.qdrant import QdrantVectorStore


class TestBuildFilter:
    def test_single_string_filter(self):
        result = QdrantVectorStore._build_filter({"type": "book"})

        assert isinstance(result, models.Filter)
        assert len(result.must) == 1
        condition = result.must[0]
        assert condition.key == "metadata.type"
        assert isinstance(condition.match, models.MatchValue)
        assert condition.match.value == "book"

    def test_list_filter_uses_match_any(self):
        result = QdrantVectorStore._build_filter({"tags": ["AI", "LLM"]})

        assert len(result.must) == 1
        condition = result.must[0]
        assert condition.key == "metadata.tags"
        assert isinstance(condition.match, models.MatchAny)
        assert condition.match.any == ["AI", "LLM"]

    def test_multiple_filters(self):
        result = QdrantVectorStore._build_filter({"type": "book", "tags": ["AI"]})

        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert keys == {"metadata.type", "metadata.tags"}

    def test_empty_dict_returns_empty_filter(self):
        result = QdrantVectorStore._build_filter({})
        assert isinstance(result, models.Filter)
        assert len(result.must) == 0
