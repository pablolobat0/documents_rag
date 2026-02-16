import pytest
from pydantic import ValidationError

from src.domain.value_objects.document_classification import DocumentClassification


class TestDocumentClassification:
    def test_valid_type(self):
        dc = DocumentClassification(type="book")
        assert dc.type == "book"

    def test_valid_tags(self):
        dc = DocumentClassification(tags=["AI", "LLM"])
        assert dc.tags == ["AI", "LLM"]

    def test_none_defaults(self):
        dc = DocumentClassification()
        assert dc.type is None
        assert dc.tags is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            DocumentClassification(type="invalid_type")

    def test_invalid_tag_raises(self):
        with pytest.raises(ValidationError):
            DocumentClassification(tags=["AI", "nonexistent"])

    def test_all_valid_types(self):
        for t in ("book", "recipe", "project", "prompt", "concept"):
            dc = DocumentClassification(type=t)
            assert dc.type == t

    def test_all_valid_tags(self):
        all_tags = [
            "AI",
            "LLM",
            "investment",
            "attention",
            "rag",
            "transformers",
            "psychology",
        ]
        dc = DocumentClassification(tags=all_tags)
        assert dc.tags == all_tags

    def test_empty_tags_list(self):
        dc = DocumentClassification(tags=[])
        assert dc.tags == []

    def test_type_with_tags(self):
        dc = DocumentClassification(type="concept", tags=["AI", "rag"])
        assert dc.type == "concept"
        assert dc.tags == ["AI", "rag"]
