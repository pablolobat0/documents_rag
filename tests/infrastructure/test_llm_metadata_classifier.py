from unittest.mock import MagicMock

from src.domain.value_objects.document_classification import DocumentClassification
from src.infrastructure.processing.llm_metadata_classifier import (
    LlmMetadataClassifier,
)


class TestLlmMetadataClassifier:
    def test_returns_classification_from_llm(self):
        expected = DocumentClassification(type="book", tags=["AI"])
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = expected
        mock_llm.with_structured_output.return_value = mock_structured

        classifier = LlmMetadataClassifier(llm=mock_llm)
        result = classifier.classify("some content about AI books")

        assert result == expected
        mock_llm.with_structured_output.assert_called_once_with(DocumentClassification)

    def test_returns_empty_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = RuntimeError("fail")

        classifier = LlmMetadataClassifier(llm=mock_llm)
        result = classifier.classify("content")

        assert result.type is None
        assert result.tags is None

    def test_returns_empty_on_non_classification_result(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = "not a classification"
        mock_llm.with_structured_output.return_value = mock_structured

        classifier = LlmMetadataClassifier(llm=mock_llm)
        result = classifier.classify("content")

        assert result.type is None
        assert result.tags is None
