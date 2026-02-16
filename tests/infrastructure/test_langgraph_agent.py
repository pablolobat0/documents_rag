from unittest.mock import MagicMock, create_autospec

from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.retrieved_document import RetrievedDocument
from src.infrastructure.agent.langgraph import LanggraphAgent


class TestRetrieveDocumentsFilters:
    def _make_agent(self, mock_vector_store):
        mock_llm = MagicMock()
        mock_checkpointer = MagicMock()
        # Prevent graph compilation from failing
        mock_checkpointer.config_specs = []

        agent = LanggraphAgent(
            llm=mock_llm,
            vector_store=mock_vector_store,
            checkpointer=mock_checkpointer,
        )
        return agent

    def test_no_filters_passes_none(self):
        mock_vs = create_autospec(VectorStorePort, instance=True)
        mock_vs.search.return_value = []
        agent = self._make_agent(mock_vs)

        agent._retrieve_documents("test query")

        mock_vs.search.assert_called_once_with("test query", 10, filters=None)

    def test_type_filter_passed(self):
        mock_vs = create_autospec(VectorStorePort, instance=True)
        mock_vs.search.return_value = []
        agent = self._make_agent(mock_vs)

        agent._retrieve_documents("test query", type="book")

        mock_vs.search.assert_called_once_with(
            "test query", 10, filters={"type": "book"}
        )

    def test_tags_filter_passed(self):
        mock_vs = create_autospec(VectorStorePort, instance=True)
        mock_vs.search.return_value = []
        agent = self._make_agent(mock_vs)

        agent._retrieve_documents("test query", tags=["AI", "LLM"])

        mock_vs.search.assert_called_once_with(
            "test query", 10, filters={"tags": ["AI", "LLM"]}
        )

    def test_both_filters_passed(self):
        mock_vs = create_autospec(VectorStorePort, instance=True)
        mock_vs.search.return_value = []
        agent = self._make_agent(mock_vs)

        agent._retrieve_documents("test query", type="concept", tags=["rag"])

        mock_vs.search.assert_called_once_with(
            "test query", 10, filters={"type": "concept", "tags": ["rag"]}
        )

    def test_filters_with_results_triggers_reranking(self):
        mock_vs = create_autospec(VectorStorePort, instance=True)
        mock_vs.search.return_value = [
            RetrievedDocument(page_content="doc1 content", metadata={}),
        ]
        agent = self._make_agent(mock_vs)

        # Mock the LLM structured output for reranking
        from src.infrastructure.agent.schemas import DocumentRelevance, RankedDocuments

        mock_ranked = RankedDocuments(
            query="test",
            documents=[DocumentRelevance(index=0, is_useful=True)],
        )
        agent._llm.with_structured_output.return_value.invoke.return_value = mock_ranked

        result = agent._retrieve_documents("test query", type="book")

        assert result == ["doc1 content"]
