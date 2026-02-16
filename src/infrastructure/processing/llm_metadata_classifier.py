import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.prompts.metadata_classifier import MetadataClassifierPrompts
from src.domain.value_objects.document_classification import DocumentClassification

logger = logging.getLogger(__name__)


class LlmMetadataClassifier:
    """Classifies documents using LLM structured output."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def classify(self, content: str) -> DocumentClassification:
        try:
            structured_llm = self._llm.with_structured_output(DocumentClassification)
            result = structured_llm.invoke(
                [
                    SystemMessage(
                        content=MetadataClassifierPrompts.CLASSIFICATION_SYSTEM_PROMPT
                    ),
                    HumanMessage(content=content),
                ]
            )
            if isinstance(result, DocumentClassification):
                return result
            return DocumentClassification()
        except Exception as e:
            logger.warning("Metadata classification failed: %s", e)
            return DocumentClassification()
