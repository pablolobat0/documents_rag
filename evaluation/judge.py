import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from evaluation.prompts import (
    ANSWER_RELEVANCE_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_USER_PROMPT,
    FAITHFULNESS_SYSTEM_PROMPT,
    FAITHFULNESS_USER_PROMPT,
    format_contexts,
)
from evaluation.schemas import JudgeScore

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-based judge for evaluating RAG answer quality."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._structured_llm = llm.with_structured_output(JudgeScore)

    def evaluate_faithfulness(
        self, question: str, contexts: list[str], answer: str
    ) -> JudgeScore:
        """Score how well the answer is grounded in the retrieved contexts."""
        try:
            result = self._structured_llm.invoke(
                [
                    SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
                    HumanMessage(
                        content=FAITHFULNESS_USER_PROMPT.format(
                            question=question,
                            contexts=format_contexts(contexts),
                            answer=answer,
                        )
                    ),
                ]
            )
            if isinstance(result, JudgeScore):
                return result
            return JudgeScore(score=0, reasoning="Evaluation failed: unexpected output")
        except Exception as e:
            logger.error("Faithfulness evaluation failed: %s", e)
            return JudgeScore(score=0, reasoning=f"Evaluation failed: {e}")

    def evaluate_answer_relevance(self, question: str, answer: str) -> JudgeScore:
        """Score how well the answer addresses the question."""
        try:
            result = self._structured_llm.invoke(
                [
                    SystemMessage(content=ANSWER_RELEVANCE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=ANSWER_RELEVANCE_USER_PROMPT.format(
                            question=question,
                            answer=answer,
                        )
                    ),
                ]
            )
            if isinstance(result, JudgeScore):
                return result
            return JudgeScore(score=0, reasoning="Evaluation failed: unexpected output")
        except Exception as e:
            logger.error("Answer relevance evaluation failed: %s", e)
            return JudgeScore(score=0, reasoning=f"Evaluation failed: {e}")
