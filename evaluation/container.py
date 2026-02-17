from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.redis import RedisSaver

from evaluation.judge import LLMJudge
from evaluation.settings import COLLECTION_NAME, EvalSettings
from src.infrastructure.agent.langgraph import LanggraphAgent
from src.infrastructure.storage.qdrant import QdrantVectorStore


class EvalContainer:
    """Lightweight DI container for evaluation, built from EvalSettings."""

    def __init__(self, eval_settings: EvalSettings) -> None:
        self._settings = eval_settings

        self._embeddings = OllamaEmbeddings(
            model=eval_settings.embeddings_model,
            base_url=eval_settings.ollama_url,
        )

        self.vector_store = QdrantVectorStore(
            url=eval_settings.qdrant_url,
            collection_name=COLLECTION_NAME,
            embeddings=self._embeddings,
        )

        self._chat_model = ChatOllama(
            model=eval_settings.model,
            base_url=eval_settings.ollama_url,
        )

        checkpointer = RedisSaver(eval_settings.redis_url)

        self.agent = LanggraphAgent(
            llm=self._chat_model,
            vector_store=self.vector_store,
            checkpointer=checkpointer,
            retrieval_num_documents=eval_settings.retrieval_num_documents,
        )

        judge_model_name = eval_settings.judge_model or eval_settings.model
        self.judge = LLMJudge(
            ChatOllama(model=judge_model_name, base_url=eval_settings.ollama_url)
        )
