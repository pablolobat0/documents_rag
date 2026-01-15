from typing import Any, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.domain.value_objects.chat_message import ChatMessage


class OllamaLLM:
    """Ollama LLM implementation. Implements LLMPort and EmbeddingsPort."""

    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embeddings_model: str,
        summary_model: str | None = None,
        image_captioning_model: str | None = None,
    ):
        self.base_url = base_url
        self.chat_model_name = chat_model
        self.embeddings_model_name = embeddings_model
        self.summary_model_name = summary_model or chat_model
        self.image_captioning_model_name = image_captioning_model or chat_model

        self._chat_llm = ChatOllama(model=chat_model, base_url=base_url)
        self._summary_llm = ChatOllama(model=self.summary_model_name, base_url=base_url)
        self._embeddings = OllamaEmbeddings(model=embeddings_model, base_url=base_url)
        self._image_captioning_llm = ChatOllama(
            model=self.image_captioning_model_name, base_url=base_url
        )

    def _convert_messages(
        self, messages: Union[list[ChatMessage], list[dict]]
    ) -> list:
        """Convert domain ChatMessage or dict to LangChain message types."""
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
        return langchain_messages

    def invoke(self, messages: list[ChatMessage]) -> str:
        """Invoke LLM with conversation messages and return response content."""
        langchain_messages = self._convert_messages(messages)
        response = self._chat_llm.invoke(langchain_messages)
        return str(response.content)

    def invoke_structured(
        self,
        messages: Union[list[ChatMessage], list[dict]],
        schema: type,
    ) -> Any:
        """Invoke LLM with messages and parse response into structured schema."""
        langchain_messages = self._convert_messages(messages)
        structured_llm = self._chat_llm.with_structured_output(schema)
        return structured_llm.invoke(langchain_messages)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        return self._embeddings.embed_query(text)

    def get_chat_model(self) -> ChatOllama:
        """Get the underlying chat model (for internal infrastructure use)."""
        return self._chat_llm

    def get_summary_model(self) -> ChatOllama:
        """Get the summary model (for internal infrastructure use)."""
        return self._summary_llm

    def get_image_captioning_model(self) -> ChatOllama:
        """Get the image captioning model (for internal infrastructure use)."""
        return self._image_captioning_llm

    def get_embeddings(self) -> OllamaEmbeddings:
        """Get the embeddings model (for internal infrastructure use)."""
        return self._embeddings
