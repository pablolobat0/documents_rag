from langchain_ollama import ChatOllama, OllamaEmbeddings


class OllamaAdapter:
    """Adapter for Ollama LLM and embeddings."""

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

    def get_chat_model(self) -> ChatOllama:
        return self._chat_llm

    def get_summary_model(self) -> ChatOllama:
        return self._summary_llm

    def get_image_captioning_model(self) -> ChatOllama:
        return self._image_captioning_llm

    def get_embeddings(self) -> OllamaEmbeddings:
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
