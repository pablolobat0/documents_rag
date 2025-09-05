import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "qwen3:1.7b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


class Agent:
    def __init__(self) -> None:
        self.llm = ChatOllama(model=MODEL)
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

        client = QdrantClient(url=QDRANT_URL)

        vector_size = len(embeddings.embed_query("sample text"))

        if not client.collection_exists("test"):
            client.create_collection(
                collection_name="test",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="test",
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_documents",
            "Search and return information about Lilian Weng blog posts.",
        )

        self.tools = [retriever_tool]

        builder = StateGraph(MessagesState)
        builder.add_node("generate_query_or_respond", self.generate_query_or_respond)
        builder.add_node("retrieve", ToolNode([retriever_tool]))

        builder.add_edge(START, "generate_query_or_respond")
        builder.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        builder.add_edge(
            "retrieve",
            "generate_query_or_respond",
        )

        self.graph = builder.compile()

    def generate_query_or_respond(self, state: MessagesState):
        response = self.llm.bind_tools(self.tools).invoke(state["messages"])

        return {"messages": [response]}

    def run(self, conversation: dict):
        return self.graph.invoke(conversation)["messages"][-1].content
