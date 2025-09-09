import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver
from src.prompts import GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage

load_dotenv()

MODEL = os.getenv("MODEL", "qwen3:1.7b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class Agent:
    def __init__(self) -> None:
        self.llm = ChatOllama(model=MODEL, base_url=OLLAMA_URL)
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_URL)
        checkpointer = RedisSaver(REDIS_URL)

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="test",
            url=QDRANT_URL,
        )

        self.retriever = vector_store.as_retriever()

        retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_documents",
            "Accesses a vector database to find and return relevant document snippets based on a query.",
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

        self.graph = builder.compile(checkpointer=checkpointer)

    def generate_query_or_respond(self, state: MessagesState):
        messages = [
            SystemMessage(content=GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT)
        ] + state["messages"]
        response = self.llm.bind_tools(self.tools).invoke(messages)

        return {"messages": [response]}

    def run(self, conversation: dict, session_id: str):
        docs = self.retriever.invoke("incidencia")
        print(docs)

        return self.graph.invoke(
            conversation, {"configurable": {"thread_id": session_id}}
        )["messages"][-1].content
