import os
import re
from typing import Any
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
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import AnyMessage

load_dotenv()

MODEL = os.getenv("MODEL", "qwen3:1.7b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.2:3b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

SUMMARY_TOKENS = 1024
TOKENS_BEFORE_SUMMARY = 4096

# TODO: add Rerank, judge model, improve tool response to store sources for the model response, evaluate other vector search functions
# add a function to check if the models are downloaded in Ollama and inject the model into de Agent


class State(MessagesState):
    context: dict[str, RunningSummary]


class LLMInputState(MessagesState):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


class Agent:
    def __init__(self) -> None:
        self.llm = ChatOllama(model=MODEL, base_url=OLLAMA_URL)
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_URL)
        checkpointer = RedisSaver(REDIS_URL)

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

        self.retriever = vector_store.as_retriever()

        retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_documents",
            "Accesses a vector database to find and return relevant document snippets based on a query.",
        )

        self.tools = [retriever_tool]

        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=ChatOllama(model=SUMMARY_MODEL, base_url=OLLAMA_URL),
            max_tokens=TOKENS_BEFORE_SUMMARY,
            max_tokens_before_summary=TOKENS_BEFORE_SUMMARY,
            max_summary_tokens=SUMMARY_TOKENS,
        )

        builder = StateGraph(State)
        builder.add_node("generate_query_or_respond", self.generate_query_or_respond)
        builder.add_node("retrieve", ToolNode([retriever_tool]))
        builder.add_node("summarize", summarization_node)

        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "generate_query_or_respond")
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

    def generate_query_or_respond(self, state: LLMInputState):
        summarize_messages = state["summarized_messages"]
        messages = state["messages"]
        system_prompt = GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT

        # Avoid two system messages
        if summarize_messages and isinstance(summarize_messages[0], SystemMessage):
            system_prompt = f"{system_prompt}\n\n{summarize_messages[0].content}"
            summarize_messages = summarize_messages[1:]

        # Put the tool call result in the array
        tool_message = []
        if messages and isinstance(messages[-1], ToolMessage):
            tool_message = [messages[-2], messages[-1]]

        if tool_message:
            messages = (
                [SystemMessage(content=system_prompt)]
                + summarize_messages
                + tool_message
            )
        else:
            messages = [SystemMessage(content=system_prompt)] + summarize_messages

        response = self.llm.bind_tools(self.tools).invoke(messages)

        # Remove thinking
        if isinstance(response, AIMessage):
            cleaned_content = re.sub(
                r"<think>.*?</think>", "", str(response.content), flags=re.DOTALL
            ).strip()
            response = AIMessage(
                content=cleaned_content,
                additional_kwargs=response.additional_kwargs,
                response_metadata=response.response_metadata,
                id=response.id,
                tool_calls=response.tool_calls,
                usage_metadata=response.usage_metadata,
            )

        return {"messages": [response]}

    def run(self, conversation: dict, session_id: str):
        return self.graph.invoke(
            conversation, {"configurable": {"thread_id": session_id}}
        )["messages"][-1].content
