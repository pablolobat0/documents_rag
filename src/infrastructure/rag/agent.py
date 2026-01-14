import re

from langchain.tools import Tool
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.infrastructure.adapters.ollama_adapter import OllamaAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.adapters.redis_adapter import RedisAdapter
from src.infrastructure.rag.prompts import (
    GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT,
    RERANK_SYSTEM_PROMPT,
)
from src.infrastructure.rag.schemas import RankedDocuments


class Agent:
    """LangGraph-based RAG Agent with dependency injection."""

    def __init__(
        self,
        ollama_adapter: OllamaAdapter,
        qdrant_adapter: QdrantAdapter,
        redis_adapter: RedisAdapter,
    ) -> None:
        self.ollama_adapter = ollama_adapter
        self.qdrant_adapter = qdrant_adapter
        self.redis_adapter = redis_adapter

        self.llm = ollama_adapter.get_chat_model()
        self.retriever = qdrant_adapter.get_retriever(search_type="mmr", k=10)
        self.vector_store = qdrant_adapter.get_vector_store()
        checkpointer = redis_adapter.get_checkpointer()

        # Create custom tool with retrieval
        retriever_tool = Tool(
            name="search_documents",
            description="Searches through uploaded documents to find relevant information based on a query. Returns document snippets that can be used to answer user questions.",
            func=self.retrieve_documents,
        )

        self.tools = [retriever_tool]

        # Build simplified graph without summarization
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
        builder.add_edge("retrieve", "generate_query_or_respond")

        self.graph = builder.compile(checkpointer=checkpointer)

    def generate_query_or_respond(self, state: MessagesState):
        messages = state["messages"]
        system_prompt = GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT

        # Build messages with system prompt
        llm_messages = [SystemMessage(content=system_prompt)] + list(messages)

        # Invoke LLM with tools bound
        response = self.llm.bind_tools(self.tools).invoke(llm_messages)

        # Remove thinking tags from response
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

    def retrieve_documents(self, query: str) -> list[str]:
        """Search documents and return relevant snippets based on the query."""
        try:
            retrieved_docs = self.retriever.invoke(query)

            if not retrieved_docs:
                return []

            documents_content = [doc.page_content for doc in retrieved_docs]

            structured_llm = self.llm.with_structured_output(RankedDocuments)

            formatted_docs = "\n\n".join(
                [
                    f"Document {i}: {content}"
                    for i, content in enumerate(documents_content)
                ]
            )

            rerank_prompt = f"{RERANK_SYSTEM_PROMPT}\n\nQuery: {query}\n\nRetrieved Documents:\n{formatted_docs}"

            result = structured_llm.invoke([SystemMessage(content=rerank_prompt)])

            if hasattr(result, "documents"):
                useful_docs = []
                for doc_rel in result.documents:
                    if doc_rel.is_useful and doc_rel.index < len(documents_content):
                        useful_docs.append(documents_content[doc_rel.index])

                return useful_docs
            else:
                return []

        except Exception as e:
            return [f"Error during retrieval and re-ranking: {e}"]

    def run(self, conversation: dict, session_id: str) -> str:
        """Run the agent with a conversation and session ID."""
        return self.graph.invoke(
            conversation, {"configurable": {"thread_id": session_id}}
        )["messages"][-1].content
