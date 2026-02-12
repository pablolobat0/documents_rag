import logging
import re

from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.prompts.agent import AgentPrompts
from src.domain.value_objects.chat_message import ChatMessage
from src.infrastructure.agent.schemas import RankedDocuments

logger = logging.getLogger(__name__)

THINKING_TAG_PATTERNS = [
    r"<think>.*?</think>",
    r"<thinking>.*?</thinking>",
    r"<reasoning>.*?</reasoning>",
    r"<reflection>.*?</reflection>",
]


class LanggraphAgent:
    """LangGraph-based RAG Agent. Implements AgentPort."""

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStorePort,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._llm = llm
        self.vector_store = vector_store

        retriever_tool = Tool(
            name="search_documents",
            description="Searches through uploaded documents to find relevant information based on a query. Returns document snippets that can be used to answer user questions.",
            func=self._retrieve_documents,
        )

        self._tools = [retriever_tool]

        builder = StateGraph(MessagesState)
        builder.add_node("generate_query_or_respond", self._generate_query_or_respond)
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

        self._graph = builder.compile(checkpointer=checkpointer)

    def _clean_llm_response(self, content: str) -> str:
        """Remove thinking/reasoning tags from LLM responses.

        Handles various formats like <think>, <thinking>, <reasoning>, etc.
        Returns cleaned content with tags removed.
        """
        original_length = len(content)
        for pattern in THINKING_TAG_PATTERNS:
            content = re.sub(pattern, "", content, flags=re.DOTALL)
        cleaned = content.strip()
        if len(cleaned) != original_length:
            logger.debug("Cleaned thinking tags from LLM response")
        return cleaned

    def _generate_query_or_respond(self, state: MessagesState):
        messages = state["messages"]

        llm_messages = [
            SystemMessage(content=AgentPrompts.GENERATE_QUERY_OR_RESPOND_SYSTEM_PROMPT),
            *messages,
        ]

        response = self._llm.bind_tools(self._tools).invoke(llm_messages)

        if isinstance(response, AIMessage):
            cleaned_content = self._clean_llm_response(str(response.content))
            response = AIMessage(
                content=cleaned_content,
                additional_kwargs=response.additional_kwargs,
                response_metadata=response.response_metadata,
                id=response.id,
                tool_calls=response.tool_calls,
                usage_metadata=response.usage_metadata,
            )

        return {"messages": [response]}

    def _retrieve_documents(self, query: str) -> list[str]:
        """Search documents and return relevant snippets based on the query."""
        try:
            retrieved_docs = self.vector_store.search(query, 10)

            if not retrieved_docs:
                return ["No relevant documents found for this query."]

            documents_content = [doc.page_content for doc in retrieved_docs]

            structured_llm = self._llm.with_structured_output(RankedDocuments)

            user_prompt = AgentPrompts.format_reranker_prompt(query, documents_content)

            result = structured_llm.invoke(
                [
                    SystemMessage(content=AgentPrompts.RERANK_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            if isinstance(result, RankedDocuments):
                useful_docs = []
                for doc_rel in result.documents:
                    if doc_rel.is_useful and doc_rel.index < len(documents_content):
                        useful_docs.append(documents_content[doc_rel.index])

                if not useful_docs:
                    return ["No relevant documents found for this query."]
                return useful_docs
            else:
                return ["No relevant documents found for this query."]

        except Exception as e:
            logger.error("Error during retrieval and re-ranking: %s", e)
            return ["Unable to search documents at this time."]

    def run(self, messages: list[ChatMessage], session_id: str) -> str:
        """Run the agent with conversation messages and session ID."""
        langchain_messages = self._convert_messages(messages)
        result = self._graph.invoke(
            {"messages": langchain_messages},
            {"configurable": {"thread_id": session_id}},
        )
        return result["messages"][-1].content

    def _convert_messages(self, messages: list[ChatMessage]) -> list:
        """Convert domain ChatMessage to LangChain message types."""
        langchain_messages = []
        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
        return langchain_messages
