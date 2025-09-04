import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")


class Agent:
    def __init__(self) -> None:
        self.llm = ChatOllama(model=MODEL_NAME)

        builder = StateGraph(MessagesState)
        builder.add_node("generate_query_or_respond", self.generate_query_or_respond)

        builder.add_edge(START, "generate_query_or_respond")
        builder.add_edge("generate_query_or_respond", END)
        self.graph = builder.compile()

    def generate_query_or_respond(self, state: MessagesState):
        response = self.llm.invoke(state["messages"])

        return {"messages": [response]}

    def run(self, query: str):
        return self.graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )[
            "messages"
        ][-1]
