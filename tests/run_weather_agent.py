"""Python port of the TypeScript LangGraph weather-agent example.

Faithful equivalent of the pasted @langchain/langgraph snippet:
a ReAct-style agent (agent <-> tools loop) with a fake `search` tool.
"""

from __future__ import annotations

import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

load_dotenv()


def build_model(tools):
    """Build the chat model from env vars so it's fully customizable.

    Env vars (all optional except the API key):
      OPENAI_MODEL       model name          (default: gpt-4.1)
      OPENAI_BASE_URL    OpenAI-compatible endpoint, e.g. a local/DeepSeek/
                         vLLM/Ollama server. Leave unset for OpenAI itself.
      OPENAI_TEMPERATURE sampling temperature (default: 0)
      OPENAI_API_KEY     API key
    """
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    return model.bind_tools(tools)


# --- State: messages channel with a concat/append reducer ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- Tool: same behaviour as the TS `searchTool` ---
@tool
def search(query: str) -> str:
    """Call to surf the web."""
    q = query.lower()
    if "sf" in q or "san francisco" in q:
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]
tool_node = ToolNode(tools)

model = build_model(tools)


# --- Conditional edge: continue to tools or end ---
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


# --- Agent node ---
async def call_model(state: AgentState) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


# --- Wire up the graph ---
workflow = (
    StateGraph(AgentState)
    .add_node("agent", call_model)
    .add_node("tools", tool_node)
    .add_edge("__start__", "agent")
    .add_conditional_edges("agent", should_continue)
    .add_edge("tools", "agent")
)

app = workflow.compile()


async def main() -> None:
    print(
        f"Using model={os.getenv('OPENAI_MODEL', 'gpt-4.1')!r} "
        f"base_url={os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1 (default)'}"
    )
    final_state = await app.ainvoke(
        {"messages": [HumanMessage(content="what is the weather in sf")]},
        {"configurable": {"thread_id": "42"}},
    )
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
