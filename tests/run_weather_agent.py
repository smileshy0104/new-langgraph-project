"""LangGraph 天气 Agent 示例（由 TypeScript 版 @langchain/langgraph 片段移植而来）。

这是一个 ReAct 风格的智能体：模型和工具之间循环（agent <-> tools），
配合一个模拟的 `search` 工具，用来演示 LangGraph 的最小可运行图。

运行流程：
    用户提问 -> agent 节点(调用大模型) -> 模型决定是否调用工具
      ├─ 需要工具 -> tools 节点(执行 search) -> 再回到 agent 节点
      └─ 无需工具 -> 结束，输出最终回答
"""

from __future__ import annotations

import asyncio          # 运行异步 main 入口
import operator         # 提供 operator.add，用作 messages 的合并规则
import os               # 读取环境变量（模型名、base_url 等）
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv                                    # 加载 .env 文件
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool                            # 用装饰器把普通函数变成工具
from langgraph.graph import StateGraph                           # 构建状态图
from langgraph.prebuilt import ToolNode                          # 预置的工具执行节点
from langchain_openai import ChatOpenAI                          # OpenAI/兼容接口的聊天模型

# 把 .env 里的变量（OPENAI_API_KEY、OPENAI_MODEL、OPENAI_BASE_URL 等）读进环境变量
load_dotenv()


def build_model(tools):
    """根据环境变量构建聊天模型，便于自定义模型/服务而无需改代码。

    可用的环境变量（除 API key 外均可选）：
      OPENAI_MODEL       模型名称              （默认：gpt-4.1）
      OPENAI_BASE_URL    OpenAI 兼容的接口地址，例如 DeepSeek / vLLM / Ollama
                         等自建或第三方服务；用官方 OpenAI 时留空即可。
                         注意：地址通常要以 /v1 结尾。
      OPENAI_TEMPERATURE 采样温度              （默认：0，越低越确定）
      OPENAI_API_KEY     API 密钥
    """
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),              # 取不到时用默认模型
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),  # 字符串转成浮点数
        base_url=os.getenv("OPENAI_BASE_URL") or None,           # 未设置则走官方 OpenAI
    )
    # bind_tools：把工具的 schema 告诉模型，模型才知道有哪些工具可调用
    return model.bind_tools(tools)


# --- 状态定义：messages 通道，用 operator.add 做“追加合并” ---
class AgentState(TypedDict):
    # Annotated[..., operator.add] 表示每个节点返回的 messages 会“拼接”到已有列表后面，
    # 而不是覆盖，从而保留完整的对话历史。
    # operator.add 是一个函数，用于“合并”多个消息，这里用它来“追加”消息
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- 工具定义：与 TS 版 `searchTool` 行为一致（这里是写死的假数据） ---
@tool
def search(query: str) -> str:
    """Call to surf the web."""  # 该 docstring 会作为工具描述提供给模型
    q = query.lower()
    # 命中旧金山相关关键词就返回“有雾”，否则返回“晴天”
    if "sf" in q or "san francisco" in q:
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]              # 工具列表（此处只有一个）
tool_node = ToolNode(tools)   # 把工具封装成图里的一个节点，负责执行模型请求的工具调用

model = build_model(tools)    # 构建并绑定了工具的模型实例


# --- 条件边：根据最后一条消息决定“继续调用工具”还是“结束” ---
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]  # 取出最新一条消息
    # 如果模型的回复里带有工具调用请求，就走到 tools 节点去执行
    # isinstance(last_message, AIMessage)：判断消息是不是模型回复
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    # 否则说明模型已给出最终答案，结束整个图
    return "__end__"


# --- agent 节点：调用大模型，返回模型的回复 ---
async def call_model(state: AgentState) -> dict:
    # async invoke：异步调用模型，返回模型回复
    response = await model.ainvoke(state["messages"])  # 把完整历史发给模型
    # 返回值会按 AgentState 的合并规则追加到 messages 中
    return {"messages": [response]}


# --- 组装状态图：定义节点与它们之间的流转关系 ---
workflow = (
    StateGraph(AgentState)
    .add_node("agent", call_model)              # 注册 agent 节点（调模型）
    .add_node("tools", tool_node)               # 注册 tools 节点（执行工具）
    .add_edge("__start__", "agent")             # 入口 -> agent
    .add_conditional_edges("agent", should_continue)  # agent 之后按条件分支
    .add_edge("tools", "agent")                 # 工具执行完再回到 agent
)

app = workflow.compile()  # 编译成可运行的应用


# --- 运行入口 ---
async def main() -> None:
    # 先打印当前使用的模型与接口地址，方便确认自定义配置是否生效
    print(
        f"Using model={os.getenv('OPENAI_MODEL', 'gpt-4.1')!r} "
        f"base_url={os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1 (default)'}"
    )
    # 以一条用户提问触发整个图；thread_id 用于标识会话（多轮/持久化时有用）
    final_state = await app.ainvoke(
        {"messages": [HumanMessage(content="what is the weather in sf")]},
        {"configurable": {"thread_id": "42"}},
    )
    # 打印最后一条消息的内容，即模型给出的最终回答
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())  # 启动异步事件循环并执行 main
