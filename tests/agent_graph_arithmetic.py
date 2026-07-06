"""算术 Agent —— LangGraph 图（StateGraph）API 版本。

用 add / multiply / divide 三个工具，构建一个 ReAct 风格智能体：
    START -> llm_call(模型) --有工具调用--> tool_node(执行工具) -> llm_call
                          \--无工具调用--> END(回复用户)

说明：原示例模型是 "claude-sonnet-4-6"（需 Anthropic key）。这里改为读环境变量，
默认复用项目里已配置的 OpenAI 兼容端点（如 gpt-5.5 @ kkidc），开箱即跑。
"""

# Step 1: 定义工具与模型 -------------------------------------------------------
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()  # 读取 .env（OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL 等）

# 用环境变量构建模型，便于自定义。若要用官方 Claude，可改成：
#   model = init_chat_model("claude-sonnet-4-6", temperature=0)
model = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-4.1"),
    model_provider="openai",                      # 走 OpenAI 兼容协议
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL") or None,  # 自定义端点，留空则用官方 OpenAI
    api_key=os.getenv("OPENAI_API_KEY"),
)


# 定义工具（@tool 装饰器把函数变成可被模型调用的工具，docstring 作为工具描述）
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# 把工具绑定到模型，并建立“名字 -> 工具”的映射，供 tool_node 查表执行
tools = [add, multiply, divide]
# tool_by_name 映射，方便 tool_node 查表执行
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# Step 2: 定义状态 ------------------------------------------------------------
import operator
from typing import Annotated

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    # messages 用 operator.add 做“追加合并”，保留完整对话历史
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int  # 记录模型被调用的次数


# Step 3: 定义模型节点 --------------------------------------------------------
from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    """LLM 决定是否调用工具。"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Step 4: 定义工具节点 --------------------------------------------------------
from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """执行模型请求的工具调用。"""
    result = []
    # 遍历最新一条 AI 消息里的所有工具调用请求
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]          # 按名字取出工具
        observation = tool.invoke(tool_call["args"])     # 用参数执行工具
        # 工具返回的是数字，ToolMessage 的 content 需为字符串，这里用 str() 转换
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


# Step 5: 定义分支逻辑（是否结束）--------------------------------------------
from typing import Literal

from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """根据模型是否发起工具调用，决定继续还是结束。"""
    last_message = state["messages"][-1]
    # 模型发起了工具调用 -> 去 tool_node 执行
    if last_message.tool_calls:
        return "tool_node"
    # 否则结束，把最终答案回复用户
    return END


# Step 6: 构建并运行 Agent ----------------------------------------------------
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 添加边，连接各节点
agent_builder.add_edge(START, "llm_call")             # 入口 -> 模型节点
agent_builder.add_conditional_edges(                  # 模型节点后按条件分支
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")       # 工具执行完回到模型节点

agent = agent_builder.compile()  # 编译成可运行的 agent


def save_graph_png(path: str = "agent_graph.png") -> None:
    """尝试把流程图导出为 PNG（需要联网 mermaid.ink，失败则跳过，不影响运行）。"""
    try:
        png = agent.get_graph(xray=True).draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png)
        print(f"[graph] 流程图已保存到 {path}")
    except Exception as e:  # 无网络 / 无 graphviz 时静默跳过
        print(f"[graph] 跳过流程图导出：{type(e).__name__}: {e}")


if __name__ == "__main__":
    from langchain.messages import HumanMessage

    save_graph_png()  # 可选：导出流程图

    # 触发 agent：让它计算 3 + 4
    messages = [HumanMessage(content="Add 3 and 4.")]
    result = agent.invoke({"messages": messages})

    # 依次打印每条消息（用户提问、模型的工具调用、工具结果、模型最终回答）
    for m in result["messages"]:
        m.pretty_print()
    print(f"\n[llm_calls] 模型共被调用 {result['llm_calls']} 次")
