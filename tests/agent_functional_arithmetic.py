"""算术 Agent —— LangGraph 函数式 API（@entrypoint / @task）版本。

与图 API 版本功能相同，但用更“命令式”的写法：
  - @task 装饰的函数是可并行、可重试的最小工作单元（调模型、调工具）
  - @entrypoint 装饰的函数是整个工作流入口，内部用普通 while 循环编排

说明：原示例模型是 "claude-sonnet-4-6"（需 Anthropic key）。这里改为读环境变量，
默认复用项目里已配置的 OpenAI 兼容端点（如 gpt-5.5 @ kkidc），开箱即跑。
"""

# Step 1: 定义工具与模型 -------------------------------------------------------
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()

# 若要用官方 Claude，可改成： model = init_chat_model("claude-sonnet-4-6", temperature=0)
model = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-4.1"),
    model_provider="openai",
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    api_key=os.getenv("OPENAI_API_KEY"),
)


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


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# 相关导入 --------------------------------------------------------------------
from langchain.messages import HumanMessage, SystemMessage, ToolCall
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages


# Step 2: 定义模型节点（task）------------------------------------------------
@task
def call_llm(messages: list[BaseMessage]):
    """LLM 决定是否调用工具。"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )


# Step 3: 定义工具节点（task）------------------------------------------------
@task
def call_tool(tool_call: ToolCall):
    """执行单个工具调用。"""
    tool = tools_by_name[tool_call["name"]]
    # 传入完整的 ToolCall 字典时，tool.invoke 会直接返回一个 ToolMessage
    return tool.invoke(tool_call)


# Step 4: 定义 Agent 主流程（entrypoint）-------------------------------------
@entrypoint()
def agent(messages: list[BaseMessage]):
    # 先让模型响应一次
    model_response = call_llm(messages).result()

    while True:
        # 模型没有再发起工具调用 -> 结束循环
        if not model_response.tool_calls:
            break

        # 并行执行本轮所有工具调用（call_tool 返回 future，先全部提交再收结果）
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # 把“模型的回复 + 工具结果”追加进消息历史，再让模型继续
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    # 把最终回复也追加进历史并返回
    messages = add_messages(messages, model_response)
    return messages


# 运行 -----------------------------------------------------------------------
if __name__ == "__main__":
    input_messages = [HumanMessage(content="Add 3 and 4.")]

    # 方式一：一次性拿到最终结果并逐条打印
    final_messages = agent.invoke(input_messages)
    for m in final_messages:
        m.pretty_print()

    # 方式二（可选）：流式观察每一步产生的状态更新
    # for chunk in agent.stream(input_messages, stream_mode="updates"):
    #     print(chunk, "\n")
