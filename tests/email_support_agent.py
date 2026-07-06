"""客户支持邮件 Agent —— 来自 LangChain 文档《Thinking in LangGraph》的案例。

整体流程（路由决策发生在各节点内部，通过 Command(goto=...) 表达）：

    START -> read_email -> classify_intent
                              ├─ billing / critical -> human_review
                              ├─ question / feature  -> search_documentation -> draft_response
                              ├─ bug                 -> bug_tracking        -> draft_response
                              └─ 其它(complex)        -> draft_response
    draft_response ├─ 高优先级/复杂 -> human_review ├─ 批准 -> send_reply -> END
                   └─ 否则          -> send_reply    └─ 拒绝 -> END(人工接手)

说明：原文档模型是 ChatOpenAI(model="gpt-5-nano")。这里改为读环境变量，
默认复用项目里已配置的 OpenAI 兼容端点（如 gpt-5.5 @ kkidc），开箱即跑。
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field

load_dotenv()  # 读取 .env（OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL）

# 用环境变量构建模型，便于自定义端点/模型
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


# --- 状态设计（只存原始数据，不存格式化后的文本/提示词）---------------------
class EmailClassification(TypedDict):
    """邮件分类结果的结构。"""
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    # 原始邮件数据（后面无法重建）
    email_content: str
    sender_email: str
    email_id: str
    # 分类结果（多个下游节点会用到）
    classification: EmailClassification | None
    # 原始检索/接口结果（重新拉取代价高）
    search_results: list[str] | None
    customer_history: dict | None
    # 生成内容
    draft_response: str | None
    messages: list[str] | None


# --- 节点：读取邮件 ----------------------------------------------------------
def read_email(state: EmailAgentState) -> dict:
    """提取并解析邮件内容（生产环境这里会对接邮件服务）。"""
    return {
        "messages": [f"Processing email: {state['email_content']}"]
    }


# --- 节点：分类意图并路由 ----------------------------------------------------
def classify_intent(
    state: EmailAgentState,
) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """用 LLM 对邮件做结构化分类，再据此路由到不同节点。"""
    # 让模型直接返回符合 EmailClassification 结构的字典
    # method="function_calling"：强制走工具调用协议做结构化输出，兼容性最好
    # （部分 OpenAI 兼容端点不支持默认的 json_schema 模式）
    structured_llm = llm.with_structured_output(
        EmailClassification, method="function_calling"
    )

    # 提示词在节点内即时拼装，而不是存进 state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """
    classification = structured_llm.invoke(classification_prompt)

    # 根据分类结果决定下一步走哪个节点
    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        goto = "human_review"
    elif classification["intent"] in ["question", "feature"]:
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # 返回 Command：一步内同时做两件事
    #   ① update：把分类结果写入共享状态的 classification 字段，
    #      供下游节点（search_documentation / draft_response / human_review）读取，
    #      无需再次调用模型分类。注意这里存的是原始分类 dict，而非格式化后的文本。
    #   ② goto：动态指定下一个要执行的节点（由上面的 if/elif 算出）。
    #      正因为路由写在节点内部，图组装时 classify_intent 后面才无需固定边。
    return Command(update={"classification": classification}, goto=goto)


# --- 节点：检索文档（外部服务，配了重试策略）--------------------------------
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """在知识库中检索相关信息（这里用写死的假数据演示）。"""
    classification = state.get("classification") or {}
    _query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    # 存原始检索结果，而不是格式化后的文本
    search_results = [
        "Reset password via Settings > Security > Change Password",
        "Password must be at least 12 characters",
        "Include uppercase, lowercase, numbers, and symbols",
    ]
    return Command(update={"search_results": search_results}, goto="draft_response")


# --- 节点：Bug 工单 ----------------------------------------------------------
def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """在缺陷跟踪系统里创建/更新工单（这里用假工单号演示）。"""
    ticket_id = "BUG-12345"  # 实际会调 API 创建
    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response",
    )


# --- 节点：起草回复，并按质量/优先级路由 ------------------------------------
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """结合上下文生成回复，并决定是否需要人工审核。"""
    classification = state.get("classification") or {}

    # 从原始 state 数据即时拼装上下文
    context_sections = []
    # 检索结果
    if state.get("search_results"):
        formatted_docs = "\n".join(f"- {doc}" for doc in state["search_results"])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")
    # 客户历史
    if state.get("customer_history"):
        context_sections.append(
            f"Customer tier: {state['customer_history'].get('tier', 'standard')}"
        )

    # 提示词在节点内即时拼装，而不是存进 state
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """
    response = llm.invoke(draft_prompt)

    # 高优先级或复杂问题走人工审核
    needs_review = (
        classification.get("urgency") in ["high", "critical"]
        or classification.get("intent") == "complex"
    )
    # 低优先级问题直接发送
    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": response.content}, goto=goto)


# --- 节点：人工审核（interrupt 必须放在最前，之前的代码恢复时会重跑）--------
def human_review(state: EmailAgentState) -> Command[Literal["send_reply"]]:
    """暂停等待人工审核，并根据人工决定路由。"""
    classification = state.get("classification") or {}

    human_decision = interrupt(
        {
            "email_id": state.get("email_id", ""),
            "original_email": state.get("email_content", ""),
            "draft_response": state.get("draft_response", ""),
            "urgency": classification.get("urgency"),
            "intent": classification.get("intent"),
            "action": "Please review and approve/edit this response",
        }
    )

    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get(
                    "edited_response", state.get("draft_response", "")
                )
            },
            goto="send_reply",
        )
    # 拒绝：人工直接接手，结束
    return Command(update={}, goto=END)


# --- 节点：发送回复 ----------------------------------------------------------
def send_reply(state: EmailAgentState) -> dict:
    """发送邮件回复（这里用打印演示）。"""
    print(f"Sending reply: {state['draft_response'][:100]}...")
    return {}


# --- 组装图 ------------------------------------------------------------------
def build_app():
    workflow = StateGraph(EmailAgentState)

    workflow.add_node("read_email", read_email)
    # 节点：分类（意图识别）
    workflow.add_node("classify_intent", classify_intent)
    # 检索是外部服务，加重试策略应对瞬时故障
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3), # 尝试 3 次
    )
    workflow.add_node("bug_tracking", bug_tracking)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_reply", send_reply)

    # 只需要几条“固定”边，其余路由由节点内 Command(goto=...) 决定
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")
    workflow.add_edge("send_reply", END)

    # 用 interrupt 需要 checkpointer 来保存/恢复状态
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# --- 运行示例：一个需要人工审核的紧急账单问题 -------------------------------
if __name__ == "__main__":
    app = build_app()

    initial_state = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_001",
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": None,
    }

    # thread_id 用于把这次会话的所有状态关联在一起，支持中断后恢复
    config = {"configurable": {"thread_id": "customer_123"}}

    # 第一次运行：会在 human_review 处因 interrupt 暂停
    result = app.invoke(initial_state, config)
    interrupts = result.get("__interrupt__", [])
    print("=== 图已暂停，等待人工审核 ===")
    for it in interrupts:
        print("interrupt payload:", it.value)

    # 模拟人工审核：批准并给出编辑后的回复
    human_response = Command(
        resume={
            "approved": True,
            "edited_response": (
                "We sincerely apologize for the double charge. "
                "I've initiated an immediate refund..."
            ),
        }
    )

    # 恢复执行：从 human_review 继续，走到 send_reply -> END
    final_state = app.invoke(human_response, config)
    print("\n=== Email sent successfully! ===")
    print("最终草稿:", final_state.get("draft_response"))
