"""
LangGraph 手动版 Agent —— 用 StateGraph 重写出 legacy/agent_native.py 的循环

📚 新手知识点：手写循环 ↔ LangGraph 图的对应关系

  先把 legacy/agent_native.py 的 chat() 在脑子里拆成两件事：
    A. 问模型（chat.completions.create）
    B. 如果模型要调工具，就执行工具、把结果喂回去，再回到 A

  LangGraph 的思路：把这个"循环"显式地画成一张图 ——

       ┌─────────┐  有 tool_calls   ┌─────────┐
       │ agent节点 │ ───────────────→ │ tools节点 │
       └─────────┘                  └─────────┘
            ↑                           │
            │       执行结果（回到 agent） │
            └───────────────────────────┘
            │ 没有 tool_calls → END（给出最终回答）

  对照表（本文件 ↔ legacy/agent_native.py）：
    · agent 节点            ↔  for 循环体里的 client.chat.completions.create(...)
    · tools 节点(ToolNode)  ↔  for call in assistant_msg.tool_calls: execute_tool(...)
    · should_continue 条件边 ↔  "情形 A / 情形 B" 的 if not tool_calls 分支
    · recursion_limit       ↔  MAX_TOOL_ROUNDS 安全阀
    · AgentState.messages   ↔  self.messages 对话历史（短期记忆）
    · add_messages 合并器    ↔  列表一直往后 append 的简化声明版

  框架没有魔法：它只是帮你把"while 循环 + if 分支"换成了"图 + 条件边"，
  换来的是状态管理、节点复用、可视化和后续加并行/人工审核等能力的扩展点。
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_config
from tools import TOOLS

SYSTEM_PROMPT = """你是一名 AI 学习助教，正在教一位编程新手理解 Agent。
规则：
1. 需要精确计算、查询天气/时间/概念资料时，必须使用工具，不要凭记忆瞎编。
2. 解释 Agent、ReAct、Function Calling、提示词、Token、LangGraph、RAG 等概念时，必须先调用 lookup_knowledge，只根据检索到的资料回答。
3. 回答使用简体中文，风格友好、简洁，适合初学者。
4. 如果用户的问题超出了你的工具能力，坦白说明，并告诉他本项目可以怎么扩展工具。"""

# 手写版里有 MAX_TOOL_ROUNDS = 6 这道保险；LangGraph 的对应物是 recursion_limit：
# 图每经过一个节点算一步，agent→tools→agent→tools… 超过上限会抛
# GraphRecursionError，防止模型陷入死循环无限烧钱。
RECURSION_LIMIT = 25


class AgentState(TypedDict):
    """图的"共享状态"：所有节点读它、也写它。

    📚 对照手写版：这就是 self.messages 对话历史。
    Annotated[..., add_messages] 声明了合并规则 —— 节点返回的新消息
    不是覆盖整个列表，而是通过 add_messages 追加（并按 id 去重），
    相当于手写版里一条条 messages.append(...) 的"声明式"写法。
    """

    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    """对外接口与 legacy/agent_native.py 的 Agent 类保持一致，
    main.py 里 native / graph / preset 三种模式共用同一套调用方式。"""

    def __init__(self, verbose: bool = True):
        cfg = get_config()
        # ChatOpenAI 同样支持 OpenAI 兼容端点：DeepSeek / Qwen 换个 base_url 即可，
        # 和手写版里 OpenAI SDK 的 api_key/base_url 两个参数一一对应。
        self.llm = ChatOpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=0.3,  # 低温度 = 回答更稳定，适合工具调用场景（同手写版）
        ).bind_tools(TOOLS)  # 把工具"说明书"交给模型（同手写版的 tools=TOOL_SCHEMAS）
        self.model = cfg["model"]  # 给 main.py 展示用（同手写版）
        self.verbose = verbose

        # ------------------------------------------------------------------
        # 一、定义节点：agent 节点 = "问模型一次"
        #
        #   完全对应手写版的这一句：
        #     response = self.client.chat.completions.create(
        #         model=self.model, messages=self.messages, tools=TOOL_SCHEMAS)
        #     self.messages.append(response...)
        #   在 LangGraph 里，节点函数返回 {"messages": [新消息]}，
        #   add_messages 会自动把它追加进状态。
        # ------------------------------------------------------------------
        def agent_node(state: AgentState) -> dict:
            reply = self.llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
            return {"messages": [reply]}

        # ------------------------------------------------------------------
        # 二、定义条件边：走"执行工具"还是"结束"
        #
        #   对应手写版的分支判断：
        #     if not assistant_msg.tool_calls:  → 情形 A，return 最终回答
        #     else:                            → 情形 B，逐个执行工具
        # ------------------------------------------------------------------
        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"  # 情形 B：去 tools 节点
            return END  # 情形 A：模型认为可以直接回答了，循环结束

        # ------------------------------------------------------------------
        # 三、组装图
        #
        #   ToolNode(tools) 对应手写版的 for call in tool_calls: execute_tool(...)
        #   一整段：解析参数 JSON、查注册表、执行函数、异常兜底、
        #   把结果打包成 role="tool"（ToolMessage）的消息带 tool_call_id 回传 ——
        #   这些"脏活"它全替你做了，且严格遵循 OpenAI 的配对协议。
        # ------------------------------------------------------------------
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(TOOLS))
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")  # 工具执行完，回到 agent 再问模型一次

        # checkpointer：按 thread_id 保存每个会话的状态，
        # 让多轮对话的"记忆"由框架托管（对照手写版：自己维护 self.messages 列表）。
        # 学习项目用内存版即可；生产环境会换成 SQLite/Postgres 等持久化后端。
        self.graph = builder.compile(checkpointer=MemorySaver())

        # thread_id 相当于"会话号"：同一个 id 共享一份对话记忆。
        # /reset 时换一个 id，就等于"失忆重来"。
        self.thread_id = "session-1"
        self._run_config = self._make_config()

    def _make_config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": RECURSION_LIMIT}

    def reset_memory(self) -> None:
        """对应 main.py 的 /reset：手写版是 self.messages = self.messages[:1]，
        这里换个 thread_id，相当于开一个全新的空白会话。"""
        self.thread_id = f"session-{id(object())}"
        self._run_config = self._make_config()

    def chat(self, user_input: str) -> str:
        """处理一条用户消息，内部可能经历多轮"agent↔tools 绕圈"，返回最终回答。

        用 stream 而不是 invoke，是为了在 verbose 模式下逐个节点打印过程，
        复刻手写版 _say() 的教学体验（对功能没有任何影响）。
        """
        final_state = {"messages": []}
        for chunk in self.graph.stream({"messages": [("user", user_input)]}, self._run_config):
            for node_name, update in chunk.items():
                final_state.update(update)
                if not self.verbose:
                    continue
                # 每经过一个节点打印一行灰色过程信息，和 legacy 版风格一致
                if node_name == "agent":
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for call in msg.tool_calls:
                                args = json.dumps(call["args"], ensure_ascii=False) if call["args"] else "{}"
                                self._say(f"agent 节点：请求调用工具 {call['name']}，参数 {args}")
                elif node_name == "tools":
                    for msg in update.get("messages", []):
                        self._say(f"tools 节点结果：{msg.content}")
        last = final_state["messages"][-1]
        return last.content or "(模型返回了空内容)"

    def _say(self, text: str) -> None:
        """打印内部执行过程，方便新手观察图每个节点在干什么。"""
        if self.verbose:
            print(f"  \033[90m⚙ {text}\033[0m")  # 灰色，和正式回答区分开
