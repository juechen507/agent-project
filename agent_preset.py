"""
LangGraph 预制版 Agent —— create_react_agent 一行搞定

📚 新手知识点：预制件 = 手动版的全部逻辑打包

  读完 agent_graph.py 再看这里，你会恍然大悟：
  create_react_agent(model, tools) 内部构建的正是那张
  agent ↔ tools 的循环图，连 ToolNode、条件判断都一模一样。

  三个实现的演进关系：
    legacy/agent_native.py  手写 while 循环   —— 理解原理
    agent_graph.py          手动 StateGraph   —— 理解框架在做什么
    本文件                  框架预制件        —— 工程上真正干活的样子

  什么时候该用哪个？
    - 学习/面试：能徒手写出 native 版循环；
    - 需要自定义流程（人工审核、并行分支、多 Agent 协作）：写 agent_graph.py 这类手动图；
    - 就是标准"模型+工具"问答：直接用预制件，别重复造轮子。
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import get_config
from tools import TOOLS

SYSTEM_PROMPT = """你是一名 AI 学习助教，正在教一位编程新手理解 Agent。
规则：
1. 需要精确计算、查询天气/时间/概念资料时，必须使用工具，不要凭记忆瞎编。
2. 回答使用简体中文，风格友好、简洁，适合初学者。
3. 如果用户的问题超出了你的工具能力，坦白说明，并告诉他本项目可以怎么扩展工具。"""

RECURSION_LIMIT = 25  # 同 agent_graph.py：预制件内部也是这套图，安全阀同样生效


class Agent:
    """对外接口与另外两种模式保持一致（chat / reset_memory / verbose）。"""

    def __init__(self, verbose: bool = True):
        cfg = get_config()
        # 注意：这里传给 create_react_agent 的是"没 bind_tools 的原始模型"，
        # 绑定工具这一步预制件内部自己完成 —— 它把整个 ReAct 模式打包了。
        llm = ChatOpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=0.3,
        )
        self.graph = create_react_agent(
            llm,
            TOOLS,
            prompt=SYSTEM_PROMPT,  # 系统提示词，对应手写版 messages 列表的第一条
            checkpointer=MemorySaver(),  # 同样靠 thread_id 托管对话记忆
        )
        self.model = cfg["model"]  # 给 main.py 展示用（同手写版）
        self.verbose = verbose
        self.thread_id = "session-1"

    def reset_memory(self) -> None:
        self.thread_id = f"session-{id(object())}"

    def chat(self, user_input: str) -> str:
        config: dict[str, Any] = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        # 预制件封装了循环细节，这里干脆用 invoke 一次性拿最终状态；
        # 想看中间过程请对比 agent_graph.py 的 stream 写法。
        state = self.graph.invoke({"messages": [("user", user_input)]}, config)
        last = state["messages"][-1]
        if self.verbose and isinstance(last, AIMessage):
            for call in last.tool_calls:
                print(f"  \033[90m⚙ （预制件内部调用了 {call['name']}，过程被框架封装，看不到逐节点细节）\033[0m")
        return last.content or "(模型返回了空内容)"
