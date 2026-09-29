# Agent（智能体）

Agent（智能体）= 能自主决定「先做什么、再做什么」的程序。
它以大模型为大脑，通过循环调用工具来完成单轮对话做不到的任务。

普通聊天机器人是一问一答：用户提问 → 模型直接生成文字 → 结束。
Agent 多了「行动权」：模型可以先调用计算器、查天气、检索知识库，观察结果后再决定下一步，直到任务完成。

本项目里三种实现都是同一个 Agent：
- native：手写 OpenAI SDK 循环（`legacy/agent_native.py`）
- graph：LangGraph 手动 StateGraph（`agent_graph.py`）
- preset：`create_react_agent` 预制件（`agent_preset.py`）
