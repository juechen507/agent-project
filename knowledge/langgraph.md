# LangGraph

LangGraph 把 Agent 循环画成一张状态图：节点干一件事，边决定下一步去哪。

本项目 `agent_graph.py` 里只有两个节点：
- agent 节点：问大模型一次
- tools 节点：执行模型请求的工具

条件边 `should_continue` 判断最后一条 AI 消息有没有 tool_calls：
有就去 tools，没有就结束并给出最终回答。

checkpointer（本项目用内存版 MemorySaver）按 thread_id 保存会话状态，
相当于手写版自己维护的 `self.messages` 列表。`/reset` 换一个 thread_id 就是失忆重来。
