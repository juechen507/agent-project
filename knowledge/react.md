# ReAct

ReAct 是 Reasoning + Acting 的缩写，是 Agent 最经典的运行模式：
模型先思考（Reasoning），再行动（Acting，即调用工具），观察结果后再思考，循环往返直到任务完成。

对照本项目：
- 思考：把对话历史和工具说明书发给大模型
- 行动：模型返回 tool_calls，由 Python 真正执行函数
- 观察：把工具返回的字符串写回 messages，再问模型一次

手写版对应 `for round_no in range(...)` 循环；LangGraph 对应 agent 节点与 tools 节点之间的绕圈。
