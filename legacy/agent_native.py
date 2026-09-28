"""
Agent 核心模块 —— 整个项目最重要的一段代码
（LangGraph 重构后，本手写版已归档到 legacy/，作为 `native` 模式保留，
 用于和 agent_graph.py / agent_preset.py 对比学习）

📚 新手知识点：Agent 循环（ReAct Loop）

  普通聊天机器人：  用户提问 → 模型回答 → 结束（一来一回）

  Agent：          用户提问 → 模型思考
                                ↓ 需要工具？
                          调用工具 → 把结果喂回模型 → 继续思考 →（循环）
                                ↓ 不需要了
                              给出最终回答

  这个"思考→行动→观察"的循环就是 Agent 的灵魂。
  本文件不到 100 行，把它读透，你就理解了 90% 的 Agent 框架（LangChain、
  OpenAI Agents SDK 等）内部在做什么 —— 框架只是额外加了记忆、并行、护栏等。

  messages 列表是 Agent 的"短期记忆"：
    - system  消息：角色设定（每次都带上）
    - user    消息：用户说的话
    - assistant 消息：模型的回复（可能带 tool_calls 调用请求）
    - tool    消息：工具执行的结果
  所谓"对话记忆"，无非就是把这个列表一直往后追加、整体重发给模型。
"""

from openai import OpenAI

from config import get_config
from legacy.tools_raw import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """你是一名 AI 学习助教，正在教一位编程新手理解 Agent。
规则：
1. 需要精确计算、查询天气/时间/概念资料时，必须使用工具，不要凭记忆瞎编。
2. 回答使用简体中文，风格友好、简洁，适合初学者。
3. 如果用户的问题超出了你的工具能力，坦白说明，并告诉他本项目可以怎么扩展工具。"""

# 安全阀：限制单轮对话中工具调用的最大次数。
# 没有这道保险，模型可能陷入"调工具→不满意→再调"的死循环，无限烧钱。
# 所有生产级 Agent 都有类似的 max iterations / recursion limit 设计。
MAX_TOOL_ROUNDS = 6


class Agent:
    def __init__(self, verbose: bool = True):
        cfg = get_config()
        # OpenAI SDK 是各家兼容端点的"事实标准"，
        # 换个 base_url 就能无缝使用 DeepSeek / Qwen 等国内模型。
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model = cfg["model"]
        self.verbose = verbose  # 是否打印思考过程（新手建议开着，直观）

        # 对话历史，Agent 的"记忆"。第一条永远是系统提示词。
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _say(self, text: str) -> None:
        """打印内部执行过程，方便新手观察 Agent 每一步在干什么。"""
        if self.verbose:
            print(f"  \033[90m⚙ {text}\033[0m")  # 灰色，和正式回答区分开

    def chat(self, user_input: str) -> str:
        """处理一条用户消息，内部可能经历多轮"工具调用循环"，返回最终回答。"""
        self.messages.append({"role": "user", "content": user_input})

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,  # 把工具"说明书"交给模型
                temperature=0.3,     # 低温度 = 回答更稳定，适合工具调用场景
            )
            assistant_msg = response.choices[0].message

            # 把模型的回复（可能是"我要调用工具"）追加进对话历史。
            # 注意：这一步不能省！下一条 tool 结果必须通过 tool_call_id
            # 与这条回复里的调用请求对应起来，模型才知道结果对应哪个调用。
            # model_dump() 把 SDK 对象转成普通 dict，并去掉值为 None 的字段。
            self.messages.append(assistant_msg.model_dump(exclude_none=True))

            # 情形 A：模型没有请求工具 —— 说明它认为可以直接回答了，循环结束
            if not assistant_msg.tool_calls:
                return assistant_msg.content or "(模型返回了空内容)"

            # 情形 B：模型请求调用工具 —— 逐个执行，把结果喂回去，进入下一轮
            # （模型也可能一次请求多个工具，即"并行工具调用"，这里同样支持）
            for call in assistant_msg.tool_calls:
                self._say(f"第 {round_no} 轮：调用工具 {call.function.name}"
                          f"，参数 {call.function.arguments or '{}'}")
                result = execute_tool(call.function.name, call.function.arguments)
                self._say(f"工具结果：{result}")

                # 按照 OpenAI 协议，工具结果要以 role="tool" 的消息回传，
                # 并带上原始的 call_id 做配对
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

        # 超过最大轮数仍未得到答案
        self.messages.append({"role": "user", "content": "（工具调用次数已达上限，请直接根据已有信息回答）"})
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages,
        )
        return response.choices[0].message.content or "（达到调用上限，且没有可用的回答）"
