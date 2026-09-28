"""
工具模块 —— Agent 的"手和脚"

📚 新手知识点（本节是理解 Agent 的关键）：

  大模型本身只会"生成文字"，它不会算数（容易算错）、不能联网、不知道现在几点。
  所谓"工具调用（Function Calling / Tool Use）"，就是：

    1. 我们用 JSON Schema 告诉模型："你有这几个工具可以用"；
    2. 模型遇到需要时，不直接回答，而是输出一条结构化的"调用请求"：
       "请帮我调用 calculator，参数是 {expression: '3.14 * 2 * 2'}"；
    3. 我们的代码真正执行这个函数，把结果喂回给模型；
    4. 模型拿到结果后，组织成自然语言回答用户。

  所以每个工具 = 一个普通 Python 函数 + 一份描述它的 JSON Schema。
  函数的 docstring 会被模型看到，写得越清楚，模型用得越准。
"""

import datetime
import json

# ------------------------------------------------------------------
# 一、工具的"实现"：就是普普通通的 Python 函数
# ------------------------------------------------------------------

# 模拟天气数据。真实项目里这里会调用气象 API（如 wttr.in、和风天气）。
# 用假数据的好处：新手跑示例不需要额外的 API Key，结果可预测、好调试。
FAKE_WEATHER = {
    "北京": {"weather": "晴", "temp_c": 22, "wind": "西北风 3 级"},
    "上海": {"weather": "多云", "temp_c": 26, "wind": "东南风 2 级"},
    "杭州": {"weather": "小雨", "temp_c": 24, "wind": "微风"},
    "深圳": {"weather": "雷阵雨", "temp_c": 30, "wind": "南风 4 级"},
}

# 迷你知识库：Agent 项目里最常见的 RAG（检索增强生成）的极简形态。
# 真实 RAG 会用向量相似度检索，这里用关键词匹配演示核心思想：
# "让模型先查资料，再基于资料回答"，减少胡编乱造。
KNOWLEDGE_BASE = {
    "什么是agent": (
        "Agent（智能体）= 能自主决定'先做什么、再做什么'的程序。"
        "它以大模型为大脑，通过循环调用工具来完成单轮对话做不到的任务。"
    ),
    "什么是react": (
        "ReAct 是 Reasoning + Acting 的缩写，是 Agent 最经典的运行模式："
        "模型先思考(Reasoning)，再行动(Acting，即调用工具)，观察结果后再思考，循环往复直到任务完成。"
    ),
    "什么是function calling": (
        "Function Calling（工具调用）是让大模型以 JSON 格式表达'我想调用哪个函数、传什么参数'的机制，"
        "真正的执行由我们的代码完成，模型本身不运行任何代码。"
    ),
    "什么是提示词": (
        "提示词（Prompt）是发给大模型的输入文本。通常分为系统提示词（设定角色和规则）"
        "和用户提示词（本次的具体问题）。"
    ),
    "什么是token": (
        "Token 是大模型处理文本的最小单位，约等于一个词或半个汉字。"
        "API 按 token 计费，模型有最大上下文长度限制（如 128K token）。"
    ),
}


def get_time() -> str:
    """查询当前的日期和时间。当用户问"现在几点""今天几号"时使用。"""
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S {weekdays[now.weekday()]}")


def get_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为中文城市名，如"北京"、"杭州"。"""
    data = FAKE_WEATHER.get(city)
    if data is None:
        # 把"查不到"也作为信息返回给模型，让它能礼貌地告知用户
        available = "、".join(FAKE_WEATHER)
        return f"暂无 {city} 的天气数据，目前支持的城市有：{available}"
    return f"{city}：{data['weather']}，气温 {data['temp_c']}°C，{data['wind']}"


def calculate(expression: str) -> str:
    """计算数学表达式，参数 expression 是纯数学表达式字符串，如 "(3+5)*2"、"2**10"。支持 + - * / ** 和括号。"""
    # 安全提示：eval 能执行任意代码，生产环境绝不能直接 eval 用户输入。
    # 这里演示一个新手可理解的防护：只允许数字、运算符和白名单函数名。
    allowed_chars = set("0123456789+-*/(). %")
    if not set(expression) <= allowed_chars:
        return "表达式包含不允许的字符，只支持四则运算和取余。"
    try:
        # __builtins__=None 禁用所有内置函数，expression 参数限定只能是个表达式
        result = eval(expression, {"__builtins__": None}, {})
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算出错：{exc}"


def lookup_knowledge(topic: str) -> str:
    """在内置知识库中查询 AI/Agent 相关概念的解释。参数 topic 是要查询的关键词，如"Agent"、"ReAct"、"token"。"""
    # 注意匹配方向：知识库的 key（如"什么是react"）比用户传的 topic（如"react"）更长，
    # 所以要用 "key 包含 topic" 来判断，而不是反过来。这是一个很好的新手 bug 教学案例。
    topic_normalized = topic.lower().replace("什么是", "").strip()
    for key, value in KNOWLEDGE_BASE.items():
        if topic_normalized and (topic_normalized in key or key in topic_normalized):
            return value
    return f"知识库中没有找到关于「{topic}」的条目。"


# ------------------------------------------------------------------
# 二、工具的"说明书"：OpenAI 格式的 JSON Schema
#
#   模型就是照着这份说明书决定"用哪个工具、传什么参数"的。
#   description 写得越具体，模型调用得越准确 —— 这是最值得体会的一点。
# ------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询当前的日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "中文城市名，例如 北京、杭州"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "精确计算数学表达式，凡是涉及计算的问题都应使用此工具而不是心算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，例如 (3+5)*2"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_knowledge",
            "description": "查询 AI / Agent 基础概念的解释（什么是 Agent、ReAct、Function Calling、提示词、token）",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "要查询的概念关键词"},
                },
                "required": ["topic"],
            },
        },
    },
]

# 工具注册表：函数名字 -> 函数本体。
# 模型返回调用请求时，我们根据它给的 name 在这里查找真正要执行的函数。
TOOL_REGISTRY = {
    "get_time": get_time,
    "get_weather": get_weather,
    "calculate": calculate,
    "lookup_knowledge": lookup_knowledge,
}


def execute_tool(name: str, arguments: str) -> str:
    """
    根据模型给出的工具名和 JSON 参数，执行对应函数并返回结果字符串。

    注意：工具返回值必须是字符串 —— 因为最终要塞回 messages 里喂给模型，
    而模型只能"读文本"。复杂结果（如字典）通常 json.dumps 一下再返回。
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"错误：不存在名为 {name} 的工具"

    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return "错误：工具参数不是合法的 JSON"

    try:
        return str(func(**args))
    except Exception as exc:
        # 把异常也变成文本喂回给模型，模型会自己纠正用法或向用户道歉，
        # 而不是让整个程序崩掉 —— 这是 Agent 健壮性的常见做法。
        return f"工具执行出错：{exc}"
