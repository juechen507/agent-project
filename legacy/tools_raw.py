"""
工具模块 —— Agent 的"手和脚"
（LangGraph 重构后，本手写版工具定义已归档到 legacy/，仅供 agent_native.py 使用；
 新版 @tool 定义见根目录 tools.py）

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
import random

import requests

from rag import retrieve_knowledge

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

def get_time() -> str:
    """查询当前的日期和时间。当用户问"现在几点""今天几号"时使用。"""
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S {weekdays[now.weekday()]}")


def get_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为中文城市名，如"北京"、"杭州"。"""
    """
    data = FAKE_WEATHER.get(city)
    if data is None:
        # 把"查不到"也作为信息返回给模型，让它能礼貌地告知用户
        available = "、".join(FAKE_WEATHER)
        return f"暂无 {city} 的天气数据，目前支持的城市有：{available}"
    return f"{city}：{data['weather']}，气温 {data['temp_c']}°C，{data['wind']}"
    """
    url = f"https://wttr.in/${city}?format=j1"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()  # 状态码不是200抛异常
        data = resp.json()

        # 当前天气在 current_condition 数组第0项
        current = data["current_condition"][0]
        weather_info = {
            "城市": city,
            "天气描述": current["weatherDesc"][0]["value"],
            "温度(℃)": current["temp_C"],
            "体感温度(℃)": current["FeelsLikeC"],
            "湿度(%)": current["humidity"],
            "风速(km/h)": current["windspeedKmph"],
            "降水(mm)": current["precipMM"],
            "气压": current["pressure"],
        }
        return weather_info

    except requests.exceptions.RequestException as e:
        print(f"网络请求异常: {e}")
        return None
    except KeyError as e:
        print(f"返回数据缺少字段: {e}")
        return None


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
    """在本地知识库中做 RAG 语义检索。解释 Agent、ReAct、Function Calling、提示词、Token、LangGraph、RAG 等概念时必须使用。参数 topic 请尽量用用户的原话或完整问题。"""
    return retrieve_knowledge(topic)


def get_random_joke() -> str:
    """讲一个随机的笑话。"""
    jokes = [
        "为什么程序员不喜欢在户外工作？因为那里有bug。",
        "程序员和数学家哪个更喜欢在户外？因为那里有bug。",
        "π 是无理数，所以它很长很长，没有重复的数字。",
    ]
    return random.choice(jokes)


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
            "description": "在本地知识库中做 RAG 语义检索。解释 Agent、ReAct、Function Calling、提示词、Token、LangGraph、RAG 等概念时必须使用。参数请尽量用用户原话或完整问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "用户的原话或完整问题，不要只传单个词"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_joke",
            "description": "讲一个随机的笑话",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
    "get_random_joke": get_random_joke,
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
