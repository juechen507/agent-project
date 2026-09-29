"""
工具模块（LangGraph 版）—— Agent 的"手和脚"

📚 新手知识点：从"手写 JSON Schema"到 @tool 装饰器

  legacy/tools_raw.py 里，每个工具 = 一个普通函数 + 一份手写 JSON Schema，
  还要自己维护 TOOL_REGISTRY 注册表和 execute_tool 分发函数。

  LangChain 提供的 @tool 装饰器把这些全部自动化：
    1. 函数的 docstring  →  自动变成 Schema 里的 description（模型看的说明书）；
    2. 函数签名 + 类型注解 →  自动推导出 parameters 的 JSON Schema；
    3. 工具调用分发、参数解析、异常兜底 → 由 LangGraph 的 ToolNode 统一处理。

  所以这里只剩"纯函数 + 一句 @tool"。对照 legacy/tools_raw.py 阅读，
  能直观感受到框架到底帮你省掉了哪些样板代码。

  不变的原则（两边通用）：
    - docstring 写得越清楚，模型用得越准；
    - 工具返回值必须是字符串 —— 模型只能"读文本"，
      复杂结果（如字典）要 json.dumps 之后再返回。
"""

import datetime
import json
import random

import requests
from langchain_core.tools import tool

from rag import retrieve_knowledge

# ------------------------------------------------------------------
# 工具的"实现"：普通 Python 函数 + @tool 装饰器
# ------------------------------------------------------------------


@tool
def get_time() -> str:
    """查询当前的日期和时间。当用户问"现在几点""今天几号"时使用。"""
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S {weekdays[now.weekday()]}")


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为中文城市名，如"北京"、"杭州"。数据来源 wttr.in，无需额外 API Key。"""
    # 注意：原手写版这里误写成 f"https://wttr.in/${city}"，多了个 $ 符号，
    # 且查不到时返回 None（模型只能读文本，None 会破坏协议）。
    # 重构时一并修复：URL 用干净的 f-string，所有分支都返回字符串。
    url = f"https://wttr.in/{city}?format=j1"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()  # 状态码不是 200 抛异常
        data = resp.json()

        # 当前天气在 current_condition 数组第 0 项
        current = data["current_condition"][0]
        weather_info = {
            "城市": city,
            "天气描述": current["weatherDesc"][0]["value"],
            "温度(℃)": current["temp_C"],
            "体感温度(℃)": current["FeelsLikeC"],
            "湿度(%)": current["humidity"],
            "风速(km/h)": current["windspeedKmph"],
            "降水(mm)": current["precipMM"],
        }
        # 字典必须序列化成字符串再喂回模型（ensure_ascii=False 保留中文）
        return json.dumps(weather_info, ensure_ascii=False)
    except requests.exceptions.RequestException as exc:
        # 把失败原因也作为文本返回，让模型能礼貌地告知用户，而不是程序崩掉
        return f"查询 {city} 的天气失败：{exc}"
    except (KeyError, IndexError, ValueError) as exc:
        return f"wttr.in 返回数据格式异常，缺少字段：{exc}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，参数 expression 是纯数学表达式字符串，如 "(3+5)*2"。支持 + - * / 和括号。凡是涉及计算的问题都应使用此工具而不是心算。"""
    # 安全提示：eval 能执行任意代码，生产环境绝不能直接 eval 用户输入。
    # 这里演示一个新手可理解的防护：只允许数字、运算符和白名单字符。
    allowed_chars = set("0123456789+-*/(). %")
    if not set(expression) <= allowed_chars:
        return "表达式包含不允许的字符，只支持四则运算和取余。"
    try:
        # __builtins__=None 禁用所有内置函数，expression 参数限定只能是个表达式
        result = eval(expression, {"__builtins__": None}, {})
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算出错：{exc}"


@tool
def lookup_knowledge(topic: str) -> str:
    """在本地知识库中做 RAG 语义检索。解释 Agent、ReAct、Function Calling、提示词、Token、LangGraph、RAG 等概念时必须使用。参数 topic 请尽量用用户的原话或完整问题，不要只传单个词。"""
    # 检索实现见 rag.py：切块 → TF-IDF 向量 → 余弦相似度 Top-K。
    return retrieve_knowledge(topic)


@tool
def get_random_joke() -> str:
    """讲一个随机的笑话。当用户想放松一下、或者明确要求讲笑话时使用。"""
    jokes = [
        "为什么程序员不喜欢在户外工作？因为那里有bug。",
        "程序员和数学家哪个更喜欢在户外？因为那里有bug。",
        "π 是无理数，所以它很长很长，没有重复的数字。",
    ]
    return random.choice(jokes)


# 工具列表：bind_tools() 和 ToolNode 都吃这个数组，
# 相当于 legacy 版里 TOOL_SCHEMAS + TOOL_REGISTRY 的合体。
TOOLS = [get_time, get_weather, calculate, lookup_knowledge, get_random_joke]
