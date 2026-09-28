# 🤖 Mini Agent —— 300 行代码看懂 AI Agent 的原理

一个**专为编程新手打造**的最小可运行 AI Agent 示例。不使用任何 Agent 框架，
只用 OpenAI 官方 SDK + 4 个手写工具，把 LangChain 等框架帮你"藏起来"的东西
全部摊开给你看。

> 学完这个项目，你能回答这三个面试常考题：
> 1. Agent 和普通聊天机器人有什么区别？—— `agent.py` 里的那个 `for` 循环
> 2. 什么是 Function Calling？—— `tools.py` 里的 JSON Schema + 注册表
> 3. Agent 的"记忆"是怎么回事？—— `messages` 列表的追加与重发

## 一、项目结构（按阅读顺序）

| 文件 | 行数 | 职责 | 新手建议 |
|---|---|---|---|
| `config.py` | ~80 | 读取 `.env` 里的 API Key | 了解 dotenv 原理 |
| `tools.py` | ~190 | 4 个工具函数 + JSON Schema 说明书 | **重点**：工具 = 函数 + 描述 |
| `agent.py` | ~100 | Agent 核心循环（ReAct） | ⭐ **全项目灵魂，必读** |
| `main.py` | ~70 | 命令行对话入口 | 看分层设计即可 |

```
用户提问 ──► agent.py 的 chat()
                 │
                 ▼
        把「问题 + 工具说明书」发给大模型
                 │
     ┌── 模型说要调工具？──┐
     │                     ▼
     │            tools.py 执行函数（计算器/天气/时间/知识库）
     │                     │ 结果塞回 messages
     │                     ▼
     │              重新发给模型 ◄───（循环，最多 6 轮）
     └── 模型给出最终回答 ──► 打印给用户
```

## 二、快速开始（3 分钟）

### 第 1 步：准备 API Key

任选一家（都兼容 OpenAI 接口格式）：

| 提供商 | 申请地址 | 说明 |
|---|---|---|
| DeepSeek | https://platform.deepseek.com | 推荐：国内直连、便宜 |
| 通义千问 | https://bailian.console.aliyun.com | 新用户有免费额度 |
| OpenAI | https://platform.openai.com | 需要网络条件 |

### 第 2 步：安装依赖并配置

```bash
cd agent-project
python3 -m venv .venv && source .venv/bin/activate   # 创建并激活虚拟环境
pip install -r requirements.txt                      # 只有 openai 一个依赖
cp .env.example .env                                 # 然后编辑 .env 填入你的 Key
```

### 第 3 步：开跑！

```bash
python3 main.py
```

试着输入：

```
你> 帮我算一下 (128 + 256) * 3，再看看现在几点
  ⚙ 第 1 轮：调用工具 calculate，参数 {"expression": "(128 + 256) * 3"}
  ⚙ 工具结果：(128 + 256) * 3 = 1152
  ⚙ 第 1 轮：调用工具 get_time，参数 {}
  ⚙ 工具结果：2026-09-27 10:30:00 星期日
Agent> 计算结果是 1152，现在是周日上午 10:30 ……
```

灰色的 `⚙` 行就是 Agent 的"思考过程"——你会亲眼看到模型**自己决定**
该用哪个工具、传什么参数，这正是 Agent 的核心魅力。

内置斜杠命令：`/reset` 清空记忆、`/verbose` 开关思考过程、`/help` 帮助、`/quit` 退出。

## 三、核心概念速查

- **Agent（智能体）**：以大模型为决策中枢，能自主规划"调用哪些工具、按什么顺序"
  的程序。区别于聊天机器人的一点：**它有权决定下一步做什么**。
- **Function Calling（工具调用）**：模型以结构化 JSON 表达"我要调用 X 函数、参数是 Y"，
  真正的执行发生在你的 Python 代码里。模型从不运行代码，别被名字吓到。
- **ReAct 循环**：Reason（思考）→ Act（调工具）→ Observe（看结果）→ 再思考，
  直到模型认为可以直接回答。对应 `agent.py` 中的 `for round_no in ...` 循环。
- **messages 列表 = 记忆**：`system / user / assistant / tool` 四种角色轮流追加，
  每次请求把整个列表重发给模型（模型本身无状态，"记住"全靠重放历史）。
- **安全阀**：`MAX_TOOL_ROUNDS = 6` 防止模型无限调工具烧钱——生产级 Agent 必备。

## 四、动手改造（学习最好的方式是改代码）

按难度递增，每个练习都只动一两个文件：

1. ⭐ **加一个新工具**：写一个 `get_random_joke()` 讲笑话函数，
   在 `TOOL_SCHEMAS` 注册、加入 `TOOL_REGISTRY`，问 Agent"讲个笑话"验证。
   （做完你会彻底理解"工具 = 函数 + 说明书"） -- done
2. ⭐ **改人设**：修改 `agent.py` 里的 `SYSTEM_PROMPT`，把它变成一个严格的高数老师，
   观察回答风格的变化。（体会系统提示词的作用）
3. ⭐⭐ **换模型**：在 `.env` 里设置 `LLM_MODEL`，对比不同模型的工具调用准确率。
4. ⭐⭐ **真实天气**：把 `tools.py` 的 `FAKE_WEATHER` 换成调用免费接口
   `https://wttr.in/城市?format=j1`（用 `requests` 库），不加 Key 也能用。 -- done
5. ⭐⭐⭐ **升级成 Web 界面**：用 20 行 FastAPI 或 Gradio 替换 `main.py` 的
   while 循环，`Agent` 类一行不用改。（体会分层设计的价值）

## 五、下一步学什么

理解了本项目的裸实现后，再去学框架会事半功倍：

- **OpenAI Agents SDK** / **LangGraph**：工业界的 Agent 循环编排
- **MCP（Model Context Protocol）**：标准化的工具接入协议，可把本项目的工具做成 MCP Server
- **RAG**：把 `lookup_knowledge` 的关键词匹配换成向量检索（推荐试 `chromadb`）
- **多 Agent 协作**：让若干个本项目的 Agent 各司其职互相调用

## 常见问题

- **报 `❌ 未找到任何 API Key`**：没建 `.env`，或 `.env` 里 Key 还是占位符。
- **报 401 / 认证错误**：Key 复制不完整，注意不要带空格或换行。
- **用了虚拟环境但 `pip install` 后仍提示 `ModuleNotFoundError: openai`**：
  确认终端里已 `source .venv/bin/activate`（提示符前应显示 `(.venv)`）。
- **模型不调用工具直接瞎编答案**：正常现象，小模型偶尔会这样；
  可尝试更强模型，或把工具 `description` 写得更明确（这是最常见的调优手段）。
