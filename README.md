# 🤖 Mini Agent —— 看懂 AI Agent 的原理：手写循环 → LangGraph → 预制件

一个**专为编程新手打造**的最小可运行 AI Agent 示例。
同一套工具、同一个入口，给你三份实现：先用 OpenAI SDK 裸写一遍把框架
帮你"藏起来"的东西全部摊开，再用 LangGraph 重写一遍看框架到底做了什么，
最后用预制件看工程上真正干活的样子。

> 学完这个项目，你能回答这三个面试常考题：
> 1. Agent 和普通聊天机器人有什么区别？—— `legacy/agent_native.py` 里的那个 `for` 循环
> 2. 什么是 Function Calling？—— `legacy/tools_raw.py` 的 JSON Schema + 注册表，以及 `tools.py` 里 `@tool` 的自动化
> 3. Agent 的"记忆"是怎么回事？—— `messages` 列表的追加与重发

> 🆕 LangGraph 重构后，项目同时保留三种实现，用 `/mode` 随时热切换对比：
> `native`（手写循环）→ `graph`（LangGraph 手动 StateGraph，默认）→ `preset`（框架预制件）。
> 推荐按"手写 → 图 → 预制件"的顺序读，体会框架到底帮你封装了什么。

## 一、项目结构（按阅读顺序）

| 文件 | 职责 | 新手建议 |
|---|---|---|
| `config.py` | 读取 `.env` 里的 API Key（三种模式共用） | 了解 dotenv 原理 |
| `legacy/agent_native.py` | ⭐ 手写 OpenAI SDK 的 Agent 核心循环（ReAct） | **先读它**：全项目灵魂 |
| `legacy/tools_raw.py` | 手写版工具：函数 + JSON Schema 说明书 + 注册表 | 重点：工具 = 函数 + 描述 |
| `tools.py` | LangChain `@tool` 版工具（graph/preset 模式共用） | 对照 legacy 版看框架省了什么 |
| `agent_graph.py` | LangGraph 手动 StateGraph（agent/tools 节点 + 条件边） | ⭐ **核心教学文件**：逐行对照手写循环 |
| `agent_preset.py` | `create_react_agent` 预制版，约 60 行 | 感受"打包后的终态" |
| `main.py` | 命令行入口，`/mode` 切换三种实现 | 看分层设计即可 |

### 手写循环 ↔ LangGraph 对照表（读 `agent_graph.py` 的最佳地图）

| `legacy/agent_native.py` 里的代码 | LangGraph 对应概念 | 在 `agent_graph.py` 的位置 |
|---|---|---|
| `for` 循环体中的 `client.chat.completions.create(...)` | `agent` 节点 | `agent_node()` |
| `if not assistant_msg.tool_calls: return`（情形 A/B 分支） | 条件边 | `should_continue()` |
| `for call in tool_calls: execute_tool(...)` 整段 | `tools` 节点 | `ToolNode(TOOLS)`（框架预制） |
| `self.messages` 对话历史 + 逐条 `append` | 图状态 `AgentState` + `add_messages` 合并器 | `class AgentState` |
| `MAX_TOOL_ROUNDS = 6` 安全阀 | `recursion_limit` | `_make_config()` |
| （无 —— 手写版自己维护列表） | checkpointer：按 `thread_id` 托管会话记忆 | `MemorySaver()` |
| （手写 JSON Schema + 注册表 + 分发） | `@tool` 装饰器自动生成说明书 | `tools.py` |

```
用户提问 ──► agent_graph.py 的 chat()（native/preset 模式结构相同）
                 │
                 ▼
        agent 节点：把「问题 + 工具说明书」发给大模型
                 │
     ┌── 模型说要调工具？（条件边 should_continue）──┐
     │                     ▼
     │            tools 节点（ToolNode）执行函数（计算器/天气/时间/知识库）
     │                     │ 结果作为 ToolMessage 进状态
     │                     ▼
     │              回到 agent 节点 ◄───（图绕圈，受 recursion_limit 保护）
     └── 模型给出最终回答 → END ──► 打印给用户
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
pip install -r requirements.txt                      # openai + langchain-openai + langgraph
cp .env.example .env                                 # 然后编辑 .env 填入你的 Key
```

### 第 3 步：开跑！

```bash
python3 main.py
```

试着输入：

```
你> 帮我算一下 (128 + 256) * 3，再看看现在几点
  ⚙ agent 节点：请求调用工具 calculate，参数 {"expression": "(128 + 256) * 3"}
  ⚙ tools 节点结果：(128 + 256) * 3 = 1152
  ⚙ agent 节点：请求调用工具 get_time，参数 {}
  ⚙ tools 节点结果：2026-09-28 10:30:00 星期日
Agent> 计算结果是 1152，现在是周日上午 10:30 ……
```

灰色的 `⚙` 行就是 Agent 的"思考过程"——你会亲眼看到模型**自己决定**
该用哪个工具、传什么参数，这正是 Agent 的核心魅力。
（`native` 模式打印的是"第 N 轮调用工具…"，`preset` 模式过程被框架封装、打印最少。）

内置斜杠命令：`/mode native|graph|preset` 切换实现、`/reset` 清空记忆、`/verbose` 开关思考过程、`/help` 帮助、`/quit` 退出。

## 三、核心概念速查

- **Agent（智能体）**：以大模型为决策中枢，能自主规划"调用哪些工具、按什么顺序"
  的程序。区别于聊天机器人的一点：**它有权决定下一步做什么**。
- **Function Calling（工具调用）**：模型以结构化 JSON 表达"我要调用 X 函数、参数是 Y"，
  真正的执行发生在你的 Python 代码里。模型从不运行代码，别被名字吓到。
- **ReAct 循环**：Reason（思考）→ Act（调工具）→ Observe（看结果）→ 再思考，
  直到模型认为可以直接回答。对应手写版的 `for round_no in ...` 循环，
  也对应 `agent_graph.py` 里 agent↔tools 两个节点之间的"绕圈"。
- **messages 列表 = 记忆**：`system / user / assistant / tool` 四种角色轮流追加，
  每次请求把整个列表重发给模型（模型本身无状态，"记住"全靠重放历史）。
  LangGraph 里它就是 `AgentState.messages` + `add_messages` 合并器。
- **安全阀**：手写版的 `MAX_TOOL_ROUNDS = 6` 与 LangGraph 的 `recursion_limit`
  本质相同 —— 防止模型无限调工具烧钱，生产级 Agent 必备。
- **StateGraph（状态图）**：LangGraph 的核心抽象 —— 把 Agent 循环显式画成
  "节点（干一件事）+ 边（决定下一步去哪）"的图。好处：可并行、可插入人工审核、
  可视化，逻辑和手写 while 循环完全等价（见 `agent_graph.py` 逐行对照注释）。
- **checkpointer（检查点）**：按 `thread_id` 保存图的状态快照，让多轮对话记忆
  由框架托管。本项目用内存版 `MemorySaver`（进程退出即丢），且没持久化到磁盘 ——
  故意保持轻量；想做跨进程记忆，换 `langgraph-checkpoint-sqlite` 两行代码即可。

## 四、动手改造（学习最好的方式是改代码）

按难度递增，每个练习都只动一两个文件：

1. ⭐ **加一个新工具**：写一个函数加上 `@tool` 装饰器、丢进 `tools.py` 的 `TOOLS` 列表，
   问 Agent"讲个笑话"验证；再对照 `legacy/tools_raw.py` 里手写 Schema + 注册表的同一段逻辑，
   彻底理解"@tool 帮你省了什么"。（做完你会彻底理解"工具 = 函数 + 说明书"） -- done
2. ⭐ **改人设**：修改 `agent_graph.py` 里的 `SYSTEM_PROMPT`，把它变成一个严格的高数老师，
   观察回答风格的变化。（体会系统提示词的作用）
3. ⭐ **三模式对比实验**：同一个问题分别用 `/mode native`、`/mode graph`、`/mode preset` 问一遍，
   对比灰色过程行的可见度 —— 直观感受"封装"到底封装了什么。
4. ⭐⭐ **换模型**：在 `.env` 里设置 `LLM_MODEL`，对比不同模型的工具调用准确率。
5. ⭐⭐ **给图加节点**：在 `agent_graph.py` 的 agent 与 tools 之间插一个"人工审核"节点
   （用 `interrupt_before`），工具执行前先问你"允许吗？"。（LangGraph 相比手写循环的真正优势）
6. ⭐⭐⭐ **升级成 Web 界面**：用 20 行 FastAPI 或 Gradio 替换 `main.py` 的
   while 循环，`Agent` 类一行不用改。（体会分层设计的价值）

## 五、下一步学什么

理解了本项目的裸实现后，再去学框架会事半功倍：

- **LangGraph 进阶**：本项目 `agent_graph.py` 只是入门图，继续学并行分支、
  子图、`interrupt` 人工审核、持久化 checkpointer（本项目已用内存版，换 SQLite 即可跨进程记忆）
- **OpenAI Agents SDK**：另一条框架路线，对照着学更能体会设计取舍
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
