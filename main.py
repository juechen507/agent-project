"""
命令行入口 —— 运行 `python3 main.py` 开始和 Agent 对话

📚 新手知识点：
  这个文件刻意做得极简：一个 while 循环读输入、调 agent.chat()、打印结果。
  Agent 的全部复杂性都在 agent_*.py 和 tools.py 里，入口只是"壳"。

  LangGraph 重构后，这里同时挂三种 Agent 实现，用 /mode 随时热切换：
    native  手写 OpenAI SDK 循环（legacy/agent_native.py）—— 理解原理
    graph   LangGraph 手动 StateGraph（agent_graph.py）    —— 理解框架（默认）
    preset  create_react_agent 预制件（agent_preset.py）   —— 工程实践
  同一个问题在三种模式下各问一遍，回答一致、只有"过程可见度"不同 ——
  这正是体会框架封装了什么的最佳方式。
"""

# 模式名 -> (导入 Agent 的函数, 一句话说明)。延迟导入，用哪个才加载哪个。
MODES = {
    "native": (
        lambda: __import__("legacy.agent_native", fromlist=["Agent"]).Agent,
        "手写 OpenAI SDK 循环 → legacy/agent_native.py + legacy/tools_raw.py",
    ),
    "graph": (
        lambda: __import__("agent_graph", fromlist=["Agent"]).Agent,
        "LangGraph 手动 StateGraph → agent_graph.py + tools.py",
    ),
    "preset": (
        lambda: __import__("agent_preset", fromlist=["Agent"]).Agent,
        "LangGraph 预制件 create_react_agent → agent_preset.py",
    ),
}

# 开场告诉新手可以试什么问题，降低"第一句话不知道说什么"的门槛
EXAMPLE_QUESTIONS = """可以试试这些问题（观察灰色的工具调用过程）：
  · 帮我算一下 (128 + 256) * 3 等于多少
  · 现在几点了？杭州天气怎么样
  · 什么是 ReAct？智能体怎么自己决定下一步？
  · RAG 和关键词匹配有什么区别？
  · 北京今天适合出门吗？
  · 能给我讲个笑话？（考验模型会不会自己组合用多个工具）"""

# 支持的斜杠命令
COMMANDS = """命令：/mode native|graph|preset 切换实现  /reset 清空对话记忆  /verbose 开关思考过程  /quit 退出"""


def make_agent(mode: str, verbose: bool):
    """按模式创建 Agent 实例。三种实现对外接口一致：chat / reset_memory / verbose。"""
    agent_cls = MODES[mode][0]()
    agent = agent_cls(verbose=verbose)
    # native 版历史遗留：重置记忆直接截断 messages 列表。
    # 为保持统一，给它补一个 reset_memory()，LangGraph 版则各自有实现。
    if not hasattr(agent, "reset_memory"):
        def reset_memory():
            agent.messages = agent.messages[:1]
        agent.reset_memory = reset_memory
    return agent


def main() -> None:
    mode = "graph"  # 默认用 LangGraph 手动版：既能看到图结构，又是新代码主入口
    verbose = True
    try:
        agent = make_agent(mode, verbose)
    except SystemExit:
        raise  # config.py 里已给出友好的缺 Key 提示，直接透出

    print("🤖 迷你学习 Agent 已启动（LangGraph 重构版，按 Ctrl+C 或输入 /quit 退出）")
    print(f"   当前模型：{agent.model}")
    print(f"   当前模式：{mode}（{MODES[mode][1]}）\n")
    print(EXAMPLE_QUESTIONS + "\n")

    while True:
        try:
            user_input = input("\n\033[36m你>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！愿你对 Agent 的理解又深了一层 🎓")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "/q"):
            print("再见！愿你对 Agent 的理解又深了一层 🎓")
            break
        if user_input == "/help":
            print(f"{COMMANDS}\n")
            for name, (_, desc) in MODES.items():
                marker = "（当前）" if name == mode else ""
                print(f"  /mode {name}{marker}：{desc}")
            print(f"\n{EXAMPLE_QUESTIONS}")
            continue
        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1] not in MODES:
                print(f"  \033[90m⚙ 用法：/mode {'|'.join(MODES)}\033[0m")
                continue
            new_mode = parts[1]
            if new_mode == mode:
                print(f"  \033[90m⚙ 已经处于 {mode} 模式\033[0m")
                continue
            # 热切换实现：只换 Agent 实例。提醒新手——各模式的对话记忆互相独立，
            # （memory 属于各自实例），切换本身就是"换了一个大脑"。
            mode = new_mode
            try:
                agent = make_agent(mode, verbose)
            except SystemExit:
                raise
            print(f"  \033[90m⚙ 已切换到 {mode} 模式：{MODES[mode][1]}\033[0m")
            print("  \033[90m⚙ 提示：切换实现后对话记忆不互通，相当于重新开聊\033[0m")
            continue
        if user_input == "/reset":
            # 重置记忆：native 版 = messages 恢复到只剩系统提示词；
            #        LangGraph 版 = 换一个 thread_id 开新会话
            agent.reset_memory()
            print("  \033[90m⚙ 对话记忆已清空\033[0m")
            continue
        if user_input == "/verbose":
            verbose = not verbose
            agent.verbose = verbose
            print(f"  \033[90m⚙ 思考过程已{'开启' if verbose else '关闭'}\033[0m")
            continue

        try:
            answer = agent.chat(user_input)
        except Exception as exc:
            # 网络抖动、Key 失效等不应该弄崩整个程序，打印错误继续下一轮
            print(f"\n\033[31m出错了：{exc}\033[0m")
            continue

        print(f"\n\033[32mAgent>\033[0m {answer}")


if __name__ == "__main__":
    main()
