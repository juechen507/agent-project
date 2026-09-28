"""
命令行入口 —— 运行 `python3 main.py` 开始和 Agent 对话

📚 新手知识点：
  这个文件刻意做得极简：一个 while 循环读输入、调 agent.chat()、打印结果。
  Agent 的全部复杂性都在 agent.py 和 tools.py 里，入口只是"壳"。
  以后你想做 Web 界面（FastAPI/Gradio），也只需要把这里的 while 循环
  换成一个 HTTP 接口，其余代码原封不动 —— 这就是分层的好处。
"""

from agent import Agent

# 开场告诉新手可以试什么问题，降低"第一句话不知道说什么"的门槛
EXAMPLE_QUESTIONS = """可以试试这些问题（观察灰色的工具调用过程）：
  · 帮我算一下 (128 + 256) * 3 等于多少
  · 现在几点了？杭州天气怎么样
  · 什么是 ReAct？
  · 北京今天适合出门吗？（考验模型会不会自己组合用多个工具）"""

# 支持的斜杠命令
COMMANDS = """命令：/reset 清空对话记忆  /verbose 开关思考过程  /quit 退出"""


def main() -> None:
    try:
        agent = Agent()
    except SystemExit:
        raise  # config.py 里已给出友好的缺 Key 提示，直接透出

    print("🤖 迷你学习 Agent 已启动（按 Ctrl+C 或输入 /quit 退出）")
    print(f"   当前模型：{agent.model}\n")
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
            print(f"{COMMANDS}\n{EXAMPLE_QUESTIONS}")
            continue
        if user_input == "/reset":
            # 重置记忆 = 把 messages 恢复到只剩系统提示词
            agent.messages = agent.messages[:1]
            print("  \033[90m⚙ 对话记忆已清空\033[0m")
            continue
        if user_input == "/verbose":
            agent.verbose = not agent.verbose
            print(f"  \033[90m⚙ 思考过程已{'开启' if agent.verbose else '关闭'}\033[0m")
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
