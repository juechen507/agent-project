"""
配置模块 —— Agent 的"环境变量中枢"

📚 新手知识点：
  大模型 API（如 OpenAI、DeepSeek、Qwen）都需要一个 API Key 才能调用。
  习惯做法是把密钥写进 .env 文件，而不是硬编码在代码里，
  这样密钥不会被提交到 Git 仓库泄露。

  本项目不依赖第三方 dotenv 库，而是手写了一个 ~20 行的 .env 解析器，
  让你理解 dotenv 类库背后其实做了什么。
"""

import os
from pathlib import Path

# __file__ 是当前文件的路径，parent 是 config.py 所在目录（即项目根目录）
ROOT_DIR = Path(__file__).parent


def load_dotenv(env_file: Path | None = None) -> None:
    """
    读取 .env 文件，把每一行 KEY=VALUE 写入进程环境变量。

    这就是 python-dotenv 库的核心原理，去掉了很多边界情况，只保留主干：
      1. 文件不存在就静默跳过（比如用户还没创建 .env）
      2. 忽略空行和 # 开头的注释行
      3. 已经存在的环境变量不覆盖（命令行传的优先级更高）
    """
    env_file = env_file or ROOT_DIR / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def get_config() -> dict:
    """
    汇总 Agent 需要的配置，返回一个字典。

    支持的提供商（任选其一，都兼容 OpenAI 接口格式）：
      - OpenAI:    OPENAI_API_KEY
      - DeepSeek:  DEEPSEEK_API_KEY   (https://api.deepseek.com)
      - 阿里云Qwen: QWEN_API_KEY       (兼容模式端点)
      - 自定义:     LLM_API_KEY + LLM_BASE_URL + LLM_MODEL
    """
    load_dotenv()

    # 按优先级依次检查各家提供商的 Key
    providers = [
        # (环境变量名, base_url, 默认模型名)
        ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
        ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
        ("QWEN_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ]

    for env_name, base_url, default_model in providers:
        api_key = os.environ.get(env_name)
        if api_key:
            return {
                "api_key": api_key,
                # 允许用户通过 LLM_BASE_URL / LLM_MODEL 覆盖默认值
                "base_url": os.environ.get("LLM_BASE_URL", base_url),
                "model": os.environ.get("LLM_MODEL", default_model),
            }

    # 一个 Key 都没找到：给新手一个清晰的错误提示，而不是让程序崩溃
    raise SystemExit(
        "\n❌ 未找到任何 API Key。\n"
        "   请复制 .env.example 为 .env，然后填入你自己的 Key：\n"
        "       cp .env.example .env\n"
        "   详细步骤见 README.md 的「第 1 步：准备 API Key」。"
    )
