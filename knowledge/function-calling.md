# Function Calling（工具调用）

Function Calling（工具调用）是让大模型以 JSON 格式表达「我想调用哪个函数、传什么参数」的机制。
真正的执行由我们的代码完成，模型本身不运行任何代码。

每个工具 = 一个普通 Python 函数 + 一份说明书：
- 手写版：自己写 JSON Schema + `TOOL_REGISTRY` 注册表（见 `legacy/tools_raw.py`）
- LangGraph 版：`@tool` 装饰器根据 docstring 和类型注解自动生成说明书（见 `tools.py`）

常见误区：名字里有 Function，但模型并没有「调用」你的函数，它只是在生成一段结构化请求。
