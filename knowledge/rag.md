# RAG（检索增强生成）

RAG = Retrieval-Augmented Generation：先检索资料，再让模型基于资料生成回答，减少胡编乱造。

四步流水线（本项目 `rag.py`）：
1. Load：读取 `knowledge/` 目录下的 Markdown 文档
2. Chunk：按段落切成小块，方便精确命中
3. Embed + Index：用 TF-IDF 把文本变成向量，写入 `.rag_index.json`（设 `RAG_BACKEND=chroma` 时，同样的向量改写入 Chroma 持久化向量库 `chroma_db/`）
4. Retrieve：把用户问题也变成向量，按余弦相似度取出最相关的几块，作为工具结果交给模型

和关键词匹配的区别：用户说「智能体怎么自己决定下一步」也能检索到 Agent 文档，
不必精确命中「什么是agent」这种固定标题。
想扩充知识库，往 `knowledge/` 里加 `.md` 文件即可，下次查询会自动重建索引。
