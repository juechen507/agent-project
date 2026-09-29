"""
RAG 模块 —— 检索增强生成里的「检索」一半

📚 新手知识点：模型为什么需要 RAG？

  大模型的参数里没有你的私有文档，也容易对细节一本正经地胡说。
  RAG 的做法不是「再训练模型」，而是每次回答前先查资料：

      文档 → 切块 → 变成向量 → 写入本地索引     （知识文件一改就重建）
                         ↓
      用户问题 → 同样变成向量 → 按余弦相似度取 Top-K → 交给模型当上下文

  本文件只做检索，生成仍由 Agent 完成：lookup_knowledge 把检索结果
  以纯文本返回，模型读完再组织成回答。三种 Agent 模式共用这一套。

  向量怎么来的？（本项目故意不用 ChromaDB / OpenAI Embedding）
    对中文用「单字 + 二字词」，对英文用单词，再做 TF-IDF。
    这样零依赖、可打印、改一行就能看懂「语义相近」到底在算什么。
    生产环境会换成神经网络 embedding；公式（切块 → 向量 → 相似度）是一样的。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from config import ROOT_DIR

KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
INDEX_PATH = ROOT_DIR / ".rag_index.json"
TOP_K = 3
# 余弦相似度低于此值视为不相关（TF-IDF 向量较稀疏，阈值不宜过高）
MIN_SCORE = 0.08

_index: dict | None = None


def retrieve_knowledge(query: str, top_k: int = TOP_K) -> str:
    """给工具调用用的入口：返回可直接喂给模型的检索结果文本。"""
    query = (query or "").strip()
    if not query:
        return "查询内容为空，请提供要检索的问题或关键词。"

    hits = search(query, top_k=top_k)
    if not hits:
        return (
            f"知识库中没有找到与「{query}」足够相关的资料。"
            "可以换个问法，或在 knowledge/ 目录新增 Markdown 后再问。"
        )

    lines = [
        "以下是从本地知识库检索到的资料，请只根据这些内容回答，不要编造未出现的事实。",
        "",
    ]
    for i, hit in enumerate(hits, start=1):
        lines.append(f"【资料 {i}｜来源 {hit['source']}｜相关度 {hit['score']:.2f}】")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines).strip()


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    index = _load_index()
    query_vec = _tfidf_vector(_tokenize(query), index["idf"])
    scored: list[dict] = []
    for chunk in index["chunks"]:
        score = _cosine(query_vec, chunk["vector"])
        if score < MIN_SCORE:
            continue
        scored.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "score": score,
            }
        )
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


def load_chunks() -> list[dict]:
    """把 knowledge/*.md 切成带来源的文本块。按空行分段，过长再按句号切开。"""
    chunks: list[dict] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(_split_markdown(text)):
            chunks.append({"id": f"{path.stem}-{i}", "source": path.name, "text": piece})
    return chunks


def _iter_markdown_files() -> list[Path]:
    if not KNOWLEDGE_DIR.is_dir():
        return []
    return sorted(p for p in KNOWLEDGE_DIR.glob("*.md") if p.is_file())


def _split_markdown(text: str, max_chars: int = 420) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= max_chars:
            buf = para
        else:
            chunks.extend(_split_long(para, max_chars))
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])", text)
    out: list[str] = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = f"{buf}{sent}" if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = sent if len(sent) <= max_chars else sent[:max_chars]
    if buf:
        out.append(buf)
    return out


def _tokenize(text: str) -> list[str]:
    """英文按单词；中文同时保留单字和相邻二字，兼顾精确命中与短句语义。"""
    lowered = text.lower().replace("什么是", " ").replace("什么叫", " ")
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def _term_freq(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, float] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0.0) + 1.0
    return freq


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _term_freq(tokens)
    length = max(len(tokens), 1)
    return {term: (count / length) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _corpus_fingerprint() -> str:
    hasher = hashlib.md5()
    files = _iter_markdown_files()
    if not files:
        hasher.update(b"empty")
        return hasher.hexdigest()
    for path in files:
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _build_index() -> dict:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"knowledge/ 目录是空的，请放入 .md 文件：{KNOWLEDGE_DIR}")

    tokenized = [_tokenize(c["text"]) for c in chunks]
    doc_count = len(tokenized)
    df: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    # 加 1 平滑，避免某个词只出现一次时 idf 变成 0
    idf = {term: math.log((doc_count + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    serialized = []
    for chunk, tokens in zip(chunks, tokenized):
        serialized.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "vector": _tfidf_vector(tokens, idf),
            }
        )
    return {"fingerprint": _corpus_fingerprint(), "idf": idf, "chunks": serialized}


def _load_index() -> dict:
    global _index
    fingerprint = _corpus_fingerprint()
    if _index is not None and _index.get("fingerprint") == fingerprint:
        return _index

    if INDEX_PATH.exists():
        cached = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            _index = cached
            return _index

    print("  \033[90m⚙ 正在为 knowledge/ 建立向量索引…\033[0m")
    built = _build_index()
    INDEX_PATH.write_text(json.dumps(built, ensure_ascii=False), encoding="utf-8")
    _index = built
    return built


if __name__ == "__main__":
    # 不调大模型，单独验证检索：python3 rag.py
    demo = "智能体怎么自己决定下一步？和普通聊天机器人有什么区别"
    print(retrieve_knowledge(demo))
    print("\n---\n")
    print(retrieve_knowledge("Function Calling 是模型自己在跑代码吗"))
