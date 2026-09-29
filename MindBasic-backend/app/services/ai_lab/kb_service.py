"""
心理教练知识库检索（本地 BM25）
================================

为什么不用 Dify 的知识库检索
---------------------------

Dify 的语义 / 混合检索需要一个 **embedding 模型**；账号里只有 DeepSeek，
而 DeepSeek 不提供 embedding（实测语义检索直接报
``Model provider langgenius/openai/openai quota exceeded``）。
退到「经济模式关键词检索」也不行：实测连片段自身提取出的关键词都搜不到
（`教练` / `PCC` / `coaching` 全部 0 命中），该索引实际不可用。

所以这里把检索搬到平台侧：**jieba 分词 + Okapi BM25**，纯本地计算，
不依赖任何模型 API、不产生调用费用、可离线跑。

检索质量不足时的补强（可选，默认关闭）
------------------------------------

设置 ``KB_RERANK_ENABLED=true`` 后，先用 BM25 召回 ``KB_RERANK_CANDIDATES`` 条，
再让 DeepSeek 从中挑出最相关的 ``top_k`` 条。用 DeepSeek 当重排器，
替代原来那个用不了的 ``qwen3-rerank``。

数据流
------

1. 离线：``python scripts/build_kb_index.py`` 扫描书目录 → 切块 → 分词 →
   BM25 倒排索引 → 落到 ``data/kb_index.pkl``（构建一次，之后秒级加载）；
2. 在线：每轮通话拿到 ASR 文本后 ``search()`` 取 top-k，
   拼成参考资料随 ``knowledge_context`` 传给 Dify 的「普通心理教练」节点。
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[3]

# 书目录：默认取仓库根下的「心理教练智能体知识库-精选15本」
DEFAULT_SOURCE_DIR = _HERE.parents[5] / "心理教练智能体知识库-精选15本"
# 索引缓存：构建一次，之后启动直接加载
DEFAULT_INDEX_PATH = _BACKEND_ROOT / "data" / "kb_index.pkl"

CHUNK_SIZE = int(os.environ.get("KB_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("KB_CHUNK_OVERLAP", "100"))
BM25_K1 = 1.5
BM25_B = 0.75

# 合并版/合集与单本内容重复，同时索引会让同一段文字被重复命中。
# 命中任一标记的文件在构建时跳过，跳过的文件会在构建统计里列出来。
_SKIP_NAME_MARKERS = ("合并", "合集", "汇总", "整理版", "全本", "全集")


def source_dir() -> Path:
    return Path(os.environ.get("KB_SOURCE_DIR") or DEFAULT_SOURCE_DIR)


def index_path() -> Path:
    return Path(os.environ.get("KB_INDEX_PATH") or DEFAULT_INDEX_PATH)


# ─── 分词 ────────────────────────────────────────────────────────────────────
_STOPWORDS = set(
    "的 了 是 在 和 与 也 就 都 而 及 或 一个 我们 你们 他们 这 那 有 我 你 他 她 它 "
    "不 没 很 会 要 把 被 让 给 对 从 到 为 上 下 里 中 后 前 之 其 等 可以 这样 那样 "
    "什么 怎么 因为 所以 但是 如果 已经 还是 就是 只是 一些 这些 那些 the a an is are "
    "of to in and or for on with that this it be as at by".split()
)

_tokenizer = None


def _get_tokenizer():
    """惰性加载 jieba（首次加载约 1 秒，避免拖慢后端启动）。"""
    global _tokenizer
    if _tokenizer is None:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        _tokenizer = jieba
    return _tokenizer


def tokenize(text: str) -> list[str]:
    """中文分词 + 归一化：去停用词、去纯标点、英文转小写。"""
    tokens: list[str] = []
    for raw in _get_tokenizer().lcut(text or ""):
        t = raw.strip().lower()
        if not t or t in _STOPWORDS:
            continue
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in t):
            continue
        tokens.append(t)
    return tokens


# ─── 切块 ────────────────────────────────────────────────────────────────────
def _book_title(path: Path) -> str:
    name = path.stem
    # 去掉「5.769.心理书籍教育书籍-」这类前缀，只留书名
    for sep in ("-", "—"):
        if sep in name:
            name = name.split(sep, 1)[1] or name
            break
    return name.strip() or path.stem


def iter_chunks(path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """把一本书切成带章节标题的片段，产出 ``(章节, 片段正文)``。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    section = ""
    buffer = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip() or section
        buffer += line + "\n"
        while len(buffer) >= chunk_size:
            yield section, buffer[:chunk_size]
            # 保留尾部做重叠，避免答案正好被切断
            buffer = buffer[max(0, chunk_size - overlap):]
    if len(buffer.strip()) >= 60:  # 太短的尾块多为版权/目录噪声
        yield section, buffer


def build_index(
    src: Path | None = None,
    out: Path | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict:
    """扫描书目录并构建 BM25 倒排索引，写入 ``out``。返回构建统计。"""
    src = Path(src or source_dir())
    out = Path(out or index_path())
    if not src.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{src}")

    all_files = sorted(src.glob("*.md"))
    files = [p for p in all_files if not any(m in p.name for m in _SKIP_NAME_MARKERS)]
    skipped = [p.name for p in all_files if p not in files]
    if skipped:
        log.info("跳过 %d 个重复内容文件：%s", len(skipped), skipped)
    if not files:
        raise FileNotFoundError(f"{src} 下没有 .md 文件")

    t0 = time.time()
    chunks: list[dict] = []
    postings: dict[str, array] = {}
    doc_len: list[int] = []

    for path in files:
        title = _book_title(path)
        for section, body in iter_chunks(path, chunk_size, overlap):
            doc_id = len(chunks)
            tokens = tokenize(body)
            if not tokens:
                continue
            chunks.append({"text": body.strip(), "book": title, "section": section})
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                arr = postings.get(term)
                if arr is None:
                    postings[term] = array("i", [doc_id, tf])
                else:
                    arr.append(doc_id)
                    arr.append(tf)

    if not chunks:
        raise RuntimeError("切块结果为空，请检查书目录内容")

    avgdl = sum(doc_len) / len(doc_len)
    payload = {
        "version": 1,
        "built_at": t0,
        "source_dir": str(src),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": chunks,
        "postings": postings,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "n_docs": len(chunks),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {
        "books": len(files),
        "skipped": skipped,
        "chunks": len(chunks),
        "terms": len(postings),
        "avg_doc_len": round(avgdl, 1),
        "seconds": round(time.time() - t0, 1),
        "index_path": str(out),
        "index_mb": round(out.stat().st_size / 1024 / 1024, 1),
    }
    return stats


# ─── 检索 ────────────────────────────────────────────────────────────────────
@dataclass
class _Index:
    chunks: list[dict]
    postings: dict[str, array]
    doc_len: list[int]
    avgdl: float
    n_docs: int
    meta: dict = field(default_factory=dict)


_index: _Index | None = None
_index_loaded = False
_index_loaded_from: Path | None = None
_lock = threading.Lock()


def load_index(path: Path | None = None, *, force: bool = False) -> _Index | None:
    """加载索引（进程内缓存）。索引不存在时返回 ``None`` 且不抛异常。"""
    global _index, _index_loaded, _index_loaded_from
    p = Path(path or index_path())
    with _lock:
        # 换了索引文件必须重新加载，否则会拿到上一个路径的内容
        if _index_loaded and not force and _index_loaded_from == p:
            return _index
        if not p.is_file():
            log.warning("知识库索引不存在：%s（先跑 scripts/build_kb_index.py）", p)
            _index, _index_loaded, _index_loaded_from = None, True, p
            return None
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)
            _index = _Index(
                chunks=payload["chunks"],
                postings=payload["postings"],
                doc_len=payload["doc_len"],
                avgdl=payload["avgdl"],
                n_docs=payload["n_docs"],
                meta={k: payload[k] for k in ("built_at", "source_dir", "chunk_size")},
            )
            log.info("知识库索引已加载：%d 片段 / %d 词条", _index.n_docs, len(_index.postings))
        except Exception as exc:
            log.error("知识库索引加载失败：%s", exc)
            _index = None
        _index_loaded, _index_loaded_from = True, p
        return _index


def reset_cache() -> None:
    """清空进程内索引缓存（测试 / 重建索引后调用）。"""
    global _index, _index_loaded, _index_loaded_from
    with _lock:
        _index, _index_loaded, _index_loaded_from = None, False, None


def is_ready(path: Path | None = None) -> bool:
    return load_index(path) is not None


def search(query: str, top_k: int = 4, *, path: Path | None = None) -> list[dict]:
    """BM25 检索，返回 ``[{text, book, section, score}]``（按分数降序）。"""
    idx = load_index(path)
    if idx is None:
        return []
    terms = tokenize(query)
    if not terms:
        return []

    scores: dict[int, float] = {}
    k1, b, avgdl, n = BM25_K1, BM25_B, idx.avgdl, idx.n_docs
    for term in set(terms):
        arr = idx.postings.get(term)
        if not arr:
            continue
        df = len(arr) // 2
        idf = ((n - df + 0.5) / (df + 0.5))
        import math

        idf = math.log(1.0 + max(idf, 1e-9))
        for i in range(0, len(arr), 2):
            doc_id, tf = arr[i], arr[i + 1]
            dl = idx.doc_len[doc_id]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / denom

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {
            "text": idx.chunks[d]["text"],
            "book": idx.chunks[d]["book"],
            "section": idx.chunks[d]["section"],
            "score": round(s, 4),
        }
        for d, s in ranked
    ]


def context_block(query: str, top_k: int = 4, *, path: Path | None = None) -> str:
    """把检索结果拼成可直接塞进提示词的「参考资料」文本；无命中返回空串。"""
    hits = search(query, top_k, path=path)
    if not hits:
        return ""
    lines = []
    for i, hit in enumerate(hits, 1):
        where = hit["book"] + (f" · {hit['section']}" if hit["section"] else "")
        lines.append(f"[资料{i}]《{where}》\n{hit['text']}")
    return "\n\n".join(lines)


def stats(path: Path | None = None) -> dict:
    """索引概况，供自检脚本 / 运维接口使用。"""
    idx = load_index(path)
    if idx is None:
        return {"ready": False}
    return {
        "ready": True,
        "chunks": idx.n_docs,
        "terms": len(idx.postings),
        "avg_doc_len": round(idx.avgdl, 1),
        "source_dir": idx.meta.get("source_dir"),
    }


# ─── DeepSeek 增强：查询扩展 + 重排 ──────────────────────────────────────────
# 纯 BM25 对「我最近总是睡不着，一躺下就想工作上的事」这类长口语句子质量一般：
# 关键词被稀释，而 BM25 也不认识「睡不着 ≈ 失眠」。这里用 DeepSeek 补上两步，
# 全程只用 DeepSeek，不需要 embedding。
KB_RERANK_ENABLED = os.environ.get("KB_RERANK_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
)
KB_RECALL_CANDIDATES = int(os.environ.get("KB_RECALL_CANDIDATES", "12"))
KB_LLM_TIMEOUT = int(os.environ.get("KB_LLM_TIMEOUT", "20"))

_EXPAND_SYSTEM = (
    "你是中文心理学科研检索助手。从用户的话里提取 4~8 个用于检索心理学/教练类书籍的"
    "关键词，要包含同义词与更书面的说法（例如「睡不着」→「失眠/入睡困难」）。"
    "只输出 JSON 数组，不要任何解释，例如：[\"失眠\",\"入睡困难\",\"焦虑\"]"
)
_RERANK_SYSTEM = (
    "你是检索排序助手。给定用户问题和若干候选资料片段，挑出与问题最相关的片段，"
    "按相关度从高到低输出它们的编号。只输出 JSON 数组，例如：[3,7,1]。"
    "如果都不相关，输出 []"
)


def _deepseek_json(system: str, user: str, *, max_tokens: int = 200) -> object:
    """调用 DeepSeek 并解析出 JSON；失败抛异常，由调用方降级。"""
    import json

    import requests

    from app.services.ai_lab import config as _cfg

    resp = requests.post(
        f"{_cfg.DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {_cfg.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _cfg.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=KB_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"未返回 JSON 数组：{content[:120]}")
    return json.loads(content[start : end + 1])


def expand_query(query: str) -> list[str]:
    """用 DeepSeek 把口语 query 扩成检索关键词；失败时返回空列表（调用方用原句）。"""
    try:
        terms = _deepseek_json(_EXPAND_SYSTEM, query, max_tokens=120)
    except Exception as exc:
        log.warning("查询扩展失败，退回原句检索：%s", exc)
        return []
    if not isinstance(terms, list):
        return []
    return [str(t).strip() for t in terms if str(t).strip()][:8]


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """用 DeepSeek 对 BM25 召回结果重排；失败时保留原顺序。"""
    if len(hits) <= top_k:
        return hits
    listing = "\n".join(
        f"[{i}] 《{h['book']}》{h['text'][:160].replace(chr(10), ' ')}"
        for i, h in enumerate(hits, 1)
    )
    try:
        order = _deepseek_json(
            _RERANK_SYSTEM, f"用户问题：{query}\n\n候选资料：\n{listing}", max_tokens=80
        )
    except Exception as exc:
        log.warning("重排失败，保留 BM25 顺序：%s", exc)
        return hits[:top_k]
    picked: list[dict] = []
    seen: set[int] = set()
    for item in order if isinstance(order, list) else []:
        try:
            idx = int(item) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(hits) and idx not in seen:
            seen.add(idx)
            picked.append(hits[idx])
        if len(picked) >= top_k:
            break
    # 模型挑不满时用 BM25 顺序补齐
    for i, hit in enumerate(hits):
        if len(picked) >= top_k:
            break
        if i not in seen:
            picked.append(hit)
    return picked


def retrieve(
    query: str,
    top_k: int = 4,
    *,
    path: Path | None = None,
    use_llm: bool | None = None,
) -> list[dict]:
    """完整检索链：查询扩展 → BM25 召回 → DeepSeek 重排 → top_k。

    任一步失败都会降级，最差等同于纯 BM25，不会中断通话。
    """
    llm = KB_RERANK_ENABLED if use_llm is None else use_llm
    if llm:
        terms = expand_query(query)
        recall_query = " ".join(terms) if terms else query
    else:
        recall_query = query
    candidates = search(recall_query, KB_RECALL_CANDIDATES if llm else top_k, path=path)
    if llm and candidates:
        candidates = rerank(query, candidates, top_k)
    return candidates[:top_k]


def context_block_smart(query: str, top_k: int = 4, *, path: Path | None = None) -> str:
    """``context_block`` 的增强版：走查询扩展 + 重排。无命中返回空串。"""
    hits = retrieve(query, top_k, path=path)
    if not hits:
        return ""
    lines = []
    for i, hit in enumerate(hits, 1):
        where = hit["book"] + (f" · {hit['section']}" if hit["section"] else "")
        lines.append(f"[资料{i}]《{where}》\n{hit['text']}")
    return "\n\n".join(lines)
