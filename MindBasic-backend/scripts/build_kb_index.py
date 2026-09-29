"""
构建心理教练知识库检索索引（本地 BM25）
========================================

扫描书目录 → 切块 → jieba 分词 → BM25 倒排索引 → 写入 ``data/kb_index.pkl``。
纯本地计算，不调用任何模型 API。书籍没变时只需构建一次。

用法（在 `MindBasic-backend/` 目录下执行）::

    python scripts/build_kb_index.py
    python scripts/build_kb_index.py --source "D:/books" --out data/kb_index.pkl
    python scripts/build_kb_index.py --query "一躺下就想起工作的事，睡不着"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.ai_lab import kb_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建知识库 BM25 索引")
    parser.add_argument("--source", default=None, help="书目录（默认取仓库里的 15 本）")
    parser.add_argument("--out", default=None, help="索引输出路径")
    parser.add_argument("--chunk-size", type=int, default=kb_service.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=kb_service.CHUNK_OVERLAP)
    parser.add_argument("--query", default=None, help="构建完顺便做一次检索验证")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    print("=" * 66)
    print("构建知识库索引")
    print("=" * 66)
    src = Path(args.source) if args.source else kb_service.source_dir()
    out = Path(args.out) if args.out else kb_service.index_path()
    print(f"书目录：{src}")
    print(f"索引输出：{out}")
    print(f"切块：{args.chunk_size} 字 / 重叠 {args.overlap} 字\n")

    stats = kb_service.build_index(src, out, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"书籍 {stats['books']} 本")
    if stats.get("skipped"):
        print(f"跳过重复内容文件 {len(stats['skipped'])} 个：")
        for name in stats["skipped"]:
            print(f"  - {name}")
    print(f"片段 {stats['chunks']:,} 条")
    print(f"词条 {stats['terms']:,} 个")
    print(f"平均片段长度 {stats['avg_doc_len']} 词")
    print(f"耗时 {stats['seconds']}s")
    print(f"索引文件 {stats['index_mb']} MB → {stats['index_path']}")

    if args.query:
        kb_service.reset_cache()
        print("\n" + "-" * 66)
        print(f"检索验证：{args.query}")
        print("-" * 66)
        hits = kb_service.search(args.query, args.top_k)
        if not hits:
            print("（无命中）")
            return 3
        for i, hit in enumerate(hits, 1):
            print(f"\n[{i}] score={hit['score']}  《{hit['book']}》{(' · ' + hit['section']) if hit['section'] else ''}")
            print("    " + hit["text"][:180].replace("\n", " "))
    print("\n完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
