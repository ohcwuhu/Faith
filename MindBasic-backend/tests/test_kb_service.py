"""本地知识库检索（jieba + BM25）测试。

背景：账号里只有 DeepSeek，没有 embedding 模型，Dify 的语义检索用不了；
Dify 经济模式的关键词检索实测也搜不出内容。检索改由平台侧自建，这里锁定行为。
"""

from pathlib import Path

from app.services.ai_lab import kb_service


def _make_corpus(tmp_path: Path) -> Path:
    """造一个小语料：两本"书"，主题分离，便于断言命中来源。"""
    src = tmp_path / "books"
    src.mkdir()
    (src / "5.1.心理书籍-睡眠改善手册.md").write_text(
        "# 第一章 睡眠卫生\n"
        "入睡困难常常和睡前认知唤醒有关：一躺下就开始盘算明天的工作，大脑被迫保持警觉，"
        "于是越想越睡不着。刺激控制技术要求把床只留给睡眠，不要在卧室处理工作。\n"
        "# 第二章 放松训练\n"
        "渐进式肌肉放松与腹式呼吸可以降低生理唤醒水平，帮助更快入睡。\n",
        encoding="utf-8",
    )
    (src / "5.2.心理书籍-团队沟通指南.md").write_text(
        "# 第一章 会议沟通\n"
        "跨部门协作时，先对齐目标再讨论方案，可以显著减少无效争论与返工。\n"
        "# 第二章 反馈技巧\n"
        "给出负面反馈时对事不对人，描述具体行为而不是评价人格。\n",
        encoding="utf-8",
    )
    return src


def test_tokenize_drops_stopwords_and_punctuation():
    tokens = kb_service.tokenize("我 的 失眠 ， 真的 很 难受 ！")
    assert "失眠" in tokens
    assert "难受" in tokens
    # 停用词与纯标点不入索引
    assert "的" not in tokens
    assert not any(t in {",", "，", "！"} for t in tokens)


def test_build_and_search_returns_on_topic_chunk(tmp_path):
    src = _make_corpus(tmp_path)
    out = tmp_path / "index.pkl"
    stats = kb_service.build_index(src, out, chunk_size=200, overlap=40)
    assert stats["books"] == 2
    assert stats["chunks"] >= 2
    assert out.is_file()

    hits = kb_service.search("一躺下就想工作，睡不着", top_k=2, path=out)
    assert hits, "应当有命中"
    assert "睡眠" in hits[0]["book"]
    assert "睡不着" in hits[0]["text"] or "入睡" in hits[0]["text"]


def test_book_title_strips_numeric_prefix():
    p = Path("5.769.心理书籍教育书籍-专业级教练 (PCC)认证手册.md")
    assert kb_service._book_title(p) == "专业级教练 (PCC)认证手册"


def test_context_block_empty_when_index_missing(tmp_path):
    missing = tmp_path / "nope.pkl"
    assert kb_service.search("失眠", path=missing) == []
    assert kb_service.context_block("失眠", path=missing) == ""
    assert kb_service.load_index(missing) is None


def test_context_block_formats_sources(tmp_path):
    src = _make_corpus(tmp_path)
    out = tmp_path / "index.pkl"
    kb_service.build_index(src, out, chunk_size=200, overlap=40)
    block = kb_service.context_block("团队沟通里怎么给反馈", top_k=1, path=out)
    assert block.startswith("[资料1]")
    assert "沟通" in block


def test_retrieve_falls_back_to_bm25_when_llm_disabled(tmp_path):
    """关掉 LLM 增强时必须等价于纯 BM25，不能因此报错。"""
    src = _make_corpus(tmp_path)
    out = tmp_path / "index.pkl"
    kb_service.build_index(src, out, chunk_size=200, overlap=40)
    hits = kb_service.retrieve("睡不着", top_k=2, path=out, use_llm=False)
    assert hits and "睡眠" in hits[0]["book"]
