"""危机风险分级规则测试（纯逻辑，不依赖数据库）。

说明：用例只使用 ``assert``，不依赖 pytest 夹具或装饰器，
因此既可以用 ``pytest tests/test_crisis_grading.py`` 执行，
也可以在无 pytest 环境下直接用 Python 调用各 ``test_*`` 函数。
"""

from app.services.crisis_rules import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    LEVEL_NONE,
    ModalitySignals,
    assess_crisis,
    detect_crisis,
)


def test_high_risk_expressions_are_detected():
    """明确的自伤/自杀表达必须判为 HIGH。"""
    cases = [
        "我有点不想活了",
        "我想过自杀，遗书都写好了",
        "想离开这个世界，谁都别拦我",
        "我准备了安眠药，不想再醒过来了",
        "不想再撑了，我已经决定明天就去天台",
    ]
    for text in cases:
        result = assess_crisis(text)
        assert result.level == LEVEL_HIGH, f"{text} -> {result.level}"
        assert result.flagged is True


def test_denial_and_hypothesis_downgrade_to_low():
    """显式否认只留痕，不建立工单。"""
    cases = [
        "我不会自杀的，只是最近很累",
        "我没有想死，只是压力大",
        "我从来没想过自杀",
    ]
    for text in cases:
        result = assess_crisis(text)
        assert result.level == LEVEL_LOW, f"{text} -> {result.level}"
        assert result.flagged is False


def test_third_party_and_quote_capped_at_medium():
    """转述他人或影视剧情不应判为本人高危。"""
    cases = [
        "我朋友说他想自杀，我不知道怎么办",
        "电视剧里的主角想死，剧情很压抑",
    ]
    for text in cases:
        result = assess_crisis(text)
        assert result.level == LEVEL_MEDIUM, f"{text} -> {result.level}"


def test_hyperbole_and_hyperbole_traces():
    """口语夸张表达降级为低风险留痕，不建立工单。"""
    result = assess_crisis("这题难得我想死，算了先去吃饭")
    assert result.level in (LEVEL_NONE, LEVEL_LOW), result.level
    assert result.flagged is False


def test_recovery_context_downgrades_medium_evidence():
    """出现缓解/恢复表述时，强负性证据降级为低风险。"""
    cases = [
        "我以前想过放弃，但现在好多了，只是偶尔会难过",
        "难受的时候会想消失一下，不过睡一觉就好了",
    ]
    for text in cases:
        result = assess_crisis(text)
        assert result.level == LEVEL_LOW, f"{text} -> {result.level}"
        assert result.flagged is False


def test_medium_expression_needs_self_reference():
    """描述外部事件而非本人时不应升级为风险。"""
    assert assess_crisis("作业多到崩溃").level == LEVEL_NONE
    assert assess_crisis("我今天崩溃了，撑不下去").level == LEVEL_MEDIUM


def test_negation_does_not_leak_across_clauses():
    """否定语义不应跨越标点影响后续从句。"""
    result = assess_crisis("我最近很不好，想过自杀")
    assert result.level == LEVEL_HIGH, result.level


def test_modality_consensus_only_yields_low_without_text_evidence():
    """仅有多模态负性一致信号时判 LOW，不建立工单。"""
    signals = ModalitySignals(
        voice_emotion="sad",
        voice_confidence=0.8,
        facial_emotion="sad",
        facial_confidence=0.7,
        facial_frames=12,
    )
    result = assess_crisis("最近有点累", signals=signals)
    assert result.level == LEVEL_LOW, result.level
    assert result.flagged is False


def test_high_risk_is_not_created_by_modality_signals_alone():
    """不变量：模态信号不能把无文本证据的样本推到 HIGH。"""
    signals = ModalitySignals(
        voice_emotion="sad",
        voice_confidence=0.95,
        facial_emotion="fearful",
        facial_confidence=0.95,
        facial_frames=30,
    )
    result = assess_crisis("今天有点累", signals=signals)
    assert result.level != LEVEL_HIGH, result.level


def test_duplicate_keyword_sources_do_not_double_score():
    """环境变量关键词与内置规则重叠时不得重复计分。"""
    baseline = assess_crisis("我想过自杀")
    with_extra = assess_crisis("我想过自杀", extra_high_keywords=("自杀", "想死"))
    assert with_extra.risk_score == baseline.risk_score


def test_assessment_payload_shape():
    """评估结果需提供接口与数据库消费的结构化字段。"""
    payload = assess_crisis("我想过自杀").to_dict()
    assert payload["level"] == LEVEL_HIGH
    assert payload["flagged"] is True
    assert payload["levelLabel"] == "高风险"
    assert payload["riskScore"] >= 70
    assert payload["reasons"]


def test_detect_crisis_backward_compatible():
    """旧接口语义保持不变：命中为 True，无风险为 False。"""
    assert detect_crisis("我想死") is True
    assert detect_crisis("今天天气不错") is False
    assert detect_crisis(None) is False
