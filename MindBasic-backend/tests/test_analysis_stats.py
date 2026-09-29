"""留痕统计聚合测试：分位数、口径映射与聚合结果。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

from app.services.analysis_stats_service import (
    LATENCY_KEYS,
    AnalysisRow,
    aggregate_rows,
    normalize_dify_risk,
    percentile,
)


# ============================================================
#  分位数
# ============================================================
def test_percentile_handles_empty_and_single():
    assert percentile([], 0.5) is None
    assert percentile([2.0], 0.5) == 2.0


def test_percentile_interpolates_linearly():
    """P50 / P95 按线性插值计算，便于复核。"""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(values, 0.95) == 4.8
    assert percentile(values, -1.0) == 1.0  # 越界收敛到端点
    assert percentile(values, 2.0) == 5.0


# ============================================================
#  Dify 风险口径映射
# ============================================================
def test_normalize_dify_risk_maps_three_levels():
    """Dify 三级口径映射到平台四级，未提供时返回 None。"""
    assert normalize_dify_risk("high") == "HIGH"
    assert normalize_dify_risk("Medium") == "MEDIUM"
    assert normalize_dify_risk("low") == "LOW"
    assert normalize_dify_risk(None) is None
    assert normalize_dify_risk("") is None
    assert normalize_dify_risk("critical") is None


# ============================================================
#  明细聚合
# ============================================================
def _row(**kwargs) -> AnalysisRow:
    kwargs.setdefault("status", "ok")
    return AnalysisRow(**kwargs)


def test_aggregate_rows_on_empty_input():
    """空窗口不能除零，且一致性比率应为 None（无样本可比）。"""
    result = aggregate_rows([])
    assert result["totals"]["analyses"] == 0
    assert result["totals"]["okRate"] == 0.0
    assert result["degradation"]["rate"] == 0.0
    assert result["risk"]["consistency"]["rate"] is None
    assert result["latency"]["metrics"]["e2e_seconds"]["p50"] is None


def test_aggregate_rows_counts_status_and_rates():
    rows = [
        _row(status="ok"),
        _row(status="ok"),
        _row(status="partial_success"),
        _row(status="failed"),
    ]
    result = aggregate_rows(rows, total=40)
    assert result["totals"]["analyses"] == 40
    assert result["totals"]["sampled"] == 4
    assert result["totals"]["status"] == {"ok": 2, "partial_success": 1, "failed": 1}
    assert result["totals"]["okRate"] == 0.5
    assert result["totals"]["partialSuccessRate"] == 0.25
    assert result["totals"]["failedRate"] == 0.25


def test_aggregate_rows_reports_modality_availability():
    """面部缺失、语音缺失与短文本分别统计。"""
    rows = [
        _row(facial_frames=12, voice_emotion="sad", asr_text="我今天有点累"),
        _row(facial_frames=0, voice_emotion=None, asr_text="嗯"),
        _row(facial_frames=None, voice_emotion="neutral", asr_text=""),
    ]
    result = aggregate_rows(rows)
    assert result["modalities"]["missingFacialRate"] == round(2 / 3, 4)
    assert result["modalities"]["missingVoiceRate"] == round(1 / 3, 4)
    assert result["modalities"]["shortTextRate"] == round(2 / 3, 4)


def test_aggregate_rows_normalizes_degradation_reasons():
    """降级原因需要归一化：去掉括号里的数值，只保留规则名。"""
    rows = [
        _row(weight_adjustments=[
            "no_facial_frames: facial=0",
            "facial_stability_low(0.21): facial-50%",
        ]),
        _row(weight_adjustments=["no_facial_frames: facial=0"]),
        _row(weight_adjustments=[]),
    ]
    result = aggregate_rows(rows)
    assert result["degradation"]["analysesWithAdjustment"] == 2
    assert result["degradation"]["rate"] == round(2 / 3, 4)
    top = {item["reason"]: item["count"] for item in result["degradation"]["topReasons"]}
    assert top["no_facial_frames: facial=0"] == 2
    assert top["facial_stability_low"] == 1


def test_aggregate_rows_counts_stage_and_risk():
    """阶段分布与风险分布按平台口径统计，工单触发只计 MEDIUM/HIGH。"""
    rows = [
        _row(coach_stage="exploration", risk_level="NONE"),
        _row(coach_stage="exploration", risk_level="LOW"),
        _row(coach_stage="closing", risk_level="MEDIUM"),
        _row(coach_stage="closing", risk_level="HIGH"),
    ]
    result = aggregate_rows(rows)
    assert result["stages"]["distribution"] == {"exploration": 2, "closing": 2}
    assert result["stages"]["rate"]["closing"] == 0.5
    assert result["risk"]["platform"]["distribution"] == {
        "NONE": 1, "LOW": 1, "MEDIUM": 1, "HIGH": 1,
    }
    assert result["risk"]["platform"]["flagged"] == 2
    assert result["risk"]["platform"]["flaggedRate"] == 0.5


def test_aggregate_rows_consistency_skips_unrecorded_rows():
    """只有两侧都有判定时才纳入一致性统计，避免把缺失当成不一致。"""
    rows = [
        _row(risk_level="HIGH", dify_risk_level="high"),      # 一致
        _row(risk_level="MEDIUM", dify_risk_level="high"),    # 平台偏保守
        _row(risk_level="LOW", dify_risk_level=None),         # 未记录，跳过
        _row(risk_level=None, dify_risk_level="medium"),      # 未记录，跳过
    ]
    result = aggregate_rows(rows)
    assert result["risk"]["consistency"]["comparable"] == 2
    assert result["risk"]["consistency"]["matched"] == 1
    assert result["risk"]["consistency"]["rate"] == 0.5
    assert result["risk"]["dify"]["distribution"] == {"HIGH": 2, "MEDIUM": 1}
    assert result["risk"]["dify"]["flagged"] == 3


def test_aggregate_rows_latency_percentiles():
    """各段耗时分别统计 P50 / P95，并给出样本数。"""
    rows = [
        _row(timings={"e2e_seconds": 1.0, "asr_seconds": 0.4}),
        _row(timings={"e2e_seconds": 2.0, "asr_seconds": 0.6}),
        _row(timings={"e2e_seconds": 3.0, "asr_seconds": 0.8}),
        _row(timings={"e2e_seconds": "bad-value"}),  # 非数值应被忽略
        _row(timings=None),
    ]
    result = aggregate_rows(rows)
    e2e = result["latency"]["metrics"]["e2e_seconds"]
    assert e2e["samples"] == 3
    assert e2e["p50"] == 2.0
    assert e2e["p95"] == round(2.9, 4)
    assert result["latency"]["metrics"]["asr_seconds"]["samples"] == 3
    assert result["latency"]["metrics"]["llm_total_seconds"]["samples"] == 0
    assert result["latency"]["sampled"] == 3


def test_latency_keys_cover_pipeline_stages():
    """耗时字段口径需覆盖端到端与各分段，供报告直接引用。"""
    assert set(LATENCY_KEYS) == {
        "asr_seconds",
        "multimodal_seconds",
        "llm_first_token_seconds",
        "llm_total_seconds",
        "tts_first_audio_seconds",
        "e2e_seconds",
    }
