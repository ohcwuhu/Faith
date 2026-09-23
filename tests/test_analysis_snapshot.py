"""分析留痕快照测试：保证接口响应与入库结构一致。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

from app.services.analysis_record_service import (
    SOURCE_HTTP_ANALYZE,
    SOURCE_VIDEO_CALL,
    AnalysisSnapshot,
)

_HTTP_BODY = {
    "status": "ok",
    "transcription": {"text": "今天有点累", "language": "zh", "duration_seconds": 3.2},
    "text_emotion": {"emotion": "sad", "confidence": 0.71, "probabilities": {}},
    "voice_emotion": {
        "emotion": "sad",
        "confidence": 0.66,
        "sv_cross_check": {"emotion": "sad", "agree": True, "source": "SenseVoice emo"},
    },
    "facial_emotion": {"dominant_emotion": "neutral", "confidence": 0.6, "frame_count": 9},
    "fusion": {
        "final_emotion": "sad",
        "overall_confidence": 0.64,
        "weights_used": {"text": 0.6, "voice": 0.4, "facial": 0.0},
        "weight_adjustments": ["no_facial_frames: facial=0"],
    },
    "timings": {"total_seconds": 4.1, "asr_seconds": 1.2},
    "server_info": {"sid": "sid-1"},
    "risk": {"level": "LOW", "riskScore": 10, "reasons": ["语音与面部信号一致指向持续负性情绪"]},
}


def test_snapshot_from_http_response_maps_all_fields():
    """HTTP 响应体应被完整映射为入库快照。"""
    snapshot = AnalysisSnapshot.from_http_response(_HTTP_BODY, user_id=7)
    assert snapshot.source == SOURCE_HTTP_ANALYZE
    assert snapshot.user_id == 7
    assert snapshot.session_id == "sid-1"
    assert snapshot.asr_text == "今天有点累"
    assert snapshot.text_emotion == "sad"
    assert snapshot.voice_emotion == "sad"
    assert snapshot.facial_frames == 9
    assert snapshot.fusion_emotion == "sad"
    assert snapshot.weights == {"text": 0.6, "voice": 0.4, "facial": 0.0}
    assert snapshot.risk_level == "LOW"
    assert snapshot.timings["asr_seconds"] == 1.2


def test_snapshot_handles_missing_sections():
    """缺失字段不应导致异常（部分失败的分析仍要留痕）。"""
    snapshot = AnalysisSnapshot.from_http_response({"status": "failed"}, user_id=None)
    assert snapshot.user_id is None
    assert snapshot.text_emotion is None
    assert snapshot.facial_frames is None
    assert snapshot.timings == {}


def test_row_conversion_truncates_long_text():
    """ASR 文本入库需截断，避免超长字段写入失败。"""
    snapshot = AnalysisSnapshot.from_video_call(
        user_id=3,
        session_id="sid-2",
        asr_text="测试" * 2000,
        status="ok",
    )
    row = snapshot.to_row()
    assert snapshot.source == SOURCE_VIDEO_CALL
    assert row["asr_text"] is not None
    assert len(row["asr_text"]) <= 2000
    assert row["user_id"] == 3


def test_empty_snapshot_row_has_no_asr_text():
    """空文本快照应写入 NULL，便于统计时排除空轮次。"""
    snapshot = AnalysisSnapshot.from_video_call(user_id=1, session_id="s", asr_text="")
    assert snapshot.to_row()["asr_text"] is None


def test_video_call_snapshot_carries_stage_and_session_link():
    """实时管线快照需带五阶段判定与所属会话，供阶段一致性与成效统计。"""
    snapshot = AnalysisSnapshot.from_video_call(
        user_id=5,
        session_id="sid-stage",
        asr_text="我希望先把方向定下来",
        stage={
            "stage": "goal_setting",
            "goal_clear": True,
            "action_ready": False,
            "should_summarize": False,
            "summary_reason": "核心需求逐渐清楚",
        },
        conversation_id=42,
        timings={"asr_seconds": 0.8, "e2e_seconds": 5.2},
        status="ok",
    )
    row = snapshot.to_row()
    assert row["coach_stage"] == "goal_setting"
    assert row["goal_clear"] is True
    assert row["action_ready"] is False
    assert row["should_summarize"] is False
    assert row["conversation_id"] == 42
    assert row["timings"]["e2e_seconds"] == 5.2


def test_snapshot_keeps_unknown_stage_as_none():
    """没有阶段判定时保持 NULL，避免把"未判定"写成 False 污染统计。"""
    snapshot = AnalysisSnapshot.from_video_call(
        user_id=None, session_id="sid-none", asr_text="随便聊聊",
    )
    row = snapshot.to_row()
    assert row["coach_stage"] is None
    assert row["goal_clear"] is None
    assert row["dify_risk_level"] is None
