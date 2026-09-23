"""融合权重入口测试：保证消融实验与线上使用同一条代码路径。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

from app.services.ai_lab import facial_buffer, fusion_service

_SID = "test-fusion-sid"
_T0 = 1_700_000_000.0


def _distribution(label: str, confidence: float) -> dict[str, float]:
    probs = {name: 0.0 for name in fusion_service.UNIFIED_LABELS}
    probs[label] = confidence
    rest = (1.0 - confidence) / (len(fusion_service.UNIFIED_LABELS) - 1)
    for name in fusion_service.UNIFIED_LABELS:
        if name != label:
            probs[name] = rest
    return probs


def _fuse(weights, *, facial_label="sad", facial_confidence=0.7, frames=12):
    """构造固定的三模态输入并调用融合引擎。"""
    facial_buffer.init_client(_SID)
    probs = _distribution(facial_label, facial_confidence)
    for index in range(frames):
        facial_buffer.append_frame(
            _SID,
            emotions={},
            score=55,
            raw_probs=probs,
            server_ts=_T0 + index * 0.4,
        )
    end_ts = _T0 + max(frames, 1) * 0.4
    return fusion_service.fuse(
        text_result={
            "text": "我最近状态还行",
            "emotion": "neutral",
            "confidence": 0.6,
            "probabilities": _distribution("neutral", 0.6),
        },
        voice_result={
            "emotion": "sad",
            "confidence": 0.6,
            "probabilities": _distribution("sad", 0.6),
        },
        sv_emo_result={"emotion": "neutral"},
        sid=_SID,
        record_start_ts=_T0,
        record_end_ts=end_ts,
        weights_override=weights,
    )


def test_weights_override_is_normalized():
    """指定权重会被归一化，且结果与基础权重一致。"""
    result = _fuse({"text": 0.40, "voice": 0.35, "facial": 0.25})
    weights = result["fusion"]["weights_used"]
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert weights["text"] == 0.4
    assert weights["voice"] == 0.35
    assert weights["facial"] == 0.25


def test_single_modality_arms_use_only_that_modality():
    """单模态基线：只有被选中的模态参与融合。"""
    text_only = _fuse({"text": 1.0, "voice": 0.0, "facial": 0.0})
    assert text_only["fusion"]["final_emotion"] == "neutral"

    voice_only = _fuse({"text": 0.0, "voice": 1.0, "facial": 0.0})
    assert voice_only["fusion"]["final_emotion"] == "sad"

    facial_only = _fuse({"text": 0.0, "voice": 0.0, "facial": 1.0})
    assert facial_only["fusion"]["final_emotion"] == "sad"


def test_all_zero_override_falls_back_to_base_weights():
    """全部为 0 的无效权重回退到基础权重，避免产生空融合。"""
    result = _fuse({"text": 0.0, "voice": 0.0, "facial": 0.0})
    weights = result["fusion"]["weights_used"]
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert result["fusion"]["final_emotion"] in fusion_service.UNIFIED_LABELS


def test_missing_facial_modality_is_reported():
    """无面部帧时按无视觉模态处理，流程不中断。"""
    result = _fuse(None, frames=0)
    facial = result["facial_emotion"]
    assert facial["frame_count"] == 0
    assert result["fusion"]["final_emotion"] in fusion_service.UNIFIED_LABELS
