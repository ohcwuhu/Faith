"""线索冲突检测测试：让"冲突时先澄清"成为可复核的行为。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

from app.services.ai_lab import facial_buffer, fusion_service

_SID = "test-conflict-sid"
_T0 = 1_700_000_000.0


def _distribution(label: str, confidence: float) -> dict[str, float]:
    """构造一个以 ``label`` 为主、其余标签均分的概率分布。"""
    probs = {name: 0.0 for name in fusion_service.UNIFIED_LABELS}
    probs[label] = confidence
    rest = (1.0 - confidence) / (len(fusion_service.UNIFIED_LABELS) - 1)
    for name in fusion_service.UNIFIED_LABELS:
        if name != label:
            probs[name] = rest
    return probs


def _fuse(*, text_label, text_conf, voice_label, voice_conf, facial_label, facial_conf,
          frames=12, include_facial=True):
    """构造三模态输入并调用融合引擎。"""
    facial_buffer.init_client(_SID)
    probs = _distribution(facial_label, facial_conf)
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
            "text": "我今天把事情做完了一部分",
            "emotion": text_label,
            "confidence": text_conf,
            "probabilities": _distribution(text_label, text_conf),
        },
        voice_result={
            "emotion": voice_label,
            "confidence": voice_conf,
            "probabilities": _distribution(voice_label, voice_conf),
        },
        sv_emo_result={"emotion": voice_label},
        sid=_SID,
        record_start_ts=_T0,
        record_end_ts=end_ts,
        include_facial=include_facial,
    )


def test_consistent_modalities_report_low_conflict():
    """三个模态都指向同一情绪时，冲突分应接近 0，且不建议澄清。"""
    result = _fuse(
        text_label="sad", text_conf=0.7,
        voice_label="sad", voice_conf=0.7,
        facial_label="sad", facial_conf=0.7,
    )
    conflict = result["fusion"]["conflict"]
    assert conflict["score"] < fusion_service._CONFLICT_LOW
    assert conflict["level"] == "low"
    assert conflict["needs_clarification"] is False
    assert conflict["modal_emotions"]["text"] == "sad"
    assert conflict["reason"] == "各模态判断基本一致"


def test_opposite_modalities_require_clarification():
    """文本说开心、语调与面部说悲伤时，应判为冲突并建议澄清。"""
    result = _fuse(
        text_label="happy", text_conf=0.9,
        voice_label="sad", voice_conf=0.9,
        facial_label="sad", facial_conf=0.9,
    )
    conflict = result["fusion"]["conflict"]
    assert conflict["score"] >= fusion_service._CONFLICT_LOW
    assert conflict["needs_clarification"] is True
    assert conflict["level"] in {"medium", "high"}
    assert "text-voice" in conflict["pairwise"]
    assert "不一致" in conflict["reason"]


def test_conflict_is_bounded_and_deterministic():
    """冲突分恒在 0–1 之间，且同一输入两次调用结果一致。"""
    kwargs = dict(
        text_label="angry", text_conf=0.8,
        voice_label="neutral", voice_conf=0.5,
        facial_label="surprised", facial_conf=0.6,
    )
    first = _fuse(**kwargs)["fusion"]["conflict"]
    second = _fuse(**kwargs)["fusion"]["conflict"]
    assert 0.0 <= first["score"] <= 1.0
    assert first == second


def test_consent_off_excludes_facial_modality():
    """未授权多模态时，面部不参与融合，并留下授权痕证。"""
    result = _fuse(
        text_label="sad", text_conf=0.8,
        voice_label="sad", voice_conf=0.8,
        facial_label="happy", facial_conf=0.95,
        include_facial=False,
    )
    fusion = result["fusion"]
    assert fusion["weights_used"]["facial"] == 0.0
    assert "multimodal_consent_off: facial=0" in fusion["weight_adjustments"]
    assert result["facial_emotion"]["frame_count"] == 0
    assert "facial" not in fusion["conflict"]["modal_emotions"]


def test_single_modality_cannot_judge_conflict():
    """只有一个模态可用时，不应报出冲突（也没法判断）。"""
    facial_buffer.init_client(_SID)
    result = fusion_service.fuse(
        text_result={
            "text": "最近有点累",
            "emotion": "sad",
            "confidence": 0.8,
            "probabilities": _distribution("sad", 0.8),
        },
        voice_result={},
        sv_emo_result={},
        sid=_SID,
        record_start_ts=_T0,
        record_end_ts=_T0 + 1,
        weights_override={"text": 0.6, "voice": 0.4, "facial": 0.0},
    )
    conflict = result["fusion"]["conflict"]
    assert conflict["score"] == 0.0
    assert conflict["needs_clarification"] is False
    assert conflict["reason"] == "可用模态不足，无法判断线索是否冲突"


__all__ = []
