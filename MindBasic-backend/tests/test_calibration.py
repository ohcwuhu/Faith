"""置信度校准测试：保证温度缩放正确、默认不改变线上行为。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

import math
import os
from contextlib import contextmanager

from app.services.ai_lab import calibration, facial_buffer, fusion_service

_SID = "test-calibration-sid"
_T0 = 1_700_000_000.0


@contextmanager
def _env(**values):
    """临时设置环境变量并在退出时还原。"""
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _distribution(label: str, confidence: float) -> dict[str, float]:
    probs = {name: 0.0 for name in fusion_service.UNIFIED_LABELS}
    probs[label] = confidence
    rest = (1.0 - confidence) / (len(fusion_service.UNIFIED_LABELS) - 1)
    for name in fusion_service.UNIFIED_LABELS:
        if name != label:
            probs[name] = rest
    return probs


# ============================================================
#  基础数学性质
# ============================================================
def test_temperature_one_is_identity():
    """T=1 等价于不校准，分布逐项不变。"""
    source = _distribution("sad", 0.42)
    result = calibration.apply_temperature(source, 1.0)
    assert set(result) == set(source)
    for label in source:
        assert abs(result[label] - source[label]) < 1e-12


def test_sharpening_increases_peak_and_preserves_order():
    """T<1 锐化：最大概率上升，类别顺序不变，分布仍归一。"""
    source = _distribution("sad", 0.42)
    result = calibration.apply_temperature(source, 0.2)

    assert result["sad"] > source["sad"]
    assert abs(sum(result.values()) - 1.0) < 1e-9
    # 单调变换：排序与 argmax 均不变
    assert max(result, key=result.get) == max(source, key=source.get)
    assert sorted(result, key=result.get) == sorted(source, key=source.get)


def test_smoothing_decreases_peak():
    """T>1 平滑：最大概率下降，向均匀分布靠拢。"""
    source = _distribution("sad", 0.6)
    result = calibration.apply_temperature(source, 5.0)

    assert result["sad"] < source["sad"]
    assert result["sad"] > 1.0 / len(source)  # 仍高于均匀分布
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_uniform_distribution_stays_uniform():
    """均匀分布在任何温度下都不变。"""
    uniform = {name: 1.0 / len(fusion_service.UNIFIED_LABELS) for name in fusion_service.UNIFIED_LABELS}
    for temperature in (0.1, 0.5, 1.0, 3.0, 10.0):
        result = calibration.apply_temperature(uniform, temperature)
        for label in uniform:
            assert abs(result[label] - uniform[label]) < 1e-9


def test_all_zero_distribution_falls_back_to_uniform():
    """全零分布无法取对数，按均匀分布处理而不是崩溃。"""
    zero = {name: 0.0 for name in fusion_service.UNIFIED_LABELS}
    result = calibration.apply_temperature(zero, 0.3)
    assert abs(sum(result.values()) - 1.0) < 1e-9
    for value in result.values():
        assert abs(value - 1.0 / len(zero)) < 1e-9


def test_invalid_temperature_is_noop():
    """非法温度（0/负数/NaN）回退为不校准，避免把分布打坏。"""
    source = _distribution("sad", 0.42)
    for temperature in (0.0, -1.0, float("nan"), 0.001):
        result = calibration.apply_temperature(source, temperature)
        for label in source:
            assert abs(result[label] - source[label]) < 1e-9


def test_empty_distribution_returns_empty():
    """空分布直接返回空，不抛异常。"""
    assert calibration.apply_temperature({}, 0.5) == {}


# ============================================================
#  配置读取
# ============================================================
def test_default_config_is_disabled():
    """默认关闭校准：写入格式规定的开关未打开时保持原始行为。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None,
              FUSION_CALIBRATION_SOURCE=None):
        config = calibration.load_calibration_config()
    assert config.enabled is False
    assert config.temperature == 1.0


def test_config_reads_env():
    """开启后按环境变量读取温度与来源说明。"""
    with _env(FUSION_CALIBRATION_ENABLED="true", FUSION_TEMPERATURE="0.85",
              FUSION_CALIBRATION_SOURCE="test-set n=180"):
        config = calibration.load_calibration_config()
    assert config.enabled is True
    assert abs(config.temperature - 0.85) < 1e-9
    assert config.source == "test-set n=180"


def test_invalid_env_temperature_falls_back_to_one():
    """环境变量写错时回退到 1.0，不让服务因配置错误而异常。"""
    with _env(FUSION_CALIBRATION_ENABLED="true", FUSION_TEMPERATURE="abc"):
        config = calibration.load_calibration_config()
    assert config.temperature == 1.0

    with _env(FUSION_CALIBRATION_ENABLED="true", FUSION_TEMPERATURE="-3"):
        config = calibration.load_calibration_config()
    assert config.temperature == 1.0


def test_calibrate_disabled_returns_original_distribution():
    """校准关闭时返回原分布，并说明当前未启用。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None):
        result, meta = calibration.calibrate(_distribution("sad", 0.42))
    assert meta["enabled"] is False
    assert meta["temperature"] == 1.0
    assert abs(result["sad"] - 0.42) < 1e-9


def test_explicit_temperature_overrides_config():
    """显式传入温度时应生效，供消融与敏感性分析使用。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None):
        result, meta = calibration.calibrate(_distribution("sad", 0.42), temperature=0.1)
    assert meta["enabled"] is True
    assert abs(meta["temperature"] - 0.1) < 1e-6
    assert result["sad"] > 0.42


# ============================================================
#  温度拟合
# ============================================================
def test_fit_temperature_recovers_sharpening():
    """对被人为锐化过的分布，标定应给出明显小于 1 的温度。"""
    base = _distribution("sad", 0.30)
    sharpened = calibration.apply_temperature(base, 0.25)
    distributions = [dict(sharpened) for _ in range(12)]

    temperature, nll = calibration.fit_temperature(distributions, ["sad"] * 12)
    assert temperature < 1.0
    assert math.isfinite(nll)


def test_fit_temperature_on_uniform_data_does_not_crash():
    """无法区分的输入不应导致异常。"""
    uniform = {name: 1.0 / len(fusion_service.UNIFIED_LABELS) for name in fusion_service.UNIFIED_LABELS}
    temperature, nll = calibration.fit_temperature([dict(uniform)] * 5, ["sad"] * 5)
    assert math.isfinite(temperature)
    assert math.isfinite(nll)


# ============================================================
#  融合服务集成
# ============================================================
def _fuse(**kwargs):
    """构造固定三模态输入并调用融合引擎。"""
    facial_buffer.init_client(_SID)
    probs = _distribution("sad", 0.7)
    for index in range(12):
        facial_buffer.append_frame(
            _SID,
            emotions={},
            score=55,
            raw_probs=probs,
            server_ts=_T0 + index * 0.4,
        )
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
        record_end_ts=_T0 + 12 * 0.4,
        **kwargs,
    )


def test_fusion_output_exposes_calibration_fields():
    """融合结果需同时给出校准前后分布与置信度，便于留痕与复核。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None):
        result = _fuse()
    fusion = result["fusion"]

    for key in ("overall_confidence", "raw_confidence", "probabilities",
                "probabilities_raw", "calibration"):
        assert key in fusion, f"缺少字段 {key}"
    assert fusion["calibration"]["enabled"] is False
    # 未校准时两者应一致
    assert abs(fusion["overall_confidence"] - fusion["raw_confidence"]) < 1e-9


def test_fusion_calibration_preserves_argmax():
    """校准不改变最终情绪判定，只改变置信度。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None):
        raw = _fuse(temperature=1.0)["fusion"]
        calibrated = _fuse(temperature=0.2)["fusion"]

    assert raw["final_emotion"] == calibrated["final_emotion"]
    assert calibrated["overall_confidence"] > raw["overall_confidence"]
    assert calibrated["calibration"]["enabled"] is True
    assert abs(calibrated["calibration"]["temperature"] - 0.2) < 1e-6
    # 原始分布保持可追溯
    assert abs(raw["probabilities_raw"]["sad"] - calibrated["probabilities_raw"]["sad"]) < 1e-9


def test_fusion_default_matches_uncalibrated_behaviour():
    """默认配置下线上行为与历史一致（overall_confidence 仍是原始最大概率）。"""
    with _env(FUSION_CALIBRATION_ENABLED=None, FUSION_TEMPERATURE=None):
        fusion = _fuse()["fusion"]
    assert abs(fusion["overall_confidence"] - fusion["probabilities_raw"][fusion["final_emotion"]]) < 0.01
