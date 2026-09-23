"""多模态融合的置信度校准（温度缩放）。

为什么需要
----------
``fusion_service`` 原先直接输出"融合后的最大概率"作为置信度。多分类概率的
最大分量并不等于"判对的概率"，因此该数值既不等于真实正确率，也不能直接拿来
做阈值判断。实测（``experiments/run_calibration.py``）显示未校准置信度
显著低于实际准确率，ECE 约 0.42。

做法
----
温度缩放（temperature scaling）是最小侵入的校准方法：只有一个参数 T，
在标注数据上以最小化 NLL 拟合。

    p'_i = softmax( log(p_i) / T )

T = 1 等价于不校准；T < 1 锐化分布（修正欠自信）；T > 1 平滑分布（修正过度自信）。
温度缩放是**单调变换**，不改变 argmax，因此识别结果与准确率完全不变，
校准只让置信度变得可用。

参数来源与可追溯性
------------------
本模块**默认不启用**（``FUSION_CALIBRATION_ENABLED=false``，T=1.0），
因为标定参数必须来自真实标注数据。开启方式::

    FUSION_CALIBRATION_ENABLED=true
    FUSION_TEMPERATURE=0.85
    FUSION_CALIBRATION_SOURCE=2026-10 校园体验标注集 n=180 双人标注 kappa=0.81

``FUSION_CALIBRATION_SOURCE`` 会随融合结果一并返回并写入留痕表，
使"这个温度从哪来"始终可复核，这是技术报告与答辩现场最容易被追问的一点。
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_log = logging.getLogger("ai-lab.calibration")

#: 概率下界，避免 log(0)
_EPS = 1e-12

#: 温度候选网格（对数等距，0.01-20）：供标定脚本与单测复用
TEMPERATURE_GRID: tuple[float, ...] = tuple(
    10 ** (math.log10(0.01) + i * (math.log10(20.0) - math.log10(0.01)) / 399)
    for i in range(400)
)

#: 标定温度取值下限（低于该值不再具有实际意义）
MIN_TEMPERATURE = 0.01


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """一次校准的配置快照，随结果一并返回以便追溯。"""

    enabled: bool
    temperature: float
    method: str = "temperature_scaling"
    source: str = ""


def load_calibration_config() -> CalibrationConfig:
    """从环境变量读取校准配置。

    每次调用都重新读取，便于测试与运行期调整；默认关闭，即保持原始行为。
    """
    enabled = os.environ.get("FUSION_CALIBRATION_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    raw_temperature = os.environ.get("FUSION_TEMPERATURE", "1.0").strip()
    try:
        temperature = float(raw_temperature)
    except ValueError:
        _log.warning("FUSION_TEMPERATURE=%r 不是合法数值，回退为 1.0（不校准）", raw_temperature)
        temperature = 1.0
    if not math.isfinite(temperature) or temperature < MIN_TEMPERATURE:
        _log.warning(
            "FUSION_TEMPERATURE=%.4f 超出有效范围 [%.2f, +inf)，回退为 1.0（不校准）",
            temperature, MIN_TEMPERATURE,
        )
        temperature = 1.0

    return CalibrationConfig(
        enabled=enabled,
        temperature=temperature,
        source=os.environ.get("FUSION_CALIBRATION_SOURCE", "").strip(),
    )


def _softmax(values: Sequence[float]) -> list[float]:
    """数值稳定的 softmax。"""
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def apply_temperature(
    distribution: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    """对概率分布做温度缩放，返回归一化后的新分布。

    只有概率没有 logits 时，标准做法是取 ``log(p)`` 作为 logit 的等价形式：
    ``softmax(log(p) / T)`` 在 T=1 时还原原始分布。

    Args:
        distribution: 类别到概率的映射。全零或非有限值按均匀分布处理。
        temperature: 温度，必须为正；非法取值按不校准处理。
    """
    labels = list(distribution.keys())
    if not labels:
        return {}

    if not math.isfinite(temperature) or temperature < MIN_TEMPERATURE:
        temperature = 1.0
    if abs(temperature - 1.0) < 1e-12:
        return {label: float(distribution[label]) for label in labels}

    raw = [float(distribution[label]) for label in labels]
    if not any(value > 0 for value in raw):
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}

    logits = [math.log(max(value, _EPS)) / temperature for value in raw]
    scaled = _softmax(logits)
    total = sum(scaled)
    if not math.isfinite(total) or total <= 0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {label: value / total for label, value in zip(labels, scaled, strict=False)}


def calibrate(
    distribution: Mapping[str, float],
    *,
    config: CalibrationConfig | None = None,
    temperature: float | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """按配置校准分布，返回 ``(校准后分布, 元信息)``。

    Args:
        distribution: 原始类别概率分布。
        config: 校准配置；缺省读取环境变量。
        temperature: 显式指定温度，优先于 ``config``（消融实验与敏感性分析使用）。
    """
    active = config or load_calibration_config()
    use_temperature = active.temperature if temperature is None else float(temperature)
    enabled = active.enabled or temperature is not None

    if not enabled:
        meta = {
            "enabled": False,
            "method": active.method,
            "temperature": 1.0,
            "source": active.source,
        }
        return {label: float(value) for label, value in distribution.items()}, meta

    calibrated = apply_temperature(distribution, use_temperature)
    meta = {
        "enabled": True,
        "method": active.method,
        "temperature": round(use_temperature, 6),
        "source": active.source,
    }
    return calibrated, meta


def negative_log_likelihood(
    distributions: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
) -> float:
    """多分类 NLL（标定目标函数，越低越好）。"""
    if not distributions:
        return 0.0
    total = 0.0
    for distribution, gold in zip(distributions, gold_labels, strict=False):
        total -= math.log(max(float(distribution.get(gold, 0.0)), _EPS))
    return total / len(distributions)


def fit_temperature(
    distributions: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
    grid: Sequence[float] = TEMPERATURE_GRID,
) -> tuple[float, float]:
    """在标注数据上以最小化 NLL 为目标搜索温度。

    这是线上线下共用的标定实现：``experiments/run_calibration.py`` 调用它产出
    写入 ``FUSION_TEMPERATURE`` 的取值，保证"报告里的数字"与"线上行为"同源。

    Returns:
        ``(最优温度, 该温度下的 NLL)``
    """
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in grid:
        scaled = [apply_temperature(dist, temperature) for dist in distributions]
        value = negative_log_likelihood(scaled, gold_labels)
        if value < best_nll:
            best_temperature, best_nll = temperature, value
    return best_temperature, best_nll


__all__ = [
    "CalibrationConfig",
    "MIN_TEMPERATURE",
    "TEMPERATURE_GRID",
    "apply_temperature",
    "calibrate",
    "fit_temperature",
    "load_calibration_config",
    "negative_log_likelihood",
]
