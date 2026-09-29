"""
Dify 智能体客户端（AI 实验室）
==============================

与 Dify 对话型应用（Chatflow / Agent）交互的公共逻辑，供视频通话管线调用：

- ``fetch_input_types()``：读取并缓存应用的入参声明（``GET /parameters``）；
- ``normalize_inputs()``：按声明把平台侧的值转换成 Dify 期望的类型；
- ``is_enabled()`` / ``probe()``：连通性判断与自检，供 ``scripts/check_dify.py`` 排障。

为什么要做入参类型转换
----------------------

Dify 服务 API 会逐个校验入参类型，类型不符时**整个工作流一步都不跑**，直接把错误
原样返回给调用方::

    (type 'text-input') goal_clear in input form must be a string

平台侧 ``goal_clear`` / ``action_ready`` / ``should_summarize_hint`` /
``modality_conflict`` 语义上是布尔，Python 的 ``True`` 序列化成 JSON ``true``；
如果工作流把它们声明成文本输入，请求就必然被拒。反过来，若工作流声明成布尔
（checkbox），传字符串同样会被拒。所以这里以**工作流实际声明**为准来转换，两边都能跑。

另有一个容易踩的细节：直接 ``str(True)`` 得到 ``"True"``（首字母大写），与工作流里
判断的 ``"true"`` 对不上，必须显式小写。
"""

from __future__ import annotations

import logging
import os
import threading
import time

import requests

from app.services.ai_lab import config as _cfg

log = logging.getLogger(__name__)

# 入参声明缓存：工作流改动不频繁，没必要每轮通话都查一次
_PARAMS_TTL_SECONDS = 300
_cache: dict[str, object] = {"types": None, "fetched_at": 0.0}
_lock = threading.Lock()

_BOOL_TYPES = {"checkbox", "boolean"}
_NUMBER_TYPES = {"number"}

_TRUE_WORDS = {"true", "1", "yes", "y", "on", "是", "开启", "打开", "真"}
_FALSE_WORDS = {"false", "0", "no", "n", "off", "否", "关闭", "关", "假", ""}


def api_base() -> str:
    """Dify 服务 API 根地址（去掉尾部斜杠）。"""
    return (_cfg.DIFY_API_BASE or "").rstrip("/")


def api_key() -> str:
    return (_cfg.DIFY_API_KEY or "").strip()


def is_enabled() -> bool:
    """是否配置了 Dify 应用密钥。未配置时调用方应回退到别的 LLM。"""
    return bool(api_key())


def fetch_parameters(timeout: int = 10) -> dict:
    """读取应用的入参声明与能力开关（``GET /parameters``）。

    返回原始 JSON；调用失败时抛异常，由调用方决定降级方式。
    """
    resp = requests.get(
        f"{api_base()}/parameters",
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_input_types(parameters: dict) -> dict[str, str]:
    """把 ``/parameters`` 的 ``user_input_form`` 压成 ``{变量名: 类型}``。"""
    types: dict[str, str] = {}
    for form in parameters.get("user_input_form") or []:
        if not isinstance(form, dict):
            continue
        for form_type, item in form.items():
            if isinstance(item, dict) and item.get("variable"):
                types[item["variable"]] = form_type
    return types


def fetch_input_types(*, force: bool = False, timeout: int = 10) -> dict[str, str]:
    """带缓存的入参声明查询；失败时返回上一次的结果（可能是空字典）。"""
    now = time.time()
    with _lock:
        cached = _cache.get("types")
        fresh = (now - float(_cache.get("fetched_at") or 0)) < _PARAMS_TTL_SECONDS
        if cached is not None and fresh and not force:
            return dict(cached)  # type: ignore[arg-type]
    try:
        types = _extract_input_types(fetch_parameters(timeout=timeout))
    except Exception as exc:
        log.warning("Dify 入参声明查询失败，沿用缓存（可能为空）：%s", exc)
        with _lock:
            return dict(_cache.get("types") or {})  # type: ignore[arg-type]
    with _lock:
        _cache["types"] = types
        _cache["fetched_at"] = time.time()
    return dict(types)


def reset_cache() -> None:
    """清空入参声明缓存（测试用；工作流改了变量类型后可调用）。"""
    with _lock:
        _cache["types"] = None
        _cache["fetched_at"] = 0.0


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return bool(text)


def normalize_value(value: object, declared_type: str | None) -> object:
    """把单个值转换成 ``declared_type`` 要求的类型。

    ``declared_type`` 为 ``None``（工作流未声明该变量）时按文本处理——Dify 会忽略
    未声明的入参，转换只是为了不触发表单校验。
    """
    if declared_type in _BOOL_TYPES:
        return _to_bool(value)
    if declared_type in _NUMBER_TYPES:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return 0
    # text-input / paragraph / select / 未声明：一律转文本
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        # 0.0 -> "0.0" 看着别扭，且与工作流里的数字字面量不易比对
        return str(int(value))
    return value if isinstance(value, str) else str(value)


def normalize_inputs(
    inputs: dict[str, object],
    *,
    input_types: dict[str, str] | None = None,
) -> dict[str, object]:
    """按工作流声明的类型批量转换入参。

    传入 ``input_types`` 可跳过网络查询（测试用）；否则用带缓存的 ``/parameters``。
    """
    types = fetch_input_types() if input_types is None else input_types
    return {key: normalize_value(value, types.get(key)) for key, value in inputs.items()}


def probe() -> dict:
    """连通性自检：返回入参声明与能力开关，不发起对话。"""
    parameters = fetch_parameters()
    return {
        "api_base": api_base(),
        "key_prefix": (api_key()[:4] + "***") if api_key() else "",
        "declared_inputs": _extract_input_types(parameters),
        "retriever_enabled": bool((parameters.get("retriever_resource") or {}).get("enabled")),
    }


# ─── 熔断 ────────────────────────────────────────────────────────────────────
# Dify 工作流本身可能不稳定（例如结构化输出解析失败导致整条链路 400）。
# 连续失败达到阈值就先冷却一段时间，避免每一轮通话都先白等一次 Dify，
# 冷却结束后自动放行重试；任意一次成功立刻复位。
_CIRCUIT_THRESHOLD = int(os.environ.get("DIFY_CIRCUIT_THRESHOLD", "3"))
_CIRCUIT_COOLDOWN = float(os.environ.get("DIFY_CIRCUIT_COOLDOWN", "120"))

_circuit_lock = threading.Lock()
_consecutive_failures = 0
_circuit_open_until = 0.0


def circuit_open() -> bool:
    """熔断是否处于打开状态（打开期间应跳过 Dify 直接走备用供应商）。"""
    with _circuit_lock:
        return time.time() < _circuit_open_until


def circuit_reason() -> str:
    with _circuit_lock:
        if time.time() >= _circuit_open_until:
            return ""
        return f"连续失败 {_consecutive_failures} 次，冷却 {int(_circuit_open_until - time.time())}s"


def record_success() -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        if _consecutive_failures or _circuit_open_until:
            log.info("Dify 恢复正常，熔断复位")
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def record_failure() -> None:
    """记录一次整轮无产出的失败；达到阈值则打开熔断。"""
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures += 1
        if _CIRCUIT_THRESHOLD > 0 and _consecutive_failures >= _CIRCUIT_THRESHOLD:
            _circuit_open_until = time.time() + _CIRCUIT_COOLDOWN
            log.warning(
                "Dify 连续失败 %d 次，熔断 %.0fs（期间自动走备用供应商）",
                _consecutive_failures, _CIRCUIT_COOLDOWN,
            )


def reset_circuit() -> None:
    """手动复位熔断（测试 / 排障用）。"""
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0
