"""Dify 入参类型转换测试。

回归背景：Dify 服务 API 会逐个校验入参类型，声明成 ``text-input`` 的变量收到
JSON 布尔值会直接拒绝整次请求（``(type 'text-input') xxx must be a string``），
工作流一步都不跑。这里锁定转换规则，避免改动后重新踩坑。
"""

from app.services.ai_lab import dify_service


def test_bool_to_text_is_lowercase():
    """布尔值转文本时必须小写，工作流里判断的是 "true" / "false"。"""
    assert dify_service.normalize_value(True, "text-input") == "true"
    assert dify_service.normalize_value(False, "text-input") == "false"
    # 未声明的变量同样按文本处理，且不能出现 Python 的 "True" / "False"
    assert dify_service.normalize_value(True, None) == "true"


def test_checkbox_keeps_bool():
    """工作流声明成布尔时，字符串要还原成真布尔，否则会被 Dify 拒收。"""
    assert dify_service.normalize_value("true", "checkbox") is True
    assert dify_service.normalize_value("false", "checkbox") is False
    assert dify_service.normalize_value("开启", "boolean") is True
    assert dify_service.normalize_value("关闭", "boolean") is False
    assert dify_service.normalize_value(True, "checkbox") is True


def test_number_and_text_coercion():
    assert dify_service.normalize_value(0.66, "number") == 0.66
    assert dify_service.normalize_value("0.66", "number") == 0.66
    assert dify_service.normalize_value(True, "number") == 1
    # 数字声明成文本时不能把 0.0 发成 "0.0"（与工作流里的字面量对不上）
    assert dify_service.normalize_value(0.0, "text-input") == "0"
    assert dify_service.normalize_value(0.6, "text-input") == "0.6"
    assert dify_service.normalize_value(None, "text-input") == ""
    assert dify_service.normalize_value("焦虑", "text-input") == "焦虑"


def test_normalize_inputs_uses_declared_types():
    declared = {
        "goal_clear": "text-input",
        "action_ready": "checkbox",
        "live_score": "number",
        "user_utterance": "paragraph",
    }
    raw = {
        "goal_clear": False,
        "action_ready": True,
        "live_score": 0.6,
        "user_utterance": "我最近总是睡不着",
    }
    assert dify_service.normalize_inputs(raw, input_types=declared) == {
        "goal_clear": "false",
        "action_ready": True,
        "live_score": 0.6,
        "user_utterance": "我最近总是睡不着",
    }


def test_extract_input_types_reads_user_input_form():
    parameters = {
        "user_input_form": [
            {"text-input": {"variable": "current_stage", "type": "text-input"}},
            {"checkbox": {"variable": "goal_clear", "type": "checkbox"}},
            {"number": {"variable": "live_score", "type": "number"}},
            {"paragraph": {"variable": "asr_text", "type": "paragraph"}},
        ]
    }
    assert dify_service._extract_input_types(parameters) == {
        "current_stage": "text-input",
        "goal_clear": "checkbox",
        "live_score": "number",
        "asr_text": "paragraph",
    }


def test_is_enabled_follows_api_key(monkeypatch):
    monkeypatch.setattr(dify_service, "api_key", lambda: "")
    assert dify_service.is_enabled() is False
    monkeypatch.setattr(dify_service, "api_key", lambda: "app-xxx")
    assert dify_service.is_enabled() is True


def test_circuit_opens_after_consecutive_failures():
    """连续失败到阈值就熔断，避免每轮通话都先白等一次 Dify。"""
    dify_service.reset_circuit()
    assert dify_service.circuit_open() is False
    for _ in range(dify_service._CIRCUIT_THRESHOLD - 1):
        dify_service.record_failure()
    assert dify_service.circuit_open() is False
    dify_service.record_failure()
    assert dify_service.circuit_open() is True
    assert "连续失败" in dify_service.circuit_reason()


def test_circuit_resets_on_success():
    dify_service.reset_circuit()
    for _ in range(dify_service._CIRCUIT_THRESHOLD):
        dify_service.record_failure()
    dify_service.record_success()
    assert dify_service.circuit_open() is False
    assert dify_service.circuit_reason() == ""
