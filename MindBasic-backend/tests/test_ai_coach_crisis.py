"""AI 教练接口的危机处置接线测试。

验证三件事：
1. 命中风险时向模型注入安全处置指令；
2. 命中风险时调用建档服务（此处用替身函数，避免依赖数据库）；
3. 无论是否命中，响应都返回结构化 ``risk`` 字段。

用例只使用 ``assert``，便于在无 pytest 环境下直接调用。
"""

import asyncio
from types import SimpleNamespace
from typing import Any

from app.api.v1 import ai_coach


class _FakeResponse:
    """模拟上游模型返回。"""

    status_code = 200

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "我在听，你愿意多说一点吗？"}}],
            "model": "deepseek-chat",
            "usage": {"total_tokens": 42},
        }


def _user() -> SimpleNamespace:
    return SimpleNamespace(id=1001, nickname="测试用户")


def _run_chat(text: str, monkeypatch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """以替身执行一次教练对话，返回（响应, 记录到的调用信息）。"""
    captured: dict[str, Any] = {"history": None, "flagged": [], "prompts": []}

    def fake_completion(api_key: str, history: list[dict[str, str]]) -> _FakeResponse:
        captured["history"] = history
        return _FakeResponse()

    async def fake_flag(user_id, source, flagged_text, **kwargs):
        captured["flagged"].append((user_id, source, flagged_text, kwargs.get("assessment")))
        return True

    original_completion = ai_coach._request_completion
    original_key = ai_coach.config.DEEPSEEK_API_KEY
    original_flag = ai_coach.crisis_service.flag_crisis_safely
    ai_coach._request_completion = fake_completion
    ai_coach.config.DEEPSEEK_API_KEY = "test-key"
    ai_coach.crisis_service.flag_crisis_safely = fake_flag
    try:
        request = ai_coach.ChatRequest(messages=[ai_coach.ChatMessage(role="user", content=text)])
        response = asyncio.run(ai_coach.chat(request, _user()))
    finally:
        ai_coach._request_completion = original_completion
        ai_coach.config.DEEPSEEK_API_KEY = original_key
        ai_coach.crisis_service.flag_crisis_safely = original_flag

    captured["prompts"] = [m["content"] for m in captured["history"] or []]
    return response, captured


def test_high_risk_message_triggers_flag_and_directive():
    """高风险表达应建档并向模型注入安全处置指令。"""
    response, captured = _run_chat("我想过自杀", {})

    assert captured["flagged"], "未调用危机建档"
    user_id, source, text, assessment = captured["flagged"][0]
    assert user_id == 1001
    assert source == "AI_COACH"
    assert text == "我想过自杀"
    assert assessment is not None and assessment.flagged is True

    assert any(ai_coach.CRISIS_DIRECTIVE in prompt for prompt in captured["prompts"]), "未注入安全指令"
    assert response["risk"]["level"] == "HIGH"
    assert response["risk"]["flagged"] is True
    assert response["reply"]


def test_benign_message_does_not_flag():
    """普通表达不建档，也不注入安全指令，但仍返回 risk 字段。"""
    response, captured = _run_chat("最近在准备考试，有点紧张", {})

    assert captured["flagged"] == []
    assert not any(ai_coach.CRISIS_DIRECTIVE in prompt for prompt in captured["prompts"])
    assert response["risk"]["level"] == "NONE"
    assert response["risk"]["flagged"] is False


def test_medium_risk_message_flags_without_hotline_directive():
    """中风险同样建档，并注入安全指令以便模型先稳定情绪。"""
    response, captured = _run_chat("我很绝望，撑不下去", {})

    assert captured["flagged"], "中风险未建档"
    assert captured["flagged"][0][3].level == "MEDIUM"
    assert any(ai_coach.CRISIS_DIRECTIVE in prompt for prompt in captured["prompts"])
    assert response["risk"]["level"] == "MEDIUM"
