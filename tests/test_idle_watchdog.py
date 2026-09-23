"""长时间无语音的空闲看门狗测试。

背景：VAD 只在"用户已经开口"之后才判静音结束，所以用户一直不出声时，
通话会永远停在聆听态。前端有 75s 的自动收尾，但标签页被浏览器挂起时
前端的定时器不可靠，因此服务端也需要一个兜底（每 60s 扫一次）。

这里只测纯逻辑（不依赖 socketio、不依赖数据库）。
"""

import asyncio
import time

from app.services.ai_lab import realtime_session as rt


def _make_call(sid: str, *, idle_seconds: float, state: str = rt.STATE_LISTENING):
    session = rt.get_session(sid)
    session.state = state
    session.idle_timeout_notified = False
    session.last_activity_at = time.time() - idle_seconds
    return session


def test_idle_calls_only_reports_live_and_silent_sessions():
    """只报"还在通话中且长时间没语音"的会话。"""
    silent = _make_call("idle-silent", idle_seconds=400)
    talking = _make_call("idle-talking", idle_seconds=5)
    closed = _make_call("idle-closed", idle_seconds=400, state=rt.STATE_IDLE)
    try:
        expired = dict(rt.idle_calls(max_idle_seconds=300))
        assert "idle-silent" in expired
        assert expired["idle-silent"] >= 300
        assert "idle-talking" not in expired, "刚说过话的会话不应被判超时"
        assert "idle-closed" not in expired, "已结束的会话不应重复收尾"
    finally:
        for sid in ("idle-silent", "idle-talking", "idle-closed"):
            rt.remove_session(sid)
    assert silent is not None and talking is not None and closed is not None


def test_touch_resets_idle_timer():
    """任何一次语音活动（收到音频、说完一轮、打断、改授权）都会重置计时。"""
    session = _make_call("idle-touch", idle_seconds=400)
    try:
        assert rt.idle_calls(max_idle_seconds=300), "构造的超时会话应被检出"
        session.touch()
        assert not rt.idle_calls(max_idle_seconds=300), "touch 后不应再判超时"
    finally:
        rt.remove_session("idle-touch")


def test_sweep_calls_handler_once_per_session():
    """收尾回调只触发一次，并返回被收尾的 sid；异常不影响其他会话。"""
    _make_call("idle-a", idle_seconds=400)
    _make_call("idle-b", idle_seconds=400)
    _make_call("idle-c", idle_seconds=400)
    called: list[tuple[str, int]] = []

    async def handler(sid: str, idle_seconds: int) -> None:
        called.append((sid, idle_seconds))
        if sid == "idle-b":
            raise RuntimeError("模拟收尾失败")

    rt.set_idle_timeout_handler(handler)
    try:
        ended = asyncio.run(rt.sweep_idle_calls(max_idle_seconds=300))
        assert sorted(ended) == ["idle-a", "idle-c"], ended
        assert sorted(sid for sid, _ in called) == ["idle-a", "idle-b", "idle-c"]
        # 已通知过的会话不会在下一轮重复触发（b 虽然失败也不重复轰炸）
        assert asyncio.run(rt.sweep_idle_calls(max_idle_seconds=300)) == []
        assert len(called) == 3
    finally:
        rt.set_idle_timeout_handler(None)
        for sid in ("idle-a", "idle-b", "idle-c"):
            rt.remove_session(sid)


def test_sweep_without_handler_is_safe():
    """没有注册回调时（例如单测环境）不能抛异常。"""
    _make_call("idle-nohandler", idle_seconds=400)
    try:
        rt.set_idle_timeout_handler(None)
        assert asyncio.run(rt.sweep_idle_calls(max_idle_seconds=300)) == []
    finally:
        rt.remove_session("idle-nohandler")


def test_remove_session_does_not_wipe_inflight_turn():
    """断线清理不能就地清空对象：实时管线还持有它，清空会丢本轮会话关联。

    真实故障现象：用户点结束通话/断线时正好有管线在跑，
    ``conversation_id`` 被清成 None → 助手回复不入库、留痕丢掉会话关联。
    """
    session = rt.get_session("inflight")
    session.conversation_id = 999
    session.turn_index = 3
    session.add_chat_message("user", "我在听")

    rt.remove_session("inflight")

    assert session.conversation_id == 999, "会话对象被就地清空，管线会丢会话关联"
    assert session.turn_index == 3
    assert session.get_chat_history(), "对话历史被清空，管线后半程拿不到上下文"
    assert not rt.has_session("inflight"), "字典里的会话应已移除"
