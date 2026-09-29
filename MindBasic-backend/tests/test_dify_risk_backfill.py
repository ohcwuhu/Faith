"""Dify 自判风险等级的解析与留痕补写测试。

背景：平台侧四级风险与 Dify 工作流三级风险的一致性统计，要求
``multimodal_analysis_records.dify_risk_level`` 有值。该字段由实时管线在
收到 Dify 结束事件后补写，这里分别验证"解析"与"补写"两段。
"""

import asyncio
import time

from sqlalchemy import delete, select

from app.db.session import SessionLocal
from app.models.analysis import MultimodalAnalysisRecord
from app.services.ai_lab.socket_events import _extract_dify_risk
from app.services.analysis_record_service import (
    AnalysisSnapshot,
    save_snapshot,
    update_dify_risk_level,
)


# ============================================================
#  解析：不同工作流的暴露位置都要认得
# ============================================================
def test_extract_from_workflow_finished_outputs():
    """结束节点的输出变量是最常见的暴露位置。"""
    payload = {
        "event": "workflow_finished",
        "data": {"status": "succeeded", "outputs": {"risk_level": "high", "answer": "..."}},
    }
    assert _extract_dify_risk(payload) == "high"


def test_extract_from_message_end_metadata():
    """message_end 的 metadata 里带等级时也要取到。"""
    payload = {"event": "message_end", "metadata": {"riskLevel": "Medium"}}
    assert _extract_dify_risk(payload) == "Medium"


def test_extract_ignores_unrelated_events():
    """普通消息事件没有等级，不应误判。"""
    assert _extract_dify_risk({"event": "message", "answer": "我在听"}) is None
    assert _extract_dify_risk({"event": "message_end", "metadata": {"usage": {"tokens": 12}}}) is None
    assert _extract_dify_risk(None) is None
    assert _extract_dify_risk("not-a-dict") is None


def test_extract_ignores_structured_value():
    """结构化对象不能当成等级写库（只接受字符串/数字）。"""
    payload = {"data": {"outputs": {"risk_level": {"level": "high"}}}}
    assert _extract_dify_risk(payload) is None


# ============================================================
#  补写：写入的是"该会话最近一行"，且值被规范成小写
# ============================================================
def test_update_dify_risk_level_backfills_latest_row():
    """补写应命中该会话最近一轮留痕，并把值规范为小写。"""
    marker = f"pytest-dify-risk-{int(time.time() * 1000)}"
    snapshot = AnalysisSnapshot.from_video_call(
        user_id=None,
        session_id=marker,
        asr_text="我最近总是提不起劲",
        risk={"level": "MEDIUM", "riskScore": 45},
        status="ok",
    )
    assert asyncio.run(save_snapshot(snapshot)) is True

    assert asyncio.run(update_dify_risk_level(marker, "HIGH")) is True

    db = SessionLocal()
    try:
        row = db.scalar(
            select(MultimodalAnalysisRecord).where(
                MultimodalAnalysisRecord.session_id == marker
            )
        )
        assert row is not None
        assert row.dify_risk_level == "high", row.dify_risk_level
        assert row.risk_level == "MEDIUM", "平台侧等级不应被覆盖"

        db.execute(
            delete(MultimodalAnalysisRecord).where(
                MultimodalAnalysisRecord.session_id == marker
            )
        )
        db.commit()
    finally:
        db.close()


def test_update_dify_risk_level_is_safe_without_row():
    """没有对应留痕行时返回 False，不抛异常（留痕失败不得影响通话）。"""
    assert asyncio.run(update_dify_risk_level("pytest-no-such-session", "high")) is False
    assert asyncio.run(update_dify_risk_level("pytest-no-such-session", None)) is False
    assert asyncio.run(update_dify_risk_level("", "high")) is False
