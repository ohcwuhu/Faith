"""分析留痕与危机建档的数据库集成测试。

与其他集成测试一致，直连开发库并自建自清：
验证"写入真实 MySQL → 字段可完整读回"的链路，
而不是只验证内存中的对象构造。
"""

import asyncio
import time
from datetime import datetime, timedelta

from sqlalchemy import delete, select

from app.db.session import SessionLocal
from app.models.analysis import MultimodalAnalysisRecord
from app.models.crisis import CrisisFlag, CrisisFollowUp
from app.models.notification import Notification
from app.models.user import User
from app.services import crisis_service
from app.services.analysis_record_service import AnalysisSnapshot, save_snapshot


def _sample_user_id() -> int | None:
    """取一个普通用户用于风险建档（避免把管理员当作被监测对象）。"""
    db = SessionLocal()
    try:
        row = db.scalar(
            select(User.id)
            .where(User.role == "USER", User.deleted_at.is_(None))
            .order_by(User.id)
            .limit(1)
        )
        return int(row) if row else None
    finally:
        db.close()


def test_analysis_snapshot_written_and_read_back():
    """留痕服务应把完整快照写入 MySQL，并能原样读回关键字段。"""
    marker = f"pytest-analysis-{int(time.time() * 1000)}"
    snapshot = AnalysisSnapshot.from_video_call(
        user_id=_sample_user_id(),
        session_id=marker,
        asr_text="今天有点累，但还撑得住",
        asr_emotion="neutral",
        text_emotion={"emotion": "sad", "confidence": 0.66},
        voice_emotion={"emotion": "sad", "confidence": 0.61},
        fusion={"final_emotion": "sad", "overall_confidence": 0.58,
                "weights_used": {"text": 0.5, "voice": 0.3, "facial": 0.2},
                "weight_adjustments": ["test_adjustment"]},
        facial_emotion={"dominant_emotion": "neutral", "confidence": 0.7, "frame_count": 11},
        risk={"level": "LOW", "riskScore": 10, "reasons": ["测试用例"]},
        timings={"multimodal_seconds": 1.23},
    )

    assert asyncio.run(save_snapshot(snapshot)) is True

    db = SessionLocal()
    try:
        row = db.scalar(
            select(MultimodalAnalysisRecord).where(
                MultimodalAnalysisRecord.session_id == marker
            )
        )
        assert row is not None, "留痕未写入数据库"
        assert row.source == "VIDEO_CALL"
        assert row.asr_text == "今天有点累，但还撑得住"
        assert row.fusion_emotion == "sad"
        assert row.facial_frames == 11
        assert row.weights["text"] == 0.5
        assert row.weight_adjustments == ["test_adjustment"]
        assert row.risk_level == "LOW"
        assert row.risk_reasons == ["测试用例"]
        assert row.timings["multimodal_seconds"] == 1.23

        db.execute(
            delete(MultimodalAnalysisRecord).where(
                MultimodalAnalysisRecord.session_id == marker
            )
        )
        db.commit()
    finally:
        db.close()


def test_crisis_flag_safely_writes_graded_record():
    """独立建档入口应写入分级工单，并把判定依据落到跟进留痕。"""
    user_id = _sample_user_id()
    if user_id is None:
        return  # 开发库无普通用户时跳过（不视为失败）

    started_at = datetime.now().replace(microsecond=0) - timedelta(minutes=1)

    db = SessionLocal()
    try:
        # 清理同一用户既有的同来源工单与通知，保证用例可重复执行
        existing = list(db.scalars(
            select(CrisisFlag.id).where(
                CrisisFlag.user_id == user_id,
                CrisisFlag.source == crisis_service.SOURCE_AI_COACH,
            )
        ))
        if existing:
            db.execute(delete(CrisisFollowUp).where(CrisisFollowUp.crisis_id.in_(existing)))
            db.execute(delete(CrisisFlag).where(CrisisFlag.id.in_(existing)))
        db.execute(
            delete(Notification).where(
                Notification.user_id == user_id,
                Notification.type == "CRISIS",
                Notification.created_at >= started_at,
            )
        )
        db.commit()
    finally:
        db.close()

    flagged = asyncio.run(crisis_service.flag_crisis_safely(
        user_id,
        crisis_service.SOURCE_AI_COACH,
        "最近什么都提不起兴趣，觉得自己很多余",
    ))
    assert flagged is True, "中风险表达未建立工单"

    db = SessionLocal()
    try:
        flag = db.scalar(
            select(CrisisFlag)
            .where(
                CrisisFlag.user_id == user_id,
                CrisisFlag.source == crisis_service.SOURCE_AI_COACH,
            )
            .order_by(CrisisFlag.id.desc())
            .limit(1)
        )
        assert flag is not None
        assert flag.level == "MEDIUM", flag.level
        assert flag.status == "OPEN"

        follow_ups = list(db.scalars(
            select(CrisisFollowUp).where(CrisisFollowUp.crisis_id == flag.id)
        ))
        assert [f.action for f in follow_ups] == ["DETECT"]
        assert "MEDIUM" in follow_ups[0].note
        assert "强负性表达" in follow_ups[0].note

        # 用户侧应收到关怀提示（中风险不推送紧急热线）
        user_notice = db.scalar(
            select(Notification).where(
                Notification.user_id == user_id,
                Notification.type == "CRISIS",
                Notification.created_at >= started_at,
            )
        )
        assert user_notice is not None
        assert user_notice.title == "关怀提示"

        # 清理本用例产生的数据
        db.execute(delete(CrisisFollowUp).where(CrisisFollowUp.crisis_id == flag.id))
        db.execute(delete(CrisisFlag).where(CrisisFlag.id == flag.id))
        db.execute(
            delete(Notification).where(
                Notification.user_id == user_id,
                Notification.type == "CRISIS",
                Notification.created_at >= started_at,
            )
        )
        db.commit()
    finally:
        db.close()
