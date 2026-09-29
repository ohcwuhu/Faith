"""危机处理 SOP：分级检测 → 建档 → 值班接管 → 跟进留痕 → 结案。

本模块负责"持久化与协同"：调用 :mod:`app.services.crisis_rules` 完成纯规则判定，
再依据判定结果建立工单、通知值班人员与用户、记录等级升级留痕。
判定口径集中在 crisis_rules，接口层、实时音视频管线与离线评测脚本共用同一套规则。

分级响应口径
------------
    HIGH   建档 + 通知值班人员 + 向用户下发紧急求助提示（热线/线下支持）
    MEDIUM 建档 + 通知值班人员 + 向用户下发支持性提示（不推送紧急热线）
    LOW    不建档、不打扰用户，仅留日志与结构化分析记录

事务约定：本模块只做 ``flush``，不主动 ``commit``，由调用方统一提交，
避免打断调用方尚未完成的写入。
"""

import logging
import os
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError
from app.db.session import AsyncSessionLocal
from app.models.crisis import CrisisFlag, CrisisFollowUp
from app.models.user import User
from app.services.crisis_rules import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    LEVEL_NONE,
    LEVEL_ORDER,
    CrisisAssessment,
    ModalitySignals,
    assess_crisis,
    detect_crisis as _detect_crisis,
)
from app.services.notification_service import notify
from app.services.system_config_service import get_config_value
from app.utils.time import to_iso, utcnow_naive

_log = logging.getLogger("crisis-service")

#: 环境变量追加的高危关键词（与历史 CRISIS_KEYWORDS 配置兼容，作为规则表的补充）
CRISIS_KEYWORDS = [
    k.strip()
    for k in os.environ.get(
        "CRISIS_KEYWORDS",
        "自杀,想死,不想活,活不下去,结束生命,伤害自己,自残,不想存在,遗书,想离开这个世界,解脱",
    ).split(",")
    if k.strip()
]

#: 同一用户 + 同一来源的去重窗口（分钟）
DEDUP_WINDOW_MIN = 10

#: 支持的来源标识
SOURCE_CHAT = "CHAT"
SOURCE_EMOTION_JOURNAL = "EMOTION_JOURNAL"
SOURCE_COMMUNITY = "COMMUNITY"
SOURCE_AI_COACH = "AI_COACH"
SOURCE_VIDEO_CALL = "VIDEO_CALL"
SOURCE_AI_LAB = "AI_LAB"

DEFAULT_HIGH_RISK_HINT = (
    "如你正处于心理危机，请立即拨打全国心理援助热线 12356（24 小时），"
    "或前往就近医疗机构；也可以联系你信任的人陪着你。"
)
DEFAULT_MEDIUM_RISK_HINT = (
    "我们注意到你最近可能不太轻松。如果你愿意，可以随时和教练聊聊，"
    "或联系学校的心理支持中心；如情况变得紧急，请拨打 12356。"
)


def detect_crisis(text: str | None) -> bool:
    """兼容旧接口：是否达到建档标准（MEDIUM 及以上）。"""
    return _detect_crisis(text)


def assess(text: str | None, signals: dict | None = None) -> CrisisAssessment:
    """按统一规则表评估风险等级（供接口层与实时管线复用）。"""
    return assess_crisis(
        text,
        signals=ModalitySignals.from_mapping(signals),
        extra_high_keywords=CRISIS_KEYWORDS,
    )


def _format_reasons(assessment: CrisisAssessment) -> str:
    """把判定依据拼接为工单可读文本。"""
    return "；".join(assessment.reasons) if assessment.reasons else "规则命中"


async def _resolve_hint(db: AsyncSession, key: str, default: str) -> str:
    """读取配置中的提示语，配置缺失或异常时回退内置文案。

    建档流程不能因为一个配置键缺失而失败——那会让本该被看见的用户沉默。
    """
    try:
        return await get_config_value(db, key) or default
    except Exception as exc:  # noqa: BLE001 - 配置问题不得影响危机建档
        _log.warning("[Crisis] 读取配置 %s 失败，改用内置文案: %s", key, exc)
        return default


async def maybe_flag_crisis(
    db: AsyncSession,
    user_id: int,
    source: str,
    text: str,
    *,
    signals: dict | None = None,
    assessment: CrisisAssessment | None = None,
) -> CrisisFlag | None:
    """按风险等级建档并通知；同来源 10 分钟内去重，等级升高时升级工单。

    Args:
        db: 异步会话（调用方负责 commit）。
        user_id: 触发风险的用户。
        source: 来源标识，取值见 ``SOURCE_*`` 常量。
        text: 触发风险的文本快照。
        signals: 可选的模态信号（语音/面部情绪及置信度）。
        assessment: 已计算好的评估结果，避免重复判定。

    Returns:
        新建或已存在的危机工单；未达到建档等级时返回 ``None``。
    """
    result = assessment or assess(text, signals)
    if not result.flagged:
        if result.level != LEVEL_NONE:
            _log.info(
                "[Crisis] 留痕未建档 | user=%s source=%s level=%s score=%s",
                user_id, source, result.level, result.risk_score,
            )
        return None

    snapshot = (text or "").strip()[:500]
    recent_flag = await db.scalar(
        select(CrisisFlag)
        .where(
            CrisisFlag.user_id == user_id,
            CrisisFlag.source == source,
            CrisisFlag.status != "RESOLVED",
            CrisisFlag.created_at > utcnow_naive() - timedelta(minutes=DEDUP_WINDOW_MIN),
        )
        .order_by(CrisisFlag.created_at.desc())
        .limit(1)
    )

    if recent_flag is not None:
        # 去重窗口内重复触发：等级升高时升级工单并留痕，否则保持静默
        if LEVEL_ORDER.get(result.level, 0) > LEVEL_ORDER.get(recent_flag.level, 0):
            previous_level = recent_flag.level
            recent_flag.level = result.level
            recent_flag.content = snapshot or recent_flag.content
            db.add(CrisisFollowUp(
                crisis_id=recent_flag.id,
                actor_id=None,
                actor_role="SYSTEM",
                action="DETECT",
                note=(
                    f"风险等级由 {previous_level} 升级为 {result.level}"
                    f"（评分 {result.risk_score}）：{_format_reasons(result)}"
                )[:500],
            ))
            await db.flush()
            _log.warning(
                "[Crisis] 工单升级 | user=%s source=%s id=%s %s→%s",
                user_id, source, recent_flag.id, previous_level, result.level,
            )
        return recent_flag

    flag = CrisisFlag(
        user_id=user_id,
        source=source,
        level=result.level,
        content=snapshot or "（无文本快照）",
        status="OPEN",
    )
    db.add(flag)
    await db.flush()
    db.add(CrisisFollowUp(
        crisis_id=flag.id,
        actor_id=None,
        actor_role="SYSTEM",
        action="DETECT",
        note=(
            f"系统自动检测：{result.level}（评分 {result.risk_score}）；"
            f"{_format_reasons(result)}"
        )[:500],
    ))

    admins = list(
        await db.scalars(
            select(User).where(
                User.role == "ADMIN",
                User.status == "ENABLED",
                User.deleted_at.is_(None),
            )
        )
    )
    urgent = result.level == LEVEL_HIGH
    for admin in admins:
        await notify(
            db,
            admin.id,
            "CRISIS",
            f"{result.level_label_cn}预警",
            (
                f"用户 #{user_id} 在「{source}」触发{result.level_label_cn}"
                f"（评分 {result.risk_score}），{'请尽快处理' if urgent else '请关注并跟进'}。"
            ),
        )

    config_key = "emergency_hint" if urgent else "support_hint"
    default_hint = DEFAULT_HIGH_RISK_HINT if urgent else DEFAULT_MEDIUM_RISK_HINT
    hint = await _resolve_hint(db, config_key, default_hint)
    await notify(
        db,
        user_id,
        "CRISIS",
        "紧急求助提示" if urgent else "关怀提示",
        hint,
    )

    await db.flush()
    _log.warning(
        "[Crisis] 建档 | user=%s source=%s id=%s level=%s score=%s",
        user_id, source, flag.id, result.level, result.risk_score,
    )
    return flag


async def flag_crisis_safely(
    user_id: int,
    source: str,
    text: str,
    *,
    signals: dict | None = None,
    assessment: CrisisAssessment | None = None,
) -> bool:
    """在独立事务中建档，供没有数据库会话的调用方使用（实时管线、HTTP 分析接口）。

    建档失败（数据库不可用等）只记录日志，不影响对话主流程。

    Returns:
        ``True`` 表示已建档或已存在同窗口工单；``False`` 表示未达建档等级或写入失败。
    """
    try:
        async with AsyncSessionLocal() as db:
            flag = await maybe_flag_crisis(
                db,
                user_id,
                source,
                text,
                signals=signals,
                assessment=assessment,
            )
            await db.commit()
            return flag is not None
    except Exception as exc:  # noqa: BLE001 - 危机建档失败不得中断对话
        _log.warning(
            "[Crisis] 独立建档失败(已忽略) | user=%s source=%s err=%s",
            user_id, source, exc,
        )
        return False


async def _get_flag_or_404(db: AsyncSession, crisis_id: int) -> CrisisFlag:
    flag = await db.get(CrisisFlag, crisis_id)
    if flag is None:
        raise AppError(404, "CRISIS_NOT_FOUND", "危机记录不存在")
    return flag


async def list_crisis_flags(
    db: AsyncSession,
    status: str | None,
    page: int,
    page_size: int,
) -> tuple[list[CrisisFlag], int]:
    stmt = select(CrisisFlag)
    if status:
        stmt = stmt.where(CrisisFlag.status == status)
    total = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(
        await db.scalars(
            stmt.order_by(CrisisFlag.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return rows, total


async def crisis_to_out(db: AsyncSession, flag: CrisisFlag, users: dict[int, User]) -> dict:
    user = users.get(flag.user_id)
    assignee = users.get(flag.assigned_admin_id) if flag.assigned_admin_id else None
    return {
        "id": flag.id,
        "user": {
            "id": flag.user_id,
            "nickname": user.nickname if user else "",
            "phone": user.phone if user else "",
        },
        "source": flag.source,
        "level": flag.level,
        "content": flag.content,
        "status": flag.status,
        "assignedAdminId": flag.assigned_admin_id,
        "assignedAdminName": assignee.nickname if assignee else "",
        "resolvedAt": to_iso(flag.resolved_at),
        "createdAt": to_iso(flag.created_at),
    }


async def assign_crisis(db: AsyncSession, admin: User, crisis_id: int) -> CrisisFlag:
    flag = await _get_flag_or_404(db, crisis_id)
    if flag.status == "RESOLVED":
        raise AppError(409, "INVALID_STATE_TRANSITION", "已结案记录不可再指派")
    flag.status = "FOLLOWING"
    flag.assigned_admin_id = admin.id
    db.add(CrisisFollowUp(
        crisis_id=flag.id,
        actor_id=admin.id,
        actor_role="ADMIN",
        action="ASSIGN",
        note=f"{admin.nickname} 接管处理",
    ))
    await db.commit()
    await db.refresh(flag)
    return flag


async def add_crisis_follow_up(db: AsyncSession, admin: User, crisis_id: int, note: str) -> CrisisFollowUp:
    flag = await _get_flag_or_404(db, crisis_id)
    if flag.status == "RESOLVED":
        raise AppError(409, "INVALID_STATE_TRANSITION", "已结案记录不可再跟进，请重新开启")
    if not note.strip():
        raise AppError(400, "VALIDATION_ERROR", "跟进内容不能为空")
    if flag.status == "OPEN":
        flag.status = "FOLLOWING"
        flag.assigned_admin_id = admin.id
    follow_up = CrisisFollowUp(
        crisis_id=flag.id,
        actor_id=admin.id,
        actor_role="ADMIN",
        action="FOLLOW_UP",
        note=note.strip()[:500],
    )
    db.add(follow_up)
    await db.commit()
    await db.refresh(follow_up)
    return follow_up


async def resolve_crisis(db: AsyncSession, admin: User, crisis_id: int, note: str) -> CrisisFlag:
    flag = await _get_flag_or_404(db, crisis_id)
    if flag.status == "RESOLVED":
        raise AppError(409, "INVALID_STATE_TRANSITION", "该记录已结案")
    flag.status = "RESOLVED"
    flag.resolved_at = utcnow_naive()
    db.add(CrisisFollowUp(
        crisis_id=flag.id,
        actor_id=admin.id,
        actor_role="ADMIN",
        action="RESOLVE",
        note=note.strip()[:500] or "结案",
    ))
    await db.commit()
    await db.refresh(flag)
    return flag


async def list_crisis_follow_ups(db: AsyncSession, crisis_id: int, users: dict[int, User]) -> list[dict]:
    await _get_flag_or_404(db, crisis_id)
    rows = list(
        await db.scalars(
            select(CrisisFollowUp)
            .where(CrisisFollowUp.crisis_id == crisis_id)
            .order_by(CrisisFollowUp.created_at.asc())
        )
    )
    return [
        {
            "id": r.id,
            "actorId": r.actor_id,
            "actorRole": r.actor_role,
            "actorName": users[r.actor_id].nickname if r.actor_id in users else "系统",
            "action": r.action,
            "note": r.note,
            "createdAt": to_iso(r.created_at),
        }
        for r in rows
    ]


__all__ = [
    "CRISIS_KEYWORDS",
    "DEDUP_WINDOW_MIN",
    "SOURCE_CHAT",
    "SOURCE_EMOTION_JOURNAL",
    "SOURCE_COMMUNITY",
    "SOURCE_AI_COACH",
    "SOURCE_VIDEO_CALL",
    "SOURCE_AI_LAB",
    "LEVEL_NONE",
    "LEVEL_LOW",
    "LEVEL_MEDIUM",
    "LEVEL_HIGH",
    "CrisisAssessment",
    "ModalitySignals",
    "assess",
    "detect_crisis",
    "maybe_flag_crisis",
    "list_crisis_flags",
    "crisis_to_out",
    "assign_crisis",
    "add_crisis_follow_up",
    "resolve_crisis",
    "list_crisis_follow_ups",
]
