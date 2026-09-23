"""AI 自我教练对话记录接口。

实时对话本身走 SocketIO（见 ``app/services/ai_lab/socket_events.py``），
本模块提供"对话之后"的读写能力：

    GET  /ai-conversations                        会话列表（本人）
    GET  /ai-conversations/{id}                   会话详情（含逐轮消息与阶段判定）
    POST /ai-conversations/{id}/summary           生成阶段总结草稿（不落库）
    POST /ai-conversations/{id}/summary/confirm   确认总结并写入情绪日记（幂等）
    DELETE /ai-conversations/{id}                 删除本人会话（日记保留，解除关联）
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_async_db, get_current_user
from app.api.response import ok, paginated
from app.models.ai_conversation import AiConversation
from app.models.user import User
from app.schemas.ai_conversation import (
    AiConversationOut,
    AiMessageOut,
    SummaryConfirmIn,
    SummaryConfirmOut,
    SummaryDraftOut,
)
from app.services.ai_conversation_service import (
    confirm_summary,
    delete_conversation,
    generate_summary_draft,
    get_ai_conversation_or_404,
    list_ai_conversations,
    list_ai_messages,
)
from app.services.coach_stage_service import STAGE_LABELS_CN
from app.utils.time import to_iso

router = APIRouter(prefix="/ai-conversations", tags=["ai-conversations"])


def _conversation_out(conversation: AiConversation) -> dict:
    return AiConversationOut(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        message_count=conversation.message_count,
        turn_count=conversation.turn_count or 0,
        final_stage=conversation.final_stage,
        final_stage_label=STAGE_LABELS_CN.get(conversation.final_stage or ""),
        max_risk_level=conversation.max_risk_level,
        summary=conversation.summary,
        summary_confirmed_at=to_iso(conversation.summary_confirmed_at),
        journal_id=conversation.journal_id,
        created_at=to_iso(conversation.created_at) or "",
        updated_at=to_iso(conversation.updated_at) or "",
    ).model_dump(by_alias=True)


def _message_out(message: dict) -> dict:
    return AiMessageOut(
        id=message["id"],
        role=message["role"],
        content=message["content"],
        emotion=message["emotion"],
        turn_index=message["turn_index"],
        stage=message["stage"],
        stage_label=STAGE_LABELS_CN.get(message["stage"] or ""),
        goal_clear=message["goal_clear"],
        action_ready=message["action_ready"],
        should_summarize=message["should_summarize"],
        risk_level=message["risk_level"],
        fusion_emotion=message["fusion_emotion"],
        fusion_confidence=message["fusion_confidence"],
        timings=message["timings"],
        created_at=message["created_at"] or "",
    ).model_dump(by_alias=True)


@router.get("")
async def my_ai_conversations(
    request: Request,
    page: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=50, alias="pageSize"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    items, total = await list_ai_conversations(db, user.id, page, pageSize)
    out = [
        AiConversationOut(
            **item,
            final_stage_label=STAGE_LABELS_CN.get(item.get("final_stage") or ""),
        ).model_dump(by_alias=True)
        for item in items
    ]
    return ok(paginated(out, total, page, pageSize), trace_id=request.state.trace_id)


@router.get("/{conversation_id}")
async def ai_conversation_detail(
    conversation_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    conv = await get_ai_conversation_or_404(db, user.id, conversation_id)
    messages = await list_ai_messages(db, conv.id)
    return ok(
        {
            "conversation": _conversation_out(conv),
            "messages": [_message_out(m) for m in messages],
        },
        trace_id=request.state.trace_id,
    )


@router.delete("/{conversation_id}", status_code=204)
async def ai_conversation_delete(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
) -> None:
    """删除本人会话记录（消息一并删除，已确认的情绪日记保留）。"""
    conv = await get_ai_conversation_or_404(db, user.id, conversation_id)
    await delete_conversation(db, conv)


@router.post("/{conversation_id}/summary", status_code=201)
async def ai_conversation_summary(
    conversation_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """生成阶段总结草稿（不落库；用户确认后经 emotion-journals 或 confirm 接口写入）。"""
    draft = await generate_summary_draft(db, user, conversation_id)
    return ok(
        SummaryDraftOut(
            mood_type=draft["mood_type"],
            content=draft["content"],
            source=draft["source"],
            conversation_id=draft["conversation_id"],
            final_stage=draft["final_stage"],
            final_stage_label=STAGE_LABELS_CN.get(draft["final_stage"] or ""),
            turn_count=draft["turn_count"],
            already_confirmed=draft["already_confirmed"],
            stage_draft=draft["stage_draft"],
        ).model_dump(by_alias=True),
        trace_id=request.state.trace_id,
    )


@router.post("/{conversation_id}/summary/confirm", status_code=201)
async def ai_conversation_summary_confirm(
    conversation_id: int,
    body: SummaryConfirmIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """确认阶段总结：写入会话总结，并生成（或更新）关联的情绪日记。

    与前端"先取草稿、再由 emotion-journals 提交"的流程等价，
    重复调用是幂等的：同一会话只会产生一篇日记。
    """
    conv = await get_ai_conversation_or_404(db, user.id, conversation_id)
    journal = await confirm_summary(
        db, conv, content=body.content, mood_type=body.moodType,
    )
    return ok(
        SummaryConfirmOut(
            conversation_id=int(conv.id),
            summary=conv.summary or body.content,
            summary_confirmed_at=to_iso(conv.summary_confirmed_at),
            journal_id=int(journal.id),
            journal_created_at=to_iso(journal.created_at),
            mood_type=journal.mood_type,
        ).model_dump(by_alias=True),
        trace_id=request.state.trace_id,
    )
