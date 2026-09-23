"""AI 自我教练对话记录。"""

from typing import Literal

from pydantic import Field

from app.schemas.base import ApiModel

MoodType = Literal["CALM", "HAPPY", "ANXIOUS", "DOWN", "IRRITATED", "OTHER"]


class AiConversationOut(ApiModel):
    """会话列表项（含阶段与风险摘要，便于回看本次对话的推进情况）。"""

    id: int
    title: str
    status: Literal["ACTIVE", "ENDED", "ABANDONED"]
    message_count: int
    turn_count: int = 0
    final_stage: str | None = None
    final_stage_label: str | None = None
    max_risk_level: str | None = None
    summary: str | None = None
    summary_confirmed_at: str | None = None
    journal_id: int | None = None
    created_at: str
    updated_at: str


class AiMessageOut(ApiModel):
    """单轮消息；阶段、风险与耗时字段用于回看与阶段一致性复核。"""

    id: int
    role: Literal["USER", "ASSISTANT"]
    content: str
    emotion: dict | None = None
    turn_index: int = 0
    stage: str | None = None
    stage_label: str | None = None
    goal_clear: bool | None = None
    action_ready: bool | None = None
    should_summarize: bool | None = None
    risk_level: str | None = None
    fusion_emotion: str | None = None
    fusion_confidence: float | None = None
    timings: dict | None = None
    created_at: str


class SummaryDraftOut(ApiModel):
    """阶段总结草稿：由后端生成，需用户确认后才会写入情绪日记。"""

    mood_type: MoodType
    content: str
    source: Literal["SELF_COACHING"] = "SELF_COACHING"
    conversation_id: int
    final_stage: str | None = None
    final_stage_label: str | None = None
    turn_count: int = 0
    already_confirmed: bool = False
    stage_draft: str | None = None


class SummaryConfirmIn(ApiModel):
    """确认阶段总结（用户可修改草稿内容与情绪类型）。"""

    content: str = Field(min_length=1, max_length=500)
    moodType: MoodType = "OTHER"


class SummaryConfirmOut(ApiModel):
    """确认结果：会话总结 + 生成的情绪日记。"""

    conversation_id: int
    summary: str
    summary_confirmed_at: str | None = None
    journal_id: int
    journal_created_at: str | None = None
    mood_type: MoodType
