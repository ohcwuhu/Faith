"""AI 自我教练对话持久化：会话 + 消息（记录可回看/继续/复核）。

三张表分工：

    ai_conversations  一次自我教练通话（起止时间、轮次、最终阶段、最高风险等级、总结）
    ai_messages       每一轮的用户表达与 AI 回复，附带阶段判定、风险分级与分段耗时
    consent_records   麦克风 / 摄像头 / 多模态分析的授权存证（伦理审核依据）

与 ``multimodal_analysis_records`` 的关系：后者记录"识别结果"，前者记录"对话过程"，
两者通过 ``conversation_id`` 关联，使对话可以回溯到当时的多模态判定。
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, JSON

from app.db.base import Base


class AiConversation(Base):
    """AI 自我教练会话（一轮完整视频通话）。

    ``status`` 取值 ``ACTIVE`` / ``ENDED`` / ``ABANDONED``：客户端异常断开时不会触发
    ``vc_stop``，这类会话由维护任务标记为 ``ABANDONED``，保证会话状态可统计。
    """

    __tablename__ = "ai_conversations"
    __table_args__ = (
        Index("idx_ai_conv_user", "user_id", "status", "created_at"),
        Index("idx_ai_conv_client", "client_session_id"),
        Index("idx_ai_conv_journal", "journal_id"),
        CheckConstraint(
            "status IN ('ACTIVE','ENDED','ABANDONED')",
            name="chk_ai_conv_status",
        ),
    )

    id = Column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_ai_conv_user"),
        nullable=False,
    )
    title = Column(String(100), nullable=False, server_default="自我教练对话")
    status = Column(
        String(16),
        nullable=False,
        server_default="ACTIVE",
        comment="ACTIVE/ENDED/ABANDONED（异常中断）",
    )
    client_session_id = Column(
        String(64),
        nullable=True,
        comment="SocketIO sid，用于把实时态与库内会话对应起来",
    )
    message_count = Column(Integer, nullable=False, server_default=text("0"))
    turn_count = Column(Integer, nullable=False, server_default=text("0"), comment="已完成的对话轮次")
    final_stage = Column(
        String(24),
        nullable=True,
        comment="最后一轮五阶段判定：opening/exploration/goal_setting/action_planning/closing",
    )
    max_risk_level = Column(
        String(8),
        nullable=True,
        comment="本次会话出现过的最高平台风险等级 NONE/LOW/MEDIUM/HIGH",
    )
    consent = Column(
        JSON,
        nullable=True,
        comment="{mic, camera, multimodal, policyVersion, grantedAt} 授权快照",
    )
    summary = Column(Text, nullable=True, comment="用户确认后的阶段总结")
    summary_confirmed_at = Column(DATETIME(fsp=3), nullable=True)
    ended_at = Column(DATETIME(fsp=3), nullable=True)
    journal_id = Column(BIGINT(unsigned=True), nullable=True, comment="由本对话生成的情绪日记 ID")
    created_at = Column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
    )
    updated_at = Column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
        onupdate=text("CURRENT_TIMESTAMP(3)"),
    )


class AiMessage(Base):
    """AI 自我教练会话消息。

    每一轮写入两条记录（USER + ASSISTANT），共享同一 ``turn_index``；
    阶段判定与风险分级挂在轮次上，便于做"阶段判定 vs 人工标签"的一致性实验。
    """

    __tablename__ = "ai_messages"
    __table_args__ = (
        Index("idx_ai_msg_conv", "conversation_id", "created_at"),
        Index("idx_ai_msg_stage", "stage"),
    )

    id = Column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("ai_conversations.id", ondelete="CASCADE", name="fk_ai_msg_conv"),
        nullable=False,
    )
    role = Column(String(16), nullable=False, comment="USER/ASSISTANT")
    content = Column(Text, nullable=False)
    emotion = Column(JSON, nullable=True, comment="该轮情绪上下文快照")
    turn_index = Column(Integer, nullable=False, server_default=text("0"), comment="轮次序号，从 1 开始")
    stage = Column(String(24), nullable=True, comment="该轮五阶段判定")
    goal_clear = Column(Boolean, nullable=True, comment="目标是否已经清楚")
    action_ready = Column(Boolean, nullable=True, comment="是否已形成可执行行动")
    should_summarize = Column(Boolean, nullable=True, comment="是否满足收束条件")
    summary_reason = Column(String(255), nullable=True, comment="收束判断依据")
    risk_level = Column(String(8), nullable=True, comment="平台风险分级 NONE/LOW/MEDIUM/HIGH")
    risk_score = Column(Integer, nullable=True)
    fusion_emotion = Column(String(16), nullable=True, comment="多模态融合情绪（英文标签）")
    fusion_confidence = Column(Float, nullable=True)
    timings = Column(JSON, nullable=True, comment="该轮分段耗时（秒）")
    created_at = Column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
    )


class ConsentRecord(Base):
    """多模态采集授权存证。

    赛题包含科技伦理审核，需要能证明"语音与图像是在用户授权后才处理的"。
    每次通话开始写入一行，撤销时补 ``revoked_at``。
    """

    __tablename__ = "consent_records"
    __table_args__ = (
        Index("idx_consent_user_time", "user_id", "granted_at"),
        Index("idx_consent_client", "client_session_id"),
    )

    id = Column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="SET NULL", name="fk_consent_user"),
        nullable=True,
    )
    client_session_id = Column(String(64), nullable=True)
    conversation_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("ai_conversations.id", ondelete="SET NULL", name="fk_consent_conversation"),
        nullable=True,
    )
    policy_version = Column(String(32), nullable=False, comment="服务协议/隐私政策版本")
    scopes = Column(
        JSON,
        nullable=False,
        comment="{mic, camera, multimodal, basis} 各项布尔值与授权来源",
    )
    source = Column(String(24), nullable=True, comment="授权入口：VIDEO_CALL 等")
    ip = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    granted_at = Column(
        DATETIME(fsp=3),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(3)"),
    )
    revoked_at = Column(DATETIME(fsp=3), nullable=True)
