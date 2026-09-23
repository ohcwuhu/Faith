"""add AI evidence ledgers (analysis records, consent records, stage/risk columns)

Revision ID: d7e8f9a0b1c2
Revises: b7c8d9e0f1a2
Create Date: 2026-09-21

本次迁移把"一次 AI 成长对话"与"每一轮的多模态识别结果"变成可统计的数据资产：

1. ``multimodal_analysis_records``：单次分析的输入、三模态输出、融合权重、
   置信度校准、线索冲突、风险分级与分段耗时；
2. ``consent_records``：麦克风 / 摄像头 / 多模态授权存证（科技伦理审核依据）；
3. ``ai_conversations``：补实时会话标识、轮次、最终阶段、最高风险等级、
   授权快照与阶段总结（状态增加 ``ABANDONED``，用于异常断开的会话收尾）；
4. ``ai_messages``：补轮次序号与逐轮的阶段判定、风险分级、分段耗时；
5. ``system_configs``：补 ``support_hint``（中风险关怀提示语，缺失会导致建档失败）。
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = "d7e8f9a0b1c2"
down_revision: Union[str, Sequence[str], None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ── 1) 多模态分析留痕 ─────────────────────────────────────
    op.create_table(
        "multimodal_analysis_records",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False),
        sa.Column("user_id", mysql.BIGINT(unsigned=True), nullable=True),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("asr_text", mysql.TEXT(), nullable=True),
        sa.Column("asr_emotion", sa.String(length=16), nullable=True),
        sa.Column("text_emotion", sa.String(length=16), nullable=True),
        sa.Column("text_confidence", sa.Float(), nullable=True),
        sa.Column("voice_emotion", sa.String(length=16), nullable=True),
        sa.Column("voice_confidence", sa.Float(), nullable=True),
        sa.Column("facial_emotion", sa.String(length=16), nullable=True),
        sa.Column("facial_confidence", sa.Float(), nullable=True),
        sa.Column("facial_frames", sa.Integer(), nullable=True),
        sa.Column("fusion_emotion", sa.String(length=16), nullable=True),
        sa.Column("fusion_confidence", sa.Float(), nullable=True),
        sa.Column("weights", mysql.JSON(), nullable=True),
        sa.Column("weight_adjustments", mysql.JSON(), nullable=True),
        sa.Column("calibration", mysql.JSON(), nullable=True),
        sa.Column("conflict", mysql.JSON(), nullable=True),
        sa.Column("risk_level", sa.String(length=8), nullable=True),
        sa.Column("risk_score", sa.Integer(), nullable=True),
        sa.Column("risk_reasons", mysql.JSON(), nullable=True),
        sa.Column("dify_risk_level", sa.String(length=8), nullable=True),
        sa.Column("coach_stage", sa.String(length=24), nullable=True),
        sa.Column("goal_clear", sa.Boolean(), nullable=True),
        sa.Column("action_ready", sa.Boolean(), nullable=True),
        sa.Column("should_summarize", sa.Boolean(), nullable=True),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=True),
        sa.Column("timings", mysql.JSON(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'ok'"),
        ),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=3),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(3)"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_mmar_user", ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["ai_conversations.id"],
            name="fk_mmar_conversation",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("idx_mmar_user_time", "multimodal_analysis_records", ["user_id", "created_at"])
    op.create_index("idx_mmar_source_time", "multimodal_analysis_records", ["source", "created_at"])
    op.create_index("idx_mmar_risk_time", "multimodal_analysis_records", ["risk_level", "created_at"])
    op.create_index("idx_mmar_stage_time", "multimodal_analysis_records", ["coach_stage", "created_at"])
    op.create_index("idx_mmar_conversation", "multimodal_analysis_records", ["conversation_id"])

    # ── 2) 授权存证 ───────────────────────────────────────────
    op.create_table(
        "consent_records",
        sa.Column("id", mysql.BIGINT(unsigned=True), autoincrement=True, nullable=False),
        sa.Column("user_id", mysql.BIGINT(unsigned=True), nullable=True),
        sa.Column("client_session_id", sa.String(length=64), nullable=True),
        sa.Column("conversation_id", mysql.BIGINT(unsigned=True), nullable=True),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("scopes", mysql.JSON(), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column(
            "granted_at",
            mysql.DATETIME(fsp=3),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(3)"),
        ),
        sa.Column("revoked_at", mysql.DATETIME(fsp=3), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_consent_user", ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["ai_conversations.id"],
            name="fk_consent_conversation",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("idx_consent_user_time", "consent_records", ["user_id", "granted_at"])
    op.create_index("idx_consent_client", "consent_records", ["client_session_id"])

    # ── 3) 会话表补字段 ───────────────────────────────────────
    op.add_column(
        "ai_conversations",
        sa.Column(
            "client_session_id",
            sa.String(length=64),
            nullable=True,
            comment="SocketIO sid",
        ),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("final_stage", sa.String(length=24), nullable=True),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("max_risk_level", sa.String(length=8), nullable=True),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("consent", mysql.JSON(), nullable=True),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("summary", mysql.TEXT(), nullable=True),
    )
    op.add_column(
        "ai_conversations",
        sa.Column("summary_confirmed_at", mysql.DATETIME(fsp=3), nullable=True),
    )
    op.create_index("idx_ai_conv_client", "ai_conversations", ["client_session_id"])
    # 状态增加 ABANDONED（异常断开由维护任务收尾）
    op.create_check_constraint(
        "chk_ai_conv_status",
        "ai_conversations",
        "status IN ('ACTIVE','ENDED','ABANDONED')",
    )

    # ── 4) 消息表补字段 ───────────────────────────────────────
    op.add_column(
        "ai_messages",
        sa.Column("turn_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("ai_messages", sa.Column("stage", sa.String(length=24), nullable=True))
    op.add_column("ai_messages", sa.Column("goal_clear", sa.Boolean(), nullable=True))
    op.add_column("ai_messages", sa.Column("action_ready", sa.Boolean(), nullable=True))
    op.add_column("ai_messages", sa.Column("should_summarize", sa.Boolean(), nullable=True))
    op.add_column(
        "ai_messages",
        sa.Column("summary_reason", sa.String(length=255), nullable=True),
    )
    op.add_column("ai_messages", sa.Column("risk_level", sa.String(length=8), nullable=True))
    op.add_column("ai_messages", sa.Column("risk_score", sa.Integer(), nullable=True))
    op.add_column(
        "ai_messages",
        sa.Column("fusion_emotion", sa.String(length=16), nullable=True),
    )
    op.add_column("ai_messages", sa.Column("fusion_confidence", sa.Float(), nullable=True))
    op.add_column("ai_messages", sa.Column("timings", mysql.JSON(), nullable=True))
    op.create_index("idx_ai_msg_stage", "ai_messages", ["stage"])

    # ── 5) 中风险关怀提示语种子 ───────────────────────────────
    connection = op.get_bind()
    exists = connection.execute(
        sa.text("SELECT COUNT(*) FROM system_configs WHERE config_key = 'support_hint'")
    ).scalar()
    if not exists:
        system_configs = sa.table(
            "system_configs",
            sa.column("config_key", sa.String),
            sa.column("config_value", sa.JSON),
            sa.column("description", sa.String),
        )
        op.bulk_insert(
            system_configs,
            [
                {
                    "config_key": "support_hint",
                    "config_value": (
                        "我们注意到你最近可能不太轻松。如果你愿意，可以随时和教练聊聊，"
                        "或联系学校的心理支持中心；如情况变得紧急，请拨打 12356。"
                    ),
                    "description": "关怀提示（中风险时展示）",
                },
            ],
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(sa.text("DELETE FROM system_configs WHERE config_key = 'support_hint'"))

    op.drop_index("idx_ai_msg_stage", table_name="ai_messages")
    op.drop_column("ai_messages", "timings")
    op.drop_column("ai_messages", "fusion_confidence")
    op.drop_column("ai_messages", "fusion_emotion")
    op.drop_column("ai_messages", "risk_score")
    op.drop_column("ai_messages", "risk_level")
    op.drop_column("ai_messages", "summary_reason")
    op.drop_column("ai_messages", "should_summarize")
    op.drop_column("ai_messages", "action_ready")
    op.drop_column("ai_messages", "goal_clear")
    op.drop_column("ai_messages", "stage")
    op.drop_column("ai_messages", "turn_index")

    op.drop_constraint("chk_ai_conv_status", "ai_conversations", type_="check")
    op.drop_index("idx_ai_conv_client", table_name="ai_conversations")
    op.drop_column("ai_conversations", "summary_confirmed_at")
    op.drop_column("ai_conversations", "summary")
    op.drop_column("ai_conversations", "consent")
    op.drop_column("ai_conversations", "max_risk_level")
    op.drop_column("ai_conversations", "final_stage")
    op.drop_column("ai_conversations", "turn_count")
    op.drop_column("ai_conversations", "client_session_id")

    # 直接删表：MySQL 不允许先删被外键引用的索引（idx_consent_user_time 覆盖 user_id），
    # 索引会随表一起消失。
    op.drop_table("consent_records")

    # 同理：idx_mmar_user_time 覆盖 fk_mmar_user，先删索引会被 MySQL 拒绝。
    op.drop_table("multimodal_analysis_records")
