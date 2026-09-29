"""SQLAlchemy models — import all modules to register tables on Base.metadata.

所有模型模块都必须在这里导入：`Base.metadata` 是 Alembic autogenerate 的对比基准，
漏掉一个模块会让迁移脚本"提议删除"那些表。
"""

from app.models import (  # noqa: F401
    ai_conversation,
    analysis,
    chat,
    coach,
    community,
    compliance,
    content,
    crisis,
    email_code,
    file,
    growth,
    notification,
    user,
    v1_1,
)
