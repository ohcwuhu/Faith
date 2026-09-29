"""管理端：AI 留痕统计接口。

为技术报告"应用成效"与"系统性能"两节提供直接可引用的数据出口。
仅管理员可访问；返回口径中的 ``sampled`` 说明分位数基于多少条明细样本。
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_async_db, require_role
from app.api.response import ok
from app.models.user import User
from app.services.analysis_stats_service import multimodal_overview

router = APIRouter(prefix="/admin/stats", tags=["admin-ai-stats"])


@router.get("/multimodal")
async def multimodal_stats(
    request: Request,
    days: int = Query(default=30, ge=1, le=365, description="统计窗口（天）"),
    source: str | None = Query(
        default=None,
        pattern="^(HTTP_ANALYZE|VIDEO_CALL)$",
        description="留痕来源，缺省为全部",
    ),
    admin: User = Depends(require_role("ADMIN")),
    db: AsyncSession = Depends(get_async_db),
) -> dict:
    """多模态留痕与 AI 教练会话的统计概览。

    返回内容：

    - ``totals``      分析总数、抽样条数、状态分布与成功率
    - ``degradation`` 触发降级的分析与原因分布（动态融合降级机制的运行证据）
    - ``modalities``  各模态缺失率
    - ``latency``     分段耗时分位数（P50 / P95）
    - ``risk``        平台与 Dify 风险分布及两侧一致性
    - ``stages``      五阶段分布
    - ``coaching``    会话数、消息数、去重用户数
    - ``daily``       逐日分析量趋势
    """
    overview = await multimodal_overview(db, days=days, source=source)
    return ok(overview, trace_id=request.state.trace_id)
