"""真实运行数据导出：把线上留痕变成技术报告可以直接引用的表格。

与管理端接口 ``GET /api/v1/admin/stats/multimodal`` 使用**同一个聚合函数**
（``analysis_stats_service.multimodal_overview``），因此脚本导出的数字
与后台页面看到的完全一致，不会出现"报告一套、系统一套"。

输出：

* ``results/runtime_stats.json``        原始聚合结果（可追溯）
* ``results/runtime_stats.md``          Markdown 表格（直接贴进技术报告）
* ``results/runtime_latency.csv``       分段耗时分位数
* ``results/runtime_risk.csv``          风险分布与两侧一致性
* ``results/runtime_stages.csv``        五阶段分布

用法::

    python experiments/run_runtime_stats.py
    python experiments/run_runtime_stats.py --days 7 --source VIDEO_CALL

前提：环境变量 ``DATABASE_URL`` 指向线上/本地库（见 .env.example）。
没有真实运行数据时脚本会正常输出空表，并在报告顶部注明样本为 0。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from _bootstrap import RESULT_DIR, ensure_app_importable

ensure_app_importable()

from metrics import markdown_table, write_csv  # noqa: E402


async def collect(days: int, source: str | None) -> dict[str, Any]:
    """从数据库聚合留痕；数据库不可用时抛出带说明的异常。"""
    from app.db.session import AsyncSessionLocal
    from app.services.analysis_stats_service import multimodal_overview

    async with AsyncSessionLocal() as db:
        return await multimodal_overview(db, days=days, source=source)


def render_report(overview: dict[str, Any], *, days: int, source: str | None) -> str:
    """把聚合结果渲染为可粘贴的 Markdown。"""
    totals = overview.get("totals", {})
    latency = overview.get("latency", {})
    risk = overview.get("risk", {})
    stages = overview.get("stages", {})
    modalities = overview.get("modalities", {})
    degradation = overview.get("degradation", {})
    coaching = overview.get("coaching", {})

    lines: list[str] = ["# 真实运行数据统计", ""]
    lines.append(
        f"- 统计窗口：最近 {days} 天"
        + (f"；来源过滤：{source}" if source else "；来源：全部")
    )
    lines.append(f"- 分析总次数：{totals.get('analyses', 0)}（明细抽样 {totals.get('sampled', 0)} 条）")
    if not totals.get("analyses"):
        lines.append("")
        lines.append("> 窗口内没有留痕数据：该表需在真实使用系统后重新导出。")
    lines.append("")

    lines.append("## 运行稳定性与降级")
    lines.append("")
    lines.append(markdown_table(
        ["指标", "数值"],
        [
            ["成功（ok）占比", totals.get("okRate", 0.0)],
            ["部分成功（partial_success）占比", totals.get("partialSuccessRate", 0.0)],
            ["失败（failed）占比", totals.get("failedRate", 0.0)],
            ["触发权重调整的分析数", degradation.get("analysesWithAdjustment", 0)],
            ["触发权重调整比例", degradation.get("rate", 0.0)],
        ],
    ))
    lines.append("")
    reasons = degradation.get("topReasons") or []
    if reasons:
        lines.append(markdown_table(
            ["降级原因", "次数"],
            [[item.get("reason", ""), item.get("count", 0)] for item in reasons],
        ))
        lines.append("")

    lines.append("## 模态可用性")
    lines.append("")
    lines.append(markdown_table(
        ["指标", "比例"],
        [
            ["面部模态缺失率", modalities.get("missingFacialRate", 0.0)],
            ["语调模态缺失率", modalities.get("missingVoiceRate", 0.0)],
            ["文本过短率", modalities.get("shortTextRate", 0.0)],
        ],
    ))
    lines.append("")

    lines.append("## 分段耗时（毫秒）")
    lines.append("")
    metrics = latency.get("metrics", {})
    if metrics:
        lines.append(markdown_table(
            ["阶段", "样本数", "P50", "P95"],
            [
                [
                    name,
                    payload.get("samples", 0),
                    payload.get("p50"),
                    payload.get("p95"),
                ]
                for name, payload in metrics.items()
            ],
        ))
    else:
        lines.append("（窗口内没有耗时留痕）")
    lines.append("")

    lines.append("## 风险判别与两侧一致性")
    lines.append("")
    consistency = risk.get("consistency", {})
    lines.append(markdown_table(
        ["口径", "分布", "触发数", "触发比例"],
        [
            [
                "平台四级",
                json.dumps(risk.get("platform", {}).get("distribution", {}), ensure_ascii=False),
                risk.get("platform", {}).get("flagged", 0),
                risk.get("platform", {}).get("flaggedRate", 0.0),
            ],
            [
                "Dify 判定",
                json.dumps(risk.get("dify", {}).get("distribution", {}), ensure_ascii=False),
                risk.get("dify", {}).get("flagged", 0),
                risk.get("dify", {}).get("flaggedRate", 0.0),
            ],
        ],
    ))
    lines.append("")
    lines.append(
        f"- 两侧都有判定：{consistency.get('comparable', 0)} 条；"
        f"一致：{consistency.get('matched', 0)} 条；"
        f"一致率：{consistency.get('rate') if consistency.get('rate') is not None else '—'}"
    )
    lines.append("")

    lines.append("## 五阶段分布")
    lines.append("")
    distribution = stages.get("distribution", {})
    if distribution:
        lines.append(markdown_table(
            ["阶段", "次数", "占比"],
            [
                [stage, count, stages.get("rate", {}).get(stage, 0.0)]
                for stage, count in sorted(distribution.items(), key=lambda item: -item[1])
            ],
        ))
    else:
        lines.append("（窗口内没有阶段留痕）")
    lines.append("")

    lines.append("## AI 教练会话")
    lines.append("")
    if coaching:
        lines.append(markdown_table(
            ["指标", "数值"],
            [[key, value] for key, value in coaching.items()],
        ))
    else:
        lines.append("（窗口内没有会话数据）")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出真实运行数据统计")
    parser.add_argument("--days", type=int, default=30, help="统计窗口天数（默认 30）")
    parser.add_argument(
        "--source",
        default=None,
        choices=["HTTP_ANALYZE", "VIDEO_CALL"],
        help="只统计某一来源（默认全部）",
    )
    args = parser.parse_args()

    try:
        overview = asyncio.run(collect(args.days, args.source))
    except Exception as exc:  # noqa: BLE001 — 需要给出可执行的排错指引
        raise SystemExit(
            "无法从数据库读取留痕数据，请检查 DATABASE_URL 与数据库是否可连接。\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report = render_report(overview, days=args.days, source=args.source)
    (RESULT_DIR / "runtime_stats.md").write_text(report, encoding="utf-8")
    (RESULT_DIR / "runtime_stats.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    latency_metrics = overview.get("latency", {}).get("metrics", {})
    write_csv(
        RESULT_DIR / "runtime_latency.csv",
        ["metric", "samples", "p50", "p95"],
        [
            {
                "metric": name,
                "samples": payload.get("samples", 0),
                "p50": payload.get("p50"),
                "p95": payload.get("p95"),
            }
            for name, payload in latency_metrics.items()
        ],
    )
    write_csv(
        RESULT_DIR / "runtime_risk.csv",
        ["side", "level", "count"],
        [
            {"side": side, "level": level, "count": count}
            for side in ("platform", "dify")
            for level, count in (overview.get("risk", {}).get(side, {}).get("distribution", {}) or {}).items()
        ],
    )
    write_csv(
        RESULT_DIR / "runtime_stages.csv",
        ["stage", "count", "rate"],
        [
            {
                "stage": stage,
                "count": count,
                "rate": overview.get("stages", {}).get("rate", {}).get(stage, 0.0),
            }
            for stage, count in (overview.get("stages", {}).get("distribution", {}) or {}).items()
        ],
    )

    print(report)
    for name in ("runtime_stats.md", "runtime_stats.json", "runtime_latency.csv",
                 "runtime_risk.csv", "runtime_stages.csv"):
        print(f"[输出] {RESULT_DIR / name}")


if __name__ == "__main__":
    main()
