"""阶段判定一致性实验。

对标注语料逐条运行**线上同一份**阶段引擎
（``app.services.coach_stage_service.decide_stage``），输出：

1. 阶段一致性：准确率、宏平均 F1、混淆矩阵；
2. 标注者间一致性：Cohen's kappa（把"引擎"视为第二位标注者）；
3. 布尔线索一致性：``goal_clear`` / ``action_ready`` 的准确率与 kappa；
4. 错误分析：逐条导出"期望阶段 → 判定阶段 + 命中线索"，便于定位规则缺陷。

用法::

    python experiments/run_stage_agreement.py
    python experiments/run_stage_agreement.py --data experiments/data/stage_labels.jsonl

数据格式（JSONL，每行一条）::

    {
      "id": "ST01",
      "source": "synthetic_seed",          # 数据来源，报告需如实标注
      "history": [{"role": "user", "content": "..."}],
      "utterance": "本轮用户表达",
      "turn_index": 2,
      "expected_stage": "goal_setting",
      "expected_goal_clear": true,          # 可选
      "expected_action_ready": false,       # 可选
      "note": "判定要点"
    }

注意：仓库内附带的样例集为**合成种子数据**，只用于跑通流程与回归，
不能作为技术报告中的准确率结论；结论必须来自人工标注语料。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from app.services.coach_stage_service import (  # noqa: E402
    STAGE_LABELS_CN,
    STAGES,
    decide_stage,
)
from metrics import (  # noqa: E402
    accuracy,
    classification_report,
    cohens_kappa,
    confusion_matrix,
    macro_f1,
    markdown_table,
    write_csv,
)


def load_samples(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 标注语料。"""
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:  # 明确报出坏行，避免静默丢样本
                raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
    if not samples:
        raise SystemExit(f"{path} 中没有样本")
    return samples


def run_case(sample: dict[str, Any]) -> dict[str, Any]:
    """对单条样本执行阶段判定，返回"期望 vs 判定"的对照行。"""
    history = sample.get("history") or []
    utterance = str(sample.get("utterance") or "")
    turn_index = int(sample.get("turn_index") or (len(history) // 2 + 1))
    decision = decide_stage(history, utterance, turn_index=turn_index)

    expected_stage = str(sample.get("expected_stage") or "")
    row: dict[str, Any] = {
        "id": sample.get("id", ""),
        "source": sample.get("source", "unspecified"),
        "turn_index": turn_index,
        "utterance": utterance,
        "expected_stage": expected_stage,
        "expected_stage_cn": STAGE_LABELS_CN.get(expected_stage, expected_stage),
        "predicted_stage": decision.stage,
        "predicted_stage_cn": decision.stage_label_cn,
        "stage_match": expected_stage == decision.stage,
        "goal_clear": decision.goal_clear,
        "action_ready": decision.action_ready,
        "should_summarize": decision.should_summarize,
        "evidence": "；".join(decision.evidence),
        "note": sample.get("note", ""),
    }
    if "expected_goal_clear" in sample:
        row["expected_goal_clear"] = bool(sample["expected_goal_clear"])
        row["goal_match"] = row["expected_goal_clear"] == decision.goal_clear
    if "expected_action_ready" in sample:
        row["expected_action_ready"] = bool(sample["expected_action_ready"])
        row["action_match"] = row["expected_action_ready"] == decision.action_ready
    return row


def build_report(rows: list[dict[str, Any]]) -> str:
    """把对照行渲染为可直接贴进技术报告的 Markdown。"""
    gold = [str(row["expected_stage"]) for row in rows]
    pred = [str(row["predicted_stage"]) for row in rows]
    labels = [stage for stage in STAGES if stage in set(gold) | set(pred)]

    accuracy_value = accuracy(gold, pred)
    macro_f1_value = macro_f1(gold, pred, labels)
    kappa_value = cohens_kappa(gold, pred, labels)

    lines: list[str] = ["# 阶段判定一致性", ""]
    lines.append(f"- 样本数：{len(rows)}")
    sources = sorted({str(row["source"]) for row in rows})
    lines.append(f"- 数据来源：{'、'.join(sources)}")
    lines.append(f"- 阶段准确率：{accuracy_value:.4f}")
    lines.append(f"- 阶段宏平均 F1：{macro_f1_value:.4f}")
    lines.append(f"- Cohen's kappa：{kappa_value:.4f}")
    lines.append("")

    lines.append("## 分级指标")
    lines.append("")
    per_class = classification_report(gold, pred, labels)
    lines.append(markdown_table(
        ["阶段", "精确率", "召回率", "F1", "样本数"],
        [
            [
                STAGE_LABELS_CN.get(str(item["label"]), item["label"]),
                item["precision"],
                item["recall"],
                item["f1"],
                item["support"],
            ]
            for item in per_class
        ],
    ))
    lines.append("")

    lines.append("## 混淆矩阵（行=人工标签，列=引擎判定）")
    lines.append("")
    matrix = confusion_matrix(gold, pred, labels)
    header = ["人工\\判定", *[STAGE_LABELS_CN.get(stage, stage) for stage in labels]]
    body = [
        [STAGE_LABELS_CN.get(stage, stage)] + [matrix[stage][other] for other in labels]
        for stage in labels
    ]
    lines.append(markdown_table(header, body))
    lines.append("")

    # 布尔线索：只在有标注的样本上统计，避免把"未标注"算成错误
    lines.append("## 布尔线索一致性")
    lines.append("")
    bool_rows: list[list[object]] = []
    for field, expected_key in (
        ("goal_clear", "expected_goal_clear"),
        ("action_ready", "expected_action_ready"),
    ):
        pairs = [
            (str(row[expected_key]), str(row[field]))
            for row in rows
            if expected_key in row
        ]
        if not pairs:
            continue
        gold_pairs = ["true" if value == "True" else "false" for value, _ in pairs]
        pred_pairs = ["true" if value == "True" else "false" for _, value in pairs]
        bool_rows.append([
            field,
            len(pairs),
            f"{accuracy(gold_pairs, pred_pairs):.4f}",
            f"{cohens_kappa(gold_pairs, pred_pairs, ['true', 'false']):.4f}",
        ])
    if bool_rows:
        lines.append(markdown_table(["线索", "标注样本数", "准确率", "kappa"], bool_rows))
    else:
        lines.append("（语料未标注布尔线索）")
    lines.append("")

    missed = [row for row in rows if not row["stage_match"]]
    lines.append(f"## 错误分析（{len(missed)} 条不一致）")
    lines.append("")
    if missed:
        lines.append(markdown_table(
            ["ID", "人工标签", "引擎判定", "命中线索", "用户表达"],
            [
                [
                    row["id"],
                    row["expected_stage_cn"],
                    row["predicted_stage_cn"],
                    row["evidence"] or "—",
                    str(row["utterance"])[:40],
                ]
                for row in missed
            ],
        ))
    else:
        lines.append("无不一致样本。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段判定一致性实验")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "stage_labels.jsonl",
        help="标注语料路径（JSONL）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULT_DIR / "stage_agreement_report.md",
        help="Markdown 报告输出路径",
    )
    args = parser.parse_args()

    samples = load_samples(args.data)
    rows = [run_case(sample) for sample in samples]

    for path, headers in (
        (
            RESULT_DIR / "stage_agreement_details.csv",
            [
                "id", "source", "turn_index", "utterance",
                "expected_stage", "expected_stage_cn",
                "predicted_stage", "predicted_stage_cn", "stage_match",
                "goal_clear", "action_ready", "should_summarize",
                "evidence", "note",
            ],
        ),
    ):
        write_csv(path, headers, [{key: row.get(key, "") for key in headers} for row in rows])

    report = build_report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    gold = [str(row["expected_stage"]) for row in rows]
    pred = [str(row["predicted_stage"]) for row in rows]
    labels = [stage for stage in STAGES if stage in set(gold) | set(pred)]
    print(report)
    print(f"\n[完成] 样本 {len(rows)} 条 | 准确率 {accuracy(gold, pred):.4f} "
          f"| 宏平均 F1 {macro_f1(gold, pred, labels):.4f} "
          f"| kappa {cohens_kappa(gold, pred, labels):.4f}")
    print(f"[输出] {args.out}")
    print(f"[输出] {RESULT_DIR / 'stage_agreement_details.csv'}")


if __name__ == "__main__":
    main()
