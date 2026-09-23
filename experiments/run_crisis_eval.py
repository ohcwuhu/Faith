"""危机风险分级评测。

对标注语料逐条运行线上同一份规则（``app.services.crisis_rules.assess_crisis``），
输出两个层面的指标：

1. 分级指标：四级（NONE/LOW/MEDIUM/HIGH）的逐类精确率、召回率、F1 与混淆矩阵；
2. 处置指标：以"是否需要建立工单"（MEDIUM/HIGH）为二分类，
   报告漏报率与误报率——这是危机预警最需要向评审说明的两个数字。

用法::

    python experiments/run_crisis_eval.py
    python experiments/run_crisis_eval.py --data experiments/data/crisis_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from app.services.crisis_rules import (  # noqa: E402
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    LEVEL_NONE,
    assess_crisis,
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

LEVELS: tuple[str, ...] = (LEVEL_NONE, LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH)


def load_samples(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 标注语料。"""
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run(data_path: Path) -> int:
    """执行分级评测并输出报表。"""
    samples = load_samples(data_path)
    if not samples:
        print(f"未在 {data_path} 读取到样本")
        return 1

    gold = [sample["gold"] for sample in samples]
    results = [assess_crisis(sample["text"]) for sample in samples]
    pred = [result.level for result in results]

    print("# 危机风险分级评测")
    print()
    print(f"样本文件：`{data_path.name}`，样本数：{len(samples)}")
    print()
    print("> 注意：data/ 内为流程验证用合成语料，正式结论需替换为真实标注数据。")
    print()

    # ---- 分级指标 ----
    report = classification_report(gold, pred, LEVELS)
    print("## 分级指标")
    print()
    print(markdown_table(
        ["等级", "精确率", "召回率", "F1", "样本数"],
        [[row["label"], row["precision"], row["recall"], row["f1"], row["support"]] for row in report],
    ))
    print()
    print(
        f"整体准确率：**{accuracy(gold, pred):.4f}**；"
        f"宏平均F1：**{macro_f1(gold, pred, LEVELS):.4f}**；"
        f"Cohen's kappa：**{cohens_kappa(gold, pred, LEVELS):.4f}**"
    )
    print()

    matrix = confusion_matrix(gold, pred, LEVELS)
    print("## 混淆矩阵")
    print()
    print(markdown_table(
        ["真实\\预测", *LEVELS],
        [[level, *[matrix[level][p] for p in LEVELS]] for level in LEVELS],
    ))
    print()

    # ---- 处置指标（是否建档）----
    gold_flag = [level in (LEVEL_MEDIUM, LEVEL_HIGH) for level in gold]
    pred_flag = [level in (LEVEL_MEDIUM, LEVEL_HIGH) for level in pred]
    tp = sum(1 for g, p in zip(gold_flag, pred_flag) if g and p)
    fp = sum(1 for g, p in zip(gold_flag, pred_flag) if not g and p)
    fn = sum(1 for g, p in zip(gold_flag, pred_flag) if g and not p)
    tn = sum(1 for g, p in zip(gold_flag, pred_flag) if not g and not p)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print("## 处置指标（是否需要建立危机工单）")
    print()
    print(markdown_table(
        ["指标", "含义", "取值"],
        [
            ["命中率/召回率", "应建档样本中被正确识别", f"{recall:.4f}"],
            ["漏报率", "应建档但被判定为无需建档", f"{1 - recall:.4f}"],
            ["精确率", "建档样本中确实应建档", f"{precision:.4f}"],
            ["误报率", "无需建档却被建档", f"{false_positive_rate:.4f}"],
            ["混淆计数", "TP / FP / FN / TN", f"{tp} / {fp} / {fn} / {tn}"],
        ],
    ))
    print()

    # ---- 错误样例 ----
    errors = [
        {
            "id": sample["id"],
            "text": sample["text"],
            "gold": sample["gold"],
            "pred": result.level,
            "score": result.risk_score,
            "reasons": "；".join(result.reasons),
        }
        for sample, result in zip(samples, results)
        if result.level != sample["gold"]
    ]
    if errors:
        print("## 判定不一致样例")
        print()
        print(markdown_table(
            ["id", "文本", "标注", "预测", "评分", "依据"],
            [[e["id"], e["text"], e["gold"], e["pred"], e["score"], e["reasons"]] for e in errors],
        ))
        print()
    else:
        print("## 判定不一致样例")
        print()
        print("无（当前语料全部判定一致）")
        print()

    # ---- 落盘 ----
    write_csv(
        RESULT_DIR / "crisis_eval_levels.csv",
        ["label", "precision", "recall", "f1", "support"],
        report,
    )
    write_csv(
        RESULT_DIR / "crisis_eval_details.csv",
        ["id", "text", "gold", "pred", "score", "reasons"],
        [
            {
                "id": sample["id"],
                "text": sample["text"],
                "gold": sample["gold"],
                "pred": result.level,
                "score": result.risk_score,
                "reasons": "；".join(result.reasons),
            }
            for sample, result in zip(samples, results)
        ],
    )
    print(f"明细已写入 `{RESULT_DIR}`")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="危机风险分级评测")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "crisis_samples.jsonl",
        help="标注语料 JSONL 路径",
    )
    args = parser.parse_args()
    return run(args.data)


if __name__ == "__main__":
    raise SystemExit(main())
