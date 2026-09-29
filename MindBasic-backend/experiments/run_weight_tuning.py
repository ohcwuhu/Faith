"""融合权重标定实验：数据标定权重 vs 线上规则化动态权重。

线上权重由 6 条启发式规则给出（见 ``fusion_service._compute_dynamic_weights``），
阈值全部为经验值。本脚本回答一个评审会问的问题：

    "把权重交给数据去选，能不能比手写规则更好？"

做法：

1. 在权重单纯形上按 ``--step`` 网格搜索（默认 0.05，共 231 组）；
2. 用 **K 折交叉验证** 选择权重：权重只在训练折上选，在校验折上评估，
   避免"在全部样本上挑最好权重"造成的高估；
3. 对照三组方案：基础固定权重（0.40/0.35/0.25）、数据标定权重、线上动态权重；
4. 分场景（模态冲突 / 面部缺失 / 低语调置信度 / 短文本）比较，
   因为动态权重的价值通常体现在这些子集，而不是整体平均。

用法::

    python experiments/run_weight_tuning.py
    python experiments/run_weight_tuning.py --step 0.1 --folds 4

数据：复用 ``experiments/data/fusion_samples.jsonl`` 的既有字段约定。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from run_fusion_ablation import (  # noqa: E402
    LABELS,
    build_modality_payloads,
    derive_subset_tags,
    load_samples,
    predict,
    prime_facial_buffer,
)
from metrics import (  # noqa: E402
    accuracy,
    expected_calibration_error,
    macro_f1,
    markdown_table,
    write_csv,
)

# 线上基础权重（与 fusion_service 常量保持一致）
BASE_WEIGHTS = {"text": 0.40, "voice": 0.35, "facial": 0.25}


def weight_grid(step: float) -> list[dict[str, float]]:
    """在权重单纯形上枚举候选权重（三元组之和恒为 1）。"""
    steps = round(1.0 / step)
    candidates: list[dict[str, float]] = []
    for text in range(steps + 1):
        for voice in range(steps - text + 1):
            facial = steps - text - voice
            candidates.append({
                "text": round(text / steps, 4),
                "voice": round(voice / steps, 4),
                "facial": round(facial / steps, 4),
            })
    return candidates


def evaluate(
    samples: Sequence[dict[str, Any]],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """在给定样本上计算一组权重的指标。``weights=None`` 表示线上动态权重。"""
    gold: list[str] = []
    pred: list[str] = []
    confidences: list[float] = []
    for sample in samples:
        label, confidence = predict(sample, weights)
        gold.append(str(sample.get("gold")))
        pred.append(label)
        confidences.append(confidence)
    correct = [g == p for g, p in zip(gold, pred)]
    return {
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, list(LABELS)),
        "ece": expected_calibration_error(confidences, correct),
    }


def pick_best(
    samples: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """在训练折上选指标最好的权重；指标相同时优先 ECE 更低者。"""
    best: tuple[dict[str, float], dict[str, float]] | None = None
    for weights in candidates:
        metrics = evaluate(samples, weights)
        if best is None or (
            metrics["macro_f1"],
            -metrics["ece"],
            metrics["accuracy"],
        ) > (
            best[1]["macro_f1"],
            -best[1]["ece"],
            best[1]["accuracy"],
        ):
            best = (weights, metrics)
    assert best is not None
    return best


def assign_folds(samples: Sequence[dict[str, Any]], folds: int) -> list[list[dict[str, Any]]]:
    """按排序后的样本轮流分折，保证分折结果可复现且分布均衡。"""
    ordered = sorted(samples, key=lambda item: str(item.get("id", "")))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for index, sample in enumerate(ordered):
        buckets[index % folds].append(sample)
    return buckets


def cross_validate(
    samples: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, float]],
    folds: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """K 折交叉验证：返回交叉验证平均指标与逐折明细。"""
    buckets = assign_folds(samples, folds)
    rows: list[dict[str, Any]] = []
    for index, test_fold in enumerate(buckets):
        train = [sample for j, bucket in enumerate(buckets) if j != index for sample in bucket]
        if not train or not test_fold:
            continue
        weights, train_metrics = pick_best(train, candidates)
        test_metrics = evaluate(test_fold, weights)
        rows.append({
            "fold": index + 1,
            "train_n": len(train),
            "test_n": len(test_fold),
            "weights": f"{weights['text']}/{weights['voice']}/{weights['facial']}",
            "train_macro_f1": round(train_metrics["macro_f1"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_ece": round(test_metrics["ece"], 4),
        })

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return round(sum(values) / len(values), 4) if values else 0.0

    return (
        {
            "accuracy": mean("test_accuracy"),
            "macro_f1": mean("test_macro_f1"),
            "ece": mean("test_ece"),
        },
        rows,
    )


def subset_rows(
    samples: Sequence[dict[str, Any]],
    tuned: dict[str, float],
) -> list[list[object]]:
    """分场景比较"数据标定权重"与"线上动态权重"。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        for tag in derive_subset_tags(sample):
            buckets.setdefault(tag, []).append(sample)

    rows: list[list[object]] = []
    for tag in sorted(buckets):
        group = buckets[tag]
        if not group:
            continue
        dynamic = evaluate(group, None)
        tuned_metrics = evaluate(group, tuned)
        rows.append([
            tag,
            len(group),
            round(dynamic["macro_f1"], 4),
            round(tuned_metrics["macro_f1"], 4),
            round(tuned_metrics["macro_f1"] - dynamic["macro_f1"], 4),
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="融合权重标定实验")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "fusion_samples.jsonl",
        help="标注样本路径（JSONL）",
    )
    parser.add_argument("--step", type=float, default=0.05, help="权重网格步长（默认 0.05）")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数（默认 5）")
    parser.add_argument("--top", type=int, default=15, help="网格结果展示条数（默认 15）")
    args = parser.parse_args()

    if args.step <= 0 or round(1.0 / args.step) * args.step < 0.999:
        raise SystemExit("--step 必须能整除 1，例如 0.05 / 0.1 / 0.2")
    if args.folds < 2:
        raise SystemExit("--folds 至少为 2")

    # 融合引擎每次调用都会写 INFO 日志；网格搜索会放大成千上万行，这里只保留告警
    logging.getLogger("fusion-service").setLevel(logging.WARNING)

    samples = load_samples(args.data)
    candidates = weight_grid(args.step)
    print(f"[数据] {args.data} 共 {len(samples)} 条样本；候选权重 {len(candidates)} 组")

    baseline = evaluate(samples, BASE_WEIGHTS)
    dynamic = evaluate(samples, None)
    cv_metrics, cv_rows = cross_validate(samples, candidates, args.folds)
    tuned, tuned_in_sample = pick_best(samples, candidates)
    tuned_label = f"{tuned['text']}/{tuned['voice']}/{tuned['facial']}"

    # 网格前 N 名（全样本口径，仅用于观察趋势，不作为结论）
    scored = [
        (weights, evaluate(samples, weights))
        for weights in candidates
    ]
    scored.sort(key=lambda item: (item[1]["macro_f1"], -item[1]["ece"]), reverse=True)

    lines: list[str] = ["# 融合权重标定实验", ""]
    lines.append(f"- 样本数：{len(samples)}（{args.data.name}）")
    lines.append(f"- 权重网格：步长 {args.step}，共 {len(candidates)} 组")
    lines.append(f"- 交叉验证：{args.folds} 折，权重只在训练折选择")
    lines.append("")

    lines.append("## 方案对比")
    lines.append("")
    lines.append(markdown_table(
        ["方案", "权重", "准确率", "宏平均 F1", "ECE", "口径"],
        [
            [
                "基础固定权重",
                "0.40/0.35/0.25",
                round(baseline["accuracy"], 4),
                round(baseline["macro_f1"], 4),
                round(baseline["ece"], 4),
                "全样本",
            ],
            [
                "数据标定权重",
                tuned_label,
                cv_metrics["accuracy"],
                cv_metrics["macro_f1"],
                cv_metrics["ece"],
                f"{args.folds} 折交叉验证",
            ],
            [
                "线上动态权重",
                "规则化（6 条）",
                round(dynamic["accuracy"], 4),
                round(dynamic["macro_f1"], 4),
                round(dynamic["ece"], 4),
                "全样本",
            ],
        ],
    ))
    lines.append("")
    lines.append(
        f"> 说明：数据标定权重在全样本上拟合的最优解为 {tuned_label}，"
        f"全样本宏平均 F1 {tuned_in_sample['macro_f1']:.4f}；"
        "交叉验证口径用于避免在全样本上调参带来的高估，报告应引用交叉验证结果。"
    )
    lines.append("")

    lines.append(f"## 网格前 {args.top} 名（全样本口径）")
    lines.append("")
    lines.append(markdown_table(
        ["权重 text/voice/facial", "准确率", "宏平均 F1", "ECE"],
        [
            [
                f"{weights['text']}/{weights['voice']}/{weights['facial']}",
                round(metrics["accuracy"], 4),
                round(metrics["macro_f1"], 4),
                round(metrics["ece"], 4),
            ]
            for weights, metrics in scored[: args.top]
        ],
    ))
    lines.append("")

    lines.append("## 交叉验证逐折明细")
    lines.append("")
    lines.append(markdown_table(
        ["折", "训练样本", "校验样本", "训练折选中权重", "训练 F1", "校验准确率", "校验宏平均 F1", "校验 ECE"],
        [
            [
                row["fold"], row["train_n"], row["test_n"], row["weights"],
                row["train_macro_f1"], row["test_accuracy"],
                row["test_macro_f1"], row["test_ece"],
            ]
            for row in cv_rows
        ],
    ))
    lines.append("")

    lines.append("## 分场景对比（数据标定权重 vs 线上动态权重）")
    lines.append("")
    lines.append(markdown_table(
        ["子集", "样本数", "线上动态 F1", "数据标定 F1", "差值"],
        subset_rows(samples, tuned),
    ))
    lines.append("")

    report = "\n".join(lines)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "weight_tuning_report.md").write_text(report, encoding="utf-8")
    write_csv(
        RESULT_DIR / "weight_tuning_grid.csv",
        ["text", "voice", "facial", "accuracy", "macro_f1", "ece"],
        [
            {
                "text": weights["text"], "voice": weights["voice"], "facial": weights["facial"],
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "ece": round(metrics["ece"], 4),
            }
            for weights, metrics in scored
        ],
    )
    write_csv(
        RESULT_DIR / "weight_tuning_cv.csv",
        ["fold", "train_n", "test_n", "weights", "train_macro_f1", "test_accuracy", "test_macro_f1", "test_ece"],
        cv_rows,
    )

    print(report)
    print(f"[输出] {RESULT_DIR / 'weight_tuning_report.md'}")
    print(f"[输出] {RESULT_DIR / 'weight_tuning_grid.csv'}（全部 {len(scored)} 组）")
    print(f"[输出] {RESULT_DIR / 'weight_tuning_cv.csv'}")


if __name__ == "__main__":
    main()
