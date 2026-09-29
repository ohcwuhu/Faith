"""评测指标工具（纯 Python 实现，无第三方依赖）。

只依赖标准库，保证在答辩现场、离线环境和 CI 中都能直接运行，
避免因为缺少 scikit-learn 而无法复现实验结论。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """准确率。"""
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    correct = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    return correct / len(y_true)


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """混淆矩阵：``matrix[gold][pred] = count``。"""
    _check_same_length(y_true, y_pred)
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in zip(y_true, y_pred):
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def precision_recall_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label: str,
) -> tuple[float, float, float, int]:
    """单个类别的精确率、召回率、F1 与支持度。"""
    _check_same_length(y_true, y_pred)
    tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
    support = sum(1 for gold in y_true if gold == label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, support


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """宏平均 F1（各类别等权，不忽略样本量小的类别）。"""
    scores = [precision_recall_f1(y_true, y_pred, label)[2] for label in labels]
    return sum(scores) / len(scores) if scores else 0.0


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """按支持度加权的 F1。"""
    _check_same_length(y_true, y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0
    weighted = 0.0
    for label in labels:
        _, _, f1, support = precision_recall_f1(y_true, y_pred, label)
        weighted += f1 * support
    return weighted / total


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Cohen's kappa：扣除随机一致后的标注一致性。"""
    _check_same_length(y_true, y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0

    observed = accuracy(y_true, y_pred)
    expected = 0.0
    for label in labels:
        gold_rate = sum(1 for gold in y_true if gold == label) / total
        pred_rate = sum(1 for pred in y_pred if pred == label) / total
        expected += gold_rate * pred_rate
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 10,
) -> float:
    """期望校准误差（ECE）：衡量置信度是否可信。

    Args:
        confidences: 每次预测的置信度，取值 0-1。
        correct: 每次预测是否正确。
        bins: 分箱数量。
    """
    _check_same_length(confidences, correct)
    total = len(confidences)
    if total == 0:
        return 0.0

    bucket_confidence = [0.0] * bins
    bucket_accuracy = [0.0] * bins
    bucket_count = [0] * bins

    for confidence, is_correct in zip(confidences, correct):
        index = min(bins - 1, max(0, int(confidence * bins)))
        bucket_confidence[index] += confidence
        bucket_accuracy[index] += 1.0 if is_correct else 0.0
        bucket_count[index] += 1

    ece = 0.0
    for index in range(bins):
        if bucket_count[index] == 0:
            continue
        mean_confidence = bucket_confidence[index] / bucket_count[index]
        mean_accuracy = bucket_accuracy[index] / bucket_count[index]
        ece += (bucket_count[index] / total) * abs(mean_confidence - mean_accuracy)
    return ece


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[dict[str, object]]:
    """逐类别指标报表（便于直接渲染成 Markdown 表格）。"""
    rows: list[dict[str, object]] = []
    for label in labels:
        precision, recall, f1, support = precision_recall_f1(y_true, y_pred, label)
        rows.append({
            "label": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    return rows


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """把二维数据渲染为 Markdown 表格。"""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """把结果写入 CSV（UTF-8 with BOM，便于 Excel 直接打开）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _check_same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"长度不一致：{len(left)} != {len(right)}")


__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "macro_f1",
    "weighted_f1",
    "cohens_kappa",
    "expected_calibration_error",
    "classification_report",
    "markdown_table",
    "write_csv",
]
