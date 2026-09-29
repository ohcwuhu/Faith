"""多模态融合消融实验。

对同一份标注样本，比较以下方案的识别效果：

    text_only      仅文本模态
    voice_only     仅语音模态
    facial_only    仅面部模态
    fixed_weight   固定权重（0.40 / 0.35 / 0.25，动态融合的对照基线）
    dynamic_weight 线上规则化动态权重（默认方案）

所有方案都调用线上同一份 ``app.services.ai_lab.fusion_service.fuse``，
只在权重入口上区分，保证实验结论与线上行为可比。

用法::

    python experiments/run_fusion_ablation.py
    python experiments/run_fusion_ablation.py --data experiments/data/fusion_samples.jsonl

输出：控制台 Markdown 报表 + ``experiments/results/`` 下的 CSV 明细。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from app.services.ai_lab import facial_buffer, fusion_service  # noqa: E402
from metrics import (  # noqa: E402
    accuracy,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    markdown_table,
    weighted_f1,
    write_csv,
)

LABELS: tuple[str, ...] = tuple(fusion_service.UNIFIED_LABELS)
OTHER_LABELS: tuple[str, ...] = tuple(label for label in LABELS)

#: 实验方案：名称 → 权重（None 表示线上的动态权重）
ARMS: dict[str, dict[str, float] | None] = {
    "text_only": {"text": 1.0, "voice": 0.0, "facial": 0.0},
    "voice_only": {"text": 0.0, "voice": 1.0, "facial": 0.0},
    "facial_only": {"text": 0.0, "voice": 0.0, "facial": 1.0},
    "fixed_weight": {"text": 0.40, "voice": 0.35, "facial": 0.25},
    "dynamic_weight": None,
}

_SID = "experiment-sid"
_FRAME_INTERVAL = 0.4
_MAX_FRAMES = 12


def load_samples(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 样本。"""
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def expand_distribution(spec: Sequence[Any] | None) -> dict[str, float]:
    """把 ``[标签, 置信度]`` 展开为 7 类概率分布。

    真实管线直接使用各模型的输出概率；实验数据只记录主标签与置信度，
    其余概率平分到其他类别，属于可复现的简化约定。
    """
    probs = {label: 0.0 for label in LABELS}
    if not spec:
        return probs

    label, confidence = str(spec[0]), float(spec[1])
    if label not in probs:
        return probs
    confidence = min(1.0, max(0.0, confidence))
    probs[label] = confidence
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    for other in OTHER_LABELS:
        if other != label:
            probs[other] = rest
    return probs


def build_modality_payloads(sample: dict[str, Any]) -> tuple[dict, dict, dict, dict, float, float]:
    """把样本展开为融合引擎需要的三路输入与时间窗。"""
    text_len = int(sample.get("text_len", 20))
    text_label = (sample.get("text") or [None])[0]
    text_probs = expand_distribution(sample.get("text"))
    text_result = {
        "text": "实" * max(text_len, 1),
        "emotion": text_label,
        "confidence": text_probs.get(text_label, 0.0) if text_label else 0.0,
        "probabilities": text_probs,
        "method": "experiment",
    }

    voice_spec = sample.get("voice") or []
    voice_probs = expand_distribution(voice_spec)
    voice_label = voice_spec[0] if voice_spec else None
    voice_result = {
        "emotion": voice_label,
        "confidence": voice_probs.get(voice_label, 0.0) if voice_label else 0.0,
        "probabilities": voice_probs,
        "method": "experiment",
    }

    facial_spec = sample.get("facial") or []
    facial_frames = int(facial_spec[2]) if len(facial_spec) > 2 else 0
    facial_probs = expand_distribution(facial_spec[:2] if facial_spec else None)
    facial_result = {"frame_count": facial_frames}

    # SenseVoice emo 是独立于 emotion2vec 的辅助信号，缺省为 neutral（不触发一致性加权）
    sv_emo_result = {
        "emotion": str(sample.get("sv") or "neutral"),
        "source": "experiment",
    }

    start_ts = 1_700_000_000.0
    end_ts = start_ts + _FRAME_INTERVAL * max(facial_frames, 1)
    return text_result, voice_result, sv_emo_result, facial_result, start_ts, end_ts


def derive_subset_tags(sample: dict[str, Any]) -> list[str]:
    """按样本特征打标签，用于分场景（子集）指标分析。

    子集分析是消融实验的关键：动态权重的作用通常体现在
    模态缺失、低置信度与模态冲突场景，而不是整体平均指标。
    """
    tags: list[str] = []
    text_spec = sample.get("text") or []
    voice_spec = sample.get("voice") or []
    facial_spec = sample.get("facial") or []

    if not facial_spec:
        tags.append("missing_facial")
    if voice_spec and float(voice_spec[1]) < 0.4:
        tags.append("low_voice_confidence")
    if int(sample.get("text_len", 20)) < 5:
        tags.append("short_text")

    dominant = {spec[0] for spec in (text_spec, voice_spec, facial_spec) if spec}
    if len(dominant) > 1:
        tags.append("modality_conflict")
    if not tags:
        tags.append("agreement")
    return tags


def prime_facial_buffer(sample: dict[str, Any], start_ts: float) -> None:
    """按样本写入面部帧，供融合引擎按时间窗读取。"""
    facial_spec = sample.get("facial") or []
    facial_buffer.init_client(_SID)
    if not facial_spec:
        return

    frames = min(int(facial_spec[2]) if len(facial_spec) > 2 else 0, _MAX_FRAMES)
    probs = expand_distribution(facial_spec[:2])
    for index in range(frames):
        facial_buffer.append_frame(
            _SID,
            emotions={},
            score=55,
            raw_probs=probs,
            server_ts=start_ts + index * _FRAME_INTERVAL,
        )


def predict(sample: dict[str, Any], weights: dict[str, float] | None) -> tuple[str, float]:
    """用指定权重跑一次融合，返回（预测标签, 置信度）。"""
    text_result, voice_result, sv_emo_result, _facial, start_ts, end_ts = build_modality_payloads(sample)
    prime_facial_buffer(sample, start_ts)

    result = fusion_service.fuse(
        text_result=text_result,
        voice_result=voice_result,
        sv_emo_result=sv_emo_result,
        sid=_SID,
        record_start_ts=start_ts,
        record_end_ts=end_ts,
        weights_override=weights,
    )
    fusion = result["fusion"]
    return fusion["final_emotion"], float(fusion["overall_confidence"])


def run(data_path: Path) -> int:
    """执行消融实验并输出报表。"""
    samples = load_samples(data_path)
    if not samples:
        print(f"未在 {data_path} 读取到样本")
        return 1

    gold = [sample["gold"] for sample in samples]
    summary_rows: list[list[Any]] = []
    detail_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[str]] = {}
    confidences: dict[str, list[float]] = {}

    for arm_name, weights in ARMS.items():
        preds: list[str] = []
        arm_confidences: list[float] = []
        for sample in samples:
            label, confidence = predict(sample, weights)
            preds.append(label)
            arm_confidences.append(confidence)
        predictions[arm_name] = preds
        confidences[arm_name] = arm_confidences
        correct_flags = [pred == true for pred, true in zip(preds, gold)]
        summary_rows.append([
            arm_name,
            round(accuracy(gold, preds), 4),
            round(macro_f1(gold, preds, LABELS), 4),
            round(weighted_f1(gold, preds, LABELS), 4),
            round(expected_calibration_error(arm_confidences, correct_flags), 4),
        ])

    for index, sample in enumerate(samples):
        detail_rows.append({
            "id": sample["id"],
            "gold": sample["gold"],
            "note": sample.get("note", ""),
            **{arm: predictions[arm][index] for arm in ARMS},
        })

    print("# 多模态融合消融实验")
    print()
    print(f"样本文件：`{data_path.name}`，样本数：{len(samples)}")
    print()
    print("> 注意：data/ 内为流程验证用合成样本，正式结论需替换为真实标注数据。")
    print()
    print(markdown_table(["方案", "准确率", "宏平均F1", "加权F1", "ECE(校准误差)"], summary_rows))
    print()

    # ---- 分场景（子集）指标：动态权重的作用通常体现在困难子集 ----
    subset_names = ["agreement", "modality_conflict", "missing_facial", "low_voice_confidence", "short_text"]
    subset_rows: list[list[Any]] = []
    subset_rows_all_labels: list[list[Any]] = []
    for subset in subset_names:
        indices = [
            index for index, sample in enumerate(samples)
            if subset in derive_subset_tags(sample)
        ]
        if not indices:
            continue
        # 子集内往往只出现部分情绪类别。把 7 类全部计入宏平均时，
        # 未出现的类别按 F1=0 参与平均，会系统性压低困难子集的指标。
        # 因此这里以"仅出现类别"为主口径，同时保留 7 类口径以便对照。
        subset_gold = [gold[index] for index in indices]
        present_labels = [label for label in LABELS if label in set(subset_gold)]
        row: list[Any] = [subset, len(indices), len(present_labels)]
        row_all: list[Any] = [subset, len(indices)]
        for arm in ARMS:
            arm_pred = [predictions[arm][index] for index in indices]
            row.append(round(macro_f1(subset_gold, arm_pred, present_labels), 4))
            row_all.append(round(macro_f1(subset_gold, arm_pred, LABELS), 4))
        subset_rows.append(row)
        subset_rows_all_labels.append(row_all)

    print("## 分场景宏平均F1（仅统计该子集内出现的情绪类别）")
    print()
    print(markdown_table(["子集", "样本数", "出现类别数", *ARMS.keys()], subset_rows))
    print()
    print("对照口径（7 类全算，未出现类别按 F1=0 计入，会压低困难子集）：")
    print()
    print(markdown_table(["子集", "样本数", *ARMS.keys()], subset_rows_all_labels))
    print()

    best_arm = max(ARMS, key=lambda arm: macro_f1(gold, predictions[arm], LABELS))
    print(f"## 逐类指标（{best_arm}）")
    print()
    report = classification_report(gold, predictions[best_arm], LABELS)
    print(markdown_table(
        ["情绪", "精确率", "召回率", "F1", "样本数"],
        [[row["label"], row["precision"], row["recall"], row["f1"], row["support"]] for row in report],
    ))
    print()

    print(f"## 混淆矩阵（{best_arm}）")
    print()
    matrix = confusion_matrix(gold, predictions[best_arm], LABELS)
    print(markdown_table(["真实\\预测", *LABELS], [[g, *[matrix[g][p] for p in LABELS]] for g in LABELS]))
    print()

    write_csv(
        RESULT_DIR / "fusion_ablation_summary.csv",
        ["arm", "accuracy", "macro_f1", "weighted_f1", "ece"],
        [
            {
                "arm": row[0],
                "accuracy": row[1],
                "macro_f1": row[2],
                "weighted_f1": row[3],
                "ece": row[4],
            }
            for row in summary_rows
        ],
    )
    write_csv(
        RESULT_DIR / "fusion_ablation_subsets.csv",
        ["subset", "samples", "present_labels", *[f"{arm}_present" for arm in ARMS],
         *[f"{arm}_all7" for arm in ARMS]],
        [
            {
                "subset": row[0],
                "samples": row[1],
                "present_labels": row[2],
                **{f"{arm}_present": row[3 + offset] for offset, arm in enumerate(ARMS)},
                **{
                    f"{arm}_all7": subset_rows_all_labels[index][2 + offset]
                    for offset, arm in enumerate(ARMS)
                },
            }
            for index, row in enumerate(subset_rows)
        ],
    )
    write_csv(
        RESULT_DIR / "fusion_ablation_details.csv",
        ["id", "gold", "note", *ARMS.keys()],
        detail_rows,
    )
    print(f"明细已写入 `{RESULT_DIR}`")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="多模态融合消融实验")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "fusion_samples.jsonl",
        help="标注样本 JSONL 路径",
    )
    args = parser.parse_args()
    return run(args.data)


if __name__ == "__main__":
    raise SystemExit(main())
