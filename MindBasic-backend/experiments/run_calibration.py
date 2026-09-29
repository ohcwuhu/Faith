"""多模态融合的置信度校准实验。

背景：``fusion_service`` 输出的 ``overall_confidence`` 直接等于融合后的最大概率，
未经校准，导致 ECE 偏高（动态融合约 0.42）。本脚本评估**温度缩放**
（temperature scaling）能否把置信度拉回可信区间，并回答两个问题：

1. 校准后 ECE / NLL / Brier 能改善多少？
2. 置信度作为"该不该降权或追问"的排序依据，是否变得更可靠？

评测协议：

    uncalibrated   线上现状（未校准）
    in_sample      在全部样本上拟合温度 T 后评估（乐观估计，仅作上界参考）
    cv_kfold       K 折交叉验证重复多次：T 只在校验折以外的样本上拟合
                   （诚实估计，报告使用这一行）

温度缩放不改变 argmax，因此准确率与宏平均 F1 完全不变——
这意味着校准是"免费"的：识别效果不动，置信度变得可用。

**本脚本不自己实现校准数学**：温度缩放、NLL 与温度拟合全部调用线上同一份
``app.services.ai_lab.calibration``，保证标定出的 ``FUSION_TEMPERATURE``
与报告里的数字同源、可复现。

用法::

    python experiments/run_calibration.py
    python experiments/run_calibration.py --data experiments/data/fusion_samples.jsonl --folds 5 --repeats 20

输出：控制台 Markdown 报表 + ``experiments/results/`` 下的 CSV。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from app.services.ai_lab import calibration as _calibration  # noqa: E402
from app.services.ai_lab import facial_buffer, fusion_service  # noqa: E402
from metrics import (  # noqa: E402
    accuracy,
    expected_calibration_error,
    macro_f1,
    markdown_table,
    write_csv,
)
from run_fusion_ablation import derive_subset_tags  # noqa: E402

LABELS: tuple[str, ...] = tuple(fusion_service.UNIFIED_LABELS)

_SID = "calibration-sid"
_FRAME_INTERVAL = 0.4
_MAX_FRAMES = 12

_T_GRID = _calibration.TEMPERATURE_GRID
_T_GRID_MIN = _calibration.MIN_TEMPERATURE


# ============================================================
#  概率工具：按标签顺序做序号 <-> 映射转换
# ============================================================
def to_distribution(probs: Sequence[float]) -> dict[str, float]:
    """把按 ``LABELS`` 排序的概率序列转为映射。"""
    return {label: float(probs[index]) for index, label in enumerate(LABELS)}


def to_sequence(distribution: dict[str, float]) -> list[float]:
    """把映射转回按 ``LABELS`` 排序的概率序列。"""
    return [float(distribution.get(label, 0.0)) for label in LABELS]


def apply_temperature_list(probs: Sequence[float], temperature: float) -> list[float]:
    """对序列形式的分部做温度缩放（内部调用线上实现）。"""
    return to_sequence(_calibration.apply_temperature(to_distribution(probs), temperature))


def brier_score(probs_list: Sequence[Sequence[float]], gold_index: Sequence[int]) -> float:
    """多分类 Brier 分数（越低越好）。"""
    if not probs_list:
        return 0.0
    total = 0.0
    for probs, index in zip(probs_list, gold_index):
        for position, prob in enumerate(probs):
            target = 1.0 if position == index else 0.0
            total += (float(prob) - target) ** 2
    return total / len(probs_list)


# ============================================================
#  数据与线上融合路径
# ============================================================
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
    """把 ``[标签, 置信度]`` 展开为 7 类概率分布（与消融脚本保持同一约定）。"""
    probs = {label: 0.0 for label in LABELS}
    if not spec:
        return probs
    label, confidence = str(spec[0]), float(spec[1])
    if label not in probs:
        return probs
    confidence = min(1.0, max(0.0, confidence))
    probs[label] = confidence
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    for other in LABELS:
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
    facial_result = {"frame_count": facial_frames}

    sv_emo_result = {
        "emotion": str(sample.get("sv") or "neutral"),
        "source": "experiment",
    }

    start_ts = 1_700_000_000.0
    end_ts = start_ts + _FRAME_INTERVAL * max(facial_frames, 1)
    return text_result, voice_result, sv_emo_result, facial_result, start_ts, end_ts


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


def fused_probabilities(sample: dict[str, Any]) -> tuple[list[float], str, float]:
    """跑一次**线上动态权重**融合，返回 (七类原始概率, 预测标签, 原始置信度)。

    权重入口不传 ``weights_override``，即与线上完全一致的行为；
    同时显式传入 ``temperature=1.0``，确保取到的是**未校准**分布——
    本脚本要自己控制标定协议，不能被线上环境变量干扰。
    """
    text_result, voice_result, sv_emo_result, _facial, start_ts, end_ts = build_modality_payloads(sample)
    prime_facial_buffer(sample, start_ts)

    result = fusion_service.fuse(
        text_result=text_result,
        voice_result=voice_result,
        sv_emo_result=sv_emo_result,
        sid=_SID,
        record_start_ts=start_ts,
        record_end_ts=end_ts,
        temperature=1.0,
    )
    fusion = result["fusion"]
    return to_sequence(fusion["probabilities_raw"]), str(fusion["final_emotion"]), float(fusion["raw_confidence"])


# ============================================================
#  评测指标
# ============================================================
def confidence_of(probs: Sequence[float]) -> float:
    """置信度 = 最大类别概率。"""
    return max(probs)


def evaluate(
    probs_list: Sequence[Sequence[float]],
    gold: Sequence[str],
    *,
    bins: int = 10,
) -> dict[str, float]:
    """计算一组概率分布的识别与校准指标。"""
    predicted = [LABELS[max(range(len(LABELS)), key=lambda i: probs[i])] for probs in probs_list]
    confidences = [confidence_of(probs) for probs in probs_list]
    correct = [pred == true for pred, true in zip(predicted, gold)]
    gold_index = [LABELS.index(true) for true in gold]

    return {
        "accuracy": accuracy(gold, predicted),
        "macro_f1": macro_f1(gold, predicted, LABELS),
        "ece": expected_calibration_error(confidences, correct, bins=bins),
        "ece_5bins": expected_calibration_error(confidences, correct, bins=5),
        "nll": _calibration.negative_log_likelihood(
            [to_distribution(probs) for probs in probs_list], gold,
        ),
        "brier": brier_score(probs_list, gold_index),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def cross_validated_calibration(
    probs_list: Sequence[Sequence[float]],
    gold: Sequence[str],
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """K 折交叉验证：T 只在训练折上拟合，在校验折上评估。

    返回汇总的诚实估计（把各折校验样本的预测汇总后统一算指标），
    以及每折 T 与每折 ECE 的分布，用于说明参数稳定性。
    """
    count = len(probs_list)
    indices = list(range(count))
    pooled_probs: list[list[float]] = [list(probs_list[i]) for i in indices]
    fold_temperatures: list[float] = []
    fold_eces: list[float] = []

    rng = random.Random(seed)
    for _ in range(repeats):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for fold in range(folds):
            test_index = shuffled[fold::folds]
            test_set = set(test_index)
            train_index = [i for i in indices if i not in test_set]
            if not train_index or not test_index:
                continue

            temperature, _ = _calibration.fit_temperature(
                [to_distribution(probs_list[i]) for i in train_index],
                [gold[i] for i in train_index],
                grid=_T_GRID,
            )
            fold_temperatures.append(temperature)

            scaled_test = [apply_temperature_list(probs_list[i], temperature) for i in test_index]
            gold_test = [gold[i] for i in test_index]
            fold_eces.append(evaluate(scaled_test, gold_test)["ece"])
            for position, i in enumerate(test_index):
                pooled_probs[i] = scaled_test[position]

    summary = evaluate(pooled_probs, gold)
    temperatures = sorted(fold_temperatures)
    eces = sorted(fold_eces)

    return {
        "metrics": summary,
        "fold_count": len(fold_temperatures),
        "t_median": temperatures[len(temperatures) // 2],
        "t_min": temperatures[0],
        "t_max": temperatures[-1],
        "ece_fold_median": eces[len(eces) // 2],
        "ece_fold_min": eces[0],
        "ece_fold_max": eces[-1],
    }


def risk_coverage_table(
    probs_list: Sequence[Sequence[float]],
    gold: Sequence[str],
    coverages: Sequence[float],
) -> list[list[Any]]:
    """风险—覆盖率表：按置信度从高到低保留一部分样本时的准确率。

    这是置信度在项目里的**实际用途**：低置信度时降权或追问。
    排序越准，保留部分的准确率越高。
    """
    pairs = sorted(
        zip(probs_list, gold),
        key=lambda item: confidence_of(item[0]),
        reverse=True,
    )
    rows: list[list[Any]] = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(pairs))))
        kept = pairs[:keep]
        predicted = [LABELS[max(range(len(LABELS)), key=lambda i: probs[i])] for probs, _ in kept]
        golds = [true for _, true in kept]
        rows.append([
            f"{coverage:.0%}",
            keep,
            round(accuracy(golds, predicted), 4),
        ])
    return rows


def subset_calibration_table(
    samples: Sequence[dict[str, Any]],
    raw_probs: Sequence[Sequence[float]],
    gold: Sequence[str],
    *,
    global_temperature: float,
    folds: int,
    repeats: int,
    seed: int,
) -> list[list[Any]]:
    """分场景校准对比：全局温度 vs 按子集单独标定温度。

    动机与消融实验一致——整体指标会被易样本主导。若全局温度是按"整体最优"拟合的，
    它可能让困难子集的置信度过于乐观。这里对每个子集分别给出：

        未校准 / 应用全局温度 / 在该子集上单独标定（交叉验证）
    """
    tags_per_sample = [derive_subset_tags(sample) for sample in samples]
    subsets = ["全部样本", "agreement", "modality_conflict",
               "missing_facial", "low_voice_confidence", "short_text"]
    rows: list[list[Any]] = []

    for subset in subsets:
        if subset == "全部样本":
            indices = list(range(len(samples)))
        else:
            indices = [i for i, tags in enumerate(tags_per_sample) if subset in tags]
        if not indices:
            continue

        sub_probs = [list(raw_probs[i]) for i in indices]
        sub_gold = [gold[i] for i in indices]

        raw = evaluate(sub_probs, sub_gold)
        global_scaled = [apply_temperature_list(probs, global_temperature) for probs in sub_probs]
        global_metrics = evaluate(global_scaled, sub_gold)

        if len(indices) >= folds * 2:
            cv = cross_validated_calibration(
                sub_probs, sub_gold, folds=folds, repeats=repeats, seed=seed,
            )
            own_ece: Any = round(cv["metrics"]["ece"], 4)
            own_t: Any = round(cv["t_median"], 4)
            own_nll: Any = round(cv["metrics"]["nll"], 4)
        else:
            own_ece, own_t, own_nll = "样本不足", "-", "-"

        rows.append([
            subset,
            len(indices),
            round(raw["accuracy"], 4),
            round(raw["mean_confidence"], 4),
            round(raw["ece"], 4),
            round(global_metrics["mean_confidence"], 4),
            round(global_metrics["ece"], 4),
            own_t,
            own_ece,
            own_nll,
        ])
    return rows


# ============================================================
#  主流程
# ============================================================
def run(data_path: Path, folds: int, repeats: int, seed: int) -> int:
    samples = load_samples(data_path)
    if not samples:
        print(f"未在 {data_path} 读取到样本")
        return 1

    gold = [str(sample["gold"]) for sample in samples]
    unknown = sorted({label for label in gold if label not in LABELS})
    if unknown:
        print(f"标注标签不在统一标签空间内：{unknown}")
        return 1

    raw_probs: list[list[float]] = []
    predictions: list[str] = []
    raw_confidences: list[float] = []
    for sample in samples:
        probs, label, confidence = fused_probabilities(sample)
        raw_probs.append(probs)
        predictions.append(label)
        raw_confidences.append(confidence)

    raw_metrics = evaluate(raw_probs, gold)

    # 全样本拟合：只能作为上界参考
    best_t, _ = _calibration.fit_temperature(
        [to_distribution(probs) for probs in raw_probs], gold, grid=_T_GRID,
    )
    in_sample_probs = [apply_temperature_list(probs, best_t) for probs in raw_probs]
    in_sample_metrics = evaluate(in_sample_probs, gold)

    # 交叉验证：报告使用这一行
    cv = cross_validated_calibration(
        raw_probs, gold, folds=folds, repeats=repeats, seed=seed,
    )
    cv_metrics = cv["metrics"]

    print("# 多模态融合置信度校准实验（温度缩放）")
    print()
    print(f"样本文件：`{data_path.name}`，样本数：{len(samples)}，类别数：{len(LABELS)}")
    print()
    print("> 数据说明：`experiments/data/` 内为流程验证用合成样本，")
    print("> 本实验用于验证校准链路的**方法与可行性**，其数值不得作为技术报告中的效果结论。")
    print("> 正式结论需要真实授权数据，且标注规模建议不少于 150 条。")
    print()
    print("> 实现说明：温度缩放与温度拟合调用线上同一份 `app.services.ai_lab.calibration`。")
    print()

    print("## 一、校准前后指标对比")
    print()
    rows = [
        ["未校准（线上现状）", round(raw_metrics["accuracy"], 4), round(raw_metrics["macro_f1"], 4),
         round(raw_metrics["ece"], 4), round(raw_metrics["ece_5bins"], 4),
         round(raw_metrics["nll"], 4), round(raw_metrics["brier"], 4),
         round(raw_metrics["mean_confidence"], 4)],
        ["温度缩放（全样本拟合，乐观）", round(in_sample_metrics["accuracy"], 4),
         round(in_sample_metrics["macro_f1"], 4), round(in_sample_metrics["ece"], 4),
         round(in_sample_metrics["ece_5bins"], 4), round(in_sample_metrics["nll"], 4),
         round(in_sample_metrics["brier"], 4), round(in_sample_metrics["mean_confidence"], 4)],
        ["温度缩放（K折交叉验证，诚实）", round(cv_metrics["accuracy"], 4),
         round(cv_metrics["macro_f1"], 4), round(cv_metrics["ece"], 4),
         round(cv_metrics["ece_5bins"], 4), round(cv_metrics["nll"], 4),
         round(cv_metrics["brier"], 4), round(cv_metrics["mean_confidence"], 4)],
    ]
    print(markdown_table(
        ["方案", "准确率", "宏平均F1", "ECE(10箱)", "ECE(5箱)", "NLL", "Brier", "平均置信度"],
        rows,
    ))
    print()
    print(f"全样本最优温度 T = {best_t:.4f}；"
          f"{folds} 折 × {repeats} 次交叉验证下 T 的中位数 = {cv['t_median']:.4f}"
          f"（范围 {cv['t_min']:.4f} – {cv['t_max']:.4f}，共 {cv['fold_count']} 折）。")
    print()
    if best_t <= _T_GRID_MIN * 1.02:
        print(f"⚠ 全样本最优温度贴在网格下界（{_T_GRID_MIN}），说明真实最优仍在网格之外："
              "这通常意味着数据本身让\"锐化\"收益过大，该数值不可直接作为结论。")
        print()
    print(f"交叉验证各折 ECE 中位数 = {cv['ece_fold_median']:.4f}"
          f"（范围 {cv['ece_fold_min']:.4f} – {cv['ece_fold_max']:.4f}）。")
    print()
    print(f"注意：准确率与宏平均 F1 在校准前后完全相同（{raw_metrics['accuracy']:.4f} / "
          f"{raw_metrics['macro_f1']:.4f}）——温度缩放是单调变换，不改变 argmax，"
          "因此校准只修置信度、不动识别结果。")
    print()

    print("## 二、置信度作为排序依据（风险—覆盖率）")
    print()
    print("按置信度从高到低保留，考察保留子集的准确率：")
    print()
    coverages = (1.0, 0.9, 0.8, 0.7)
    raw_cover = risk_coverage_table(raw_probs, gold, coverages)
    calibrated_pooled = [
        apply_temperature_list(raw_probs[i], cv["t_median"]) for i in range(len(raw_probs))
    ]
    cal_cover = risk_coverage_table(calibrated_pooled, gold, coverages)
    print(markdown_table(
        ["覆盖率", "保留样本数", "未校准准确率", "校准后准确率"],
        [
            [raw_cover[i][0], raw_cover[i][1], raw_cover[i][2], cal_cover[i][2]]
            for i in range(len(coverages))
        ],
    ))
    print()

    print("## 三、分场景校准（关键检验）")
    print()
    print("按整体拟合出来的温度，是否会让困难子集的置信度过于乐观？")
    print()
    subset_rows = subset_calibration_table(
        samples, raw_probs, gold,
        global_temperature=cv["t_median"],
        folds=folds, repeats=repeats, seed=seed,
    )
    print(markdown_table(
        ["子集", "样本数", "准确率", "未校准置信度", "未校准ECE",
         "全局T置信度", "全局T ECE", "本集T", "本集ECE", "本集NLL"],
        subset_rows,
    ))
    print()

    summary_rows = [
        {"protocol": "uncalibrated", **{k: round(v, 4) for k, v in raw_metrics.items()}},
        {"protocol": "in_sample", **{k: round(v, 4) for k, v in in_sample_metrics.items()}},
        {"protocol": "cv_kfold", **{k: round(v, 4) for k, v in cv_metrics.items()}},
    ]
    write_csv(
        RESULT_DIR / "calibration_summary.csv",
        ["protocol", "accuracy", "macro_f1", "ece", "ece_5bins", "nll", "brier", "mean_confidence"],
        summary_rows,
    )
    write_csv(
        RESULT_DIR / "calibration_details.csv",
        ["id", "gold", "pred", "raw_confidence", "calibrated_confidence", "correct"],
        [
            {
                "id": samples[i].get("id", i),
                "gold": gold[i],
                "pred": predictions[i],
                "raw_confidence": round(raw_confidences[i], 4),
                "calibrated_confidence": round(confidence_of(calibrated_pooled[i]), 4),
                "correct": predictions[i] == gold[i],
            }
            for i in range(len(samples))
        ],
    )
    write_csv(
        RESULT_DIR / "calibration_subsets.csv",
        ["subset", "samples", "accuracy", "raw_confidence", "raw_ece",
         "global_t_confidence", "global_t_ece", "own_t", "own_ece", "own_nll"],
        [
            {
                "subset": row[0], "samples": row[1], "accuracy": row[2],
                "raw_confidence": row[3], "raw_ece": row[4],
                "global_t_confidence": row[5], "global_t_ece": row[6],
                "own_t": row[7], "own_ece": row[8], "own_nll": row[9],
            }
            for row in subset_rows
        ],
    )
    print(f"明细已写入 `{RESULT_DIR}`"
          "（calibration_summary.csv / calibration_details.csv / calibration_subsets.csv）")
    print()
    print("## 四、如何把标定结果用到线上")
    print()
    print("在真实标注数据上标定后，把该温度写入后端 `.env` 即可启用：")
    print()
    print("```")
    print("FUSION_CALIBRATION_ENABLED=true")
    print(f"FUSION_TEMPERATURE={cv['t_median']:.4f}   # 请替换为真实数据标定值")
    print("FUSION_CALIBRATION_SOURCE=<标定数据集说明>")
    print("```")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="多模态融合置信度校准实验")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "fusion_samples.jsonl",
                        help="标注样本 JSONL 路径")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数（默认 5）")
    parser.add_argument("--repeats", type=int, default=20, help="交叉验证重复次数（默认 20）")
    parser.add_argument("--seed", type=int, default=20260918, help="随机种子")
    args = parser.parse_args()
    return run(args.data, args.folds, args.repeats, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
