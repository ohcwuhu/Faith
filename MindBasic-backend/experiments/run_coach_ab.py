"""成长教练 A/B 对照实验（盲评）。

回答的问题：**在同一个基础模型、同一组初始问题下，加入阶段化教练机制
是否让回复更好？**

两个实验臂：

    A 普通臂（plain）  只给通用共情倾听提示词，不知道阶段，不注入阶段线索；
    B 阶段臂（staged） 注入平台侧阶段判定（``coach_stage_service``）与
                       教练方法提示词，允许"先澄清 / 定目标 / 收束"。

流程（三个子命令，可分开执行）::

    # 1) 生成两臂回复：离线用现成回复文件，或在线调用 DeepSeek
    python experiments/run_coach_ab.py prepare --provider offline
    python experiments/run_coach_ab.py prepare --provider deepseek

    # 2) 生成盲评表（A/B 顺序按样本 ID 打散，答案存在单独的文件里）
    python experiments/run_coach_ab.py sheet

    # 3) 人工填完 winner_r1（必要时再加 winner_r2）后统计
    python experiments/run_coach_ab.py score

设计要点（评审会追问的三件事）：

* **同基础模型**：两臂使用相同的模型、温度与最大长度，只有提示词与阶段信息不同；
* **盲评**：评分表不显示哪个是 B 臂，映射关系单独存放，可在评分完成后再打开；
* **可复现**：A/B 顺序由 ``--seed`` 决定，重跑得到同一份盲评表。

评分填写方式：``winner_r1`` 列填 ``A`` / ``B`` / ``tie``；
如需双人盲评，再填 ``winner_r2``，脚本会额外报告两名评价者的一致性 kappa。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from _bootstrap import DATA_DIR, RESULT_DIR, ensure_app_importable

ensure_app_importable()

from app.services.coach_stage_service import STAGE_LABELS_CN, decide_stage  # noqa: E402
from metrics import cohens_kappa, markdown_table, write_csv  # noqa: E402

PROMPT_PATH = DATA_DIR / "coach_ab_prompts.jsonl"
RESPONSE_PATH = DATA_DIR / "coach_ab_responses.jsonl"
RAW_PATH = RESULT_DIR / "coach_ab_raw.csv"
SHEET_PATH = RESULT_DIR / "coach_ab_blind_sheet.csv"
KEY_PATH = RESULT_DIR / "coach_ab_key.json"
REPORT_PATH = RESULT_DIR / "coach_ab_report.md"

DEFAULT_SEED = 20261015

#: A 臂：通用共情倾听，不做阶段判断
PLAIN_SYSTEM_PROMPT = (
    "你是一名善于倾听的助手，用温和、简短的中文回应用户，"
    "先承接情绪，再给出一个开放式问题，不提供诊断或治疗建议。"
)

#: B 臂：阶段化成长教练（阶段判定由平台侧引擎给出并注入）
STAGED_SYSTEM_PROMPT = (
    "你是一名成长教练，用温和、简短的中文回应。只做日常成长支持，不做诊断或治疗。\n"
    "你会收到平台侧判定的当前阶段与线索，请据此回应：\n"
    "- 开始交流：帮助用户确定本次想讨论的主题；\n"
    "- 问题探索：澄清事实、感受、顾虑与资源，不急于给建议；\n"
    "- 目标形成：帮助用户说清希望发生的变化；\n"
    "- 行动规划：把目标缩小为可以开始的一步；\n"
    "- 阶段收束：整理问题、目标与下一步。\n"
    "若平台提示模态线索互相矛盾，先用一句澄清式提问确认，再继续推进。"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；坏行直接报错，避免静默丢样本。"""
    if not path.exists():
        raise SystemExit(f"缺少文件：{path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
    return rows


def stage_hint(item: dict[str, Any]) -> str:
    """用线上阶段引擎为 B 臂生成阶段提示。"""
    history = item.get("history") or []
    utterance = str(item.get("utterance") or "")
    turn_index = int(item.get("turn_index") or (len(history) // 2 + 1))
    decision = decide_stage(history, utterance, turn_index=turn_index)
    return (
        f"当前阶段：{decision.stage_label_cn}（{decision.stage}）\n"
        f"目标是否清楚：{'是' if decision.goal_clear else '否'}；"
        f"是否已形成行动：{'是' if decision.action_ready else '否'}；"
        f"建议收束：{'是' if decision.should_summarize else '否'}\n"
        f"判定依据：{'；'.join(decision.evidence) or '无线索'}"
    )


def build_messages(item: dict[str, Any], arm: str) -> list[dict[str, str]]:
    """拼装某一臂的完整消息列表。"""
    system = PLAIN_SYSTEM_PROMPT if arm == "plain" else STAGED_SYSTEM_PROMPT
    if arm == "staged":
        system = system + "\n\n" + stage_hint(item)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for message in item.get("history") or []:
        role = "assistant" if str(message.get("role")) in {"assistant", "coach"} else "user"
        messages.append({"role": role, "content": str(message.get("content") or "")})
    messages.append({"role": "user", "content": str(item.get("utterance") or "")})
    return messages


def call_deepseek(messages: Sequence[dict[str, str]], *, model: str, timeout: int = 60) -> str:
    """调用 DeepSeek Chat Completions（只用标准库，避免额外依赖）。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("在线生成需要环境变量 DEEPSEEK_API_KEY（见 .env.example）")
    payload = json.dumps({
        "model": model,
        "messages": list(messages),
        "temperature": 0.7,
        "max_tokens": 400,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 明确报出状态码，便于排查配额/鉴权
        raise SystemExit(f"DeepSeek 调用失败：HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"DeepSeek 调用失败：{exc.reason}") from exc
    choices = body.get("choices") or []
    if not choices:
        raise SystemExit(f"DeepSeek 返回为空：{body}")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def run_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    """生成两臂回复，写出 ``results/coach_ab_raw.csv``。"""
    items = load_jsonl(args.prompts)
    responses: dict[str, dict[str, str]] = {}

    if args.provider == "offline":
        if not args.responses.exists():
            raise SystemExit(
                f"离线模式需要现成回复文件：{args.responses}\n"
                "可先运行 `--provider dry-run` 导出两份提示词，人工或其它模型生成后\n"
                "按 {\"id\":..., \"plain\":..., \"staged\":...} 的 JSONL 格式放回该路径。"
            )
        for row in load_jsonl(args.responses):
            responses[str(row.get("id"))] = {
                "plain": str(row.get("plain") or ""),
                "staged": str(row.get("staged") or ""),
            }
    elif args.provider == "dry-run":
        for item in items:
            print("=" * 72)
            print(f"[{item.get('id')}] {item.get('scenario', '')}")
            print("--- A 臂（普通）---")
            for message in build_messages(item, "plain"):
                print(f"{message['role']}: {message['content']}")
            print("--- B 臂（阶段化）---")
            for message in build_messages(item, "staged"):
                print(f"{message['role']}: {message['content']}")
        print("\n[dry-run] 未生成回复；请把结果写成 coach_ab_responses.jsonl 后用 --provider offline 重跑。")
        return []
    else:
        for item in items:
            item_id = str(item.get("id"))
            print(f"[生成] {item_id} ...")
            responses[item_id] = {
                arm: call_deepseek(build_messages(item, arm), model=args.model)
                for arm in ("plain", "staged")
            }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in items:
        item_id = str(item.get("id"))
        pair = responses.get(item_id)
        if not pair or not pair.get("plain") or not pair.get("staged"):
            missing.append(item_id)
            continue
        rows.append({
            "id": item_id,
            "scenario": item.get("scenario", ""),
            "utterance": item.get("utterance", ""),
            "stage": decide_stage(
                item.get("history") or [],
                str(item.get("utterance") or ""),
                turn_index=int(item.get("turn_index") or (len(item.get("history") or []) // 2 + 1)),
            ).stage,
            "plain": pair["plain"].replace("\n", " ").strip(),
            "staged": pair["staged"].replace("\n", " ").strip(),
        })

    if missing:
        print(f"[警告] 以下样本缺少回复，未写入：{', '.join(missing)}")
    if not rows:
        raise SystemExit("没有任何可用回复，未生成结果")

    write_csv(RAW_PATH, ["id", "scenario", "utterance", "stage", "plain", "staged"], rows)
    print(f"[输出] {RAW_PATH}（{len(rows)} 条）")
    return rows


def run_sheet(args: argparse.Namespace) -> None:
    """生成盲评表：A/B 顺序按样本 ID 打散，映射单独存放。"""
    if not RAW_PATH.exists():
        raise SystemExit(f"缺少原始回复：{RAW_PATH}，请先执行 prepare")
    # 结果 CSV 由 write_csv 写出（UTF-8 with BOM），读回时必须用 utf-8-sig，
    # 否则第一个表头会带上 BOM 前缀（"\ufeffid"）。
    rows = list(csv.DictReader(RAW_PATH.open("r", encoding="utf-8-sig")))

    key: dict[str, str] = {}
    sheet: list[dict[str, Any]] = []
    for row in rows:
        # 用 样本ID + 种子 派生顺序，保证可复现且与内容无关
        digest = hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest()
        staged_is_a = int(digest[:8], 16) % 2 == 0
        key[row["id"]] = "staged" if staged_is_a else "plain"
        sheet.append({
            "id": row["id"],
            "scenario": row["scenario"],
            "utterance": row["utterance"],
            "response_a": row["staged"] if staged_is_a else row["plain"],
            "response_b": row["plain"] if staged_is_a else row["staged"],
            "winner_r1": "",
            "winner_r2": "",
            "rationale": "",
        })

    random.Random(args.seed).shuffle(sheet)  # 打乱行序，避免固定排列带来的顺序效应
    write_csv(
        SHEET_PATH,
        ["id", "scenario", "utterance", "response_a", "response_b", "winner_r1", "winner_r2", "rationale"],
        sheet,
    )
    KEY_PATH.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[输出] {SHEET_PATH}（{len(sheet)} 条，A/B 已打散）")
    print(f"[输出] {KEY_PATH}（评分完成前请勿打开，以免破坏盲评）")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """胜率的 Wilson 置信区间（小样本下比正态近似更稳）。"""
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + z ** 2 / total
    center = (phat + z ** 2 / (2 * total)) / denominator
    margin = z * ((phat * (1 - phat) / total + z ** 2 / (4 * total ** 2)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _winner_label(value: str, key: str) -> str | None:
    """把评分表里的 A/B 还原为实验臂名称。"""
    value = (value or "").strip().lower()
    if value == "tie" or value == "平" or value == "平局":
        return "tie"
    if value == "a":
        return key
    if value == "b":
        return "plain" if key == "staged" else "staged"
    return None


def run_score(args: argparse.Namespace) -> None:
    """统计盲评结果：阶段臂胜率、置信区间与双人一致性。"""
    if not SHEET_PATH.exists():
        raise SystemExit(f"缺少盲评表：{SHEET_PATH}，请先执行 sheet")
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(SHEET_PATH.open("r", encoding="utf-8-sig")))

    decided = {"staged": 0, "plain": 0, "tie": 0}
    rated = 0
    rater1: list[str] = []
    rater2: list[str] = []
    detail: list[dict[str, Any]] = []

    for row in rows:
        arm_key = key.get(row["id"], "staged")
        r1 = _winner_label(row.get("winner_r1", ""), arm_key)
        r2 = _winner_label(row.get("winner_r2", ""), arm_key)
        if r1 is None:
            continue
        rated += 1
        decided[r1] += 1
        rater1.append(r1)
        if r2 is not None:
            rater2.append(r2)
        detail.append({
            "id": row["id"],
            "scenario": row["scenario"],
            "rater1": r1,
            "rater2": r2 or "",
            "rationale": row.get("rationale", ""),
        })

    lines: list[str] = ["# 成长教练 A/B 盲评结果", ""]
    lines.append(f"- 样本数：{len(rows)}")
    lines.append(f"- 已评分：{rated}")
    lines.append("")

    if rated == 0:
        lines.append("尚未评分：请在盲评表中填写 `winner_r1`（A / B / tie）后重新运行 score。")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"[输出] {REPORT_PATH}")
        return

    decisive = decided["staged"] + decided["plain"]
    win_rate = decided["staged"] / decisive if decisive else 0.0
    low, high = wilson_interval(decided["staged"], decisive)

    lines.append("## 结果")
    lines.append("")
    lines.append(markdown_table(
        ["判定", "数量", "占比（含平局）"],
        [
            ["阶段臂更好", decided["staged"], f"{decided['staged'] / rated:.4f}"],
            ["普通臂更好", decided["plain"], f"{decided['plain'] / rated:.4f}"],
            ["平局", decided["tie"], f"{decided['tie'] / rated:.4f}"],
        ],
    ))
    lines.append("")
    lines.append(f"- 阶段臂胜率（仅计非平局，{decisive} 条）：**{win_rate:.4f}**")
    lines.append(f"- Wilson 95% 置信区间：{low:.4f} – {high:.4f}")
    if rater2:
        kappa = cohens_kappa(rater1[:len(rater2)], rater2, ["staged", "plain", "tie"])
        lines.append(f"- 两名评价者一致性 kappa（{len(rater2)} 条）：{kappa:.4f}")
    lines.append("")
    lines.append("## 逐条明细")
    lines.append("")
    lines.append(markdown_table(
        ["ID", "场景", "评价者1", "评价者2", "理由"],
        [
            [row["id"], row["scenario"], row["rater1"], row["rater2"] or "—", row["rationale"] or "—"]
            for row in detail
        ],
    ))
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_csv(
        RESULT_DIR / "coach_ab_scores.csv",
        ["id", "scenario", "rater1", "rater2", "rationale"],
        detail,
    )
    print(report)
    print(f"[输出] {REPORT_PATH}")
    print(f"[输出] {RESULT_DIR / 'coach_ab_scores.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="成长教练 A/B 盲评实验")
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["prepare", "sheet", "score", "all"],
        help="执行阶段（默认 all：prepare → sheet → score）",
    )
    parser.add_argument("--provider", default="offline", choices=["offline", "deepseek", "dry-run"])
    parser.add_argument("--model", default="deepseek-chat", help="在线生成使用的模型")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="盲评打散随机种子")
    parser.add_argument("--prompts", type=Path, default=PROMPT_PATH, help="初始问题集（JSONL）")
    parser.add_argument("--responses", type=Path, default=RESPONSE_PATH, help="离线回复文件（JSONL）")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.action in {"prepare", "all"}:
        rows = run_prepare(args)
        if not rows:
            return
    if args.action in {"sheet", "all"}:
        run_sheet(args)
    if args.action in {"score", "all"}:
        run_score(args)


if __name__ == "__main__":
    main()
