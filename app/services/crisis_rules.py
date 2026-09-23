"""危机风险分级规则（纯领域逻辑）。

本模块只做一件事：把一段文本（以及可选的模态信号）映射为结构化风险等级。
它不读写数据库、不依赖应用配置，因此接口层、SocketIO 实时管线与离线评测脚本
可以复用完全相同的判定逻辑，避免"线上判定"与"评测判定"口径不一致。

分级口径
--------
    HIGH   明确的自伤/自杀意图、计划或手段信号
    MEDIUM 强烈的负性体验、无望感、自我否定，且指向用户本人
    LOW    出现风险语汇，但被否定/假设/转述语境削弱，仅作分析留痕
    NONE   未命中任何规则

证据计分规则
------------
    明确的自伤/自杀表达（未被否定）    +70
    强负性表达（指向本人）             +35（第二条起每条 +15）
    判定为否定/假设语境的表达          -40（显式否认不应单独触发工单）
    紧迫性线索（计划/手段/时点）       +20（仅在存在明确表达时计入）
    语音与面部一致指向持续负性情绪     +10

判定不变量
----------
    1. 没有"明确的自伤/自杀表达"时，任何证据叠加都不会判为 HIGH；
    2. 同一关键词只计一次，避免重复堆分；
    3. 转述语境（如"电影里的主角想死"）封顶 MEDIUM，不按用户本人高危处理；
    4. 显式否认（如"我不会自杀"）只判 LOW，用于留痕而非打扰值班人员。

    等级阈值：>=70 判 HIGH，>=35 判 MEDIUM，>=10 判 LOW，其余 NONE。

阈值与关键词均为模块级常量，便于标注数据积累后进行网格搜索标定
（标定入口见 ``experiments/run_crisis_eval.py``）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# ============================================================
#  等级定义
# ============================================================
LEVEL_NONE = "NONE"
LEVEL_LOW = "LOW"
LEVEL_MEDIUM = "MEDIUM"
LEVEL_HIGH = "HIGH"

LEVEL_ORDER: dict[str, int] = {
    LEVEL_NONE: 0,
    LEVEL_LOW: 1,
    LEVEL_MEDIUM: 2,
    LEVEL_HIGH: 3,
}

LEVEL_LABEL_CN: dict[str, str] = {
    LEVEL_NONE: "未检出",
    LEVEL_LOW: "低风险",
    LEVEL_MEDIUM: "中风险",
    LEVEL_HIGH: "高风险",
}

#: 需要建立危机工单的等级（LOW 仅留痕，不打扰值班人员）
LEVELS_REQUIRING_FLAG: tuple[str, ...] = (LEVEL_MEDIUM, LEVEL_HIGH)

# ============================================================
#  评分权重与阈值
# ============================================================
#: 明确的自伤/自杀表达
SCORE_HIGH_KEYWORD = 70
#: 强负性表达（指向本人）
SCORE_MEDIUM_KEYWORD = 35
#: 第二条及以后的强负性表达增量
SCORE_MEDIUM_KEYWORD_EXTRA = 15
#: 判定为否定/假设语境时的减分（显式否认不应单独触发工单）
SCORE_NEGATION_PENALTY = 40
#: 紧迫性线索（计划、手段、时点）
SCORE_ESCALATION_CUE = 20
#: 临近性线索（明确时点或已下定决心的计划）
SCORE_IMMINENCE_CUE = 35
#: 语音与面部一致指向持续负性情绪
SCORE_MODALITY_CONSENSUS = 10

#: 仅"明确的自伤/自杀表达"可以判为 HIGH；其余证据叠加封顶到此分值
SCORE_CAP_WITHOUT_EXPLICIT_RISK = 60
#: 显式否认语境最多累计到此分值（即最多判 LOW，不建立工单）
SCORE_CAP_WITH_DENIAL = 30
#: 缓解/恢复语境最多累计到此分值（即最多判 LOW，不建立工单）
SCORE_CAP_WITH_RECOVERY = 30
#: 有模块信号佐证时，明确表达的加分上限（防止直接飙到 100 掩盖证据强度）
SCORE_CAP_TOTAL = 100

#: 等级阈值
THRESHOLD_HIGH = 70
THRESHOLD_MEDIUM = 35
THRESHOLD_LOW = 10

# ============================================================
#  关键词表
# ============================================================
#: 明确的自伤/自杀相关表达
HIGH_RISK_KEYWORDS: tuple[str, ...] = (
    "自杀",
    "想死",
    "去死",
    "不想活",
    "不活了",
    "活不下去",
    "活着没意思",
    "活着没意义",
    "结束生命",
    "结束自己",
    "了结自己",
    "轻生",
    "寻死",
    "自我了断",
    "伤害自己",
    "自残",
    "割腕",
    "跳楼",
    "上吊",
    "遗书",
    "不想存在",
    "想离开这个世界",
    "永远睡过去",
)

#: 高危表达的形态变体（正则）：中文常有"伤害过自己""跳下去"等屈折与搭配变化，
#: 纯关键词匹配会漏检，因此对高风险类别补充正则规则。
HIGH_RISK_PATTERNS: tuple[str, ...] = (
    r"伤[害过]{0,2}自己",
    r"自[己我]{0,1}(伤|残)",
    r"割[腕手]",
    r"跳[楼下河]",
    r"上吊|绳索|绳子上",
    r"安眠药|农药|老鼠药",
    r"不想[再]?醒(来|过来)",
    r"结束[自己我]{0,1}(的)?生命",
    r"了结[自己我]{0,1}",
    r"轻生|寻死",
    r"遗书|遗言",
)

#: 强负性表达的形态变体（正则）
MEDIUM_RISK_PATTERNS: tuple[str, ...] = (
    r"撑不[住下]去",
    r"熬不[住下]去",
    r"扛不[住下]去",
    r"不想[再]?撑",
    r"活(得|着)(好)?累",
    r"没(有)?(任何)?意思",
    r"提不起(兴趣|精神)",
    r"想(过)?放弃",
    r"多[余](的)?(人)?",
)

#: 强烈负性体验 / 无望感 / 自我否定（需指向用户本人或叠加出现）
MEDIUM_RISK_KEYWORDS: tuple[str, ...] = (
    "绝望",
    "没希望",
    "没有希望",
    "看不到希望",
    "撑不下去",
    "扛不住",
    "坚持不住",
    "熬不下去",
    "活得累",
    "活得好累",
    "崩溃",
    "想消失",
    "想逃离",
    "没有意义",
    "毫无意义",
    "多余的人",
    "是累赘",
    "拖累",
    "讨厌自己",
    "恨自己",
    "我很没用",
    "一无是处",
    "惩罚自己",
    "整夜失眠",
    "长期失眠",
    "整夜睡不着",
)

#: 否定 / 假设 / 反事实语境线索
NEGATION_CUES: tuple[str, ...] = (
    "不",
    "没",
    "别",
    "未",
    "无",
    "不会",
    "没有",
    "不是",
    "不至于",
    "从来没",
    "从不",
    "并不",
    "才不",
    "绝不",
)

#: 转述 / 引用语境线索（命中时封顶 MEDIUM，避免把他人故事判成用户本人高危）
ATTRIBUTION_CUES: tuple[str, ...] = (
    "他说",
    "她说",
    "朋友说",
    "同学说",
    "别人说",
    "新闻",
    "报道",
    "电影",
    "电视剧",
    "小说",
    "歌词",
    "游戏",
    "段子",
    "漫画",
    "视频里",
    "书里",
    "课上",
)

#: 口语夸张语境的形态（如"这题难得我想死""累得我想死"）：
#: 这类表达不表示真实自伤意图，降级为低风险留痕。
HYPERBOLE_PATTERNS: tuple[str, ...] = (
    r"(难|累|困|饿|气|笑|热|烦|忙|痛)(得|到|死我)",
    r"(笑|气|饿|困|热|累)死(我)?了",
)

#: 缓解 / 恢复语境线索：出现时最多判 LOW，不建立工单
RECOVERY_CUES: tuple[str, ...] = (
    "好多了",
    "已经好了",
    "已经过去",
    "过去了",
    "缓过来",
    "没事了",
    "睡一觉就",
    "就好了",
    "现在好",
)

#: 紧迫性线索：计划、手段、时点
ESCALATION_CUES: tuple[str, ...] = (
    "遗书",
    "已经决定",
    "决定了",
    "打算",
    "计划",
    "准备好",
    "准备好了",
    "今晚",
    "明天就",
    "现在就去",
    "最后一次",
    "楼顶",
    "天台",
    "绳子",
    "准备了刀",
    "拿着刀",
    "安眠药",
    "农药",
    "跳下去",
)

#: 临近性线索：明确的时点或已下定决心的计划。
#: 与"自身即主语的强负性表达"同时出现时，风险等级可以升级到 HIGH，
#: 对应"已经决定明天就去天台"这类带有明确计划的表达。
IMMINENCE_CUES: tuple[str, ...] = (
    "已经决定",
    "决定了",
    "已经想好",
    "想好了",
    "今晚",
    "明天",
    "现在就去",
    "最后一次",
    "就在这几天",
)

#: 第一人称线索（MEDIUM 需要指向本人）
FIRST_PERSON_CUES: tuple[str, ...] = ("我", "自己", "本人", "咱")

#: 自身即为主语的强负性表达：中文常省略主语，这些短语不需要第一人称标记
SELF_REFERENTIAL_MEDIUM_KEYWORDS: frozenset[str] = frozenset({
    "绝望",
    "撑不下去",
    "扛不住",
    "坚持不住",
    "熬不下去",
    "活得累",
    "活得好累",
    "想消失",
    "想逃离",
    "多余的人",
    "是累赘",
    "讨厌自己",
    "恨自己",
    "我很没用",
    "一无是处",
    "惩罚自己",
    "整夜失眠",
    "长期失眠",
    "整夜睡不着",
})

#: 例外：这些强负性规则描述的是对象而非说话人本人，仍需第一人称线索配合
NON_SELF_REFERENTIAL_MEDIUM_PATTERNS: frozenset[str] = frozenset({
    r"没(有)?(任何)?意思",
})

#: 多模态负性情绪标签（与 fusion_service.UNIFIED_LABELS 对齐）
NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"sad", "fearful", "disgusted"})

#: 否定线索的观察窗口（字符数，向前回溯）
_NEGATION_WINDOW = 3
#: 转述线索的观察窗口（字符数，向前回溯）
_ATTRIBUTION_WINDOW = 10

#: 紧跟在否定词之后、使其成为普通构词而非否定语义的字符
#: （例如"不好""不仅""不断"，它们并不否定其后的风险词）
_NEGATION_EXCLUSION_FOLLOWERS = frozenset(
    {"好", "错", "仅", "但", "断", "管", "安", "良", "久", "止", "得", "少"}
)

#: 预编译正则（模块导入时编译一次，避免每次判定重复编译）
_HIGH_RISK_REGEX: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in HIGH_RISK_PATTERNS
)
_MEDIUM_RISK_REGEX: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in MEDIUM_RISK_PATTERNS
)
_HYPERBOLE_REGEX: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in HYPERBOLE_PATTERNS
)

#: 强负性正则的源码集合，用于判断某次命中是否来自正则规则
_MEDIUM_RISK_REGEX_SOURCES = frozenset(
    pattern.pattern for pattern in _MEDIUM_RISK_REGEX
)


# ============================================================
#  数据结构
# ============================================================
@dataclass(frozen=True)
class ModalitySignals:
    """参与风险判定的模态信号（全部可选）。

    Attributes:
        voice_emotion: 语音情绪标签（英文统一标签，如 ``sad``）。
        voice_confidence: 语音情绪置信度，0-1。
        facial_emotion: 面部情绪标签（英文统一标签）。
        facial_confidence: 面部情绪置信度，0-1。
        facial_frames: 参与聚合的面部帧数，用于判断信号是否稳定。
    """

    voice_emotion: str | None = None
    voice_confidence: float = 0.0
    facial_emotion: str | None = None
    facial_confidence: float = 0.0
    facial_frames: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ModalitySignals":
        """从字典安全构造（容忍缺字段与类型异常）。"""
        if not data:
            return cls()

        return cls(
            voice_emotion=_as_emotion(data.get("voice_emotion")),
            voice_confidence=_as_float(data.get("voice_confidence")),
            facial_emotion=_as_emotion(data.get("facial_emotion")),
            facial_confidence=_as_float(data.get("facial_confidence")),
            facial_frames=_as_int(data.get("facial_frames")),
        )


@dataclass(frozen=True)
class CrisisAssessment:
    """结构化风险评估结果。

    Attributes:
        level: ``NONE`` / ``LOW`` / ``MEDIUM`` / ``HIGH``。
        risk_score: 0-100 的累计风险分。
        matched_keywords: 命中的关键词（去重、保持出现顺序）。
        reasons: 人类可读的判定依据，用于工单留痕与复核。
        signals: 参与判定的模态信号。
    """

    level: str
    risk_score: int
    matched_keywords: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    signals: ModalitySignals = field(default_factory=ModalitySignals)

    @property
    def flagged(self) -> bool:
        """是否需要建立危机工单。"""
        return self.level in LEVELS_REQUIRING_FLAG

    @property
    def level_label_cn(self) -> str:
        return LEVEL_LABEL_CN.get(self.level, self.level)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入数据库 / 接口响应的普通字典。"""
        return {
            "level": self.level,
            "levelLabel": self.level_label_cn,
            "riskScore": self.risk_score,
            "matchedKeywords": list(self.matched_keywords),
            "reasons": list(self.reasons),
            "flagged": self.flagged,
        }


# ============================================================
#  文本归一化
# ============================================================
_PUNCTUATION_RE = re.compile(r"[\s，。！？!,.;；:：、~～…\-—\"'“”‘’()（）\[\]【】]+")

#: 标点替换为分句边界标记：既保留索引对齐，又阻止否定语义跨从句回溯
_CLAUSE_BOUNDARY = "\u0001"


def _normalize(text: str) -> str:
    """把标点替换为分句边界，便于按从句分析否定语境。

    标点不直接删除，是为了让"我不想去死""活得好累"这类跨从句文本
    不会因为窗口回溯而互相污染，同时保留"想死"整体匹配能力。
    """
    return _PUNCTUATION_RE.sub(_CLAUSE_BOUNDARY, text)


def _lookback(
    normalized: str,
    index: int,
    window: int,
    cues: Sequence[str],
    *,
    exclusion_followers: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """回溯 ``index`` 之前的 ``window`` 个字符，返回命中的线索。

    Args:
        normalized: 归一化后的文本。
        index: 风险词起始下标。
        window: 向前回溯的字符数。
        cues: 待匹配的线索集合。
        exclusion_followers: 线索后紧接这些字符时不视为命中
            （用于排除"不好""不仅"这类构词干扰）。
    """
    start = max(0, index - window)
    context = normalized[start:index]
    # 只分析当前从句：跨越标点边界的否定词不作用于本风险词
    boundary = context.rfind(_CLAUSE_BOUNDARY)
    if boundary != -1:
        context = context[boundary + 1:]
    if not context:
        return ()
    hits: list[str] = []
    for cue in cues:
        position = context.find(cue)
        while position != -1:
            follower_index = position + len(cue)
            follower = context[follower_index] if follower_index < len(context) else ""
            if follower not in exclusion_followers:
                hits.append(cue)
                break
            position = context.find(cue, position + 1)
    return tuple(hits)


# ============================================================
#  主入口
# ============================================================
def assess_crisis(
    text: str | None,
    *,
    signals: ModalitySignals | Mapping[str, Any] | None = None,
    extra_high_keywords: Iterable[str] = (),
) -> CrisisAssessment:
    """对一段文本（配合可选模态信号）给出结构化风险分级。

    Args:
        text: 待判定文本（用户输入、ASR 转写、日记或社区内容）。
        signals: 模态信号，支持 :class:`ModalitySignals` 或等价字典。
        extra_high_keywords: 追加的高危关键词（用于承载环境变量配置，
            保持与历史 ``CRISIS_KEYWORDS`` 配置兼容）。

    Returns:
        :class:`CrisisAssessment`。文本为空时返回 ``NONE``。
    """
    modality = (
        signals
        if isinstance(signals, ModalitySignals)
        else ModalitySignals.from_mapping(signals)
    )

    raw_text = (text or "").strip()
    if not raw_text:
        return CrisisAssessment(level=LEVEL_NONE, risk_score=0, signals=modality)

    normalized = _normalize(raw_text)
    if not normalized:
        return CrisisAssessment(level=LEVEL_NONE, risk_score=0, signals=modality)

    extra = tuple(k.strip() for k in extra_high_keywords if k and k.strip())
    # 去重：环境变量配置的关键词可能与内置规则表重叠（如"自杀"）
    high_keywords = tuple(dict.fromkeys(HIGH_RISK_KEYWORDS + extra))

    score = 0
    denial_score = 0
    matched: list[str] = []
    reasons: list[str] = []
    positive_hits = 0
    attributed = False

    # 按文本区间去重：重叠命中（"不想活" 与 "不想活了"）只计一次
    consumed: list[tuple[int, int]] = []
    hyperbole_spans = [
        (match.start(), match.end()) for pattern in _HYPERBOLE_REGEX for match in pattern.finditer(normalized)
    ]
    for start, _end, keyword, _source in _collect_spans(
        normalized, high_keywords, _HIGH_RISK_REGEX, consumed
    ):
        matched.append(keyword)
        negations = _lookback(
            normalized,
            start,
            _NEGATION_WINDOW,
            NEGATION_CUES,
            exclusion_followers=_NEGATION_EXCLUSION_FOLLOWERS,
        )
        attributions = _lookback(normalized, start, _ATTRIBUTION_WINDOW, ATTRIBUTION_CUES)
        if _has_hyperbole_context(start, hyperbole_spans):
            denial_score += SCORE_HIGH_KEYWORD - SCORE_NEGATION_PENALTY
            reasons.append(f"命中「{keyword}」但属于口语夸张用法，不作高危处理")
        elif negations:
            denial_score += SCORE_HIGH_KEYWORD - SCORE_NEGATION_PENALTY
            reasons.append(f"命中「{keyword}」但处于否定/假设语境（{negations[0]}）")
        else:
            positive_hits += 1
            score += SCORE_HIGH_KEYWORD
            reasons.append(f"命中高风险表达「{keyword}」")
            if attributions:
                attributed = True
                reasons.append(f"该表达疑似转述（{attributions[0]}），不作为本人高危处理")

    # 显式否认只做留痕：累计后封顶 LOW，避免"我不会自杀"被反复升级成工单
    score += min(denial_score, SCORE_CAP_WITH_DENIAL)

    first_person = any(cue in normalized for cue in FIRST_PERSON_CUES)
    medium_hits = 0
    medium_hit_keywords: list[str] = []
    medium_hit_sources: list[str] = []
    for start, _end, keyword, source in _collect_spans(
        normalized, MEDIUM_RISK_KEYWORDS, _MEDIUM_RISK_REGEX, consumed
    ):
        negations = _lookback(
            normalized,
            start,
            _NEGATION_WINDOW,
            NEGATION_CUES,
            exclusion_followers=_NEGATION_EXCLUSION_FOLLOWERS,
        )
        if negations:
            continue
        medium_hits += 1
        medium_hit_keywords.append(keyword)
        medium_hit_sources.append(source)
        matched.append(keyword)
        reasons.append(f"命中强负性表达「{keyword}」")

    # MEDIUM 要求指向本人（显式第一人称 / 自身即主语的短语），
    # 或叠加出现两次以上，降低"作业多到崩溃"这类夸张修辞造成的误报
    self_referential = any(
        keyword in SELF_REFERENTIAL_MEDIUM_KEYWORDS for keyword in medium_hit_keywords
    ) or any(
        source in _MEDIUM_RISK_REGEX_SOURCES
        and source not in NON_SELF_REFERENTIAL_MEDIUM_PATTERNS
        for source in medium_hit_sources
    )
    if medium_hits and (first_person or self_referential or medium_hits >= 2):
        # 强负性表达可以累加，但单独存在时永远不足以判 HIGH
        medium_score = SCORE_MEDIUM_KEYWORD + SCORE_MEDIUM_KEYWORD_EXTRA * (medium_hits - 1)
        score += min(medium_score, SCORE_CAP_WITHOUT_EXPLICIT_RISK)
        if not positive_hits:
            reasons.append("强负性表达均指向用户本人")

    # 紧迫性/临近性线索只在存在正向证据时计入，避免"明天要交作业"这类无关表达被计分
    plan_evidence = bool(positive_hits) or (medium_hits > 0 and self_referential)
    plan_cue_present = False
    if plan_evidence:
        escalation = [cue for cue in ESCALATION_CUES if cue in normalized]
        if escalation:
            score += SCORE_ESCALATION_CUE
            reasons.append(f"存在紧迫性线索：{'、'.join(escalation[:3])}")
        imminence = [cue for cue in IMMINENCE_CUES if cue in normalized]
        if imminence:
            score += SCORE_IMMINENCE_CUE
            reasons.append(f"存在临近性线索：{'、'.join(imminence[:3])}")
        plan_cue_present = bool(escalation or imminence)
        if attributed and positive_hits:
            # 转述他人故事（电影/新闻/他人转述）：封顶 MEDIUM
            score = min(score, THRESHOLD_HIGH - 1)

    # 多模态一致负性信号：文本证据不足时用于提示"值得关注"，文本已高危时仅加权重
    if _has_negative_consensus(modality):
        score += SCORE_MODALITY_CONSENSUS
        reasons.append("语音与面部信号一致指向持续负性情绪")

    if not positive_hits and not (self_referential and plan_cue_present):
        # 不变量：没有明确的自伤/自杀表达时，只有"指向本人的强负性表达 + 计划/时点线索"
        # 组合才允许判为 HIGH，其余证据叠加封顶在 MEDIUM
        score = min(score, SCORE_CAP_WITHOUT_EXPLICIT_RISK)

    # 缓解/恢复语境（"现在好多了""睡一觉就好了"）：仅削弱中等证据，不影响明确的自伤表达
    if not positive_hits and any(cue in normalized for cue in RECOVERY_CUES):
        if score > SCORE_CAP_WITH_RECOVERY:
            reasons.append("上下文含缓解/恢复表述，等级上限调整为低风险")
        score = min(score, SCORE_CAP_WITH_RECOVERY)

    score = max(0, min(SCORE_CAP_TOTAL, score))
    level = _level_for_score(score)

    return CrisisAssessment(
        level=level,
        risk_score=score,
        matched_keywords=tuple(dict.fromkeys(matched)),
        reasons=tuple(dict.fromkeys(reasons)),
        signals=modality,
    )


def detect_crisis(text: str | None) -> bool:
    """兼容旧接口：是否存在需要建档的危机风险（MEDIUM 及以上）。"""
    return assess_crisis(text).flagged


# ============================================================
#  内部工具
# ============================================================
def _as_float(value: Any) -> float:
    """安全转换浮点数，异常值按 0 处理（脏数据不应中断判定）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    """安全转换整数，异常值按 0 处理。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_emotion(value: Any) -> str | None:
    """统一情绪标签大小写；空值返回 ``None``。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _find_all(haystack: str, needle: str) -> list[int]:
    """返回 ``needle`` 在 ``haystack`` 中的所有起始下标。"""
    if not needle:
        return []
    positions: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        positions.append(start)
        start = haystack.find(needle, start + len(needle))
    return positions


def _collect_spans(
    normalized: str,
    keywords: Sequence[str],
    patterns: Sequence[re.Pattern[str]],
    consumed: list[tuple[int, int]],
) -> list[tuple[int, int, str, str]]:
    """收集关键词与正则命中区间，并跳过与已计分区间重叠的命中。

    Args:
        normalized: 归一化文本。
        keywords: 待匹配的关键词集合。
        patterns: 待匹配的编译后正则。
        consumed: 已计分的区间列表，函数会原地追加本次采纳的区间，
            使高危命中与强负性命中之间也不会重复计分。

    Returns:
        ``[(start, end, 命中文本, 来源规则), ...]``，按出现位置升序。
        来源规则为命中的关键词本身或正则表达式源码，供上层区分规则类型。
    """
    spans: list[tuple[int, int, str, str]] = []
    for keyword in keywords:
        for index in _find_all(normalized, keyword):
            spans.append((index, index + len(keyword), keyword, keyword))
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            spans.append((match.start(), match.end(), match.group(0), pattern.pattern))
    spans.sort()

    accepted: list[tuple[int, int, str, str]] = []
    for start, end, text, source in spans:
        if any(start < existing_end and end > existing_start for existing_start, existing_end in consumed):
            continue
        consumed.append((start, end))
        accepted.append((start, end, text, source))
    return accepted


def _level_for_score(score: int) -> str:
    """分值 → 风险等级。"""
    if score >= THRESHOLD_HIGH:
        return LEVEL_HIGH
    if score >= THRESHOLD_MEDIUM:
        return LEVEL_MEDIUM
    if score >= THRESHOLD_LOW:
        return LEVEL_LOW
    return LEVEL_NONE


def _has_hyperbole_context(
    start: int,
    hyperbole_spans: Sequence[tuple[int, int]],
    *,
    max_gap: int = 4,
) -> bool:
    """判断风险词是否紧跟在口语夸张结构之后。

    Args:
        start: 风险词起始下标。
        hyperbole_spans: 夸张结构匹配到的区间。
        max_gap: 夸张结构结束位置到风险词之间允许的最大间隔字符数。
    """
    for _span_start, span_end in hyperbole_spans:
        if 0 <= start - span_end <= max_gap:
            return True
    return False


def _has_negative_consensus(signals: ModalitySignals) -> bool:
    """语音与面部是否一致指向持续负性情绪。"""
    voice_negative = (
        signals.voice_emotion in NEGATIVE_EMOTIONS and signals.voice_confidence >= 0.6
    )
    facial_negative = (
        signals.facial_emotion in NEGATIVE_EMOTIONS
        and signals.facial_confidence >= 0.5
        and signals.facial_frames >= 5
    )
    return bool(voice_negative and facial_negative)


__all__ = [
    "LEVEL_NONE",
    "LEVEL_LOW",
    "LEVEL_MEDIUM",
    "LEVEL_HIGH",
    "LEVEL_ORDER",
    "LEVEL_LABEL_CN",
    "LEVELS_REQUIRING_FLAG",
    "CrisisAssessment",
    "ModalitySignals",
    "assess_crisis",
    "detect_crisis",
    "HIGH_RISK_KEYWORDS",
    "MEDIUM_RISK_KEYWORDS",
    "ESCALATION_CUES",
    "NEGATION_CUES",
    "ATTRIBUTION_CUES",
    "THRESHOLD_HIGH",
    "THRESHOLD_MEDIUM",
    "THRESHOLD_LOW",
]
