# Dify 工作流配套改动说明

> 对应工作流：`心理教练对话`
> 当前导出文件：`docs/心理教练对话.yml`（2026-09-23 版本，含 `current_stage` / `platform_risk_level` 等变量）；
> 旧版 `心理教练对话(1).yml`（2026-09-02）已被替换，画布改动完成后请重新导出并覆盖本文件。
> 生效前提：后端 `.env` 已配置 `DIFY_API_KEY`（应用密钥，`app-` 开头）。未配置时后端回落到直连 DeepSeek，本说明的改动不会生效。
> 本次核验：2026-09-23，下列节点 ID 与连线均从上述导出文件实际读出，不是人工转述。
> **接入实测（2026-09-23）发现两个问题：入参类型已由后端修复，工作流结构化输出仍需你在 Dify 画布上处理，见第六节。**

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-23 | 后端完成 Dify 实接与自检（`scripts/check_dify.py`）；新增第六节：入参类型不匹配、工作流结构化输出不稳定两个实测缺陷及处置 |
| 2026-09-23 | 变量数由 5 订正为 7（补 `modality_conflict` / `modality_conflict_reason`）；补全全部节点 ID 与实测连线；测试文件名订正为仓库内实际存在的文件 |

---

## 一、为什么要改

后端每一轮通话都会把一组字段作为 `inputs` 传给工作流，位置见
`app/services/ai_lab/socket_events.py` 的 `dify_inputs`（约 859 行），实际发起的请求在约 911 行
（`POST {DIFY_API_BASE}/chat-messages`）。

其中 **7 个字段目前没有在开始节点声明**，所以后端虽然每轮都在传，工作流侧取不到值
（`{{#1786523214663.current_stage#}}` 会解析为空）：

| 变量 | 含义 | 来源 |
| --- | --- | --- |
| `current_stage` | 平台侧判定的当前阶段 | `app/services/coach_stage_service.py` |
| `goal_clear` | 目标是否已经清楚 | 同上 |
| `action_ready` | 是否已形成可执行行动 | 同上 |
| `should_summarize_hint` | 平台侧认为是否可以收束 | 同上 |
| `platform_risk_level` | 平台四级风险 `NONE/LOW/MEDIUM/HIGH` | `app/services/crisis_rules.py` |
| `modality_conflict` | 语音/文本/面部线索互相矛盾 | `fusion_service._compute_conflict` |
| `modality_conflict_reason` | 冲突原因（如"文本积极但语调低落"） | 同上 |

另外有一个真实缺陷：**`知识检索` 节点是断头的。** 它的输出没有进入任何生成节点，
只在 `retriever_resource` 元数据里返回。也就是说"教练节点接入知识库、每轮最多检索 4 条"
目前**不影响回复内容**。

这两件事一起决定了改动的目标：

1. 让工作流吃下平台侧的阶段与风险判定，使"阶段口径"只有一个权威来源；
2. 让知识检索真正进入生成回复的节点。

> 说明：本说明给出的是逐项改动清单与可粘贴的片段，**不直接交付改好的 YAML**。
> Dify 导出文件包含节点坐标、插件依赖与版本信息，手工改动后必须在画布上确认连线，
> 否则容易在导入时报错。

---

## 二、当前链路与节点对照（改动前，实测）

| 节点 | ID | 类型 |
| --- | --- | --- |
| 开始 | `1786523214663` | start |
| 安全风险识别 | `1786974707383` | llm |
| 条件分支 | `1786976628409` | if-else |
| 高风险安全支持 | `1786977451508` | llm |
| 中风险关注 | `1786980152167` | llm |
| 教练进度判断 LLM | `1787216804182` | llm |
| 条件分支 2 | `1787217119842` | if-else |
| 会话总结 | `1787217704248` | llm |
| 知识检索 | `1787131109293` | knowledge-retrieval |
| 普通心理教练 | `llm` | llm |
| 变量聚合器 | `1787032868273` | variable-aggregator |
| 直接回复 | `answer` | answer |

实测连线（`source sourceHandle -> target`）：

```
1786523214663 开始            source -> 1786974707383 安全风险识别
1786974707383 安全风险识别      source -> 1786976628409 条件分支
1786976628409 条件分支          true   -> 1786977451508 高风险安全支持
1786976628409 条件分支          2047a800-bbed-485c-b058-04bf15411caa -> 1786980152167 中风险关注
1786976628409 条件分支          false  -> 1787216804182 教练进度判断 LLM
1786977451508 高风险安全支持     source -> 1787032868273 变量聚合器
1786980152167 中风险关注        source -> 1787032868273 变量聚合器
1787216804182 教练进度判断 LLM   source -> 1787217119842 条件分支 2
1787217119842 条件分支 2        true   -> 1787217704248 会话总结
1787217119842 条件分支 2        false  -> llm 普通心理教练
1787217704248 会话总结          source -> 1787032868273 变量聚合器
llm 普通心理教练                source -> 1787032868273 变量聚合器
1787032868273 变量聚合器        source -> 1787131109293 知识检索      ← 改动三要删
1787131109293 知识检索          source -> answer 直接回复             ← 改动三要删
```

`直接回复.answer = {{#1787032868273.output#}}`（指向变量聚合器），改动后不用动。

---

## 改动一：开始节点新增 7 个变量

节点：**开始**（`1786523214663`），当前已有 13 个变量。在末尾追加以下 7 个。
**字段名必须完全一致**，大小写与下划线都不能变。

| 变量名 | 类型 | 默认值 | 候选值 |
| --- | --- | --- | --- |
| `current_stage` | 文本 | `opening` | `opening` / `exploration` / `goal_setting` / `action_planning` / `closing` |
| `goal_clear` | 布尔 | 关 | — |
| `action_ready` | 布尔 | 关 | — |
| `should_summarize_hint` | 布尔 | 关 | — |
| `platform_risk_level` | 文本 | `NONE` | `NONE` / `LOW` / `MEDIUM` / `HIGH` |
| `modality_conflict` | 布尔 | 关 | — |
| `modality_conflict_reason` | 文本 | 空 | — |

类型建议一律用「文本」与「布尔」。若把 `current_stage` 或 `platform_risk_level` 做成「下拉选项」，
必须把上表候选值全部填进选项列表，否则入参对不上。

导出文件中的对应片段（字段顺序不影响导入）：

```yaml
        - default: opening
          hint: ''
          label: current_stage
          options: []
          placeholder: ''
          required: false
          type: text-input
          variable: current_stage
        - default: false
          hint: ''
          label: goal_clear
          options: []
          placeholder: ''
          required: false
          type: checkbox
          variable: goal_clear
        - default: false
          hint: ''
          label: action_ready
          options: []
          placeholder: ''
          required: false
          type: checkbox
          variable: action_ready
        - default: false
          hint: ''
          label: should_summarize_hint
          options: []
          placeholder: ''
          required: false
          type: checkbox
          variable: should_summarize_hint
        - default: NONE
          hint: ''
          label: platform_risk_level
          options: []
          placeholder: ''
          required: false
          type: text-input
          variable: platform_risk_level
        - default: false
          hint: ''
          label: modality_conflict
          options: []
          placeholder: ''
          required: false
          type: checkbox
          variable: modality_conflict
        - default: ''
          hint: ''
          label: modality_conflict_reason
          options: []
          placeholder: ''
          required: false
          type: text-input
          variable: modality_conflict_reason
```

> 这 7 个变量都是**可选**的。工作流在未收到时按原有逻辑运行，
> 因此可以先加变量、后改分支，分两步上线。

---

## 改动二：`条件分支 2` 改为"平台提示 **或** 工作流自判"

节点：**条件分支 2**（`1787217119842`）。

改动前只有一个条件，即完全由工作流自行判断收束时机：

```yaml
      cases:
      - case_id: 'true'
        conditions:
        - comparison_operator: is
          id: 41922876-178d-4700-8d55-3594e3cebf89
          value: 'true'
          varType: object
          variable_selector:
          - '1787216804182'
          - structured_output
          - should_summarize
        id: 'true'
        logical_operator: and
```

改动后加一条平台条件，逻辑关系改为 `or`：

```yaml
      cases:
      - case_id: 'true'
        conditions:
        - comparison_operator: is
          id: 41922876-178d-4700-8d55-3594e3cebf89   # 保留原有 uuid
          value: 'true'
          varType: object
          variable_selector:
          - '1787216804182'
          - structured_output
          - should_summarize
        - comparison_operator: is
          id: <新生成 uuid>
          value: 'true'
          varType: boolean
          variable_selector:
          - '1786523214663'
          - should_summarize_hint
        id: 'true'
        logical_operator: or
```

注意两条条件的 `varType` 不同：第一条取的是 LLM 的结构化输出，为 `object`；
第二条取的是开始节点的布尔变量，为 `boolean`。

改完在画布上确认：

- `条件分支 2` 的 `true` 分支仍指向 `会话总结`；
- `false` 分支仍指向 `普通心理教练`（改动三会把它的**起点**换成 `知识检索`，终点不变）。

> 若希望完全由平台决定收束时机，可把 `logical_operator` 改为 `and`，或直接删除原有那条条件。
> **建议先保持 `or`**，观察一段时间后再收紧。

---

## 改动三：把断头的 `知识检索` 接回主链路

节点：**知识检索**（`1787131109293`，`top_k=4`）、**普通心理教练**（`llm`）、
**变量聚合器**（`1787032868273`）、**直接回复**（`answer`）。

改动前：

```
条件分支 2 ─ false ─→ 普通心理教练 ─→ 变量聚合器
变量聚合器 ──→ 知识检索 ──→ 直接回复            ← 检索跑完没人用
```

改动后：

```
条件分支 2 ─ false ─→ 知识检索 ─→ 普通心理教练 ─→ 变量聚合器
```

画布上的四步操作：

1. 删除连线 `变量聚合器 → 知识检索`；
2. 删除连线 `知识检索 → 直接回复`；
3. 把 `条件分支 2 [false] → 普通心理教练` 改成 `条件分支 2 [false] → 知识检索`
   （即删除旧连线，新增一条起点为 `条件分支 2` 的 `false` 分支、终点为 `知识检索` 的连线）；
4. 新增连线 `知识检索 → 普通心理教练`。

保留不动：`普通心理教练 → 变量聚合器`、`会话总结 → 变量聚合器`、
`高风险安全支持 → 变量聚合器`、`中风险关注 → 变量聚合器`。

`知识检索` 自身的数据集与 `top_k=4` 已经配对，不需要调整。

---

## 改动四：给 `普通心理教练` 打开检索上下文

节点：**普通心理教练**（`llm`）。当前配置为：

```yaml
        context:
          enabled: false
          variable_selector: []
```

改为：

```yaml
        context:
          enabled: true
          variable_selector:
          - '1787131109293'
          - result
```

**改动三和改动四缺一不可。** 只接线不开上下文，检索结果照样不进提示词；
只开上下文不接线，取到的是空值。

---

## 需要人工确认的一处

`知识检索` 的查询变量当前写作：

```yaml
        query_variable_selector:
        - '1786523214663'
        - sys.query
```

这个组合看着可疑。请在画布上点开该节点，确认它取到的是本轮用户说的话：
正常应为 `sys.query`，或显式改成 `{{#1786523214663.user_utterance#}}`。
这一处不影响接线正确性，但直接影响检索命不命得中。

---

## 三、验证清单

导入修改后的 DSL 后，按顺序验证（每一步都能在 Dify 的"运行历史"里看到实际走了哪个分支）：

| 步骤 | 操作 | 预期 |
| --- | --- | --- |
| 1 | 不带新变量调用 | 工作流正常返回，不报"变量缺失" |
| 2 | 传 `current_stage=goal_setting` | 走到 `教练进度判断 LLM`，回复正常 |
| 3 | 传 `should_summarize_hint=true` | 直接进入 `会话总结` 分支 |
| 4 | 传一句能被资料库命中的问题 | `知识检索` 有命中记录，**且回复内容确实体现了资料中的说法** |
| 5 | 传高风险语料 | 仍进入 `高风险安全支持`，不经过知识检索 |
| 6 | 传 `modality_conflict=true` | 依提示词设计先澄清再回应（若工作流已使用该变量） |

第 4 步是改动三、改动四是否真正生效的**唯一判据**——只看节点跑通不算数，
必须看回复内容有没有引用资料。

## 改动后的完整链路

```
用户输入
  └─ 安全风险识别 ─ 条件分支
       ├ true  → 高风险安全支持 ─┐
       ├ 其他  → 中风险关注 ─────┤
       └ false → 教练进度判断 LLM → 条件分支 2
                                    ├ true  → 会话总结 ────┐
                                    └ false → 知识检索 → 普通心理教练 ─┤
                                                                       └→ 变量聚合器 → 直接回复
```

---

## 四、后端侧验证（不需要 Dify 即可跑）

```bash
pytest tests/test_coach_stage.py -q                          # 阶段引擎规则
pytest tests/test_ai_conversations.py -q                     # 会话/消息持久化与权限
pytest tests/test_ai_conversation_evidence.py -q             # 留痕、会话收尾
pytest tests/test_ai_coach_crisis.py -q                      # 教练场景下的危机分流
pytest tests/test_analysis_stats.py tests/test_analysis_snapshot.py -q
```

当前全量测试基线：`pytest -q` → **181 passed**。

---

## 五、回滚

四处改动互相独立，可以分别回滚：

1. 删掉开始节点新增的 7 个变量 → 恢复原入参（后端会继续传，但不影响工作流）；
2. `条件分支 2` 删掉平台条件、`logical_operator` 改回 `and` → 回到工作流自判收束；
3. 知识检索改回 `变量聚合器 → 知识检索 → 直接回复`，并把 `条件分支 2 [false]` 指回 `普通心理教练`
   → 回到改动前的行为；
4. `普通心理教练` 的 `context.enabled` 改回 `false` → 检索结果不再进提示词。

回滚后平台侧的阶段判定、会话留痕与留痕统计仍然生效，只是不再影响工作流的收束时机。
改动完成后，请把新导出的 DSL 放回仓库根目录替换旧文件，作为"AI 技术应用"一项的佐证材料。

---

## 六、接入实测与两个新缺陷（2026-09-23）

后端已按 `.env` 里的 `DIFY_API_BASE` / `DIFY_API_KEY`（应用密钥 `app-` 开头）完成实接，
并新增自检脚本 `scripts/check_dify.py`（只读 `.env`，不连数据库、不起服务）：

```bash
python scripts/check_dify.py                      # 入参声明 + 类型匹配 + 一轮完整对话
python scripts/check_dify.py --stage goal_setting  # 换一个平台阶段再自检
```

### 6.1 已修复：后端入参类型与工作流声明不一致

工作流把 `goal_clear` / `action_ready` / `should_summarize_hint` / `modality_conflict`
四个变量声明成了**文本输入**（`text-input`，默认值"关闭"），而后端按布尔值发送 →
Dify 在表单校验阶段就把请求打回，**工作流一步都不跑**：

```
{"code":"invalid_param","message":"Run failed: (type 'text-input') goal_clear in input form must be a string"}
```

处置：新增 `app/services/ai_lab/dify_service.py`，按 `GET /parameters` 拿到的
**实际声明类型**转换入参（文本类布尔值转小写 `"true"` / `"false"`，布尔类转真布尔），
结果缓存 5 分钟。`socket_events.py` 在发请求前调用它，因此工作流侧把这四个变量
改成布尔、或继续用文本，后端都能跑通。

> 细节：直接 `str(True)` 会得到 `"True"`，与工作流里判断的 `"true"` 不匹配，
> 必须在转换时显式小写。

### 6.2 待你处理（需在 Dify 画布上改）：结构化输出会抓到推理内容

类型问题解决后工作流能跑起来，但**结构化输出经常取不到值**，工作流直接 400：

```
Variable ['1786974707383', 'structured_output', 'risk_level'] not found
Variable ['1787216804182', 'structured_output', 'should_summarize'] not found
```

最近 6 次同输入实测：**工作流只成功 1 次**；`安全风险识别` 的结构化输出坏 3/6，
`教练进度判断 LLM` 坏 2/6。

#### 根因（坏样本的原始输出）

同一个坏样本里，三个字段同时存在、内容互相矛盾：

| 字段 | 长度 | 内容 |
| --- | --- | --- |
| `text`（模型答案） | 95 字 | `{"risk_level":"low","risk_reason":"...","needs_safety_check":false}` ← **完全正确** |
| `reasoning_content`（思考过程） | 5338 字 | 其中一句：`Actually response format says {"type":"json_object"}` |
| `structured_output`（Dify 抽取结果） | — | `{"type": "json_object"}` ← **抓错了** |

即：**Dify 的 JSON 抽取扫到了模型思考内容里的花括号**，抓走的是模型思考时复述的
`response_format`，而不是真正的答案。正确答案一直好好地躺在 `text` 里。

`教练进度判断 LLM` 的坏样本更明显——抽取结果的 key 直接变成一段思考文本
（`"You MUST strictly adhere to following schema" and schema type boolean...`），
说明抽取跨越了 `</think>` 边界。

所以问题不在于提示词写得好不好，而是**「结构化输出」这个机制在当前模型上不可靠**：
模型一边思考一边复述 schema / response_format，抽取器就会抓错对象。

> 注：本次核对 `心理教练对话.yml` 全文，节点提示词中**没有**任何
> 「不要输出布尔值」的约束——那句约束是 Dify 服务端在启用结构化输出时注入的，
> 不在你能编辑的提示词里，所以改提示词解决不了。

#### 加剧因素：`教练进度判断 LLM` 漏配「推理格式＝分离」

对比 `心理教练对话.yml` 里的 LLM 节点：

| 节点 | `reasoning_format: separated` |
| --- | --- |
| 普通心理教练 | 有 |
| 安全风险识别 | 有 |
| 高风险安全支持 | 有 |
| 中风险关注 | 有 |
| **教练进度判断 LLM** | **缺** |
| 会话总结 | 缺（无结构化输出，风险较低） |

少了这一项，思考内容会直接混进 `text`（坏样本里 `text` 开头就是 `<think>`），
于是「从 `text` 兜底解析 JSON」这条退路对它也不成立。

#### 处置（按优先级，可叠加）

1. **改成代码节点解析，不再用结构化输出**（最稳，推荐）：在两个 LLM 节点后各接一个
   「代码」节点解析上游的 `text`，条件分支改引用代码节点的输出。代码见 6.2.1。
2. **换非推理模型**：把 `安全风险识别` / `教练进度判断 LLM` 换成不开深度思考的模型
   （当前 5 个 LLM 节点用的都是 `deepseek-v4-flash`）。没有思考内容，抽取就不会抓错。
3. **至少先补 `reasoning_format: separated`**：给 `教练进度判断 LLM`（和 `会话总结`）
   补上，与另外 4 个节点一致。这是方案 1 的前置条件，否则它的 `text` 不干净。

#### 6.2.1 代码节点（可直接粘贴）

在 Dify 里新建「代码」节点，**输入变量命名为 `text`**（取上游 LLM 节点的 `text`），
输出按需要声明（例如 `risk_level` / `risk_reason` / `needs_safety_check`）：

```python
import json, re

def main(text: str) -> dict:
    s = text or ""
    # 1) 先剥掉思考内容，避免从推理里抓错对象
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = s.replace("<!--dify-deepseek-reasoning-->", " ")

    # 2) 枚举所有配对完整的 { ... }，取最后一个能解析成 JSON 对象的
    blocks, i = [], 0
    while i < len(s):
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(s[i:j + 1])
                    break
        i += 1

    for block in reversed(blocks):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and obj:
            return obj
    return {}
```

改完用 `python scripts/check_dify.py` 连跑 5 次，全通过再上线。

### 6.3 后端兜底：Dify 失败自动降级 DeepSeek

因为工作流目前成功率很低，视频通话管线改成**Dify 优先、整轮无产出时自动降级
DeepSeek 重试一次**：

- 触发条件：Dify 返回非 200（如上面的 `invalid_param`），或 SSE 流结束但一个字都没产出；
- 且这一轮**尚未向前端推送过任何 token**（没有半截内容）才降级，避免同一轮说两遍；
- 两条路都失败时，`vc_error` 会带上两侧的真实错误，方便定位。

另加一道**熔断**（`dify_service.py`）：Dify 连续 3 次整轮无产出就先熔断 120 秒，
期间直接走 DeepSeek，不再每轮白等一次 Dify（实测一次 Dify 失败要约 9~10 秒）；
任意一次成功立刻复位。可用 `DIFY_CIRCUIT_THRESHOLD` / `DIFY_CIRCUIT_COOLDOWN`
调整，设为 0 即关闭熔断。

降级只影响可用性，不改变"配置了 Dify 就优先走 Dify"的既有约定。

---

## 附：本次改动之后的连带影响

这份改动不只是修一个连线，它是另外两件事的前置条件：

1. **两侧风险一致性统计。** `socket_events.py` 中 `dify_risk_level` 目前传 `None`
   （原因：Dify 对话 API 的响应里不返回节点结构化字段）。只有让工作流把
   `platform_risk_level` 用起来、并把自身判定结果回吐出来，`multimodal_analysis_records`
   的 `dify_risk_level` 列才有值，`analysis_stats_service` 的"平台 vs Dify 一致性"表才不是空的。
2. **教练 A/B 对照实验。** `experiments/run_coach_ab.py` 目前的 B 臂是"直连 DeepSeek +
   脚本内写死的阶段提示词"，并不经过本工作流。要让实验真正测到"Dify 工作流 + 平台阶段回传"
   这一套机制，必须先完成本说明的改动，再让 B 臂改调 `{DIFY_API_BASE}/chat-messages`。
