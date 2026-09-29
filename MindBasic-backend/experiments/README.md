# 评测工具（experiments）

本目录用于生成**可复现的实验证据**：多模态融合的消融对比、危机风险分级的准确率与
漏报/误报率、以及端到端耗时统计口径。所有脚本直接调用线上同一份代码
（`app.services.ai_lab.fusion_service`、`app.services.crisis_rules`），
保证"报告里的数字"与"系统实际行为"一致。

## 快速开始

```bash
cd MindBasic-backend
python experiments/run_fusion_ablation.py
python experiments/run_crisis_eval.py
python experiments/run_calibration.py
python experiments/run_stage_agreement.py      # 阶段判定一致性
python experiments/run_weight_tuning.py        # 权重数据标定 vs 线上规则
python experiments/run_coach_ab.py --provider dry-run   # A/B 盲评（先导出提示词）
python experiments/run_runtime_stats.py        # 真实运行数据（需要数据库）
```

脚本不依赖 scikit-learn 等第三方库（仅标准库 + 项目自身依赖），
结果同时打印到控制台（Markdown 表格，可直接粘贴进技术报告）并写入 `experiments/results/`。

## 目录结构

```
experiments/
├── _bootstrap.py            # 路径与最小配置引导（让脚本可独立运行）
├── metrics.py               # 指标实现：准确率 / 宏平均F1 / 混淆矩阵 / kappa / ECE
├── run_fusion_ablation.py   # 多模态融合消融实验
├── run_crisis_eval.py       # 危机风险分级评测
├── run_calibration.py       # 置信度校准实验（温度缩放）
├── run_stage_agreement.py   # 五阶段判定与人工标签的一致性
├── run_weight_tuning.py     # 融合权重的数据标定（网格搜索 + 交叉验证）
├── run_coach_ab.py          # 成长教练 A/B 盲评（普通臂 vs 阶段臂）
├── run_runtime_stats.py     # 真实运行数据导出（直连线上留痕表）
├── data/
│   ├── fusion_samples.jsonl # 融合样本（含模态缺失、低置信度、冲突场景）
│   ├── crisis_samples.jsonl # 危机分级标注语料
│   ├── stage_labels.jsonl   # 阶段判定标注语料（合成种子）
│   └── coach_ab_prompts.jsonl # A/B 盲评初始问题集
└── results/                 # 运行结果（CSV 明细，自动生成）
```

## 一、多模态融合消融实验

对比方案：

| 方案 | 说明 |
| --- | --- |
| `text_only` / `voice_only` / `facial_only` | 单模态基线 |
| `fixed_weight` | 固定权重 0.40 / 0.35 / 0.25（动态融合的对照） |
| `dynamic_weight` | 线上规则化动态权重 |

实验通过 `fusion_service.fuse(..., weights_override=...)` 复用同一条融合代码路径，
只替换权重入口，避免"实验实现"与"线上实现"不一致。

输出指标：准确率、宏平均F1、加权F1、ECE（置信度校准误差），
并按 `子集` 分组统计（模态冲突 / 面部缺失 / 语音低置信度 / 短文本），
因为动态权重的作用通常体现在困难子集而非整体平均。

## 二、危机风险分级评测

两个层面的指标：

1. **分级指标**：NONE / LOW / MEDIUM / HIGH 四级逐类精确率、召回率、F1、混淆矩阵；
2. **处置指标**：以"是否需要建立工单"为二分类，报告**漏报率**与**误报率**——
   这是危机预警最需要向评审说明的数字。

调整判定口径时，修改 `app/services/crisis_rules.py` 顶部的关键词表、阈值与计分常量，
然后重跑本脚本即可得到前后对比（规则与阈值均为模块级常量，便于标定）。

## 三、置信度校准实验

`fusion_service` 输出的 `overall_confidence` 直接等于融合后的最大概率，未经校准
（动态融合的 ECE 约 0.42）。本脚本评估**温度缩放**（temperature scaling）
能否把置信度拉回可信区间，并回答两个问题：

1. 校准后 ECE / NLL / Brier 改善多少？
2. 置信度作为"该不该降权或追问"的排序依据，是否更可靠？

三种评测协议，**报告应使用第三行**：

| 协议 | 含义 |
| --- | --- |
| `uncalibrated` | 线上现状 |
| `in_sample` | 全样本拟合温度（乐观上界，仅作参考） |
| `cv_kfold` | K 折交叉验证：T 只在训练折拟合，在校验折评估 |

脚本同时输出**分场景校准对比**（全局温度 vs 按子集单独标定），
用于检查"按整体拟合的温度是否会让困难子集的置信度过于乐观"。

温度缩放与温度拟合的数学实现**不在本脚本内**，而是调用线上同一份
`app/services/ai_lab/calibration.py`。因此脚本输出的 `T` 可以直接写入后端 `.env`
的 `FUSION_TEMPERATURE` 并启用，无需再实现一遍：

```bash
FUSION_CALIBRATION_ENABLED=true
FUSION_TEMPERATURE=<脚本输出的 T>
FUSION_CALIBRATION_SOURCE=<标定数据集说明>
```

注意两个读表要点：

- 温度缩放是单调变换，**不改变 argmax**，因此准确率与宏平均 F1 校准前后完全相同；
  校准只修置信度，不动识别结果。
- 若最优 T 贴在网格下界，说明真实最优在网格之外，该数值不可直接采用——
  脚本会显式给出该警告。

## 四、阶段判定一致性实验

回答"阶段化教练真的有据可依吗"：把线上同一份规则引擎
（`app.services.coach_stage_service.decide_stage`）当作一位标注者，
与人工标签逐条比对，输出阶段准确率、宏平均 F1、混淆矩阵，
以及 `goal_clear` / `action_ready` 两个布尔线索的准确率与 kappa。

标注语料格式见 `data/stage_labels.jsonl` 顶部说明；`source` 字段用于区分
合成种子数据与人工标注数据，报告引用时必须如实标注来源。

```bash
python experiments/run_stage_agreement.py
```

输出 `results/stage_agreement_report.md`（含错误分析：每条不一致样本的
人工标签、引擎判定与命中线索）与 `results/stage_agreement_details.csv`。

## 五、融合权重标定实验

线上权重由 6 条启发式规则给出，阈值是经验值。本脚本在权重单纯形上做网格搜索，
用 **K 折交叉验证**选择权重（权重只在训练折上选），再与"基础固定权重""线上动态权重"
在整体和分场景（模态冲突 / 面部缺失 / 低语调置信度 / 短文本）上对比。

```bash
python experiments/run_weight_tuning.py --step 0.05 --folds 5
```

报告应引用交叉验证口径（`results/weight_tuning_cv.csv`），
`weight_tuning_grid.csv` 是全样本口径的完整 231 组结果，只用于观察趋势。

## 六、成长教练 A/B 盲评

回答"加入阶段化教练机制，在同一个基础模型下回复是否更好"：

| 臂 | 提示词 | 是否注入阶段线索 |
| --- | --- | --- |
| A `plain` | 通用共情倾听 | 否 |
| B `staged` | 成长教练方法 | 是（阶段判定来自线上引擎） |

三步子命令，评分前**不要打开** `results/coach_ab_key.json`：

```bash
python experiments/run_coach_ab.py prepare --provider deepseek   # 或 offline
python experiments/run_coach_ab.py sheet                         # 生成盲评表
python experiments/run_coach_ab.py score                         # 填完后统计
```

A/B 顺序由 `--seed` 与样本 ID 派生，重跑得到同一份盲评表；
`score` 输出阶段臂胜率、Wilson 95% 置信区间，
若填写了 `winner_r2` 还会报告两名评价者的一致性 kappa。
离线模式需要 `data/coach_ab_responses.jsonl`（`{"id","plain","staged"}`），
便于在无网络或答辩现场复现评分流程。

## 七、真实运行数据导出

把线上留痕聚合为报告表格，与管理端 `GET /api/v1/admin/stats/multimodal`
使用同一个聚合函数，保证"报告里的数字"与"后台看到的数字"一致。

```bash
python experiments/run_runtime_stats.py --days 30
python experiments/run_runtime_stats.py --days 7 --source VIDEO_CALL
```

输出 `runtime_stats.md`（可粘贴表格）、`runtime_stats.json`（可追溯原始值）
与 `runtime_latency.csv` / `runtime_risk.csv` / `runtime_stages.csv`。
需要可连接的 `DATABASE_URL`；窗口内没有数据时输出空表并显式提示。

## 八、数据说明（重要）

`data/` 内的样本是**流程验证用合成数据**，用于确认评测链路可运行、指标可解释；
它们**不能**作为技术报告中的效果结论。正式材料需要：

1. 用真实授权数据替换 `data/` 内容（保持 JSONL 字段结构不变）；
2. 至少两名标注者独立标注并报告一致性（脚本已提供 Cohen's kappa 计算）；
3. 把 `experiments/results/*.csv` 作为数据资料一并提交。

融合样本字段约定：

```json
{"id": "S01", "gold": "sad", "sv": "sad", "note": "标注说明",
 "text": ["sad", 0.82], "voice": ["sad", 0.74], "facial": ["sad", 0.68, 14]}
```

`text` / `voice` 为 `[主标签, 置信度]`；`facial` 为 `[主标签, 置信度, 帧数]`，
空数组表示该模态缺失；`sv` 为 SenseVoice 情绪辅助信号（缺省为 `neutral`）。
其余概率按"置信度给主标签、剩余概率均分"展开，这是数据文件保持紧凑的简化约定。

## 九、与竞赛评分项的对应关系

| 评分项 | 本目录可提供的证据 |
| --- | --- |
| 创新性 | 动态融合 vs 固定权重的消融对比、模态冲突子集增益、**阶段臂 vs 普通臂的 A/B 盲评** |
| 需求分析 | 阶段判定与人工标签的一致性（说明"阶段化教练"不是凭空设计） |
| AI技术应用 | 单模态/多模态、缺失与低置信度场景的技术处理与数据；**线索冲突检测的判定口径** |
| 应用成效 | 危机分级的漏报率、误报率；**真实运行数据（成功率、降级率、P50/P95）** |
| 附加分 | 评测代码 + 数据 + 结果 CSV（数据资料与注释代码） |

### 分场景宏平均 F1 的口径说明

子集内往往只出现部分情绪类别。`fusion_ablation_subsets.csv` 同时给出两套口径：

- `<arm>_present`：**仅在该子集出现的类别上**计算宏平均（主口径）
- `<arm>_all7`：7 类全算，未出现的类别按 F1=0 参与平均（会被系统性压低）

以现有合成数据为例，`modality_conflict` 子集的动态融合宏平均 F1：
主口径 0.9179，7 类口径 0.6557——差异全部来自"子集内不存在的类别"。
报告中建议采用主口径，并注明子集类别构成。
