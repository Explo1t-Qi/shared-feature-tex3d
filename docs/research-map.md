# Shared-Feature Tex3D Research Map

## 1. Research Goal

当前研究目标是提升 Tex3D adversarial 3D texture 在不同架构 Vision-Language-Action (VLA) 模型之间的迁移性。

核心问题不是继续增强单一 surrogate VLA 上的攻击强度，而是减少 adversarial texture 对 surrogate architecture 特有表征的过拟合。

当前核心假设是：

> 不同 VLA 架构不仅可能存在稳定的 clean shared representation structure，也可能
> 存在跨模型共同的 adversarially vulnerable / action-relevant structure。本项目优先
> 在各模型内部独立识别真正容易被攻击且与动作相关的 features/directions，再研究这些
> vulnerable structures 能否在异构 VLA 之间对齐或融合，最终用于提高
> single-surrogate adversarial texture 的迁移性。

必须严格区分：

\[
\text{shared}
\neq
\text{vulnerable}
\neq
\text{policy-relevant}
\neq
\text{transferable}.
\]

因此，clean representation similarity 只提供前置 representation evidence，不能直接
作为 vulnerability、策略相关性或攻击迁移性的证据。

---

## 2. Threat Model

### DECISION — Vulnerable-Feature Discovery + Single-Surrogate Attack

Representation discovery 阶段允许同时分析多个 VLA，例如：

- OpenVLA；
- \(\pi_{0.5}\)。

允许使用多个模型的 clean/adversarial representations 做 discovery、对齐和融合分析，
包括使用 paired clean representations 分析 shared structure。

但是正式 adversarial texture optimization 必须保持：

```text
single surrogate only
```

例如以 OpenVLA 为 surrogate：

\[
x_{\mathrm{adv}}
\rightarrow
Z^{\mathrm{vuln}}_A(x_{\mathrm{adv}})
\rightarrow
\mathcal{L}_{\mathrm{vulnerable-surrogate}}
\rightarrow
\theta_{\mathrm{texture}}.
\]

该式只冻结 single-surrogate boundary；具体 vulnerable representation 和 loss 尚未
contracted 或 authorized。

攻击阶段不得退化为：

\[
\mathcal{L}_{\mathrm{OpenVLA}}
+
\mathcal{L}_{\pi_{0.5}}
\]

这样的 multi-model ensemble attack。

Held-out VLA 只用于迁移性评估，不参与 attack-time loss、gradient 或 texture optimization。

---

## 3. Relation to Prior Shared-Representation Evidence

### FACT

UPA-RFAS 等工作提供了不同 VLA feature spaces 之间可能存在共享结构的实验证据，并使用 CCA、线性关系分析等工具研究跨模型 representation relationship。

### DECISION

本项目不把“观察到 cross-model correlation”直接等价为“已经得到 transferable attack space”。

当前默认 scientific route 冻结为 vulnerability-first：

```text
model-specific vulnerability discovery
        ↓
action-relevant vulnerable feature identification
        ↓
cross-model vulnerable feature alignment / fusion
        ↓
shared vulnerable representation
        ↓
single-surrogate vulnerable-feature loss
        ↓
Tex3D texture optimization
        ↓
held-out VLA transfer evaluation
```

此前的 clean-shared-first 路线：

```text
clean cross-model shared space / CCA
→ action-relevant shared directions
→ attack
```

不被否定或删除。它保留为 complementary analysis、ablation 和 possible alternative
route。已经完成的 clean cross-model CKA/CCA 结果继续作为稳定、可对齐 shared
structure 的前置证据，但不再作为当前默认攻击研究主线。

---

## 4. Current Model Targets and Representation Nodes

### 4.1 OpenVLA

当前 OpenVLA scientific representation nodes 为：

#### O1-S — Shared-Backbone Control

```text
SigLIP branch output
before DINO/SigLIP fusion
before multimodal projector

shape:
[256, 1152]
```

用途：

```text
shared-backbone control
```

#### O1-F — Supplementary Diagnostic

```text
concat(DINOv2, SigLIP)
before multimodal projector

shape:
[256, 2176]
```

用途：

```text
supplementary / diagnostic
```

#### O2 — Primary VLA-Adapted Representation

```text
OpenVLA multimodal projector output
before deeper Llama processing

shape:
[256, 4096]
```

O2 是当前最重要的 OpenVLA scientific representation node。

---

### 4.2 \(\pi_{0.5}\)

当前冻结模型配置为：

```text
config = "pi05_libero"
checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
```

当前 scientific extraction 使用：

```text
batch_size = 1
```

Representation nodes：

#### P1 — SigLIP Encoder Representation

```text
after final SigLIP encoder norm
before projection head

shape:
[256, 1152]
```

#### P2 — VLA-Ready Projected Visual Representation

```text
projected PaliGemma-ready visual tokens

shape:
[256, 2048]
```

---

### 4.3 Primary Pairings

### DECISION

Primary scientific pairing：

```text
O2 ↔ P2
```

问题：

> VLA-specific visual adaptation 之后，OpenVLA 与 \(\pi_{0.5}\) 是否仍保留稳定的 shared representation geometry？

Shared-backbone control：

```text
O1-S ↔ P1
```

用于判断观察到的 similarity 是否主要来自共享或近似的 SigLIP backbone。

O1-F 只做 supplementary analysis。

---

## 5. Current Pipeline Status

当前 pipeline 状态：

```text
C0 — PilotObservation schema               PASS
C1 — LIBERO/OpenVLA observation collector  PASS
C2 — OpenVLA extractor implementation      PASS
C3 — π0.5 extractor implementation         PASS
C4 — paired-manifest implementation        PASS
C5-D0 — Pilot v0.2 collector               UNIT-LEVEL PASS
C5-D0 — reduced real integration smoke     PASS
C5-D0 — Pilot v0.2 formal collection       COMPLETE
C2/C3 — Pilot v0.2 full extraction         COMPLETE
C4 — Pilot v0.2 formal paired manifest     COMPLETE
C5-A — Representation Geometry             FORMAL COMPLETE / GO
C5-B scientific contract                   FROZEN
C5-B implementation                        UNIT-LEVEL PASS
C5-B formal execution                      FORMAL COMPLETE / PASS
C5 representation-stage                    FORMAL COMPLETE / PASS
C6-A policy-sensitivity interface closure   COMPLETE / FROZEN
C5-BM scientific/engineering contract       FROZEN
C5-BM implementation                        UNIT-LEVEL PASS
C5-BM formal materialization                FORMAL COMPLETE / PASS
C6 intervention-interface contract         FINAL AUDIT PASS / FROZEN
C6 intervention-interface implementation   UNIT-LEVEL PASS
C6 intervention-interface unit validation  PASS
C6 real clean-equivalence                   PASS (OpenVLA 2/2; pi0.5 2/2)
C6 original intervention smoke              PARTIAL / HISTORICAL BLOCKED
  OpenVLA                                   BLOCKED (translation-response gate)
  pi0.5                                     PASS
C6 OpenVLA token/logit diagnostic           PASS
C6 intervention-interface closure           COMPLETE
Previous C6-B clean-shared-direction plan   DEFERRED / COMPLEMENTARY / NOT AUTHORIZED
Vulnerability-first cross-model study       NEXT / NOT CONTRACTED / NOT AUTHORIZED
Final overall research gate                OPEN / NOT DEFINED HERE
Policy/action relevance                    NOT STARTED
Transferability / Tex3D optimization       NOT STARTED / NOT AUTHORIZED
```

C1 已完成 real OpenVLA/LIBERO smoke。

C5-D0 reduced real integration smoke 已使用 LIBERO-Spatial tasks `0` 和 `1`
完成真实 OpenVLA/LIBERO 验证：每个 task 接受 official initial state `0` 的
一个成功 group，共生成 `8` 个有效 `PilotObservation`，manifest 状态为
`SMOKE_COMPLETED + null`。该结果证明 multi-task collection integration closure，
不构成 formal Pilot v0.2 dataset collection。Portable checkpoint identity 的冻结
来自后续独立、明确的 scientific decision，而不是由 smoke 结果自动推导。

C5-D0 formal collection 已按冻结协议完成全部 10 个 LIBERO-Spatial tasks、
50 个 accepted trajectory groups 和 200 个 canonical `PilotObservation`。
其 manifest 状态为 `COMPLETED + COMPLETE`。基于这 200 个冻结 observations
执行的 C2/C3 full feature extraction 已完成，C4 formal paired manifest 也已完成，
并保持 200/200 sample identity 与 source-image hash 一致。

C2 保存：

```text
O1-S [256,1152]
O1-F [256,2176]
O2   [256,4096]
```

C3 保存：

```text
P1 [256,1152]
P2 [256,2048]
```

C2/C3 的 2-sample real smoke 只证明：

```text
pipeline closure
```

不能用于正式 C5 scientific conclusion。C5-A 已在正式 C4 paired dataset 上执行
完成，TRAIN 与 HELD-OUT 均通过冻结判据，geometry-stage 结果为 `GO`。C5-B
implementation 与 formal execution 也已完成；primary `O2 ↔ P2` 99%-PCA
HELD-OUT Top5Mean 为 `0.970583518852`，one-sided empirical
`p = 0.00497512437811`，因此冻结结果为 `C5-B PASS`。由冻结 joint gate 得到
`C5 representation-stage PASS`。该结果不是最终 overall research PASS；最终
overall research gate 仍为 `OPEN / NOT DEFINED HERE`。C6-A source audit、scientific
review 与 contract review 已完成，冻结状态为
`INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES`。该 interface closure 不构成
policy/action relevance evidence。C5-BM scientific/engineering contract 已通过最终
read-only audit 并冻结；它定义新的 authoritative mapping materialization，而不声称
恢复历史未保存矩阵。其 implementation 已通过 unit-level validation，formal
materialization 已完成并通过全部验证，结果为 `PASS`。

C6 O2/P2 intervention interface 已在真实 checkpoint 上完成验证。OpenVLA 与
pi0.5 的 clean-equivalence 均为 `2 / 2 PASS`。原始 intervention smoke 结果必须
保留为：OpenVLA 在冻结的 translation-response gate 下为 `BLOCKED`，pi0.5 为
`PASS`。后续 OpenVLA token/logit diagnostic 为 `PASS`：modified O2 会可测量地
改变 downstream action-token logits，但 greedy action-token IDs 保持不变，因此
decoded translation 保持不变。该结果支持 discrete argmax/token-boundary explanation，
并完成 intervention-interface engineering closure，但不把历史 OpenVLA smoke
改写为 `PASS`，也不构成 policy/action relevance evidence。

这些 C5/C6 结果现在定位为两类既有基础证据：异构 VLA representations 中存在稳定、
可对齐的 clean shared structure；native O2/P2 features 可被显式介入并传播到
downstream computation。它们不直接证明 vulnerability 或 transferability。下一默认
主线是尚未 contracted/authorized 的 vulnerability-first cross-model feature study；
原 C6-B clean-shared-direction 计划保留为 complementary analysis / ablation，而非
失败路线。

---

## 6. Pilot Protocol Versioning

### FACT — Pilot v0.1 Is Historical and Frozen

`docs/pilot-v0.1-spec.md` 是已完成的历史 Pilot 协议。

其核心设计是：

```text
1 LIBERO-Spatial task
× 10 fixed initial states
× 20 valid-policy frames
≈ 200 observations
```

Pilot v0.1 必须保持不变，不得 retroactively 改写以匹配当前设计。

---

### DECISION — Pilot v0.2 Is the Current C5 Data Protocol

`docs/pilot-v0.2-spec.md` 是当前 C5 数据准备和分组的 authoritative scientific protocol。

当前目标：

```text
10 LIBERO-Spatial tasks
× 5 accepted successful state groups/task
× 4 observations/group
= 50 groups
= 200 observations
```

Observation source：

```text
successful clean OpenVLA rollouts
```

同一个 frozen raw observation 同时用于：

```text
OpenVLA feature extraction
π0.5 feature extraction
```

因此：

\[
x_i^{\mathrm{OpenVLA}}
=
x_i^{\pi_{0.5}}
=
x_i.
\]

Pilot v0.2 明确采样的是：

> successful clean OpenVLA rollout state distribution。

它不是：

- \(\pi_{0.5}\) on-policy distribution；
- uniform LIBERO state distribution；
- unbiased successful-state distribution。

---

## 7. Pilot v0.2 Collection Principles

### DECISION

每个 LIBERO-Spatial task：

1. 按官方 initial-state sequence 的确定性 canonical order 遍历；
2. 接受前 5 个满足协议的 successful OpenVLA rollout groups；
3. 每个 accepted trajectory 恰好采 4 个 observations。

Target relative progress：

\[
Q=\{0.10,0.40,0.70,0.90\}.
\]

对于 valid-policy trajectory length \(T\)：

\[
t_q
=
\left\lfloor q(T-1)+0.5\right\rfloor.
\]

Scientific group identity：

```text
(task_id, initial_state_id)
```

同 group 的 4 个 observations 必须始终作为一个整体处理。

Dataset status：

```text
COMPLETE
USABLE_WITH_SHORTFALL
BLOCKED
```

只有 `COMPLETE` 和 `USABLE_WITH_SHORTFALL` 可以进入正式 C5。

具体 rollout boundary、success/failure、sampling、identity、completeness 和 split 规则以 `docs/pilot-v0.2-spec.md` 为准，本 research map 不重复定义实现细节。

---

## 8. Frozen OpenVLA Collection Identity

### DECISION — Portable OpenVLA Checkpoint Identity

Pilot v0.2 已冻结 OpenVLA rollout 的主要 scientific configuration，包括：

- OpenVLA LIBERO-Spatial policy；
- `unnorm_key = "libero_spatial_no_noops"`；
- `center_crop = True`；
- camera resolution `512`；
- 10 dummy/stabilization steps；
- maximum 300 valid-policy actions；
- 4/8-bit quantization disabled；
- global OpenVLA evaluation seed `7`；
- LIBERO environment seed `0`；
- deterministic action decoding with `do_sample = False`；
- 已验证的 C1 action/preprocessing path。

portable OpenVLA checkpoint identity 已冻结为：

```text
openvla/openvla-7b-finetuned-libero-spatial
```

该 identity 按明确 scientific decision 不附加 immutable revision suffix。
机器本地 resolved checkpoint path 只作为 runtime provenance 写入 collection
manifest，不属于 portable scientific identity。

portable checkpoint identity blocker 已解除。Formal Pilot v0.2 collection 仍须通过
独立、明确授权的 full-collection contract/entrypoint 执行，不得把 reduced smoke
当作正式 collection。

---

## 9. C5-A — Representation Geometry

### STATUS — FORMAL COMPLETE / GO

C5-A 回答：

> OpenVLA 与 \(\pi_{0.5}\) 是否对同一批 observations 形成相似的 representation geometry？

已完成的 C5-A contract 保持为 C5-A scientific/statistical authority。

### FACT — Formal C5-A Result

正式 C5-A 已在 50 个 trajectory groups、200 个 paired observations 上完成：

```text
O2 ↔ P2 TRAIN
debiased Linear CKA = 0.528420895275
empirical p-value  = 0.0196078431373
split result       = PASS

O2 ↔ P2 HELD-OUT
debiased Linear CKA = 0.478168155531
empirical p-value  = 0.0196078431373
split result       = PASS

C5-A geometry-stage result = GO
```

正式输出的 split manifest、metric summary、null arrays 与 Markdown summary
已通过一致性审计。50 次 TRAIN / HELD-OUT 独立 group-block derangements 与冻结
`SeedSequence(7).spawn(2)`、PCG64 convention 完全一致。

该 C5-A 结果本身不构成最终 C5 PASS，也不曾单独冻结 C5-B 或 final joint gate；
后续独立 C5-B contract 与正式执行已经完成，并形成当前记录的
`C5 representation-stage PASS`。

### DECISION — Frozen C5-A Method

Primary：

```text
debiased Linear CKA on O2 ↔ P2
```

Robustness：

```text
Spearman RSA
```

Diagnostic：

```text
biased Linear CKA
```

Primary representation form：

```text
observation-level mean-pooled representation
```

对于：

\[
Z_i\in\mathbb{R}^{256\times D},
\]

先计算：

\[
\bar z_i
=
\frac{1}{256}
\sum_{t=1}^{256}Z_{i,t}.
\]

得到：

\[
X\in\mathbb{R}^{N\times D_A},
\qquad
Y\in\mathbb{R}^{N\times D_B}.
\]

其中：

```text
one PilotObservation = one statistical row
```

Primary CKA 前：

```text
center
no PCA
no CCA
no per-channel z-score
```

Scientific primary：

```text
O2 ↔ P2
```

Shared-backbone control：

```text
O1-S ↔ P1
```

上述三个 metric 也用于 `O1-S ↔ P1` control，但 control、RSA 和 biased CKA
均不决定 C5-A geometry-stage gate。

C5-A null 与 reproducibility decisions 已冻结为：

```text
50 group-block derangements
C5-A RNG seed = 7
independent TRAIN / HELD-OUT RNG streams
```

C5-A geometry-stage gate 已冻结为：

```text
C5-A GO iff TRAIN PASS AND HELD-OUT PASS
```

该 C5-A gate 单独只决定是否进入 C5-B discussion，不等同于后续已经正式完成的
C5 representation-stage joint gate。

CKA 用于判断 representation geometry similarity。

CKA 不用于：

- channel-to-channel correspondence；
- 学习 explicit \(W_A/W_B\)；
- 判断 individual observation 是否“shared”；
- 证明 policy relevance 或 attack transferability。

---

## 10. C5-B — Explicit Shared Linear Alignment

### STATUS — FORMAL COMPLETE / PASS

C5-B 的 scientific/statistical contract、RNG、null protocol 与 PASS/FAIL rule
保持冻结。Implementation 已通过 unit-level validation，formal execution 已在
200-record C4 paired dataset 上完成。

### DECISION — SVCCA-Style PCA/SVD → CCA

C5-B 回答：

> 如果 representation geometry 存在 similarity，是否可以学习一个能够在 held-out observations 上泛化的 explicit linear aligned subspace？

方法：

```text
node-specific PCA/SVD
        ↓
ordinary linear CCA
```

得到：

\[
H_A=Z_AW_A,
\qquad
H_B=Z_BW_B.
\]

\(W_A,W_B\) 只称为：

```text
candidate shared-space mappings
```

不能提前称为：

- transferable mappings；
- policy-relevant mappings；
- attack mappings。

Primary SVCCA 使用：

```text
position-aligned token-wise rows
```

即：

\[
[N_{\mathrm{train}},256,D]
\rightarrow
[N_{\mathrm{train}}\cdot256,D].
\]

Row correspondence：

\[
(i,t)_A
\leftrightarrow
(i,t)_B.
\]

这些 token rows 是 computational rows / repeated measurements，不得声称为独立 statistical observations。

禁止 full token-channel flatten：

\[
[N,256D].
\]

PCA：

```text
node-specific
independent between models
TRAIN-only
centering only
```

Primary：

```text
99% cumulative explained variance
```

Robustness：

```text
95%
```

CCA：

```text
ordinary linear CCA
pairing-specific
TRAIN-only
```

Held-out 阶段：

```text
no refit
no reorder
no abs(correlation)
```

Primary SVCCA summary：

```text
held-out Top5Mean
```

Supporting metrics：

```text
Top1
Top10Mean
```

### FACT — Formal C5-B Result

Primary `O2 ↔ P2`, 99%-PCA formal result：

```text
TRAIN Top5Mean                 0.977299206880
HELD-OUT Top5Mean              0.970583518852
HELD-OUT null median           0.941086355712
one-sided empirical p          0.00497512437811
C5-B result                    PASS
C5 representation-stage       PASS
```

The minimum attainable empirical p-value under `R = 200` is `1 / 201`, matching
the observed primary p-value. The 95%-PCA robustness configuration and
`O1-S ↔ P1` control also completed, but neither determines the frozen C5-B gate.

This result establishes only the completed representation-stage conclusion. It
does not establish policy/action relevance, causal relevance, transferability, or
attack effectiveness.

---

## 11. Group-Aware Split and Shuffled Null

### DECISION — Group-Aware Split

Statistical group：

```text
(task_id, initial_state_id)
```

同一 trajectory group 的 4 observations 必须全部进入 TRAIN 或全部进入 HELD-OUT。

禁止 frame-level random split。

Pilot v0.2 使用 task-stratified deterministic split：

```text
1 held-out group/task
all remaining groups/task → TRAIN
```

`COMPLETE` dataset 时：

```text
40 TRAIN groups
10 HELD-OUT groups
160 TRAIN observations
40 HELD-OUT observations
```

最终 serialized split manifest 是 authoritative split materialization。

---

### DECISION — C5-A Group-Block Shuffled Null

Shared-representation claim 必须与 shuffled null 比较。

Pilot v0.2 中一个 statistical group 为：

```text
(task_id, initial_state_id)
```

每个 accepted group 包含 4 个按 target progress 排列的 observations：

```text
0.10
0.40
0.70
0.90
```

因此 null permutation 的最小交换单位不是单个 observation，而是完整 trajectory group。

True group correspondence：

\[
g_i^A
\leftrightarrow
g_i^B.
\]

Null group correspondence：

\[
g_i^A
\leftrightarrow
g_{\pi(i)}^B.
\]

其中同一 group 的 4 个 progress observations 必须整体移动，并保持 progress-slot 对应关系：

```text
0.10 ↔ 0.10
0.40 ↔ 0.40
0.70 ↔ 0.70
0.90 ↔ 0.90
```

C5-A null permutation 必须：

- 在 TRAIN 与 HELD-OUT 内分别独立执行；
- 禁止跨 split permutation；
- 使用 group-level derangement，即不允许任何 group 保持原始 cross-model pairing；
- 保持每个 group 内的 4 个 progress slots。

C5-A null repeats：

```text
50
```

C5-A primary evidence：

```text
TRAIN and HELD-OUT true debiased Linear CKA
vs their separate group-block shuffled null distributions
```

C5-A exact reproducibility convention：

```text
root RNG seed = 7
SeedSequence(7).spawn(2)
independent PCG64 TRAIN / HELD-OUT streams
rejection-sampled group derangements
```

C5-B 使用独立冻结的 fit-and-evaluate null：PCA 固定为 TRAIN-only transforms；
每个 null repeat 先对 π0.5 TRAIN groups 做 derangement 并重新拟合 ordinary CCA，
再使用独立 deranged HELD-OUT pairing 计算 held-out statistics。两个 pair 与 99%/95%
PCA configurations 共用同一 permutation bank。

### LIMITATION — Task Correspondence Is Also Broken

在当前 Pilot v0.2 split 中，每个 task 只有 1 个 HELD-OUT group，因此 held-out split 无法构造 within-task group permutation。

当前 group-block null 必须允许跨 task permutation。

因此该 null 同时破坏：

```text
trajectory-group correspondence
+
task correspondence
```

它可以检验 true paired cross-model structure 是否强于广义的 cross-group mismatch null，但不能单独证明：

> 在保持 task identity 不变的条件下，同一具体 trajectory/state 仍存在额外 shared structure。

若后续需要回答该更严格问题，应单独设计 task-conditioned robustness analysis，而不应为此 retroactively 修改 Pilot v0.2 collection 或 split protocol。

### DECISION — C5-A and C5-B Null RNG

C5-A 的 50 次 permutation、seed `7` 与 exact RNG convention 已由完成的
C5-A contract 冻结。

C5-B 的独立 convention 已由完成的 C5-B contract 冻结：

```text
R = 200
root RNG seed = 17
SeedSequence(17).spawn(2)
independent PCG64 TRAIN / HELD-OUT streams
rejection-sampled fixed-point-free group derangements
one shared permutation bank across all four configurations
```

C5-B 不复用 C5-A seed 或 permutation bank。

---

## 12. C5 Interpretation Rules

### DECISION — C5-A Geometry-Stage Gate

C5-A 的 geometry-stage gate 已冻结：

```text
C5-A GO
iff
TRAIN PASS AND HELD-OUT PASS
```

其中 split-level PASS 只由 `O2 ↔ P2` debiased Linear CKA 决定：

```text
true debiased Linear CKA > 0
and
one-sided empirical p-value <= 0.05
```

该 C5-A gate 只决定是否进入 C5-B explicit-alignment discussion。正式执行中
TRAIN 与 HELD-OUT 均为 `PASS`，因此冻结的 geometry-stage 结果为 `C5-A GO`。

### DECISION — C5-B Gate and C5 Representation-Stage Joint Gate

C5-B primary configuration：

```text
O2 ↔ P2
99% TRAIN-only PCA
ordinary linear CCA
TRAIN-ordered HELD-OUT Top5Mean
```

C5-B PASS iff：

```text
true HELD-OUT Top5Mean > 0
and
one-sided empirical p-value <= 0.05
```

C5 representation-stage gate 已冻结为：

```text
C5 representation-stage PASS
iff
C5-A GO AND C5-B PASS
```

正式执行已得到 `C5-A GO` 与 `C5-B PASS`，因此冻结的 joint gate 结果为：

```text
C5 representation-stage PASS
```

该结果只总结 representation analysis，不是最终 overall research gate，也不证明
policy relevance、action relevance、adversarial transferability 或 attack effectiveness。

### OPEN — Final Overall Research Gate

以下术语目前只用于 qualitative interpretation：

```text
CKA positive
SVCCA positive
SVCCA weak
near null
significantly above shuffled null
```

这些术语不构成代码中的自动 strong/weak 分类阈值。

最终 overall research PASS/FAIL 仍为：

```text
OPEN / NOT DEFINED HERE
```

C5-B implementation 与 formal execution 均已完成。当前仍禁止从
`C5 representation-stage PASS` 推导 final overall PASS/FAIL。以下解释只描述不同
CKA/SVCCA 结果模式的科学含义，不代表最终 overall research gate 已冻结。

### CKA positive + SVCCA positive

支持：

> representation geometry shared，并且该 similarity 可以形成在 held-out observations 上泛化的 explicit linear alignment。

### CKA positive + SVCCA weak

支持：

> geometry similarity 存在，但当前 linear SVCCA 不足以形成稳定 shared coordinate system。

### CKA weak + SVCCA train-high / held-out weak

提示：

> apparent CCA alignment 可能来自 high-dimensional fitting 或 train-specific structure。

### CKA near null + SVCCA near null

说明：

> 当前 selected representation nodes 不支持 shared-representation hypothesis。

不得强推 attack conclusion。

---

### Layer Interpretation

若：

```text
O1-S ↔ P1 positive
O2   ↔ P2 positive
```

说明：

```text
shared structure survives VLA-specific visual adaptation
```

若：

```text
O1-S ↔ P1 positive
O2   ↔ P2 weak
```

说明：

```text
sharing is mainly backbone-level
and decays after VLA-specific adaptation
```

若：

```text
O1-S ↔ P1 weak
O2   ↔ P2 positive
```

这是有趣结果，但必须优先排查：

- preprocessing；
- node semantics；
- alignment confounders。

---

## 13. Vulnerability-First Cross-Model Study — Next Scientific Stage

### DECISION — Vulnerability-First Main Route

当前默认 scientific route 是：

1. 分别在每个 VLA 内识别 model-specific adversarial/action-relevant vulnerable
   features 或 directions；
2. 只在这些 model-specific vulnerability 已被支持后，分析其跨模型 alignment 或
   fusion；
3. 使用得到的 shared vulnerable structure 设计未来的 single-surrogate attack loss；
4. 最后仅在 held-out VLA 上评估 transferability。

该 next stage 当前为 `NOT YET CONTRACTED / NOT AUTHORIZED`。尚未证明：

```text
model-specific vulnerable features 已找到
cross-model vulnerable structure 已存在
shared vulnerable direction 已确定
transferability 已提高
Tex3D attack 已建立
```

当前已知的只是 clean shared representation structure 存在，以及 native O2/P2
feature intervention path 可用。`shared != vulnerable != policy-relevant !=
transferable`。

### DECISION / FACT — C6-A Interface Closure

C6-A source audit、scientific review 与 final contract review 已完成。冻结状态为：

```text
INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES
```

C6-A 冻结了以下 stage boundary：

- confirmed primary cross-model action object 是 first-step translation only；
- rotation 仅可在确认 deployed robosuite `OSC_POSE` 语义后条件性加入；
- gripper 保持 separate analysis；
- gradient/JVP 可以作为 screening，但不是 policy relevance 的充分证据；
- controlled directional intervention 是更强的后续验证机制；
- 历史 C5-B mappings 未被持久化，不能声称精确恢复历史内存矩阵；
- C6-B 前必须通过显式授权的 re-fit 产生新的、版本化的 authoritative
  `O2 ↔ P2` 99%-PCA true-TRAIN mapping artifact；
- real intervention smoke 前必须先完成独立的 O2/P2 intervention-interface
  coding stage。

上述结论只完成 interface closure，不证明 policy relevance、causal relevance、
action relevance 或 transferability。

### OPEN — Previous C6-B Clean-Shared-Direction Route (Complementary)

C5 不负责证明：

```text
policy relevance
```

原 clean-shared-first 路线会从 candidate shared space 进一步寻找：

\[
S_{\mathrm{target}}
=
S_{\mathrm{shared}}
\cap
S_{\mathrm{action-relevant}}.
\]

该路线现为 `DEFERRED / RETAINED AS COMPLEMENTARY ROUTE`，用于 complementary
analysis、ablation 或 possible alternative route；它不是失败路线。其最终
sensitivity metric、native intervention vector、token scope、perturbation scale、
candidate-selection rule 或 statistical threshold 仍未冻结。

候选方向包括但不限于：

- gradient / Jacobian；
- probe；
- intervention；
- action prediction。

在该 complementary C6-B 方法被单独讨论、冻结和授权之前，不应提前选定其中任意
一种，也不得把它重新视为默认主线。

### DECISION / FACT — C5-BM Authoritative Mapping Materialization

C5-BM scientific/engineering contract 的最终 read-only audit 已通过，当前
`task.md` 是该阶段的冻结 authority。合同冻结该 materialization 的输入、
O2↔P2 99%-PCA true-TRAIN 重拟合、确定性符号规范、历史标量复现、产物完整性与
事务发布要求。

C5-BM implementation 为 `UNIT-LEVEL PASS`，formal materialization 已完成并达到
`FORMAL COMPLETE / PASS`。正式验证覆盖全部 `200 / 200` source feature pairs，确认
四个历史 C5-B 文件未变化、九个 mapping arrays 的内容哈希全部匹配，并持久化
`262` 个 canonical components；冻结的四个历史标量均以绝对差 `0.0` 复现。

该产物是新的 authoritative reusable mapping，不是历史未保存内存矩阵的恢复。
C5-BM 也不定义 native intervention vector、token scope、epsilon、C6-B sensitivity
metric 或 Tex3D loss。

### DECISION / FACT — C6 O2/P2 Intervention-Interface and Real-Smoke Closure

C6 O2/P2 intervention-interface contract 已冻结。Implementation 已达到
`UNIT-LEVEL PASS`，unit validation 为 `PASS`。后续独立授权的真实验证确认：

```text
real clean-equivalence:
  OpenVLA: 2/2 PASS
  pi0.5: 2/2 PASS

original intervention smoke:
  OpenVLA: BLOCKED under the frozen translation-response gate
  pi0.5: PASS

OpenVLA follow-up token/logit diagnostic:
  PASS
```

OpenVLA diagnostic 使用同一冻结 observation、direction 和 `alpha`，确认：

```text
modified O2
→ downstream action-token logits measurably changed
→ greedy action-token IDs remained unchanged
→ decoded translation therefore remained unchanged
```

因此 OpenVLA original intervention-smoke 的 `BLOCKED` 仍是历史事实，不得改写为
`PASS`；但它不再代表 intervention interface 的工程 blocker。当前
intervention-interface feasibility / intervention closure 为 `COMPLETE`。

该阶段只支持 O2/P2 continuation 在真实 checkpoint 上 clean-equivalent、native
O2/P2 intervention 会进入 downstream policy computation、pi0.5 小随机 P2
intervention 可产生 decoded-action response，以及 OpenVLA 小随机 O2 intervention
可产生 action-logit response但未跨越 greedy token boundary。它不支持 shared CCA
direction 已具 action relevance、两个模型具有共同 policy sensitivity、transferable
adversarial direction 已建立或 Tex3D transferable attack 已成立。`shared !=
vulnerable != policy-relevant != transferable`。原 C6-B clean-shared-direction 计划为
`DEFERRED / COMPLEMENTARY ROUTE`；当前 vulnerability-first next stage 尚未
contracted/authorized，CCA-to-native direction construction 与 Tex3D optimization
均未授权。

正式 provenance 与输出位置为：

```text
real-smoke project commit:
eefc0e652f801c20f3de29c5d53e821dd65aa978

OpenVLA diagnostic artifact/project commit:
fffea7571fcde7922b0d0abc1a56d1e88439c011

experiment_inbox/c6-real-smoke-output/
experiment_inbox/c6-openvla-logit-diagnostic/
```

---

## 14. Vulnerable-Feature Attack Stage

### DECISION — Discovery and Attack Remain Separate

Representation discovery 可以使用 OpenVLA 与 \(\pi_{0.5}\) 的 model-specific
clean/adversarial features，并可在 discovery 阶段分析跨模型 vulnerable-structure
alignment/fusion。

未来正式攻击阶段必须只使用 surrogate-side frozen vulnerable/shared structure。

例如：

\[
x_{\mathrm{adv}}
\rightarrow
Z_A(x_{\mathrm{adv}})
\rightarrow
H_A(x_{\mathrm{adv}})
\rightarrow
\mathcal{L}_{\mathrm{vulnerable-shared}}
\rightarrow
\theta_{\mathrm{texture}}.
\]

该式仅表示 single-surrogate threat-model boundary；具体 vulnerable feature、mapping
与 loss 尚未确定或授权。

Held-out VLA 不提供：

- feature；
- action；
- loss；
- gradient。

最终 shared-feature objective 必须在完成 policy/action relevance analysis 之后再设计。

当前不得直接把 CKA similarity 或 SVCCA canonical directions 当作最终 attack objective。

---

## 15. Full Research Flow

当前 authoritative research flow：

```text
Pilot v0.2 protocol freeze
        ↓
portable OpenVLA checkpoint identity frozen
        ↓
Pilot v0.2 collection
        ↓
C2 full OpenVLA extraction
        ↓
C3 full π0.5 extraction
        ↓
C4 full paired-feature manifest
        ↓
group-aware C5 split
        ↓
C5-A Linear CKA — FORMAL COMPLETE / GO
        ↓
C5-B SVCCA — FORMAL COMPLETE / PASS
        ↓
C5 representation-stage — PASS
        ↓
C6-A policy-sensitivity interface closure — COMPLETE / FROZEN
        ↓
C5-BM scientific/engineering contract — FROZEN
        ↓
C5-BM implementation — UNIT-LEVEL PASS
        ↓
C5-BM formal materialization — FORMAL COMPLETE / PASS
        ↓
O2/P2 intervention-interface contract — FINAL AUDIT PASS / FROZEN
        ↓
O2/P2 intervention-interface implementation — UNIT-LEVEL PASS
        ↓
O2/P2 intervention-interface unit validation — PASS
        ↓
C6 real clean-equivalence — PASS (OpenVLA 2/2; pi0.5 2/2)
        ↓
C6 original intervention smoke — OpenVLA HISTORICAL BLOCKED; pi0.5 PASS
        ↓
C6 OpenVLA token/logit diagnostic — PASS
        ↓
C6 intervention-interface feasibility / closure — COMPLETE
        ↓
model-specific vulnerability discovery — NEXT / NOT CONTRACTED / NOT AUTHORIZED
        ↓
action-relevant vulnerable feature identification
        ↓
cross-model vulnerable feature alignment / fusion
        ↓
shared vulnerable representation
        ↓
single-surrogate vulnerable-feature Tex3D loss
        ↓
Tex3D texture optimization
        ↓
held-out cross-VLA transfer evaluation
```

第一个 stage gate——C5-A geometry-stage gate——已经正式执行：

> `O2 ↔ P2` 是否在 TRAIN 与 HELD-OUT 上都通过冻结的 debiased Linear CKA
> group-block null criterion？

C5-A 的正式结果为 `GO`，C5-B / SVCCA 的正式结果为 `PASS`，因此冻结的 joint
gate 得到 `C5 representation-stage PASS`。C6-A interface closure 现已完成并冻结为
`INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES`。该结论不是 policy/action
relevance PASS；C5-BM contract 已冻结，implementation 已达到 `UNIT-LEVEL PASS`，
formal materialization 已正式完成并验证为 `PASS`。真实 clean-equivalence 已在
OpenVLA 与 pi0.5 上达到 `2 / 2 PASS`；原始 intervention smoke 保留 OpenVLA
`BLOCKED`、pi0.5 `PASS` 的历史结果。OpenVLA follow-up token/logit diagnostic
为 `PASS`，支持 discrete token-boundary explanation，并使 intervention-interface
closure 达到 `COMPLETE`。这仍不是 policy/action relevance PASS。最终 overall
research gate 仍为 `OPEN / NOT DEFINED HERE`。

该图在 C6 closure 后展示的是新的 vulnerability-first 默认路线。原
clean-shared-first 路线（clean CCA → action-relevant shared direction → attack）
仍完整保留为 complementary analysis / ablation / possible alternative route，不是
失败路线。C5-A、C5-B 与 C5-BM 的历史 pipeline 和结果仍是 clean shared structure
稳定且可对齐的正式证据。

如果 C5-A NO-GO，应停止并重新评估当前 frozen geometry hypothesis；不得静默
更换 C5-A metric、split、null 或 threshold。无论 C5-A 结果如何，都不能将其
单独写成最终 overall C5 PASS/FAIL。

---

## 16. Documentation and Engineering State

### FACT

当前 scientific documentation hierarchy：

```text
docs/pilot-v0.1-spec.md
    historical frozen Pilot

docs/pilot-v0.2-spec.md
    current C5 data-collection and grouping protocol

docs/research-map.md
    current research rationale, status, and roadmap

AGENTS.md
    long-lived repository/research rules and document routing

task.md
    current implementation contract
```

### CURRENT DOCUMENTATION ORDER

```text
1. pilot-v0.2-spec.md        PASS
2. task.md                   C6 INTERVENTION CLOSURE COMPLETE
3. AGENTS.md                 C6 INTERVENTION STATUS SYNCHRONIZED
4. research-map.md           C6 INTERVENTION STATUS SYNCHRONIZED
```

Pilot v0.1 remains historical and unchanged.

`AGENTS.md` distinguishes v0.1 and v0.2 routing and preserves the completed C5-A,
C5-B, representation-stage, C6-A interface-closure, C5-BM, and C6 intervention
records. The current `task.md` preserves the frozen C6 real-smoke contract and its
execution record. The intervention interface implementation is `UNIT-LEVEL PASS`,
unit validation is `PASS`, and real clean-equivalence is `PASS` for OpenVLA `2 / 2`
and pi0.5 `2 / 2`. The original intervention smoke remains OpenVLA `BLOCKED` under
the frozen translation gate and pi0.5 `PASS`; the follow-up OpenVLA token/logit
diagnostic is `PASS`, resolving the interface engineering blocker without rewriting
the historical smoke result. Formal C6-B policy/action experiments remain
`DEFERRED / RETAINED AS COMPLEMENTARY ROUTE` and are not authorized. The next
default scientific stage is the vulnerability-first cross-model feature study,
which is `NOT YET CONTRACTED / NOT AUTHORIZED`.

The C5-D0 collector has reached `UNIT-LEVEL PASS`, its reduced real integration
smoke has been audited as `PASS`, and the formal Pilot v0.2 collection has reached
`COMPLETED + COMPLETE` with 50 accepted groups and 200 canonical observations.
C2/C3 formal extraction and C4 formal pairing are complete. Formal C5-A execution
is also complete: TRAIN and HELD-OUT both passed, producing the frozen
geometry-stage result `C5-A GO`.

The portable OpenVLA checkpoint identity is frozen as
`openvla/openvla-7b-finetuned-libero-spatial` and was used by the completed formal
collection.

C5-B / SVCCA implementation and formal execution are complete, producing
`C5-B PASS` and therefore `C5 representation-stage PASS` under the frozen joint
gate. The final overall research gate remains `OPEN / NOT DEFINED HERE`;
C6-A interface closure is `COMPLETE / FROZEN`, with status
`INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES`. C5-BM contract is `FROZEN`, its
implementation is `UNIT-LEVEL PASS`, and formal materialization is `FORMAL
COMPLETE / PASS`. The intervention-interface contract is `FINAL AUDIT PASS /
FROZEN`, its implementation is `UNIT-LEVEL PASS`, and unit validation is `PASS`.
Real clean-equivalence is complete and passed for both model interfaces. The
original intervention-smoke split result and subsequent OpenVLA diagnostic are
preserved separately, and intervention-interface closure is `COMPLETE`. C6-B is
`DEFERRED / COMPLEMENTARY ROUTE`; the vulnerability-first cross-model study is
`NEXT / NOT CONTRACTED / NOT AUTHORIZED`. Model-specific vulnerability discovery,
policy/action relevance, transferability, and Tex3D optimization remain
`NOT STARTED / NOT AUTHORIZED`.

---

## 17. Current One-Line Research Summary

> Linear CKA and SVCCA establish clean alignable cross-VLA representation evidence, and the C6 O2/P2 intervention-interface closure establishes downstream intervention feasibility; the next default route is model-specific vulnerability/action-relevance discovery followed by cross-model vulnerable-structure alignment, while clean-shared-first analysis remains a complementary route and no vulnerability, transferability, or Tex3D attack conclusion has yet been established.
