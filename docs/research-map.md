# Shared-Feature Tex3D Research Map

## 1. Research Goal

当前研究目标是提升 Tex3D adversarial 3D texture 在不同架构 Vision-Language-Action (VLA) 模型之间的迁移性。

核心问题不是继续增强单一 surrogate VLA 上的攻击强度，而是减少 adversarial texture 对 surrogate architecture 特有表征的过拟合。

当前核心假设是：

> 不同 VLA 架构在视觉到动作决策的数据流中，可能存在稳定的 cross-model shared representation structure。若进一步找到其中与 policy/action 相关的 shared directions，并在单一 surrogate 上针对这些方向优化 adversarial texture，则可能提高跨架构迁移性。

必须严格区分：

\[
\text{shared}
\neq
\text{transferable}
\neq
\text{policy-relevant}.
\]

因此，representation similarity 只是研究链路的第一步，不能直接作为攻击迁移性或策略相关性的证据。

---

## 2. Threat Model

### DECISION — Shared-Feature Discovery + Single-Surrogate Attack

Representation discovery 阶段允许同时分析多个 VLA，例如：

- OpenVLA；
- \(\pi_{0.5}\)。

允许使用两边的 paired clean representations 学习或分析 shared structure。

但是正式 adversarial texture optimization 必须保持：

```text
single surrogate only
```

例如以 OpenVLA 为 surrogate：

\[
x_{\mathrm{adv}}
\rightarrow
Z_A(x_{\mathrm{adv}})
\rightarrow
W_A
\rightarrow
H_A(x_{\mathrm{adv}})
\rightarrow
\mathcal{L}_{\mathrm{shared-policy}}.
\]

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

研究路线显式拆分为：

```text
cross-model representation similarity
        ↓
explicit shared-space alignment
        ↓
policy/action relevance
        ↓
shared + action-relevant representation
        ↓
single-surrogate shared-feature loss
        ↓
Tex3D texture optimization
        ↓
held-out VLA transfer evaluation
```

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

当前 C0–C4 状态：

```text
C0 — PilotObservation schema               PASS
C1 — LIBERO/OpenVLA observation collector  PASS
C2 — OpenVLA feature extraction            PASS
C3 — π0.5 feature extraction               PASS
C4 — paired-feature manifest               PASS
```

C1 已完成 real OpenVLA/LIBERO smoke。

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

现有 2-sample real smoke 只证明：

```text
pipeline closure
```

不能用于正式 C5 scientific conclusion。

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

## 8. Current Research Blocker

### OPEN — Portable OpenVLA Checkpoint Identity

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

但 portable OpenVLA checkpoint identity 仍未冻结。

机器本地 filesystem path 不能作为 scientific checkpoint identity。

在正式 Pilot v0.2 collection 开始之前，必须将该项从：

```text
OPEN
```

转换为：

```text
DECISION
```

并使用可移植、稳定的 checkpoint identity，例如 repository/revision、artifact identity 或 checkpoint content digest。

### STOP CONDITION

在 portable OpenVLA checkpoint identity 冻结之前：

```text
formal Pilot v0.2 collection MUST NOT start
```

该 OPEN 不阻塞：

- `research-map.md` 更新；
- `AGENTS.md` 文档路由更新；
- collection contract 的准备性讨论。

但它阻塞正式 collection execution。

---

## 9. C5-A — Representation Similarity

### DECISION — Primary Method: Linear CKA

C5-A 回答：

> OpenVLA 与 \(\pi_{0.5}\) 是否对同一批 observations 形成相似的 representation geometry？

Primary method：

```text
Linear CKA
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

CKA 用于判断 representation geometry similarity。

CKA 不用于：

- channel-to-channel correspondence；
- 学习 explicit \(W_A/W_B\)；
- 判断 individual observation 是否“shared”；
- 证明 policy relevance 或 attack transferability。

---

## 10. C5-B — Explicit Shared Linear Alignment

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

### DECISION — Group-Block Shuffled Null

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

对于 token-wise SVCCA，每个 observation 的 256 tokens 必须一起移动，并保持 patch position：

\[
(obs_i, token_t)_A
\leftrightarrow
(obs_{\pi(i)}, token_t)_B.
\]

Null permutation 必须：

- 在 TRAIN 与 HELD-OUT 内分别独立执行；
- 禁止跨 split permutation；
- 使用 group-level derangement，即不允许任何 group 保持原始 cross-model pairing；
- 保持每个 group 内的 4 个 progress slots；
- 保持每个 observation 内的 patch-position correspondence。

Primary null repeats：

```text
50
```

CKA primary evidence：

```text
held-out true CKA
vs
held-out group-block shuffled CKA distribution
```

SVCCA null：

```text
reuse frozen node-specific PCA
refit CCA for each group-block shuffled TRAIN pairing
evaluate on an independently group-block shuffled HELD-OUT pairing
```

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

### OPEN — Null Permutation RNG Seed

50 次 permutation 的 RNG seed 必须在查看正式 C5 结果之前冻结，并由独立 C5 scientific contract 记录。

实现不得自行选择 seed。

---

## 12. C5 Interpretation Rules

### OPEN — Exact C5 Go/No-Go Decision Rule

以下术语目前只用于 qualitative interpretation：

```text
CKA positive
SVCCA positive
SVCCA weak
near null
significantly above shuffled null
```

它们尚未构成可执行的 PASS/FAIL 判据。

在查看任何 formal C5 result 之前，必须通过独立 C5 scientific contract 冻结至少：

- primary CKA decision statistic；
- primary SVCCA decision statistic；
- null comparison rule；
- permutation p-value、null quantile 和/或 effect-size 的使用方式；
- 显著性或 effect threshold；
- multiple representation pairings / supporting metrics 的处理方式；
- CKA 与 SVCCA 如何联合形成 C5 go/no-go conclusion；
- null permutation RNG seed。

在这些规则冻结之前：

```text
formal C5 PASS/FAIL evaluation MUST NOT begin
```

以下解释只描述不同结果模式的科学含义，不代表已经冻结的统计阈值。

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

## 13. Policy / Action Relevance — Next Stage

### OPEN

C5 不负责证明：

```text
policy relevance
```

如果 C5 得到 candidate shared space，下一阶段需要进一步寻找：

\[
S_{\mathrm{target}}
=
S_{\mathrm{shared}}
\cap
S_{\mathrm{action-relevant}}.
\]

当前尚未冻结具体 action-relevance 方法。

候选方向包括但不限于：

- gradient / Jacobian；
- probe；
- intervention；
- action prediction。

在方法正式讨论和冻结之前，不应提前选定其中任意一种。

---

## 14. Shared-Feature Attack Stage

### DECISION — Discovery and Attack Remain Separate

Representation discovery 可以使用 OpenVLA 与 \(\pi_{0.5}\) 的 paired clean features。

正式攻击阶段只使用 surrogate-side frozen mapping。

例如：

\[
x_{\mathrm{adv}}
\rightarrow
Z_A(x_{\mathrm{adv}})
\rightarrow
H_A(x_{\mathrm{adv}})
\rightarrow
\mathcal{L}_{\mathrm{shared-policy}}
\rightarrow
\theta_{\mathrm{texture}}.
\]

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
resolve portable OpenVLA checkpoint identity
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
C5-A Linear CKA
        ↓
C5-B SVCCA
        ↓
policy/action relevance analysis
        ↓
shared + action-relevant directions
        ↓
single-surrogate shared-feature Tex3D loss
        ↓
Tex3D texture optimization
        ↓
held-out cross-VLA transfer evaluation
```

第一个当前 scientific go/no-go gate 是 C5：

> selected OpenVLA / \(\pi_{0.5}\) representation nodes 是否在 held-out observations 上表现出强于预先冻结 null criterion 的 shared structure，并且 explicit alignment evidence 是否满足预先冻结的解释规则？

该 gate 的 exact PASS/FAIL decision rule 当前仍为 `OPEN`，必须在查看正式 C5 结果之前通过独立 C5 scientific contract 冻结。

如果该 gate 不通过，不应强行进入 shared-feature attack design。

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
2. research-map.md           PASS
3. AGENTS.md routing update  PASS
4. task.md                   PASS
```

Pilot v0.1 remains historical and unchanged.

`AGENTS.md` now distinguishes v0.1 and v0.2 routing and preserves stage-specific OPEN blockers.

The current `task.md` is the implementation contract under final audit. Coding must not begin until that contract passes review.

Formal Pilot v0.2 collection remains blocked until the portable OpenVLA checkpoint identity is frozen.

Formal C5 PASS/FAIL evaluation remains separately blocked until the exact C5 go/no-go rule and null RNG seed are frozen in a dedicated C5 scientific contract.

---

## 17. Current One-Line Research Summary

> Use Linear CKA to establish where cross-VLA representation geometry is shared, use SVCCA to test whether that similarity admits an explicit held-out-generalizing linear shared space, then identify action-relevant directions inside that space before designing a single-surrogate Tex3D shared-feature loss.