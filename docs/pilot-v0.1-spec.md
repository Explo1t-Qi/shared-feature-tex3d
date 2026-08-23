# Shared-Feature Feasibility Pilot v0.1

## 1. Purpose

本 Pilot 的目标是验证：

> OpenVLA 与 π0 对同一批 LIBERO observations 的视觉 representation 中，是否存在稳定、可泛化、且显著强于随机配对的 shared structure。

当前阶段只验证：

$$
\text{shared representation exists?}
$$

当前阶段**不**声称：

$$
\text{shared}
=
\text{transferable}
=
\text{policy-relevant}
$$

也不假设当前使用的 mean-pooled representation 会成为最终 adversarial texture optimization 的 shared-feature objective。

---

## 2. Scope

Pilot v0.1 是一个 **read-only representation feasibility experiment**。

当前不实现：

- shared-feature adversarial loss；
- Tex3D texture optimization；
- token-level shared attack；
- object-region shared attack；
- π0 Gemma / Action Expert feature objective；
- multi-VLA ensemble attack。

本阶段只执行：

```text
LIBERO observation collection
        ↓
OpenVLA representation extraction
        ↓
π0 representation extraction
        ↓
offline paired representation analysis
```

---

## 3. Observation Source

### DECISION

Pilot observation dataset 由 **OpenVLA rollout** 产生。

OpenVLA 和 π0 不分别执行 rollout。

原因是必须保证：

$$
x_i^{OpenVLA}=x_i^{\pi_0}=x_i
$$

即两个模型处理完全相同的外部 observation。

流程为：

```text
OpenVLA policy
      ↓
LIBERO rollout
      ↓
save observations
      ↓
      ├───────────────┐
      ↓               ↓
OpenVLA extractor   π0 extractor
      ↓               ↓
representation O     representation P
      └──── paired ───┘
```

Pilot 中发现的 representation structure 因此首先是在 OpenVLA-induced state distribution 上进行验证。

其他 rollout source，例如 π0 rollout 或 expert demonstrations，可作为未来 distribution control，而不是 Pilot v0.1 的必要组成部分。

---

## 4. Task and Initial States

### DECISION

Pilot 使用：

```text
1 LIBERO task
10 fixed official initial states
```

每个 initial state 对应一个独立 episode：

```text
state 0 → episode 0
state 1 → episode 1
...
state 9 → episode 9
```

使用不同 initial states 的目的是增加 trajectory 和 scene-state diversity，而不是测试最终 adversarial texture 的 cross-task universality。

---

## 5. Frame Sampling

每个 episode 保存：

```text
20 valid-policy frames
```

总 sample 数约为：

$$
10\times20=200
$$

### Sampling Rules

- 不从 rollout video 中重新截帧；
- 直接保存 LIBERO environment observation；
- 不使用连续密集采样；
- 跳过 dummy action / no-op / stabilization 阶段；
- 在真正的 policy execution 区间内均匀采样；
- 每个 episode 保存相同数量的 frames；
- 避免较长 episode 在 statistical analysis 中获得更高权重。

每个 sample 可记录：

$$
p=\frac{t}{T-1}
$$

作为 `normalized_episode_progress`，用于未来分析 representation 是否随任务阶段变化。

该字段目前仅作为 metadata。

---

## 6. Train / Held-Out Split

### DECISION

严格按照完整 episode 划分：

```text
episode/state 0–6 → discovery / train
episode/state 7–9 → held-out
```

约：

```text
140 train observations
60 held-out observations
```

禁止随机按 frame 划分 train/test。

同一个 episode 中的所有 frames 必须属于同一个 split。

原因是同一 trajectory 内相邻 observations 高度相关，frame-level random split 会造成严重 data leakage。

### Train-Only Rule

所有需要学习参数或统计量的步骤只能在 train episodes 上 fit，包括：

- feature centering / standardization；
- PCA / SVD；
- CCA；
- linear regression。

Held-out observations 只能执行：

```text
transform
+
evaluation
```

---

## 7. Observation Dataset

### DECISION

Collector 保存尽可能原始的 LIBERO observation。

不保存 OpenVLA 已经 resize / normalize 后的最终 policy input 作为公共输入标准。

目标是让：

```text
same raw observation
      │
      ├── OpenVLA native preprocessing
      │
      └── π0 native preprocessing
```

两个模型分别使用自己的官方 inference preprocessing。

### Required Sample Metadata

每个 observation 至少包含：

```text
sample_id
task_id
initial_state_id
episode_id
step_id
normalized_episode_progress

base_rgb_raw
wrist_rgb_raw
state
prompt
episode_success
```

需要保留足够信息，以便重建两个模型的 inference input。

---

## 8. Camera Selection

### DECISION

Pilot v0.1 的 cross-model representation comparison 只使用：

```text
base / agent-view image
```

不把 π0 wrist-camera representation 混入主分析。

原因是当前 OpenVLA LIBERO policy 主要使用 base image，而 π0 可以使用额外 wrist observations。

Pilot 需要首先保证 representation comparison 的外部视觉输入语义一致。

---

## 9. OpenVLA Representation Nodes

Pilot 提取两个 OpenVLA representation nodes。

### O1 — SigLIP Raw Representation

作为 backbone-level comparison node。

语义：

```text
SigLIP visual representation
before OpenVLA multimodal projector
```

### O2 — Projector Representation

作为 OpenVLA-adapted visual representation。

语义：

```text
vision representation
after OpenVLA projector
before deeper Llama processing
```

Pilot v0.1 不要求提取更深的 Llama language-conditioned visual hidden states。

---

## 10. π0 Representation Nodes

Pilot 提取两个 π0 representation nodes。

### P1 — V1 Raw SigLIP Representation

形状约为：

$$
[256,1152]
$$

对应：

```text
SigLIP encoder output
after final encoder normalization
before PaliGemma projection
```

### P2 — V2 Projected Representation

形状约为：

$$
[256,2048]
$$

对应：

```text
PaliGemma-ready image token embeddings
```

即：

```text
SigLIP visual representation
        ↓
Dense 1152 → 2048
        ↓
image_tokens
```

Pilot v0.1 不等待 Gemma language-conditioned V3 representation 完成 audit 后才启动。

---

## 11. Primary Cross-Model Pairs

主实验只分析：

$$
O1\leftrightarrow P1
$$

和：

$$
O2\leftrightarrow P2
$$

### O1 ↔ P1

回答：

> backbone-level visual representations 是否具有 shared structure？

### O2 ↔ P2

回答：

> representation 经过各自 VLA-specific adaptation 后，shared structure 是否仍然存在？

以下组合暂不作为 Pilot 主结果：

```text
O1 ↔ P2
O2 ↔ P1
```

未来可以作为 supplementary analysis。

---

## 12. Feature Serialization

### DECISION

Extractor 必须保存完整 token representation。

禁止在 extraction 阶段只保存 mean-pooled vector。

例如：

```text
OpenVLA

O1: [256, d1]
O2: [256, d2]
```

```text
π0

P1: [256,1152]
P2: [256,2048]
```

保存完整 token tensor 的目的是让同一 representation dataset 未来能够支持：

- global mean pooling；
- token-level analysis；
- object-region pooling；
- spatial shared-feature analysis。

---

## 13. Sample Alignment

### DECISION

所有 observation 和 representation 必须通过稳定的：

```text
sample_id
```

进行关联。

禁止依赖：

- 文件排序；
- 数组 index；
- extractor 输出顺序。

Feature record 至少包含：

```text
sample_id
model_name
checkpoint
node_name
feature_shape
feature_dtype
source_image_hash
```

统计分析前必须验证：

```text
OpenVLA.sample_id == π0.sample_id
```

并且：

```text
OpenVLA.source_image_hash == π0.source_image_hash
```

如果任一条件不成立，analysis 必须失败，而不是继续计算。

---

## 14. Artifact Organization

Pilot 数据逻辑上分成三层：

```text
observations/
features/
analysis/
```

对应：

```text
raw observations
      ↓
model representations
      ↓
statistical analysis
```

推荐：

```text
observations/
    manifest.jsonl
    ...

features/
    openvla/
    pi0/

analysis/
    pilot_v0.1/
```

Train / held-out split 必须作为独立固定 artifact 保存。

不同 analysis scripts 不允许各自重新生成随机 split。

---

## 15. Primary Representation Reduction

对于每张 observation：

$$
Z\in\mathbb R^{256\times d}
$$

Pilot 主实验使用 mean pooling：

$$
\bar z
=
\frac{1}{256}
\sum_{i=1}^{256}z_i
$$

得到：

$$
\bar z\in\mathbb R^d
$$

因此每张 observation 对应一个 frame-level representation vector。

### Interpretation

Mean-pooled analysis 回答：

> OpenVLA 与 π0 对整张 observation 的 global visual representation 是否存在稳定的 cross-model correspondence？

它不回答：

> 哪个 patch 是 transferable？

也不回答：

> 哪个 feature direction 与动作最相关？

### Important Limitation

Mean pooling 会丢失 spatial information，并可能稀释 target-object representation。

因此：

> mean-pooled CCA 是 feasibility baseline，而不是最终 shared-feature attack objective。

---

## 16. Future Representation Levels

Pilot v0.1 之后可以继续研究：

### Stage 2 — Token-Level Representation

研究：

```text
spatially corresponding visual tokens
```

之间是否存在 shared structure。

### Stage 3 — Object-Region Representation

利用目标物体 spatial mask，只分析：

```text
target-object-related visual tokens
```

这可能比 whole-image mean 更接近 Tex3D 最终 attack objective。

但 Pilot v0.1 不引入 segmentation / object-mask complexity。

---

## 17. CCA Method

### DECISION

Pilot 使用 SVCCA-style analysis，而不是直接在高维 raw representation 上执行 vanilla CCA。

若：

$$
n\ll d
$$

其中：

- $n$ = paired observations 数量；
- $d$ = representation dimension；

则高维 CCA 很容易在 train data 中获得虚假的高 canonical correlation。

例如：

$$
n\approx140
$$

但：

$$
d=1152,\ 2048,\ 4096
$$

因此首先进行 PCA / SVD 降维。

---

## 18. SVCCA Pipeline

对于 train representations：

$$
X\in\mathbb R^{n\times d_A}
$$

$$
Y\in\mathbb R^{n\times d_B}
$$

执行：

```text
train-only standardization
          ↓
independent PCA / SVD
          ↓
retain major explained variance
          ↓
CCA
```

### PCA / SVD

OpenVLA 与 π0 分别独立 fit PCA/SVD。

两侧不要求保留相同 dimension。

主设置：

$$
99\%\text{ explained variance}
$$

robustness check：

$$
95\%
$$

例如允许：

```text
OpenVLA: 4096 → 73
π0:      2048 → 58
```

### CCA

在 train pairing 上 fit：

$$
W_X,\ W_Y
$$

然后冻结：

- train standardization statistics；
- PCA/SVD transformations；
- CCA projections。

Held-out data 只能通过冻结后的 pipeline：

```text
held-out features
      ↓
train standardization
      ↓
train PCA
      ↓
train CCA projection
      ↓
held-out canonical variables
```

最终 canonical correlation 只在 held-out observations 上解释。

---

## 19. CCA Metrics

CCA 不只报告 top-1 correlation。

至少保存：

```text
rho_1
mean top-5
mean top-10
```

如果实际 canonical component 数少于 10，则只报告可用 components。

目的在于区分：

```text
one isolated highly correlated direction
```

和：

```text
multi-dimensional shared subspace
```

---

## 20. Shuffled-Pairing Null Baseline

### DECISION

必须建立随机 pairing 对照。

Correct pairing：

$$
X_i\leftrightarrow Y_i
$$

Shuffled pairing：

$$
X_i\leftrightarrow Y_{\pi(i)}
$$

其中 $\pi$ 是随机 permutation。

Pilot 建议执行：

```text
20 shuffled repetitions
```

得到 null distribution。

Shuffled experiment 用来回答：

> 如果跨模型 sample correspondence 本身是随机的，同样的 statistical pipeline 能产生多高的 apparent correlation？

PCA/SVD 是单模型变换，不依赖跨模型 pairing，因此实现时可以安全复用 train-only single-model preprocessing。

CCA fitting 和 paired evaluation 必须按照相应 correct / shuffled pairing 执行。

---

## 21. Additional Representation Metrics

Pilot 不只依赖 CCA。

### Linear CKA

回答：

> 两个 representation 的整体 sample geometry 是否相似？

### Linear Regression $R^2$

在 train 上学习：

$$
Y\approx XW+b
$$

然后在 held-out observations 上测量：

$$
R^2
$$

回答：

> 一个 VLA 的 representation 是否包含能够泛化预测另一个 VLA representation 的线性信息？

CCA、CKA 与 regression 提供互补证据。

---

## 22. Pilot Outputs

Pilot 最终控制为简单、可解释的输出。

### Main Table

至少包含：

| Representation Pair | Held-Out CCA Top-1 | Held-Out CCA Top-5 Mean | Linear CKA | Regression $R^2$ | Shuffled Baseline |
| ------------------- | -----------------: | ----------------------: | ---------: | ---------------: | ----------------: |
| O1 ↔ P1             |                    |                         |            |                  |                   |
| O2 ↔ P2             |                    |                         |            |                  |                   |

### Figure 1 — CCA Spectrum

横轴：

```text
canonical component
```

纵轴：

```text
held-out canonical correlation
```

比较：

```text
correct pairing
shuffled null baseline
```

### Figure 2 — Correct vs Shuffled

展示主要指标的：

```text
correct result
vs
shuffle distribution
```

---

## 23. Pilot Gate

Pilot 不预先规定：

$$
CCA>0.7
$$

之类的固定绝对阈值。

主要判断 empirical signal 是否稳定且明显高于 null baseline。

### PASS

满足：

1. held-out observations 上仍存在明显 correspondence；
2. correct pairing 明显强于 shuffled pairing；
3. 至少两个 representation metrics 支持 shared-structure interpretation。

通过后可以继续：

```text
token-level analysis
        ↓
object-region analysis
        ↓
deeper multimodal / policy representation audit
        ↓
shared-feature attack objective
```

### UNCERTAIN

例如：

- correct 略高于 shuffled；
- CCA / CKA / regression 结论冲突；
- O1 ↔ P1 很强但 O2 ↔ P2 明显变弱；
- PCA threshold 对结果影响极大。

此时先检查：

- sample size；
- preprocessing；
- representation extraction；
- PCA/SVD stability；
- shared-SigLIP confounder。

不直接进入 adversarial attack implementation。

### FAIL

如果：

```text
held-out correct ≈ shuffled
```

并且：

```text
CCA
CKA
Regression
```

都没有提供稳定 signal，则暂停 shared-feature attack implementation，重新检查核心假设和 candidate representation nodes。

---

## 24. Shared-SigLIP Confounder

需要特别关注：

```text
O1 ↔ P1 strong
O2 ↔ P2 weak
```

这种结果不能简单判定 Pilot failure。

它可能意味着：

> observed shared structure 主要来自共同或高度相似的 SigLIP backbone，而 VLA-specific adaptation 使两个 representation spaces 快速分化。

如果：

```text
O1 ↔ P1 strong
O2 ↔ P2 also clearly above shuffled
```

则 evidence 更强，因为 shared structure 不仅存在于底层 vision backbone。

Pilot 的核心科学问题因此不仅是：

> CCA correlation 有多高？

而是：

> shared representation structure 在 VLA representation hierarchy 中能保留到多深？

---

## 25. Relationship to Final Attack

Shared-representation discovery 与 adversarial texture optimization 是两个阶段。

Discovery 阶段允许分析多个 VLA：

$$
Z_A(x),Z_B(x)
$$

并学习 shared mapping。

之后 mapping 冻结。

正式 adversarial texture optimization 只使用 surrogate VLA。

因此本项目不是：

$$
\mathcal L
=
\mathcal L_{OpenVLA}
+
\mathcal L_{\pi_0}
$$

这样的 ensemble attack。

未来攻击形式更接近：

$$
x_{adv}=R(\theta)
$$

$$
Z_A(x_{adv})
\rightarrow
\Phi_A(Z_A(x_{adv}))
\rightarrow
\mathcal L_{shared}
$$

其中只有 surrogate model $A$ 参与 texture gradient。

---

## 26. Codex Implementation Rule

本文件描述完整 Pilot roadmap 和已经冻结的 research decisions。

**它不是授权 Codex 一次实现整个 Pilot。**

Codex 每次 coding task 必须满足：

> Only implement the explicitly requested coding contract.

不得因为本文件描述了未来模块而提前实现：

- π0 extractor；
- CCA；
- CKA；
- regression；
- Tex3D integration；
- shared-feature attack loss；
- unrelated refactors。

每轮 coding task 应明确：

```text
Goal
Allowed files
Forbidden scope
Required behavior
Tests
Stop condition
```

默认采用小规模、可完整 review 的 implementation turn。

---

## 27. Pilot v0.1 Summary

Pilot v0.1 可以概括为：

> 使用 OpenVLA 在同一 LIBERO task 的 10 个固定 initial states 上执行 rollout，采集约 200 个 paired raw observations。随后分别在独立模型环境中提取 OpenVLA SigLIP/projector 与 π0 SigLIP/projected visual-token representations。每帧首先使用 mean pooling 得到 global representation，在 train episodes 上执行 train-only PCA/SVD → CCA，并使用 Linear CKA 和 linear regression 作为辅助指标。最终只在完整 held-out episodes 上评测，并与 repeated shuffled-pairing null baseline 比较，以判断跨 VLA global visual representation 是否存在稳定、可泛化的 shared structure。

---

## 28. Current Status

### FROZEN

- Pilot research question；
- observation source；
- rollout policy；
- task scope；
- initial-state policy；
- frame sampling principle；
- train/held-out split；
- camera choice；
- OpenVLA nodes O1/O2；
- π0 nodes P1/P2；
- full-token serialization；
- sample alignment rules；
- mean-pooled primary analysis；
- SVCCA-style CCA；
- PCA explained-variance principle；
- shuffled null baseline；
- CKA / regression supporting metrics；
- PASS / UNCERTAIN / FAIL interpretation。

### NOT YET IMPLEMENTED

- Pilot observation schema；
- rollout collector；
- OpenVLA representation extractor；
- π0 representation extractor；
- paired-feature validator；
- SVCCA analysis；
- CKA analysis；
- regression analysis；
- Pilot report generation。

### NEXT STEP

将 Pilot implementation 拆分为多个独立 Codex coding contracts，并从最小的数据 schema / observation collector 开始。



## 9. OpenVLA Representation Nodes

### DECISION

Pilot v0.1 对 OpenVLA 保存三个视觉 representation nodes：

```text
O1-S
O1-F
O2
```

其中：

### O1-S — OpenVLA SigLIP Branch Representation

语义：

```text
OpenVLA fused visual backbone 中的 SigLIP branch representation
before DINO/SigLIP feature fusion
before OpenVLA multimodal projector
```

对应 OpenVLA 实际 inference path 中 SigLIP featurizer 的输出。

Pilot 预期单 sample 形状：

```text
[256, 1152]
```

O1-S 的主要作用是作为跨模型 backbone-level control node，便于后续与 π0 的 SigLIP representation 进行直接比较。

---

### O1-F — OpenVLA Fused DINOv2 + SigLIP Representation

语义：

```text
DINOv2 visual representation
        +
SigLIP visual representation
        ↓
concat along feature dimension
        ↓
full OpenVLA pre-projector visual representation
```

即：

```text
O1-F = concat(DINOv2 feature, SigLIP feature)
```

Pilot 预期单 sample 形状：

```text
[256, 2176]
```

O1-F 是 OpenVLA multimodal projector 实际接收的完整视觉 representation。

O1-F 用于区分：

```text
shared SigLIP-backbone structure
```

与：

```text
OpenVLA full fused visual structure
```

之间的差异。

当前 Pilot 不要求 O1-F 必须成为跨模型主比较节点；它首先作为 diagnostic / supplementary representation 保存。

---

### O2 — OpenVLA Projector Representation

语义：

```text
O1-F
    ↓
OpenVLA multimodal projector
    ↓
VLA-adapted visual representation
before deeper Llama processing
```

Pilot 预期单 sample 形状：

```text
[256, 4096]
```

O2 用于研究视觉 representation 经过 OpenVLA-specific multimodal adaptation 后，跨模型 shared structure 是否仍然存在。

Pilot v0.1 不要求提取更深的 Llama language-conditioned visual hidden states。

---

## 11. Primary Cross-Model Pairs

### DECISION

Pilot 主实验仍优先分析：

$$
O1\text{-}S \leftrightarrow P1
$$

和：

$$
O2 \leftrightarrow P2
$$

### O1-S ↔ P1

回答：

> 两个 VLA 中的 SigLIP-level visual representations 是否具有稳定的 shared structure？

该比较主要作为 shared-backbone-level baseline / control。

### O2 ↔ P2

回答：

> representation 经过各自 VLA-specific visual adaptation 后，shared structure 是否仍然存在？

### O1-F

O1-F 当前作为 OpenVLA-side diagnostic / supplementary node 保存。

它可以用于后续分析：

```text
O1-S vs O1-F
```

以判断加入 DINOv2 fusion 后，OpenVLA representation 的 cross-model correspondence 如何变化。

当前 Pilot 不预先冻结：

```text
O1-F ↔ P1
O1-F ↔ P2
```

为主结果。

这些组合可在 supplementary analysis 中探索，但不得替代冻结的主比较对，除非后续研究决策显式更新。

---

## 12. Feature Serialization

### DECISION

Extractor 必须保存完整 token representation。

禁止在 extraction 阶段只保存 mean-pooled vector。

例如：

```text
OpenVLA

O1-S: [256,1152]
O1-F: [256,2176]
O2:   [256,4096]
```

```text
π0

P1: [256,1152]
P2: [256,2048]
```

保存完整 token tensor 的目的是让同一 representation dataset 未来能够支持：

- global mean pooling；
- token-level analysis；
- object-region pooling；
- spatial shared-feature analysis。

---

### Multi-node Feature Record

一个 observation 对应一个模型侧 feature record。

对于 OpenVLA：

```text
{sample_id}.npz
```

单个 archive 同时保存：

```text
o1_siglip
o1_fused
o2_projected
metadata_json
```

即：

```text
one sample
    ↓
one OpenVLA feature artifact
    ├── O1-S
    ├── O1-F
    └── O2
```

不要求每个 node 单独保存为独立文件。

该设计的目的是保证同一 observation 上多个 representation nodes 的 provenance 和 identity 始终绑定。

---

## 13. Sample Alignment and Feature Provenance

### DECISION

所有 observation 和 representation 必须通过稳定的：

```text
sample_id
```

进行关联。

禁止依赖：

- 文件排序；
- 数组 index；
- extractor 输出顺序。

每个模型侧 feature record 至少包含：

```text
sample_id
source_model
checkpoint
feature_schema_version
source_image_hash
```

对于 OpenVLA 当前 C2 schema：

```text
source_model = "openvla"
feature_schema_version = "openvla_features_v1"
```

---

### Source Image Hash

`source_image_hash` 用于确认 OpenVLA 和 π0 representation 确实来自完全相同的原始视觉 observation。

Pilot v0.1 冻结：

```text
hash algorithm = SHA-256
```

hash source 为：

```text
PilotObservation.base_rgb_raw
```

在验证其为：

```text
shape = [H, W, 3]
dtype = uint8
```

后，对其 C-contiguous raw bytes 计算：

```python
hashlib.sha256(
    np.ascontiguousarray(base_rgb_raw).tobytes()
).hexdigest()
```

推荐 metadata 表示为：

```text
sha256:<hex_digest>
```

例如：

```text
sha256:0123abcd...
```

hash 必须针对 preprocessing 之前的原始 `base_rgb_raw` 计算。

不得对以下内容计算跨模型 pairing hash：

- OpenVLA resize 后图像；
- OpenVLA center-cropped 图像；
- processor normalized tensor；
- π0-specific preprocessed image。

原因是 Pilot 的 cross-model identity 定义在：

```text
same raw external observation
```

而不是：

```text
same model-specific input tensor
```

---

### Cross-model Alignment Check

在任何 OpenVLA ↔ π0 statistical analysis 前，必须验证：

```text
OpenVLA.sample_id == π0.sample_id
```

并且：

```text
OpenVLA.source_image_hash == π0.source_image_hash
```

如果任一条件不成立，analysis 必须失败，而不是继续计算。

---

### Node Metadata

由于一个 feature artifact 可以同时保存多个 representation nodes：

```text
o1_siglip
o1_fused
o2_projected
```

不再要求 feature record 使用单一：

```text
node_name
```

字段。

每个 node 的：

```text
shape
dtype
```

由对应 NumPy array 本身定义，并由 extractor 在写入和加载时进行显式验证。

当前 Pilot 不要求在 `metadata_json` 中重复保存每个 node 的 shape/dtype。

如果未来引入跨版本 schema migration，再考虑加入结构化 `nodes` metadata。