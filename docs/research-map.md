# Shared-Feature Tex3D Research Map

## 1. Research Goal

当前研究目标是提升 Tex3D adversarial 3D texture 在不同架构 Vision-Language-Action (VLA) 模型之间的迁移性。

原始 Tex3D 在 source model 上具有较强的攻击能力，但跨架构 VLA 的迁移效果仍有提升空间。本项目关注的问题不是进一步增强单一 source VLA 上的攻击强度，而是减少攻击对 source architecture 特有表征的过拟合。

---

## 2. Threat Model

本项目采用 **shared-feature discovery + single-surrogate attack** 的设置。

研究阶段允许使用多个架构不同的 VLA，例如 OpenVLA 与 $\pi_0$，分析不同模型之间是否存在稳定的共享视觉或视觉-策略表征。

在 shared representation 的定义和提取方式冻结之后，正式 adversarial texture optimization 只允许依赖一个 surrogate VLA。

形式上，shared representation discovery 可以写为：

$$
\mathcal{S}_{shared}
=
D(f_A, f_B, \ldots)
$$

其中 $D$ 表示跨模型表征分析过程。

正式攻击阶段则只使用 surrogate model $A$：

$$
\delta^\star
=
\arg\max_{\delta}
\mathcal{L}_{shared}^{A}(x,\delta)
$$

其他 VLA 不进入攻击 loss、梯度计算或 texture optimization，仅作为 unseen target model 测试迁移性。

本项目不采用：

$$
\mathcal{L}
=
\mathcal{L}_{OpenVLA}
+
\mathcal{L}_{\pi_0}
$$

这样的多模型 ensemble attack 作为主要方法。

---

## 3. Core Hypothesis

当前核心假设为：

> 不同架构 VLA 在视觉到动作决策过程中存在可识别的共享表征结构。相比攻击 surrogate model 的完整 feature space，若 adversarial texture 优先扰动这些跨模型共享且与策略行为相关的表征，则攻击更不容易过拟合单一模型架构，并可能获得更强的跨架构迁移性。

需要特别区分：

$$
\text{shared}
\neq
\text{transferable}
\neq
\text{policy-relevant}
$$

因此，发现跨模型相关 feature 只是第一步，并不能直接说明该 feature 适合作为 transferable attack objective。

---

## 4. Relation to UPA-RFAS

### FACT

UPA-RFAS 在论文 Section 3.2 中使用 CCA、线性关系分析等方式研究不同 VLA feature space 之间的关系，并提供了跨模型 representation 存在共享结构的实验依据。

### FACT

UPA-RFAS 的公开攻击训练代码没有显式：

1. 从多个 VLA 中学习 shared feature space；
2. 保存跨模型 CCA projection；
3. 将 surrogate feature 投影到显式 shared subspace；
4. 在该 shared subspace 中构造攻击 loss。

公开训练代码仍然只加载一个 surrogate OpenVLA，并在该模型自身的 `projector_features` 上计算 feature discrepancy、contrastive loss 等目标。

因此，UPA-RFAS 更接近：

$$
\text{cross-model representation evidence}
\rightarrow
\text{single-surrogate feature attack}
$$

而本项目计划研究：

$$
\text{cross-model representation evidence}
\rightarrow
\text{explicit transferable representation}
\rightarrow
\text{single-surrogate shared-feature attack}
$$

### OPEN

CCA 是否适合作为最终 shared representation 方法尚未确定。

目前只把 CCA 视为第一阶段用于验证 shared structure 是否存在的候选分析工具，而不是预先确定的方法核心。

---

## 5. Current Model Understanding

### 5.1 OpenVLA

OpenVLA inference 主链路已经完成初步代码追踪。

当前已确认的数据流包括：

```text
LIBERO observation
    ↓
RGB preprocessing
    ↓
processor(image, language)
    ↓
pixel_values + input_ids
    ↓
OpenVLA vision backbone
    ↓
multimodal projector
    ↓
Llama
    ↓
autoregressive action tokens
    ↓
continuous 7D action
```

#### FACT: Current LIBERO Default and Analysis Target

The current `Physical-Intelligence/openpi` default LIBERO deployment uses `pi05_libero`.

The current research target is the original $\pi_0$ architecture. Therefore, future experiments on $\pi_0$ must explicitly record and select the corresponding model config and checkpoint rather than relying on the default LIBERO deployment.

The following terms must not be treated as interchangeable:

- `openpi`: repository / implementation framework;
- $\pi_0$: original model architecture;
- $\pi_{0.5}$: later model architecture;
- `pi0_libero`, `pi05_libero`: specific LIBERO configurations.

Future experiment records should include at least:

- model type;
- config name;
- checkpoint;
- backend (JAX or PyTorch).

#### FACT: $\pi_0$ Visual Token Pipeline

For the original $\pi_0$, images are encoded independently inside `Pi0.embed_prefix()`:

```python
image_tokens, _ = self.PaliGemma.img(
    obs.images[name],
    train=False,
)
```

The returned `image_tokens` should be referred to more precisely as **visual token embeddings** rather than discrete visual tokens.

For the default $\pi_0$ configuration:

- image resolution: $224 \times 224$;
- SigLIP variant: `So400m/14`;
- patch size: $14 \times 14$;
- number of spatial tokens: $16 \times 16 = 256$;
- SigLIP encoder width: $1152$;
- PaliGemma / Gemma-2B width: $2048$.

The visual path is therefore:

```text
RGB image
    ↓
SigLIP patch embedding + Transformer
    ↓
raw SigLIP visual token embeddings
[B, 256, 1152]
    ↓
SigLIP output head / PaliGemma projection
1152 → 2048
    ↓
image_tokens
[B, 256, 2048]
    ↓
Gemma prefix processing
```

Therefore, at least two distinct visual-representation nodes must be distinguished:

1. raw SigLIP visual token embeddings;
2. PaliGemma-ready projected visual token embeddings.

The current `embed_prefix()` interface exposes the second representation directly.

#### FACT: Masked Padding Camera

LIBERO provides two real camera observations to $\pi_0$:

- `base_0_rgb`;
- `left_wrist_0_rgb`.

The third camera slot, `right_wrist_0_rgb`, is filled with a zero image for the original $\pi_0$ LIBERO input.

The zero image is still passed through the SigLIP image encoder and therefore still produces visual token embeddings. However, its `image_mask` is `False`, so these tokens are marked invalid in subsequent attention computation.

Therefore, representation collection must not treat the padded right-wrist tokens as valid visual observations.

---

## Current Research Gates

The immediate research path is:

```text
M1. Complete π0 visual-representation audit
        ↓
M2. Build OpenVLA ↔ π0 representation-node mapping
        ↓
M3. Audit the shared-SigLIP confounder
        ↓
M4. Implement read-only cross-model feature collection
        ↓
M5. Run held-out CCA / representation validation
        ↓
Gate: does stable cross-model shared structure exist?
        ↓
M6. Analyze transferability and policy relevance
        ↓
M7. Define single-surrogate shared-feature objective
        ↓
M8. Integrate with Tex3D
        ↓
M9. Cross-architecture transfer evaluation
```

The first go/no-go research gate is M5.

A shared-feature attack objective should not be implemented before held-out representation analysis provides evidence that stable cross-model shared structure exists.





## 12. Current Implementation Decision

### DECISION: Start Representation Feasibility Before Completing Full $\pi_0$ Audit

The current OpenVLA and $\pi_0$ code understanding is sufficient to begin the first read-only shared-representation feasibility experiment.

The first experiment does **not** require complete understanding of the $\pi_0$ Gemma, Action Expert, or flow-matching pipeline.

The currently identified visual representation nodes are sufficient to test the initial question:

> Do OpenVLA and $\pi_0$ contain stable cross-model shared representation structure for paired visual observations?

For $\pi_0$, the currently identified candidate nodes include:

- V1: SigLIP-native visual token embeddings, shape approximately $[B,256,1152]$ per camera;
- V2: PaliGemma-ready projected visual token embeddings, shape approximately $[B,256,2048]$ per camera.

Deeper representations such as language-conditioned Gemma hidden states and Action Expert representations remain part of the continuing model audit and will later be used to study policy relevance.

Therefore, model understanding and feasibility experiments will proceed in parallel.

---

### DECISION: Offline Paired Representation Extraction

OpenVLA and $\pi_0$ will not initially be loaded into the same Python runtime.

The two model stacks have substantially different dependency environments, including PyTorch/Transformers for OpenVLA and JAX/Flax/NNX for the current openpi implementation.

Instead, shared-representation discovery will use an offline paired-observation workflow.

A common observation dataset will be collected first. Each sample should preserve sufficient information to reproduce the relevant model input, including when available:

- sample identifier;
- task identifier;
- frame identifier;
- base RGB observation;
- wrist RGB observation;
- robot state;
- language instruction;
- preprocessing metadata.

The same observation samples will then be independently processed by model-specific extractors.

Conceptually:

    paired observations
          |
          +----> OpenVLA environment
          |          |
          |          v
          |     OpenVLA representations
          |
          +----> openpi environment
                     |
                     v
                pi0 representations

The representations are serialized to disk and analyzed separately.

The shared-space analysis stage therefore does not require both VLA models to coexist in the same software environment or GPU process.

---

### DECISION: Shared-Space Discovery Is Separate from Attack Optimization

During representation discovery, paired clean representations from multiple VLA models may be used.

Let the surrogate model representation be $Z_A(x)$ and another discovery model representation be $Z_B(x)$.

A shared-representation method may learn model-specific mappings:

$$
H_A = Z_A W_A
$$

$$
H_B = Z_B W_B
$$

where the mappings are selected such that $H_A$ and $H_B$ capture cross-model shared structure.

CCA is one candidate method for learning such mappings, but the final method is not yet fixed.

After discovery, the mapping associated with the surrogate model is frozen.

During the actual adversarial attack, only the surrogate VLA is used:

$$
x_{adv}
\rightarrow
Z_A(x_{adv})
\rightarrow
Z_A(x_{adv})W_A
\rightarrow
\mathcal{L}_{shared}
$$

The discovery model is not used to provide features, actions, losses, or gradients during texture optimization.

---

### DECISION: The Shared Objective Updates Texture Through the Surrogate Gradient

The shared representation is not directly optimized as an independent variable.

Let the adversarial texture parameters be $\theta$, and let the differentiable Tex3D rendering pipeline produce:

$$
x_{adv} = R(\theta)
$$

The surrogate model produces:

$$
Z_A(x_{adv})
$$

and the frozen shared-space mapping produces:

$$
H_A(x_{adv}) = Z_A(x_{adv})W_A
$$

A simple initial untargeted shared-feature objective can measure the deviation from the clean shared coordinates:

$$
\mathcal{L}_{shared}
=
\left\|
H_A(x_{adv}) - H_A(x_{clean})
\right\|^2
$$

The exact sign depends on whether the optimizer performs minimization or maximization.

The resulting gradient follows the chain:

$$
\frac{\partial \mathcal{L}_{shared}}{\partial \theta}
=
\frac{\partial \mathcal{L}_{shared}}{\partial H_A}
\frac{\partial H_A}{\partial Z_A}
\frac{\partial Z_A}{\partial x_{adv}}
\frac{\partial x_{adv}}{\partial \theta}
$$

Therefore the optimization still updates Tex3D texture parameters.

The shared representation determines which surrogate feature directions the attack objective rewards.

This differs fundamentally from the deprecated spectral approach:

- spectral approach: constrain the texture parameter space;
- current approach: select or weight transferable directions in the VLA representation space.

---

## 13. Parallel Workstreams

The project will now proceed through three coordinated workstreams.

### Workstream A: VLA Representation Audit

Continue the $\pi_0$ analysis beyond V2 to identify:

- language-conditioned Gemma visual hidden states;
- state/action-conditioned representations;
- Action Expert representations;
- candidate policy-relevant representation nodes.

In parallel, normalize the OpenVLA representation map into comparable nodes.

### Workstream B: Shared-Representation Feasibility

Implement a read-only pipeline containing:

1. paired observation collection;
2. OpenVLA representation extraction;
3. $\pi_0$ representation extraction;
4. serialized representation datasets;
5. train/held-out splits defined at the observation or trajectory level;
6. CCA / regression / CKA or related representation analysis;
7. held-out validation of discovered shared structure.

The initial feasibility stage should analyze at least both backbone-level and VLA-adapted representations where practical.

High correlation at a shared or closely related SigLIP backbone alone must not be treated as evidence of a shared VLA policy representation.

### Workstream C: Bug-Fixed Tex3D Baseline

A new Tex3D baseline should be derived from the official Tex3D implementation rather than directly using the previous modified Tex3D research branch.

Only confirmed general baseline fixes should be transplanted from the previous modified implementation.

Current confirmed categories include:

- differentiable rendering is used for attack training rather than formal policy evaluation;
- formal/live evaluation consumes real MuJoCo observations after attack texture activation;
- renderer position offset defaults to exactly zero unless measured otherwise;
- Clean Asset / Active Texture lifecycle is transactionally restored;
- OpenVLA visual preprocessing and branch ordering follow checkpoint semantics;
- renderer/background edge cases such as missing MVP are handled explicitly.

Deprecated spectral-specific modules and assumptions must not be transplanted into the new baseline unless independently justified.

The intended lineage is:

    official Tex3D
          |
          v
    bug-fixed baseline
          |
          v
    shared-feature method

The exact Git branch/import strategy will be decided before repository mutation.

---

## 14. Immediate Research Gate

The next implementation milestone is not yet a Tex3D shared-feature attack.

The immediate milestone is a read-only shared-representation feasibility pipeline.

Its success criterion is:

> A shared representation fitted on one set of paired observations must exhibit stable cross-model structure on held-out observations.

Only after this gate is passed should the project define and integrate a shared-feature attack objective into the bug-fixed Tex3D optimizer.

The deeper $\pi_0$ Gemma and Action Expert audit remains necessary for determining whether the discovered shared structure is also policy-relevant, but it does not block the first visual-representation feasibility experiment.