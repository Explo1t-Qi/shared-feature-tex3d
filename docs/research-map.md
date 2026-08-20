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