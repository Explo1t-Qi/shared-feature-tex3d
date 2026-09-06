# 实验 2：per-observation local gradient positive control

## 定位与当前状态

**DECISION（用户授权）**：本轮仅诊断当前 observation 的局部 action-z
可控性。每帧独立计算方向，不重新进行 action-relevant neuron 筛选，不发布
新的 frozen neuron 集合。没有 frozen shared direction、scope experiment、
新轨迹采集、rollout 或 texture optimization。

**FACT**：前序 `up-action-screen-75e783b764ef` 工程运行完成，但只有 1 个
候选达到旧规则，未形成冻结集合。实验 1 的逐帧 directional-D 复算也未找到
跨三个 trajectory groups 稳定的词汇候选。这尚不能区分候选空间、测量和
干预接口的问题。

当前实现及 CPU 工程检查见本文件末尾。真实 OpenVLA checkpoint 的实验 2
尚待服务器执行；CPU 合成模型结果不作为其科学结论。

## 数据、目标与边界帧

只使用原 task0、state0/1/2 的 **12 个 calibration/development observations**。
沿用 v1 / screen 的 checkpoint、样本、图像和文件哈希。原 state3/4
仅用于前序来源身份核对，不执行本轮 forward；不读取新独立验证轨迹。

对每帧，以 native clean generation 的 normalized z bin 为固定中心：

```text
D = logsumexp(z-position logits[action_value > clean_z])
  - logsumexp(z-position logits[action_value < clean_z])
```

沿用实验 1 的 action-token mapping，包括重复端点；掩码在该帧全部干预中固定。
读出使用 clean action prefix 下的 teacher-forced logits。D 用 float64
logsumexp 计算，模型仍为冻结参数、eval、BF16；只有增量变量需要梯度。

state00/step0031 与 state01/step0124 的 clean z 位于最低 bin，D undefined。
不补值、不构造梯度、不借用其他帧方向。这两帧仍执行 lexical 参照，保存
expected-z、argmax、native decoded action；D 相关字段为 null。

## 干预变量与三个条件

每个 FFN neuron 只有一个 FP32 变量，作用于 `down_proj` 的输入：

```text
a'[batch, token, i] = cast_to_native_dtype(a[batch, token, i].float() + delta[i])
```

同一 delta 在所有 token positions、prefill 和每次 decode forward 中复用。
没有 token-specific delta。梯度 `g_i = dD/d(delta_i)` 汇总所有 token 的贡献。
新 hook 不 detach activation，保留早期层增量经过后续层的完整导数。
每个方向还与原固定 offset hook 做 teacher-forced 数值一致性检查。

| 条件 | 当前帧方向 | 非零 neuron 数量 |
| --- | --- | --- |
| lexical | 原词汇排序 Top-10，各自 positive clean population std | 10 |
| candidate_gradient | 在原 232 个候选内构造当前帧稀疏上升方向 | 10 |
| broad_gradient | 同一批 FFN layers 的完整 neuron pool 内构造当前帧稀疏上升方向 | 10 |

**FACT**：232 个候选覆盖全部 32 个 FFN layers。因此 broader 可选池在本模型
中覆盖全部 32 层；没有增加层数。为了进一步控制实际干预的层分布，broader
probe 匹配该帧 candidate probe 的逐层非零数量。这是一个保守的 matched-layer
比较，不是允许 broader 自由跨层分配 10 个位置的最优控制实验。

两种 gradient 组包含相同局部目标的正控制，而不是语义证明。broader pool
包含 lexical candidates，不强制排除重叠。支持位置逐帧保存以便审计，不跨帧
排名、不输出 `frozen.json`，不可直接作为后续固定集合使用。

如果 candidate 梯度中不足 10 个非零值，该 probe 记录 unavailable；broader
的匹配层分配也随之 unavailable。这种缺失不计为 broader control 失败。

## 稀疏化、归一化与步长

令 `W_l` 为第 l 层 down projection，`c_i = ||W_l[:,i]||_2`。

1. 在允许池中按 `|g_i| / c_i` 排序；candidate 取 10 个，broader 按上述
   逐层数量取相应位置。相同值按 layer/index 决定顺序。
2. 支持内系数为 `v_i = g_i / c_i²`，其余为零。这是 value-vector 范数的
   对角预条件化；不声称求解了包含列间相关性的最优稀疏方向。
3. 定义理想每-token residual 预算 `B(v) = sqrt(sum_l ||W_l v_l||²)`。
   原 lexical `std` 向量的预算为 `B0`。两个 gradient 方向分别精确缩放到
   `B0`，不是分别缩放各个 neuron。实现先做共同幅值缩放以避免 FP32 下溢，
   再做该组合范数归一化，检查保留恰好 10 个非零值。
4. 三组统一执行 `delta = sign * eta * v`，`eta = 0.05, 0.1`，
   `sign = +1, -1`。不依据结果自动扩大步长或切换 scope。

lexical 参照保留原方向与 std 定义，但使用上述小步而非重跑 v1 的 alpha=1/2。
因此其结果不能被描述为对旧大步实验的直接复现。

理想一阶预测为 `Delta D_pred = sign * eta * sum_i(g_i v_i)`。
记录实际 `Delta D`、预测误差及比值。BF16 舍入使小步不一定接近光滑函数；
不能仅根据 autograd 的正内积认定实测有效。

## 实际 residual norm 与动作读出

对每次 hooked down projection，用同一上游输入额外计算无增量的原生
`F.linear` 输出，与实际 hooked 输出相减。每层按实际 token rows 取平方均值，
然后跨层求和开方。它包括 native dtype 的增量/输出舍入，度量局部直接注入的
FFN residual shift，不包括前层干预传播到本层输入的变化。

分别记录 teacher-forced 与 native generation 的该范数、逐层平方和、token
数量、hook 调用次数、实际改变的 activation 数量，以及实际/理想范数比。
理论预算一致不保证 BF16 下实际预算完全一致；读结果时必须同时核对。

每帧每个条件/步长/sign 保存：

- Delta D、预测值、误差与比值；Delta expected normalized z。
- teacher-forced z-position argmax token、是否改变、对应 action value 及 bin
  value 差；非 action token 时不伪造 bin。差值以 clean teacher-forced argmax
  为参照，D 的中心仍固定为 clean native z。
- native generation 全部 7 个 token、normalized / unnormalized / deployed
  action，以及 deployed z 和其他 6 维动作差。
- 理想与实际 residual norm；原接口等价、zero hook、clean continuation 和
  移除 hook 后恢复 clean 的检查。

每帧保存 JSON 与 NPZ，后者包括全部 teacher-forced logits、当前帧梯度和
局部 offsets。若全部 probe 可构造，10 个内部帧产生 120 条干预记录，2 个
边界帧产生 8 条 lexical 记录，共 128 条。

summary 按同一帧的正负小步统计双向响应，同时显示每帧结果；没有跨帧汇总
方向或筛选分数。计数用 `1e-8` 数值零容差，不是科学显著性阈值或通过门槛。

## 解释边界

- candidate local gradient 有效：232 维空间在当前 observation 下包含局部
  action-z controllable direction；不证明 up 语义或跨观测 shared direction。
- 只有 matched broader 有效：支持 lexical pre-filter 可能遗漏主要 action-z
  方向；先确认实际 norm 和一阶预测，避免误把尺度差异归因于候选来源。
- 两者都不稳定：优先检查梯度实现、BF16 舍入、小步尺度和 all-token scope。
  不直接判断模型不可控；本轮不会自动启动 scope experiment。
- D 改变、expected-z 改变与 native decoded z 改变分别报告。native generation
  的前缀可能随前两维变化，不能将 clean-prefix D 的导数当成 native z 的导数。
- 这些方向直接针对 D 构造。positive control 的目标是检查实现和有限步响应，
  其成功不是独立的语义验证，更不是跨帧稳定性或迁移性证据。

## 服务器执行与交付

入口：`scripts/up_local_gradient_server_run.sh FULL_SHA --preflight-only|--run`。
沿用已成功环境及 checkpoint，不改变依赖；`GPU_ID` 必须显式指定。
默认 v1 / screen 目录分别是服务器日志根目录下
`up-concept-v1-b8a330f7d27b` / `up-action-screen-75e783b764ef`。
可用 `UP_V1_DIR`、`UP_SCREEN_DIR`、`COLLECTION_MANIFEST` 覆盖真实路径。

```bash
DIAG_SHA=$(git rev-parse HEAD)  # 确认已同步到本次提交
bash scripts/up_local_gradient_server_run.sh "$DIAG_SHA" --preflight-only
GPU_ID=7 bash scripts/up_local_gradient_server_run.sh "$DIAG_SHA" --run
```

这里的 7 是前序设备示例，运行时使用实际空闲 GPU。preflight 不加载 checkpoint
或创建结果目录；run 先执行 CPU 测试，再执行 GPU 诊断。拒绝覆盖已有输出。
失败保留 console log 与 `failure.json`，不自动改模型、精度、scope 或步长重试。
需要真实服务器验证的部分是 BF16 autograd 路径、显存及 native generation 响应。

默认输出 `up-local-gradient-<SHA前12位>`。将同名 `.review.tar.gz` 和 console
log 同步回本地即可先审阅所有 JSON 读出；NPZ 保留服务器供需要时独立复算。
`COMPLETE` 只表示工程执行完成，科学状态固定为 `LOCAL_DIAGNOSTIC_ONLY`。

## 本机工程诊断

新增测试覆盖跨 token/调用的广播导数求和、跨层梯度传播、BF16 实际舍入测量、
固定 D 掩码、inference tensor clone、有限差分、匹配稀疏度与范数、完整单帧
三组干预、边界跳过、unavailable 状态和失败留证。合成模型使用冻结参数的两层
causal FFN，并调用本次实际 runner、原 logit builder 和原固定 offset hook。
其规模与生成实现不能代替真实 OpenVLA 验证。

CPU 数值诊断保存于
`experiment_inbox/up-concept/up-local-gradient-cpu-diagnostic/`（本地实验产物，不入 Git）。
新增 10 项测试及相关 38 项回归测试共 **48 passed**（Python 3.10.12，
torch 2.2.0+cu121，CUDA 不可见）。合成模型在 eta=0.05 下的中心有限差分
斜率相对 autograd 预测误差分别为 candidate `5.773e-6`、broader `5.754e-6`。
两组均产生双向 D 响应；该合成模型的 native decoded z 没有变化。边界帧仅有
lexical 记录且 D 为 null。另一个 BF16 测试确认理论非零小步可被舍入完全抹掉。
这些是工程验证，本次不生成真实 checkpoint 的科学判定。
