# UP 单概念干预 pilot v1

状态：v1 真实 checkpoint 运行已完成（`b8a330f7d27b`），UP decoded action 未变化。
后续已授权 [action-relevant 筛选扩展](up-action-relevant-pilot.md)。
下文原始协议及当时的验证记录保留历史语义。

## 授权与范围

用户在 2026-09-05 的讨论中同意单个 `up` 概念、小样本、少量随机对照与两个
clean 尺度校准的非零强度，并授权开始编码；2026-09-06 要求继续完成。
本合同仅覆盖 OpenVLA Spatial 的候选提取与冻结 observation 干预筛查。
历史 `task.md` 是已完成的 C6 合同，不用于覆盖本轮授权。

不运行 LIBERO rollout，不优化纹理，不使用 π0.5，不重写任何 C5/C6 历史产物。
研究问题：语义投影为 `up` 的 FFN neurons 增强后，是否比匹配随机集合更稳定地
提高 decoded translation 的 z 分量？这只是后续 rollout/texture 的候选证据。

## v1 执行参数

以下是把本轮讨论落实为可审阅实现的具体参数，不声称它们经过调参或最优：

- checkpoint：`openvla/openvla-7b-finetuned-libero-spatial`，BF16、无量化、eval。
- 原生 OpenVLA generation，`do_sample=False`；复用 C6 输入准备与解码代码。
- 复用完整 Pilot v0.2 collection 的 task 0：五个 trajectory groups，每组四帧。
- 按原始 state ID 排序，前三组的 12 帧仅用于 clean calibration；后两组的
  8 帧用于干预比较。当前产物对应 states `0,1,2` 和 `3,4`。
- 这是新的 pilot 内部分工，不改变历史 C5 split；这些数据已经用于开发，
  不声称是从未接触的正式 test，也不声称覆盖全部成功轨迹分布。
- 保留实际 episode progress；四个进度分位不解释为已标注的操作阶段。
- 一个 `up` 集合，`K=10`；三个随机集合；PCG64 seed `7`。
- 两个非零强度 `alpha=1,2`，候选和强度不根据干预输出重新选择。

## 候选提取

参考 Häon et al., *Mechanistic Interpretability for Steering Vision-Language-Action
Models* (CoRL 2025)，https://arxiv.org/abs/2509.00328 。
作者参考仓库：`Physical-AI-Safety-Institute/mechanistic-steering-vlas`，本地读取的
commit 为 `559c0f25a3cc5a20fc8b804774a88415404e0d22`。
本轮独立实现词汇投影和 hook，不复制作者的原地修改实现。

对每层 `down_proj.weight` 的列（value vectors）与 `lm_head.weight` 做 FP32
词汇投影，禁用 TF32。只接受单 token 解码后 strip/casefold 恰为 `up` 的变体，
不匹配 `upcoming`。候选的 top-10 投影 token 中必须包含这样的词。
按完整词表 softmax 中所有 `up` 变体的概率质量降序选择前十；并列时依次用
best-up-rank、层号、neuron index 决定。词表 logit 并列用 token ID 升序。

所有合格候选与 top tokens 都保存；少于十个则停止，保留候选结果，不扩词或换层。
排名不使用 observation/action；这是词汇启发式，不是行为标签或 vulnerability 证明。
小批投影避免生成完整的全神经元×词表存档。`projection-batch-size` 只影响资源使用。

## 干预与对照

对象为原生 `down_proj` 输入 `[batch, token_positions, neurons]`，不是 FFN 输出
维度，也不是 O2。作用于全部 forward token：prefill 和 autoregressive decode。
第一版不筛选特定层或 token；没有手工 phase-conditioned intervention。

Clean calibration 在原生 generation 的每次 hook 调用上收集所选 up/random neurons
的 activation。prefill 和后续 decode 调用中的每个 token row 等权，计算 population
mean/std；这是 position-level 尺度统计，不把 token 当作独立行为统计样本。
不会把 eval observations 的 activation 纳入尺度估计。

UP 干预为 `a' = a + alpha * std_clean`；与作者的 `a'=constant` 不同。
std 为零或非有限时停止，不偷偷添加下限或改换候选。输入 clone 后再修改，原生
dtype 舍入后的 actual delta 和产生的 FFN residual-shift L2 单独记录。

随机集合匹配每层 neuron 数量，排除全部 up 合格候选；每个集合内部不重复，
集合之间允许重叠。随机方向先取各 neuron 自身的 positive clean std，再逐层
缩放，使其理想 `||W_down * delta||_2` 与 UP 的同层理想变化一致。
这样控制 value-vector 范数和层分布混杂；不声称 downstream sensitivity 一致。
BF16 舍入后的实际变化不一定严格等范数，以保存的测量为准。

每个 eval observation 验证：

1. 原始 generation 与 clean O2 continuation 的 token/动作等价。
2. 只记录 hook、zero-offset hook 不改变原始动作。
3. 每个条件干预后，移除 hook 的动作恢复为 clean。
4. hook 正常/异常退出均移除；命中次数、实际变化、有限值均可追踪。

任一工程等价性失败就停止该次运行。反之，没有动作变化可以是合法的科学负结果。
全部四集合×两强度×八 eval observations，共 64 个配对条件。

## 测量与解释

主读出：原生 greedy 解码的七维动作，尤其 `deployed_action[:3]` 的有符号差。
同时记录 gripper 变化、action-token Hamming、相同 clean action prefix 下的
七个 action-position logits，以及 clean repeatability/top-2 margin。
teacher-forced logits 是边界诊断，不代替原生 autoregressive action。

主统计先对同一 trajectory 的四帧取均值，再对两个 evaluation groups 等权汇总。
同时保留逐帧、进度和各随机集合结果。两个 groups 不支持正式显著性或普适性结论，
不自动生成 scientific PASS。`1e-8` 只用于数值上是否为正的计数。

脚本将第三个平移分量称为 **decoded action z**，不称为实际位移（米）、速度或
世界坐标上移已被证明。将其解释为 world-up 需要确认服务器部署的 OSC_POSE
平移坐标、缩放和控制约定；本实验不实例化控制器，也不测量真实末端位移。

后续 review 只做一次决策：有稳定方向且优于随机对照则讨论小规模 rollout；
仅 logits 变化则分析 token boundary；没有定向优势则保留负结果并讨论候选。
不得根据本 pilot 自动执行更多强度、更多概念、rollout 或 texture optimization。

## 文件与运行

- `shared_feature/up_concept.py`：词汇候选、随机集合、clean moments、FFN hooks、组级摘要。
- `scripts/up_concept_pilot.py`：身份验证、模型编排、校准与配对证据。
- `scripts/up_concept_server_run.sh`：无 GPU preflight、指定 GPU 运行、小型结果包。
- `tests/test_up_concept*.py`：数学/干预语义和合成全链路、失败路径、Bash 入口。

本机 CPU 验证：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_up_concept.py tests/test_up_concept_runner.py
```

服务器运行前需要用户确认 Python、仓库、collection、checkpoint 和 GPU。脚本的
默认路径来自既有交接记录，均可用环境变量覆盖；preflight 不加载模型、不创建
输出目录，也不证明依赖完整或 GPU 可用。GPU_ID 没有默认值。

```bash
cd /data/xiaomengqi/src/shared-feature-tex3d
# EXPECTED_HEAD 必须填写本次交付的完整 SHA，不使用任意当前 HEAD 代替。
EXPECTED_HEAD=REPLACE_WITH_DELIVERED_40_CHARACTER_SHA
bash scripts/up_concept_server_run.sh "$EXPECTED_HEAD" --preflight-only

# 确认一个可用的物理 GPU 后，例如把 N 替换为真实编号：
GPU_ID=N bash scripts/up_concept_server_run.sh "$EXPECTED_HEAD" --run
```

可覆盖的变量：`OPENVLA_PY`、`TEX3D_ROOT`、`OPENVLA_CKPT`、`COLLECTION_MANIFEST`、
`UP_OUTPUT_ROOT`、`UP_RUN_ID`。Python 版本 gate 要求 torch `2.2.0+cu121`、
torchvision `0.17.0+cu121`、transformers `4.40.1`、tokenizers `0.19.1`。
不自动安装或改动依赖，不修改 checkpoint/XML/纹理，不自动 push/pull/SSH。

默认输出：`/data/xiaomengqi/logs/up-concept/up-concept-v1-<commit前12位>/`。
成功产生 `results.json`（engineering COMPLETE、scientific DESCRIPTIVE_ONLY）、
候选、校准、64 条配对结果、CSV/Markdown 摘要，以及单独的完整 logits NPZ。
失败保留 `failure.json`、已有阶段产物与 console；已有目录拒绝覆盖/恢复。
不要在失败后自动改 run ID 重跑，应先检查失败原因。

Bash 成功后创建同名 `.review.tar.gz`，包含小型审阅证据；完整 logits 留在服务器。
提交只包含代码/文档/测试，不包含 observation、权重、activation 或实验输出。

## 待服务器确认

- 物理 GPU 编号和显存是否足够加载 BF16 OpenVLA 并运行 FFN→vocabulary projection。
- source Python、两个仓库、collection manifest、checkpoint 和输出路径。
- 实际 OSC_POSE 对 `action[2]` 的 world-up 解释；在确认前只报告 action z。

本阶段不宣称真实 checkpoint clean-equivalence、UP 行为作用或跨模型迁移已经通过。

## 本地验证记录（2026-09-06）

- Python 3.10 / torch `2.2.0+cu121`，CUDA 禁用；新增测试与相关 C6/数据接口回归
  共 **47 passed**，其中新增 pilot 测试 21 个。
- 合成全链路覆盖 64 个配对条件、72 个 logits 文件、hook 恢复、候选不足、
  中途失败，以及摘要写入失败时不发布 `COMPLETE`。
- 实际本地 200-record collection 验证后，task 0 分组为 calibration 12 帧
  （states 0/1/2）与 evaluation 8 帧（states 3/4）。
- Bash 语法检查、CLI help 和 diff whitespace 检查通过。
- 较早的全仓回归为 **293 passed / 1 failed**（之后增加的两项测试已包含在上述
  47 项中）。唯一失败位于历史 `pilot_v02_collector.py` 异常路径使用
  Python 3.11 的 `Exception.add_note`；在隔离的基准 commit
  `f0b7b644862fc002b1992c5c4618bf8f2c7e5177` 上复现同一失败。
  本次未修改该采集模块；当前 pilot 读取既有 observation，不调用采集器。
- 尚未执行服务器 preflight、真实 checkpoint 干预或 GPU 验证。
- 服务器首次运行暴露的包引用问题已定位并修复：Tex3D 的
  `openvla_utils.py` 使用 top-level sibling import，pilot 现在会在 preflight
  和 runtime 加载前加入并检查 `openvla/experiments/robot`。
- 后续服务器运行发现 Tex3D `1aab9b0` 已将 `crop_and_resize` 从
  `openvla_utils.py` 移至 `openvla_policy_view.py`；共享预处理加载器现兼容旧函数
  与新目录布局，未修改 checkpoint 或 Tex3D checkout。
