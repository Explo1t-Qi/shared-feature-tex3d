# UP action-relevant pilot v1

用户已同意从原 UP 候选中增加动作相关筛选，并授权本轮实现。
范围为单模型冻结观测干预；不采集新轨迹、不运行 rollout、不优化纹理。
新轨迹是否已有、其来源路径仍待用户提供，筛选阶段不依赖此信息。

## 前序证据与问题

UP v1 在 `b8a330f7d27bb3dd13828f9b283c601e636410e8` 真实模型运行完成。
10 个词汇候选在 alpha 1/2 均未改变 decoded action tokens；已有 logits 的
第三位置 action 条件概率期望相对随机均值存在约 1e-4 的正差，主要由少数帧贡献。
这支持继续检查候选，不建立稳定 upward action 或迁移性结论。

本轮问题：同一词汇候选池内，正负干预的动作响应筛选能否得到更稳定的组合？
结论限于“具有 up 词汇关联且对 z 输出有作用的候选”，不宣称发现通用 up 语义。

## 筛选规则

- 候选直接读取已完成 v1 的 `candidates.json`（该次共 232 个），不重新做投影。
- 仅用原 collection task 0 的前三条轨迹（states 0/1/2，每条四帧）。
  逐个 NPZ hash 与 v1 协议比对；旧 states 3/4 不参与此次筛选，保留为开发数据。
- 对全部候选采集 clean 原生 generation activation population std。
  对每个候选分别施加 `+std` 和 `-std`，作用于 down_proj 输入的全部 forward tokens。
  零 std 候选明确排除并保存；不添加尺度下限，不根据响应翻转候选符号。
- 筛选读出是相同 clean action prefix 下第三个位置的条件期望归一化 z。
  action IDs 和 bin values 从实际 `model.vocab_size/bins/bin_centers` 构造，
  并对 clean 七维解码逐一验证。保留全词表中 action 概率质量作为解释依据。
- 每条轨迹分别取四帧 `E_z(+) - E_z(clean)` 与 `E_z(clean) - E_z(-)` 的中位数。
  对三条轨迹、两个方向共六个中位数取最小值作为分数。
- 分数必须大于 `1e-8`（数值阈值，不是显著性阈值）；按分数降序取 10 个。
  并列按 layer/index 升序。少于 10 个则保存 `INSUFFICIENT_CANDIDATES`，不发布冻结集合。
  此规则在运行前固定，不自动降低标准。
- 逐候选检查移除 hook 后 clean-prefix logits 精确恢复；逐观测检查原生动作恢复。
  保存每帧全部候选正负干预的第三位置完整词表 logits 和 clean 七位置 logits。

这不是原 steering 论文完整复现。当前使用 std 加减，不使用作者常数覆盖。

## 冻结集合与组合验证

筛选足够十个后，冻结：原词汇 Top-10、action-selected Top-10，及两者各自的
三个随机对照集合（seed 7；逐层数量匹配；排除全部词汇合格候选；随机组之间可重叠）。
随机对照不接受 action-relevant 筛选，本轮不能单独证明语义先验比随机候选池更有价值。

组合 offsets 同样仅由原校准集确定。原词汇集合保持 `std`；新集合统一缩放至
与原词汇集合相同的理想总 FFN 输出变化范数（各层平方范数求和后开方）。
每个随机集合再逐层匹配其对应目标集合的理想输出变化范数。
保存缩放后的 offsets 与实测 hook 变化；不宣称不同层分布或 downstream sensitivity 相同。
原词汇集合可复现 v1 定义的新运行，但不复用 v1 的输出代替独立验证。

验证命令只读取 `frozen.json`，不重新校准或排名。检查冻结 SHA、同一项目 commit、
同一 checkpoint 内容 hash 与 v1 输入身份。固定 alpha 1/2，八个集合。
新验证至少两条轨迹，每条四帧；两条时共 128 个条件和 136 个完整 logits NPZ。
记录七维 decoded action、delta translation、token Hamming、gripper、概率期望、
原生/continuation 等价与 hook 恢复。按轨迹等权汇总，再计算目标减匹配随机均值。
全部结果只给描述性统计，不产生 scientific PASS。

## 独立验证数据接口

沿用 `PilotObservation` NPZ 格式、成功 clean OpenVLA Spatial task 0 原始 512×512 图像。
拒绝旧 states 0–4（实际按 v1 全部 task0 身份排除）、旧 sample/image、重复帧，
并要求每组四帧来自同一 episode。state ID 必须是原始 LIBERO initial-state 索引，不能重编号。
代码不决定/执行采集；采集帧选择和来源应先确定，不按本轮干预效果挑选。

验证 manifest JSON（相对路径相对于 manifest 目录）：

```json
{
  "schema": "up_action_validation_observations_v1",
  "provenance": {
    "checkpoint_identity": "openvla/openvla-7b-finetuned-libero-spatial",
    "task_suite": "libero_spatial",
    "task_id": 0,
    "collection_description": "填写真实采集配置、代码版本、帧选择规则及来源",
    "not_used_for_selection": true
  },
  "observations": [
    {"path": "observations/实际文件名.npz", "sha256": "sha256:实际64位hash"}
  ]
}
```

示例只展示一项，实际需所有新轨迹帧。身份验证不能证明未查看过数据，独立性仍依赖实验纪律。

## 服务器入口

默认 Python/checkpoint/Tex3D/collection 路径沿用成功 v1；GPU 没有默认值。
`UP_V1_DIR` 默认 `/data/xiaomengqi/logs/up-concept/up-concept-v1-b8a330f7d27b`。
如果服务器成功产物已搬动，只需覆盖此变量。
preflight 现在会真实导入完整 runtime/preprocessing（无 GPU、无模型、无新输出），
以尽早发现前两轮遇到的 import 问题。

```bash
cd /data/xiaomengqi/src/shared-feature-tex3d
EXPECTED_HEAD=填写本轮交付的完整commit
bash scripts/up_action_server_run.sh "$EXPECTED_HEAD" screen --preflight-only
GPU_ID=填写可用编号 bash scripts/up_action_server_run.sh "$EXPECTED_HEAD" screen --run
```

筛选状态 `FROZEN` 且新数据准备好后，验证命令：

```bash
export UP_SELECTION_DIR=/data/xiaomengqi/logs/up-concept/up-action-screen-${EXPECTED_HEAD:0:12}
export UP_VALIDATION_MANIFEST=/实际路径/validation.json
bash scripts/up_action_server_run.sh "$EXPECTED_HEAD" validate --preflight-only
GPU_ID=填写可用编号 bash scripts/up_action_server_run.sh "$EXPECTED_HEAD" validate --run
```

每阶段默认输出 `up-action-<screen或validate>-<commit前12位>`，拒绝覆盖/自动恢复。
科学负结果也是完成状态，需查看 `selection_status`；不会自动进入验证。
每次完成会生成 `.review.tar.gz`（JSON 小型证据），原始 NPZ 留服务器。
`UP_OUTPUT_ROOT/UP_RUN_ID/OPENVLA_PY/TEX3D_ROOT/OPENVLA_CKPT/COLLECTION_MANIFEST`
均可用环境变量覆盖。不自动安装依赖、同步 Git、SSH 或选择 GPU。

232 候选×12 帧×2 符号 = 5568 次干预 forward；每候选还做一次移除 hook 的
clean-prefix 恢复检查（2784 次）。实际耗时待服务器测量，命令输出逐帧进度。

## 验证状态

本机 CPU 验证完成：新增 11 项测试连同 UP/C6 相关回归共 **46 passed**。
覆盖 token 映射与端点、单帧异常值、双向与跨组评分、零信号停止、hook 异常清理、
冻结 artifact 篡改、旧轨迹/重复帧/文件 hash 拒绝，以及验证阶段不允许重新拟合。
合成模型完整验证 128 个组合条件和 136 个 logits NPZ；Bash 路径空格、GPU 显式指定、
完整 commit gate 与 preflight 不创建输出目录均通过。CLI help、Bash 语法检查通过。
真实 checkpoint 的本轮筛选和独立验证尚未运行。
