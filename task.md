# C6 Real Clean-Equivalence / Intervention-Smoke Runner Contract

## 1. Current Status

```text
C6 intervention-interface contract: FROZEN
C6 interface implementation: UNIT-LEVEL PASS
C6 unit validation: PASS

C6 real-smoke protocol: REVISED / PENDING FINAL READ-ONLY AUDIT
C6 real-smoke runner implementation: AUTHORIZED
C6 real-smoke runner unit validation: AUTHORIZED
C6 real checkpoint execution: NOT AUTHORIZED

C6-B policy-sensitivity analysis: NOT AUTHORIZED
Tex3D optimization: NOT AUTHORIZED
```

本阶段只授权：

```text
1. 实现真实 smoke runner
2. 实现结果汇总工具
3. CPU-only / static / mock-based runner tests
4. 做 runner code audit
```

本阶段**不授权真实模型推理**。

------

# 2. Execution Architecture

由于 OpenVLA 与 π0.5 当前依赖不同运行环境，本 smoke 不要求合并环境。

冻结为三个独立入口：

```text
OpenVLA real-smoke runner
π0.5 real-smoke runner
result aggregator
```

建议文件：

```text
scripts/c6_openvla_real_smoke.py
scripts/c6_pi05_real_smoke.py
scripts/c6_real_smoke_summary.py
```

必要时可增加少量 repository-local helper 或 tests。

禁止为了统一运行环境修改：

```text
../openvla/**
../openpi/**
```

------

# 3. Frozen Observations

使用以下两个固定 Pilot v0.2 observations：

```text
libero_spatial__task00__state00__step0008
libero_spatial__task01__state00__step0011
```

对应正式 observation 文件：

```text
/data/xiaomengqi/src/shared-feature-tex3d/experiment_inbox/
c5_d0_pilot_v02_full_collection/observations/
libero_spatial__task00__state00__step0008.npz

/data/xiaomengqi/src/shared-feature-tex3d/experiment_inbox/
c5_d0_pilot_v02_full_collection/observations/
libero_spatial__task01__state00__step0011.npz
```

原始 observation manifest：

```text
/data/xiaomengqi/src/shared-feature-tex3d/experiment_inbox/
c5_d0_pilot_v02_full_collection/collection_manifest.json
```

对应冻结 image identity：

```text
task00:
sha256:279786daa449ca71c3e6aa2c8d6941c37814a9286d2dd2fefd53c3123390b879

task01:
sha256:b8c55f535389ba9e7ff7a1e106ed2dea696282b62b4202694857372c634f4f4a
```

runner 必须在执行前验证 observation identity。

禁止因为 smoke 结果不理想而替换 observation。

------

# 4. Model Identity

## 4.1 OpenVLA

Scientific identity：

```text
openvla/openvla-7b-finetuned-libero-spatial
```

服务器 checkpoint：

```text
/data/huangsimin/openvla-7b-finetuned-libero-spatial
```

冻结：

```text
B = 1
unnorm_key = libero_spatial_no_noops
center_crop = True
do_sample = False
```

runner 必须显式记录：

```text
scientific checkpoint identity
resolved checkpoint path
git commit
runtime/library versions
```

不能只依赖 `PreparedOpenVLAContext` 中记录的字符串判断真实 checkpoint 来源。

------

## 4.2 π0.5

Scientific identity：

```text
config:
pi05_libero

checkpoint:
gs://openpi-assets/checkpoints/pi05_libero

backend:
JAX / NNX
```

服务器 resolved checkpoint：

```text
/data/xiaomengqi/checkpoints/pi05_libero/
openpi-assets/checkpoints/pi05_libero
```

冻结：

```text
B = 1
P2 = [1,256,2048]
noise = [1,10,32]
```

runner 必须显式记录并检查：

```text
config identity
scientific checkpoint identity
resolved checkpoint path
backend
git commit
JAX/runtime versions
```

------

# 5. Runner Input Reconstruction

runner 必须从正式 `PilotObservation` NPZ 和 collection manifest 重建对应模型的真实 policy input。

不得使用 C4 feature archive 代替原始 observation。

## OpenVLA

必须复用冻结的 C1/OpenVLA policy-input semantics。

## π0.5

必须使用已经验证的 LIBERO client-side preprocessing 构造原始 `Policy.infer()` inference dict。

之后由 intervention API 执行：

```text
policy input transforms
→ Observation.from_dict
→ preprocess_observation(..., train=False)
→ image encoder
```

禁止 runner 自行创造新的 preprocessing semantics。

------

# 6. π0.5 Fixed Noise RNG

冻结 RNG implementation：

```python
rng = np.random.Generator(
    np.random.PCG64(2026)
)
```

生成：

```text
distribution:
standard normal N(0,1)

shape:
[1,10,32]

generation dtype:
float32
```

生成一次后同时供：

```text
reference
clean continuation
modified continuation
```

使用。

禁止：

```text
implicit resampling
reference 和 continuation 分别采样
根据结果重新生成 noise
```

------

# 7. Intervention Direction RNG

为避免依赖程序调用顺序，每个 `(model, observation)` 使用独立固定 seed。

冻结：

```text
OpenVLA task00: 202700
OpenVLA task01: 202701

π0.5 task00:   202710
π0.5 task01:   202711
```

每个 direction 使用：

```python
rng = np.random.Generator(
    np.random.PCG64(seed)
)

direction = rng.standard_normal(
    feature.shape,
    dtype=np.float32,
)
```

然后对整个单样本 tensor 做全局 L2 normalization：

```python
direction /= np.linalg.norm(direction)
```

不得依赖共享 RNG 连续 draw 的调用顺序决定 direction identity。

------

# 8. Synthetic Perturbation

本阶段不使用 CCA direction。

冻结 engineering perturbation：

```text
alpha = 1e-3
```

定义：

```python
intended_delta =
    alpha * ||clean_feature||_2 * direction

intended_modified =
    clean_feature + intended_delta
```

这是 smoke-only engineering scale，不是 C6-B scientific epsilon。

------

# 9. Native-Dtype Applied Perturbation

如果 intervention API 将 override 转换为模型 native dtype/device，例如 BF16，则：

```text
真正用于 smoke 指标和 PASS 判断的 perturbation
必须基于模型实际消费后的 feature。
```

定义：

```text
actual_modified =
    actual feature consumed by continuation

actual_delta =
    actual_modified - clean_native_feature
```

必须记录：

```text
||clean_native_feature||
||intended_delta||
||actual_delta||

intended relative perturbation
actual relative perturbation
```

其中 PASS/smoke metric 使用：

```text
actual_delta
```

而不是仅使用转换前 float32 `intended_delta`。

不得原地修改 prepared clean feature。

------

# 10. Correct API Call Order

## OpenVLA

正确调用顺序：

```python
prepared = prepare_openvla_context(...)

reference = run_openvla_reference(
    prepared=prepared,
)

continued = continue_openvla_from_o2(
    prepared=prepared,
    o2=prepared.o2,
)
```

随后才构造 modified O2 并执行 intervention continuation。

------

## π0.5

正确调用顺序：

```python
prepared = prepare_pi05_context(
    ...,
    noise=fixed_noise,
)

reference = run_pi05_reference(
    prepared=prepared,
)

continued = continue_pi05_from_p2(
    prepared=prepared,
    base_p2=prepared.base_p2,
)
```

随后才构造 modified base P2。

------

# 11. Clean-Equivalence Criteria

## OpenVLA

两个 observations 均要求：

```text
action_token_ids:
exact match

normalized_action:
rtol = 0
atol = 1e-8

unnormalized_action:
rtol = 0
atol = 1e-8

deployed_action:
rtol = 0
atol = 1e-8

discrete gripper:
exact match
```

------

## π0.5

两个 observations 均要求：

```text
normalized_action_chunk_32:
rtol = 0
atol = 1e-6

normalized_action_chunk:
rtol = 0
atol = 1e-6

unnormalized_action_chunk:
rtol = 0
atol = 1e-6
```

并使用完全相同的 fixed noise。

------

# 12. Clean Gate

只有对应模型 clean-equivalence 全部通过后，才允许执行该模型的 intervention smoke。

```text
OpenVLA:
2/2 clean-equivalence PASS

π0.5:
2/2 clean-equivalence PASS
```

若任一 observation FAIL：

```text
该模型 real smoke = BLOCKED
```

必须：

```text
保存 diagnostics
停止该模型 intervention smoke
不放宽 tolerance
不替换 observation
不修改 checkpoint/config
```

------

# 13. Intervention Metrics

## OpenVLA

记录：

```text
||clean O2||
||intended ΔO2||
||actual ΔO2||
actual relative perturbation

clean unnormalized_action[:3]
modified unnormalized_action[:3]

Δtranslation
||Δtranslation||_2
```

## π0.5

记录：

```text
||clean P2||
||intended ΔP2||
||actual ΔP2||
actual relative perturbation

clean unnormalized_action_chunk[0,0,:3]
modified unnormalized_action_chunk[0,0,:3]

Δtranslation
||Δtranslation||_2
```

所有 feature 和 action 必须 finite。

------

# 14. Intervention Smoke PASS

每个模型要求：

```text
clean-equivalence PASS
AND
actual modified feature differs from clean feature
AND
continuation consumes the supplied modified native feature
AND
all outputs finite
AND
at least 1/2 observations produces translation response above numerical floor
```

Numerical floor：

```text
OpenVLA:
||Δtranslation||_2 > 1e-8

π0.5:
||Δtranslation||_2 > 1e-6
```

这些仅用于排除 numerical-noise-level 无响应。

不得据此声明：

```text
policy sensitive
CCA direction relevant
shared representation action-relevant
transferable attack success
```

------

# 15. Runner Output

真实执行阶段使用新目录：

```text
experiment_inbox/c6-real-smoke-output/
```

正式执行前目标目录必须不存在或为空。

建议分阶段文件：

```text
openvla_results.json
pi05_results.json
results.json
summary.md
```

OpenVLA 和 π0.5 runner 分别只写自己的结果文件。

aggregator 在两侧结果存在后生成：

```text
results.json
summary.md
```

必须记录至少：

```text
git commit
observation IDs
observation/image hashes
checkpoint/config identities
resolved checkpoint paths
runtime versions

noise RNG implementation
noise seed
noise dtype

direction RNG implementation
per-model/per-observation direction seeds

alpha
intended perturbation metrics
actual perturbation metrics

clean-equivalence metrics
intervention metrics

per-observation results
per-model PASS/BLOCKED
overall PASS/BLOCKED
```

不得覆盖既有正式输出。

------

# 16. Authorized Runner Implementation Scope

本合同授权实现：

```text
scripts/c6_openvla_real_smoke.py
scripts/c6_pi05_real_smoke.py
scripts/c6_real_smoke_summary.py
```

以及必要的：

```text
CPU-only tests
static validation tests
small repository-local helpers
```

runner implementation 必须复用现有：

```text
openvla_intervention.py
pi05_intervention.py
```

不得复制并重新实现 intervention scientific semantics。

------

# 17. Runner Unit-Level Completion Gate

runner implementation 达到：

```text
UNIT-LEVEL PASS
```

iff：

```text
两个 model runner 均实现
AND
aggregator 实现
AND
frozen observation/config/RNG 参数被验证
AND
CPU/static tests PASS
AND
relevant regression tests PASS
AND
lint/format PASS
AND
git diff --check PASS
AND
未运行真实 checkpoint inference
```

完成后：

```text
STOP
```

进行一次 read-only runner audit。

------

# 18. Real Execution Authorization

本合同当前仍不授权真实执行。

当前状态必须保持：

```text
C6 real-smoke runner implementation: AUTHORIZED
C6 real-smoke runner unit validation: AUTHORIZED

C6 real checkpoint execution: NOT AUTHORIZED
```

只有 runner implementation + audit PASS 后，才单独授权：

```text
OpenVLA real smoke execution
π0.5 real smoke execution
result aggregation
```

并由用户在对应服务器环境执行。

------

# 19. Explicit Restrictions

禁止：

```text
修改 frozen intervention modules 的科学语义
修改 ../openvla/**
修改 ../openpi/**
修改 C5/C5-BM artifacts

替换 frozen observations
修改 tolerance 追求 PASS
修改 checkpoint/config 追求 PASS

使用 CCA directions
搜索 epsilon
搜索 sensitive direction
JVP/gradient screening
C6-B
Tex3D optimization
```

若 runner implementation 暴露现有 intervention API bug：

```text
STOP
→ 报告 blocker
→ 单独修复
→ 重新 unit audit
```

不得在 runner 内静默绕过。

------

# 20. Final Stage Boundary

runner audit PASS 后，下一步才是：

```text
C6 REAL EXECUTION AUTHORIZATION
```

真实运行成功后才允许判定：

```text
C6 real clean-equivalence / intervention smoke: PASS
```

只有该状态 PASS 后，才允许起草：

```text
C6-B action-relevant shared-direction scientific contract
```
