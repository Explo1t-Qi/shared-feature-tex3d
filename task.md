# C6 — O2/P2 Intervention-Interface Implementation Contract

## Contract Status

```text
C5-BM formal materialization: FORMAL COMPLETE / PASS
C6 intervention-interface contract: FINAL READ-ONLY AUDIT PASS / FROZEN
C6 interface implementation: UNIT-LEVEL PASS
C6 unit validation: PASS
C6 real clean-equivalence / intervention smoke: NEXT / NOT AUTHORIZED
C6-B policy-sensitivity analysis: NOT AUTHORIZED
Tex3D optimization: NOT AUTHORIZED
```

The completed unit-level validation does not establish real-checkpoint integration.

This contract defines the minimum implementation boundary for explicit continuation interfaces at OpenVLA O2 and π0.5 P2.

The interface consumes native model features. It does **not** consume CCA-space features and does not define CCA-to-native intervention directions.

---

## 1. Stage Objective

Implement explicit, non-hook continuation paths that allow controlled replacement of:

```text
OpenVLA O2:            [1, 256, 4096]
π0.5 base-camera P2:   [1, 256, 2048]
```

while holding all non-intervened policy context fixed.

This stage establishes an experimental instrument only. It does not establish policy/action relevance.

---

## 2. Explicitly Deferred Scientific Questions

Do not implement or decide:

```text
CCA component selection
CCA/shared-space → native-space perturbation construction
shared-direction ranking
epsilon selection for scientific analysis
gradient/JVP policy-sensitivity screening
C6-B sensitivity metrics
policy/action relevance claims
Tex3D losses
adversarial texture optimization
```

C6-B requires a separate scientific contract.

---

# 3. OpenVLA Interface

## 3.1 Frozen Runtime Boundary

The authoritative OpenVLA continuation path must use the same deployed inference semantics as the existing LIBERO/OpenVLA path.

Freeze:

```text
batch size: B == 1
checkpoint/model identity: openvla/openvla-7b-finetuned-libero-spatial
unnorm_key: libero_spatial_no_noops
do_sample: False
center_crop: True
image preprocessing: reuse the existing C1/OpenVLA LIBERO preprocessing path
prompt construction: reuse the existing deployed C1/OpenVLA prompt path
gripper postprocessing: identical to the existing deployed path
```

Do not create an alternative image preprocessing or prompt formatting path for the intervention interface.

The authoritative interface must not modify `../openvla`.

## 3.2 Preparation API

Implement an explicit preparation function equivalent in responsibility to:

```python
prepared = prepare_openvla_context(
    *,
    model,
    processor,
    observation,
    task_description,
    pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
    unnorm_key="libero_spatial_no_noops",
    center_crop=True,
)
```

`observation` is the C1 policy observation passed to the deployed OpenVLA `get_action()` path, not the raw MuJoCo observation. In the current C1 semantics it contains the policy-facing image/state fields, while `task_description` is supplied separately.

`pretrained_checkpoint` is explicit for prompt-branch compatibility and provenance, but this contract freezes its authoritative value to:

```text
openvla/openvla-7b-finetuned-libero-spatial
```

Changing it is outside this contract.

The preparation function must reuse existing preprocessing/prompt behavior and produce an immutable or caller-read-only prepared object containing at least:

```text
o2
  shape: [1, 256, 4096]
  node: projector output inserted after BOS and before text tokens

all fixed downstream context required to reconstruct the same multimodal sequence and decoding configuration

checkpoint/model identity
processor/preprocessing identity or configuration
unnorm_key
```

The exact Python class name may follow the repository style, but its fields and semantics must be explicit and tested.

## 3.3 Continuation API

Implement an explicit continuation function equivalent in responsibility to:

```python
result = continue_openvla_from_o2(
    *,
    prepared,
    o2,
)
```

Input contract:

```text
o2 shape: [1, 256, 4096]
finite values required
dtype/device must be compatible with the downstream OpenVLA model
```

Allowed handling:

```text
shape validation
finite-value validation
required dtype conversion
required device placement
```

Forbidden semantic handling:

```text
normalization
clipping
reprojection
automatic rescaling
CCA-specific processing
```

The supplied O2 must be the representation actually consumed by the downstream multimodal language-model path.

Forward hooks may be used only for debugging and are not an authoritative intervention mechanism.

## 3.4 OpenVLA Result Semantics

Expose one result object with unambiguous fields:

```text
action_token_ids
  shape: [1, 7]
  public type: CPU NumPy integer array
  generated action token IDs

normalized_action
  shape: [1, 7]
  public type: CPU NumPy floating array
  action-token bin-center values before checkpoint-stat unnormalization

unnormalized_action
  shape: [1, 7]
  public type: CPU NumPy floating array
  checkpoint q01/q99 unnormalized continuous action

deployed_action
  shape: [1, 7]
  public type: CPU NumPy floating array
  unnormalized action followed by the existing deployed gripper processing:
    normalize_gripper_action(..., binarize=True)
    invert_gripper_action()
```

For dimensions 1–3, primary cross-model comparison uses the checkpoint-unnormalized translation:

```python
result.unnormalized_action[:, :3]
```

Do not retain an ambiguous public definition such as “deployed action or equivalent unnormalized translation”.

---

# 4. π0.5 Interface

## 4.1 Frozen Runtime Boundary

This contract supports only the already audited LIBERO π0.5 deployment:

```text
config: pi05_libero
checkpoint: gs://openpi-assets/checkpoints/pi05_libero
backend: JAX / NNX
batch size: B == 1
```

A PyTorch backend is out of scope for this contract.

The authoritative interface must not modify `../openpi`.

## 4.2 Observation Boundary

The public preparation API accepts the single **unbatched raw LIBERO inference dict** used by the existing `pi05_libero` `Policy.infer()` path.

The authoritative preparation order is:

```text
policy input transforms
→ Observation.from_dict
→ preprocess_observation(None, observation, train=False)
→ image encoder
```

This stage freezes `B == 1`; batching multiple raw inference dicts is out of scope.

Do not define the intervention API around `PilotObservation` or around an already partially transformed observation as the public authoritative input.

## 4.3 Preparation API

Implement an explicit preparation function equivalent in responsibility to:

```python
prepared = prepare_pi05_context(
    *,
    policy,
    observation,
    noise,
)
```

where `policy` is an already loaded JAX/NNX `pi05_libero` OpenPI policy or an equivalent repository-local wrapper that exposes the same model plus input/output transforms.

Required noise:

```text
shape: [1, 10, 32]
finite floating array
explicitly supplied
reused exactly between clean and intervention continuations
no implicit resampling is allowed
```

The prepared object must be immutable or caller-read-only and contain at least:

```text
base_p2
  [1, 256, 2048]

left_p2
  [1, 256, 2048]

right_p2
  [1, 256, 2048]

right_image_mask
  False for the deployed zero-filled right-wrist stream

fixed prompt embeddings / masks / prefix context
fixed noise
input/output transform context required to reproduce Policy.infer semantics
```

Prepared feature tensors/arrays must preserve the model-native dtype/device/backend representation used by the authoritative forward path.

Only base-camera P2 is replaceable in this stage.

## 4.4 Continuation API

Implement an explicit continuation function equivalent in responsibility to:

```python
result = continue_pi05_from_p2(
    *,
    prepared,
    base_p2,
)
```

Input contract:

```text
base_p2 shape: [1, 256, 2048]
finite values required
dtype/device compatible with the downstream JAX/NNX path
```

The continuation must preserve:

```text
left/right camera representations
prompt embeddings
masks
initial Gaussian noise
Euler integration settings
checkpoint normalization statistics
output transforms
```

Allowed handling:

```text
shape validation
finite-value validation
required dtype/device conversion on the supplied override only
```

Any allowed conversion must not mutate or replace the original clean feature stored in `prepared`.

Forbidden semantic handling:

```text
normalization
clipping
reprojection
automatic rescaling
CCA-specific processing
```

## 4.5 π0.5 Result Semantics

Expose distinct fields for the different action representations:

```text
normalized_action_chunk_32
  shape: [1, 10, 32]
  public type: CPU NumPy floating array
  native model output before LIBERO dimensional trimming / output transforms

normalized_action_chunk
  shape: [1, 10, 7]
  public type: CPU NumPy floating array
  first seven normalized action dimensions before checkpoint unnormalization

unnormalized_action_chunk
  shape: [1, 10, 7]
  public type: CPU NumPy floating array
  final LIBERO action chunk after the authoritative output-transform path
```

Primary cross-model comparison object:

```python
result.unnormalized_action_chunk[:, 0, :3]
```

---

# 5. Authoritative Reference Helpers

Clean-equivalence validation requires repository-local reference helpers that expose intermediate quantities from **one authoritative forward/inference invocation**, rather than comparing quantities assembled from unrelated repeated calls.

## 5.1 OpenVLA Reference

The OpenVLA reference helper must expose, from the authoritative original-policy path or a semantically identical repository-local wrapper:

```text
action token IDs
normalized action
unnormalized action
deployed action
```

The helper must not alter decoding semantics.

## 5.2 π0.5 Reference

The π0.5 reference helper must expose, from one fixed-noise authoritative inference path:

```text
normalized_action_chunk_32 [1,10,32]
normalized_action_chunk    [1,10,7]
unnormalized_action_chunk  [1,10,7]
```

It must reuse the same `pi05_libero` input/output transforms as `Policy.infer()`.

These helpers are validation instrumentation, not new policy semantics.

---

# 6. Unit-Level Clean-Equivalence Semantics

Implementation/unit tests should verify continuation equivalence with synthetic or controlled local fixtures wherever possible.

The intended real-runtime numerical gates remain:

## OpenVLA

```text
action token IDs: exact match
discrete deployed gripper decision: exact match
continuous quantities: rtol = 0, atol = 1e-8
```

## π0.5

with exactly the same explicit noise tensor:

```text
normalized_action_chunk_32: rtol = 0, atol = 1e-6
normalized_action_chunk:    rtol = 0, atol = 1e-6
unnormalized_action_chunk:  rtol = 0, atol = 1e-6
```

However, **real GPU/server clean-equivalence execution is not authorized by this contract**. Its observations, runtime parameters and commands must be frozen separately.

---

# 7. Real Intervention Smoke — Deferred

Real clean-equivalence and real intervention smoke are a separate authorization stage.

A later smoke contract must freeze at minimum:

```text
exact observation/sample IDs
number of observations
batch size
checkpoint/runtime identities
Gaussian RNG seed
RNG implementation
perturbation dtype
whether direction normalization is global or per sample
engineering epsilon
reported quantities
output directory / overwrite safety
server command
PASS/BLOCKED criteria
```

The later smoke may use deterministic synthetic native-space perturbations only.

It must not use CCA directions and must not make policy-sensitivity claims.

No real intervention smoke is authorized now.

---

# 8. Authorized Implementation File Boundary

Preferred minimal repository-local files:

```text
shared_feature/openvla_intervention.py
shared_feature/pi05_intervention.py
tests/test_openvla_intervention.py
tests/test_pi05_intervention.py
```

`shared_feature/__init__.py` may be changed only if required to expose the new APIs.

Small adjacent repository-local helper/test files are allowed only when necessary for the explicit reference helpers or shared dataclasses. Avoid unrelated refactoring.

Forbidden modifications:

```text
../openvla/**
../openpi/**
C5-BM formal artifacts
C2/C3/C4/C5-A/C5-B formal artifacts
existing C2/C3 scientific semantics
```

---

# 9. Required Unit Tests

At minimum test:

## OpenVLA

```text
B == 1 enforcement
O2 shape validation
finite-value validation
prepared-state field/immutability semantics
supplied O2 is actually consumed
no hidden normalize/clip/reproject path
reference/result action semantic separation
clean continuation equivalence on controlled fixtures
```

## π0.5

```text
base P2 shape validation
finite-value validation
B == 1 enforcement
noise shape [1,10,32]
noise finite-value validation
no implicit noise resampling
prepared-state field/immutability semantics
supplied base P2 is actually consumed
left/right P2 and prompt/mask/noise remain fixed
32D vs 7D action semantic separation
fixed-noise continuation equivalence on controlled fixtures
```

## Regression / Integrity

```text
relevant existing tests remain passing
no upstream repository modification
no C5-BM artifact modification
no C6-B code added
```

---

# 10. Implementation Completion Gate

Following the separate implementation authorization and completed unit-level
validation, the current stage is:

```text
C6 interface implementation: UNIT-LEVEL PASS
C6 unit validation: PASS
C6 real clean-equivalence / intervention smoke: NEXT / NOT AUTHORIZED
```

Implementation reached `UNIT-LEVEL PASS` because:

```text
OpenVLA explicit continuation API implemented
AND
π0.5 explicit continuation API implemented
AND
reference helpers implemented
AND
focused unit tests pass
AND
relevant regression tests pass
AND
no upstream source repository was modified
AND
no real GPU/server intervention smoke was executed
AND
no C6-B scientific analysis was implemented
```

Then stop.

---

# 11. Final Stop Condition

This contract does not authorize real server intervention experiments.

After implementation/unit validation:

```text
STOP
```

Next required stage:

```text
C6 real clean-equivalence / intervention-smoke contract
```

Only after that stage passes may the project define a separate C6-B scientific contract for action-relevant shared-direction discovery.
