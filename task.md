You are auditing the repository for the next research stage, C6-A.

## Context

Project goal:

Identify cross-VLA shared representations between OpenVLA and π0.5, then determine which shared directions are genuinely policy/action-relevant, and eventually use those directions to design a single-surrogate transferable adversarial texture loss.

Completed stages:

- C5-A: cross-model representation geometry → GO
- C5-B: explicit shared-space alignment via PCA + ordinary CCA → PASS
- C5 representation-stage → PASS

Primary aligned representation pair:

- OpenVLA O2:
  multimodal projector output
  shape `[256, 4096]`
- π0.5 P2:
  projected PaliGemma-ready visual representation
  shape `[256, 2048]`

C5-B learned TRAIN-only PCA + CCA mappings that define paired canonical shared directions.

We are now entering:

# C6-A — Policy Sensitivity Interface Audit & Feasibility

C6-A does NOT yet search for action-relevant directions.

Its sole purpose is to determine how to rigorously measure:

> how a change at O2 / P2 affects the downstream VLA action output.

A related paper, “Mechanistic Interpretability for Steering Vision-Language-Action Models”, uses direct FFN activation intervention to establish causal links between internal semantic directions and robot behavior. We may later adopt the same intervention philosophy, but our shared directions come from CCA and are not native FFN neurons.

------

# Task

Perform a READ-ONLY code audit.

Do NOT:

- modify any source file;
- add tests;
- implement hooks;
- implement gradients;
- implement interventions;
- change documentation;
- run formal C6 experiments;
- modify C5 artifacts;
- propose a final scientific contract yet.

You may run harmless read-only inspection commands, import inspection, shape inspection, or minimal non-mutating code tracing if necessary.

The goal is to answer the questions below from the actual current codebase.

------

# Part A — OpenVLA O2 → Action Path

Trace the exact computation path beginning from the scientific node O2.

Answer:

1. Where exactly is O2 produced?
   - file
   - class/function
   - relevant tensor variable
   - exact shape/dtype if inferable
2. After O2 is produced, what exact modules consume it before action prediction?

Provide a concise path such as:

```text
O2
→ ...
→ ...
→ action-related logits
→ token decoding
→ continuous robot action
```

Use actual function/class names from the repository.

1. What is the earliest downstream point after O2 from which the model can continue inference if O2 is replaced by a modified tensor?

Determine whether:

- there is an existing continuation-style API;
- inference must instead be rerun with a hook/replacement at O2;
- or another intervention point is cleaner.

1. What is OpenVLA's actual action prediction representation?

Identify:

- logits shape;
- number/type of action tokens;
- whether autoregressive token generation is used;
- how tokens are converted to continuous robot actions;
- final continuous action shape;
- semantic meaning/order of action dimensions if defined in code/config.

1. Identify all potentially non-differentiable operations between O2 and the final continuous action:

- argmax;
- token sampling;
- discrete token IDs;
- integer indexing;
- detach;
- no_grad;
- decoding tables;
- clipping;
- numpy conversion;
- other gradient breaks.

1. Based strictly on the implementation, distinguish which of these quantities are differentiable with respect to O2:

- action logits;
- action-token probabilities;
- expected decoded action, if such an object naturally exists;
- final standard inference continuous action.

Do NOT redesign the model yet. Just report what is and is not differentiable.

------

# Part B — π0.5 P2 → Action Path

Trace the exact computation path beginning from P2.

Answer:

1. Where exactly is P2 produced?
   - file
   - class/function
   - tensor variable
   - exact shape/dtype if inferable
2. Trace the exact downstream computation:

```text
P2
→ ...
→ policy/action generation
→ action chunk
```

Use actual code symbols.

1. Determine the action-generation mechanism:

- direct regression?
- flow matching?
- iterative denoising?
- sampling?
- another mechanism?

Describe the actual implementation rather than relying on general π0.5 knowledge.

1. Determine:

- output action tensor shape;
- action horizon/chunk length;
- action dimension;
- meaning/order of dimensions;
- normalization/unnormalization logic;
- any task-specific action masks.

1. Identify stochastic inputs or iterative states involved in inference:

- sampled noise;
- random seeds;
- number of integration/denoising steps;
- scheduler;
- other stochasticity.

1. Determine whether a deterministic evaluation path can be obtained by freezing the stochastic inputs without changing the scientific algorithm.
2. Identify every potential gradient break from final action output back to P2:

- stop_gradient;
- detach;
- no_grad;
- conversion between JAX/NumPy/Python;
- integer/discrete operations;
- solver implementation details;
- explicit non-differentiable code.

1. Determine whether the final continuous action chunk is differentiable with respect to P2 under a fixed-noise/fixed-inference-path setup.

Do not implement this. Report feasibility only.

------

# Part C — Cross-Model Action Semantics

Compare the two models' final robot action representations.

Produce a table with at least:

| Property                     | OpenVLA | π0.5 |
| ---------------------------- | ------- | ---- |
| action shape                 |         |      |
| action horizon               |         |      |
| translation dimensions       |         |      |
| rotation representation      |         |      |
| gripper representation       |         |      |
| normalization                |         |      |
| clipping                     |         |      |
| stochasticity                |         |      |
| differentiability from O2/P2 |         |      |

Then answer:

1. Do OpenVLA and π0.5 expose semantically compatible physical action quantities?
2. Which action dimensions are directly comparable?
3. Which dimensions are not directly comparable without normalization or representation conversion?
4. Does either model predict action chunks while the other predicts only one step?
5. Is a common action-level sensitivity metric scientifically feasible?

Do NOT choose the final metric yet.

------

# Part D — Intervention Feasibility

C5-B canonical directions are analysis-space directions, not native model neurons.

For each model determine how a future intervention could technically be implemented.

Conceptually, future C6-B/C6-C may need:

```text
original O2/P2
+ small perturbation corresponding to one CCA-derived direction
→ continue normal policy forward
→ compare action output
```

Audit:

1. Can O2/P2 be replaced by a modified tensor in the normal computation graph?
2. If not directly, what is the cleanest technically valid intervention mechanism?
   - forward hook;
   - function argument;
   - refactored continuation function;
   - JAX intermediate replacement;
   - other.
3. Would such replacement preserve gradients?
4. Would replacing O2/P2 alter only the chosen scientific representation node, or accidentally bypass/recompute other model components?
5. Are there architectural complications because O2/P2 are token sequences `[256,D]` rather than one observation-level vector?
6. Confirm whether the stored C5-B PCA/CCA mapping can in principle be converted back into a direction in the native O2/P2 feature space.

Do not implement the inverse/native direction mapping yet; only verify feasibility.

------

# Part E — Policy Sensitivity Target Candidates

Based strictly on the audited code paths, identify the technically valid candidate quantities that C6-A could later use as the downstream "policy output" for sensitivity.

For OpenVLA, list all viable candidates, e.g. if supported by code:

- action-token logits;
- probability/logit margin;
- continuous decoded action surrogate;
- other.

For π0.5, list all viable candidates:

- final continuous action;
- denoising/flow velocity;
- intermediate action estimate;
- other.

For each candidate report:

- differentiable from O2/P2? YES/NO
- physically interpretable? HIGH/MEDIUM/LOW
- faithful to actual deployed action? HIGH/MEDIUM/LOW
- major caveats

Do NOT decide the scientific primary target yet.

------

# Part F — Determinism and Reproducibility

For both models determine what would need to be frozen for a reproducible local sensitivity experiment.

Audit:

- model eval mode;
- RNG;
- action sampling;
- flow/noise initialization;
- decoding strategy;
- dtype;
- dropout;
- observation preprocessing;
- language/task input;
- robot state/proprio input;
- action normalization stats.

State whether repeated identical input can produce bitwise-identical or numerically equivalent action outputs under a controlled setup.

------

# Part G — C5-B Mapping Availability and Provenance Audit

Audit what C5-B actually persisted.

Determine whether the formal C5-B artifacts contain sufficient information to
reconstruct and reuse the exact TRAIN-fitted PCA + CCA mappings without refitting.

Explicitly check whether the following objects are persisted:

- PCA TRAIN means;
- PCA bases / right singular vectors;
- PCA component ordering;
- CCA whitening transforms;
- CCA canonical mappings W_A / W_B;
- canonical component ordering / signs.

If any of these are not persisted, state clearly:

C5-B scientific results remain valid, but the exact fitted canonical mapping is
not currently materialized as a reusable formal artifact.

Do NOT:

- silently refit PCA;
- silently refit CCA;
- regenerate mappings and treat them as already-frozen artifacts;
- modify existing formal C5-B outputs;
- infer missing mapping tensors from summary statistics.

Then audit what would be required for a future mapping-materialization step.

Report:

1. Which existing frozen C5-B inputs would be required to deterministically
   regenerate the mappings.

2. Whether the existing implementation is sufficiently deterministic to
   reproduce the same mappings, including:
   - observation ordering;
   - token ordering;
   - float64 semantics;
   - PCA SVD ordering;
   - covariance eigendecomposition ordering;
   - CCA SVD ordering;
   - sign conventions.

3. What provenance would need to be stored with a future mapping artifact:
   - source paired manifest;
   - C5-B split manifest;
   - source feature hashes / identities;
   - PCA cutoff;
   - node identity;
   - implementation / repository commit;
   - array dtype and shapes;
   - canonical ordering and sign convention.

4. Whether a future C6 intervention can technically consume such frozen mappings
   once materialized.

Do not materialize the mappings during this audit.

Conclude one of:

C5-B MAPPING REUSE: DIRECTLY AVAILABLE

or

C5-B MAPPING REUSE: REQUIRES FORMAL MATERIALIZATION

or

C5-B MAPPING REUSE: BLOCKED

------

# Required Final Report

Return a structured audit report with these sections:

1. `Executive Summary`
2. `OpenVLA O2 → Action`
3. `π0.5 P2 → Action`
4. `Cross-Model Action Compatibility`
5. `Differentiability Audit`
6. `Intervention Feasibility`
7. `Candidate Policy Sensitivity Targets`
8. `Determinism / Reproducibility`
9. `C5-B Reuse Boundaries`
10. `Blockers / Risks`
11. `Recommended Decisions for the Human Scientific Review`

For every important claim, include:

- file path;
- function/class name;
- relevant line range or code symbol where possible.

At the end give one of:

```text
C6-A INTERFACE FEASIBILITY: CLEAR
```

or

```text
C6-A INTERFACE FEASIBILITY: BLOCKED
```

or

```text
C6-A INTERFACE FEASIBILITY: PARTIAL
```

This status is only an engineering/interface audit result.

It must NOT be interpreted as:

- policy-relevance established;
- causal relevance established;
- transferability established;
- C6-A scientific PASS.

If PARTIAL or BLOCKED, state exactly which model/path causes the issue.

Finally include:

```text
Files modified: NONE
Formal experiments executed: NONE
Implementation performed: NONE
```

Stop after the audit. Do not patch anything without explicit authorization.
