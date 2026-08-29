# C6-A Scientific / Engineering Contract

## Policy Sensitivity Interface Closure Contract — FROZEN

**Purpose:** freeze the minimum scientific and engineering conclusions required to close the C6-A interface-feasibility stage and define the explicit prerequisites for C6-B.

This document does **not** design:

- the final C6-B policy-sensitivity metric;
- the native-space synthesis/intervention vector;
- token intervention scope;
- perturbation scale \(\epsilon\);
- C6-B statistical thresholds;
- any Tex3D attack loss.

---

## 1. Stage Context

C5 established that OpenVLA O2 and π0.5 P2 contain statistically non-random shared representation structure and that a TRAIN-only PCA+CCA shared space exists.

C6 asks which shared directions are genuinely policy/action-relevant.

The current primary representation nodes are:

- **OpenVLA O2:** multimodal projector output, shape `[256, 4096]`.
- **π0.5 P2:** projected PaliGemma-ready visual representation, shape `[256, 2048]`.

The research logic must preserve the distinction:

```text
shared
≠ policy-relevant
≠ transferable
```

---

## 2. C6-A Read-Only Audit Conclusions

### 2.1 OpenVLA

O2 is the live projector output (`projected_patch_embeddings`).

The downstream deployed path is:

```text
O2
→ multimodal embeddings
→ language model
→ autoregressive action-token generation
→ discrete token decoding
→ checkpoint action unnormalization
→ deployed 7D action
→ gripper post-processing
→ LIBERO env.step(...)
```

The standard deployed path is **not differentiable** from O2 to the final continuous action because it contains:

- `generate()` under `no_grad`;
- greedy `argmax`;
- discrete token IDs;
- NumPy-based decoding / bin lookup;
- gripper binarization.

However, direct language-model logits and token probabilities are differentiable with respect to O2 when evaluated outside the standard `generate/no_grad` path with a fixed token prefix.

There is no existing public O2 continuation API.

A future intervention can technically replace the projector output, while a cleaner gradient-preserving implementation would require an explicit continuation interface downstream of O2.

### 2.2 π0.5

P2 is the projected visual representation produced by:

```python
model.PaliGemma.img(...)
```

The downstream policy path is:

```text
P2
+ other image streams
+ prompt embeddings
→ multimodal prefix
→ prefix KV cache
→ initial Gaussian action noise
→ flow-matching velocity prediction
→ Euler integration
→ normalized action trajectory
→ checkpoint quantile unnormalization
→ LiberoOutputs trimming
→ deployed LIBERO action chunk
```

For the current LIBERO configuration:

- internal action tensor: `[B, 10, 32]`;
- deployed LIBERO action chunk: `[10, 7]`;
- initial Gaussian noise is the principal stochastic input;
- once noise and all other inputs are fixed, the core computation path is deterministic.

The current dynamic JAX `lax.while_loop` supports forward-mode differentiation / JVP in the audited setup, but not reverse-mode differentiation through the final integrated `x0`.

The public `Policy.infer()` path converts outputs to NumPy and is therefore outside the JAX gradient graph.

There is no existing P2 replacement API.

A future intervention would require an explicit prefix/continuation path that accepts a modified base-image P2 while preserving:

- other visual streams;
- prompt tokens;
- masks;
- positions;
- KV-cache logic.

---

## 3. Scientific Decision 1 — Definition of Policy Sensitivity

**Decision:** policy sensitivity is **not** defined exclusively by reverse-mode gradients.

The core scientific object is the change in policy output caused by a small, controlled perturbation along a representation-space direction \(d\):

\[
h' = h + \epsilon d
\]

A directional response may later be summarized by a finite directional effect such as:

\[
S(d)
=
\frac{\|a(h+\epsilon d)-a(h)\|}{\epsilon}
\]

This expression is **illustrative only**.

C6-A does **not** freeze:

- the norm;
- endpoint;
- aggregation;
- normalization;
- perturbation schedule;
- candidate-ranking procedure.

### Evidence hierarchy

1. **Gradient / JVP**
   Used as a candidate locator or screening signal.

2. **Small controlled directional intervention**
   Used to directly measure policy-output change.

3. **Positive/negative interventions across observations and perturbation scales**
   Used as stronger evidence of policy relevance and causal participation.

Therefore:

> Gradient availability is useful, but it is not a scientific prerequisite for C6.

A large gradient/JVP is **not**, by itself, sufficient evidence that a direction is policy-relevant.

---

## 4. Scientific Decision 2 — Cross-Model Action Object

### 4.1 Confirmed primary comparison

**Decision:** the confirmed primary cross-model action object is the **first-step translation command only**.

```text
Primary confirmed comparison:

OpenVLA:
a_open[:3]

π0.5:
A_pi[0, :3]
```

where:

- `a_open` is the **OpenVLA checkpoint-unnormalized deployed action** after action-token decoding;
- `A_pi` is the **π0.5 checkpoint quantile-unnormalized action chunk after LiberoOutputs trimming**.

The models' internal normalized actions must **not** be directly compared against the other model's deployed/unnormalized actions.

### 4.2 Conditional rotation extension

Rotation dimensions 4–6 are currently **conditional**, not primary.

After the exact deployed `robosuite==1.4.0` `OSC_POSE` rotation-command semantics are confirmed, the comparison may be extended to:

```text
Conditional extension:

OpenVLA:
a_open[:6]

π0.5:
A_pi[0, :6]
```

Until that controller semantics check is complete, no claim should be made that dimensions 4–6 are strictly identical physical coordinates.

### 4.3 Gripper

The gripper dimension is analyzed separately because the deployed conventions differ:

- OpenVLA is finally binarized;
- π0.5 may remain continuous;
- the sign / opening convention differs.

Therefore gripper must not be included in the primary cross-model distance without an explicit later harmonization rule.

### 4.4 π0.5 action horizon

π0.5 predicts a 10-step action chunk while OpenVLA predicts a single step.

Therefore:

- **primary:** π0.5 first action step only;
- **later robustness / temporal analysis:** π0.5 steps 2–10 or full-chunk response.

The full π0.5 action chunk is not currently the primary cross-model action object.

---

## 5. Scientific Decision 3 — Authoritative C5-B Mapping Materialization

### 5.1 Historical limitation

The formal historical C5-B artifacts did **not** persist:

- PCA TRAIN means;
- PCA bases;
- CCA whitening transforms;
- CCA mappings \(W_A/W_B\).

Therefore it is **not possible to prove** that any newly generated matrices are element-wise identical to the historical in-memory C5-B matrices.

This is especially important because:

- SVD / eigendecomposition signs are ambiguous;
- near-degenerate directions may rotate within a subspace;
- no historical matrix artifact exists for direct equality comparison.

Therefore the project must **not** claim to “recover the exact historical C5-B mapping.”

### 5.2 Authorized re-fit and new authoritative artifact

**Decision:** before C6-B, the project will perform an explicitly authorized, versioned re-fit using:

- the frozen formal paired inputs;
- the frozen TRAIN / HELD-OUT split;
- the same C5-B estimator;
- the same scientific configuration;
- explicitly recorded numerical conventions.

This re-fit will produce a **new authoritative reusable mapping artifact** for all future C6 work.

The new artifact may be scientifically consistent with historical C5-B, but it must **not** be described as element-wise identical to the unsaved historical in-memory matrices.

### 5.3 Required materialization scope

The minimum required materialization is:

```text
node pair:
O2 ↔ P2

PCA:
99% explained variance

fit:
true, unpermuted TRAIN pairing
```

This is the primary C5-B configuration and becomes the authoritative mapping reference for the C6 mainline.

### 5.4 Intentionally excluded from required materialization

The following mappings are **not required** for the C6 mainline materialization step:

```text
O2 ↔ P2 @95% PCA robustness fit
O1-S ↔ P1 @99% PCA control fit
O1-S ↔ P1 @95% PCA control/robustness fit
all null / permuted mappings
```

These configurations remain valid parts of the C5 scientific evidence, but they are intentionally excluded from the minimum C6 intervention-reference artifact.

If future work requires them, they must be materialized explicitly under a separate declared scope.

### 5.5 Required consistency validation

The re-materialized mapping must be validated against the **quantities that were actually persisted** in the existing formal C5-B result.

At minimum, verify consistency of:

- retained PCA dimensions;
- PCA explained-variance metadata / cutoff behavior;
- historical TRAIN scalar summaries that can be recomputed from the new fit, including TRAIN Top5Mean;
- historical HELD-OUT Top1 / Top5Mean / Top10Mean;
- other persisted formal scalar metrics required to establish that the re-fit reproduces the intended C5-B scientific configuration.

For the newly fitted canonical-correlation vector:

\[
\sigma = [\sigma_1,\ldots,\sigma_K],
\]

the full vector must be saved in the new mapping artifact, but it must **not** be described as element-wise validated against the historical C5-B run because the historical full `sigma` vector was not persisted.

Where applicable:

\[
\text{TRAIN Top5Mean}
=
\frac{1}{5}\sum_{k=1}^{5}\sigma_k
\]

from the new fit should match the historical persisted TRAIN Top5Mean within the frozen numerical tolerance.

Likewise, HELD-OUT Top1 / Top5Mean / Top10Mean recomputed from the new mappings must match the corresponding historical persisted metrics within the frozen tolerance.

Validation must use explicitly frozen numerical tolerances.

This validation establishes **pipeline/scientific consistency**, not equality with the unsaved historical matrices or unsaved historical `sigma` vector.

### 5.6 Required frozen mapping contents

The new authoritative mapping artifact must persist at least:

- PCA TRAIN means for both models;
- PCA bases;
- retained PCA dimensions;
- PCA cutoff;
- any CCA whitening transforms required by the implementation;
- CCA mappings \(W_A\) and \(W_B\);
- canonical correlations `sigma`;
- `sigma` shape and dtype;
- canonical component ordering;
- explicit guarantee that `W_A[:, k]`, `W_B[:, k]`, and `sigma[k]` share the same canonical-component order;
- explicit sign canonicalization;
- node identity;
- array shapes;
- array dtypes;
- source paired-manifest identity/hash;
- TRAIN/HELD-OUT split identity/hash;
- relevant source-feature identities/hashes;
- repository/code commit;
- Python version;
- NumPy version;
- BLAS/LAPACK implementation/version where available;
- platform information;
- array-content hashes, including a content hash for `sigma`.

### 5.7 Sign canonicalization

Because CCA direction sign is mathematically ambiguous, the new authoritative artifact must impose and record a deterministic sign convention.

This is required so that future statements such as:

```text
+d_k
-d_k
```

have stable meanings across reloads and future experiments.

The sign convention belongs to **mapping materialization/provenance**.

It does **not** define the native-space intervention vector.

### 5.8 Old artifacts remain immutable

The materialization step must:

- leave the four existing formal C5-B artifacts unchanged;
- create a separate, versioned mapping artifact;
- explicitly record that a new authorized re-fit was performed.

It must not silently rewrite historical C5-B outputs.

---

## 6. Deferred Scientific Decisions

The following items remain intentionally **deferred** and must not be silently decided during mapping materialization or interface implementation:

- the final C6-B policy-sensitivity metric;
- the exact native-space intervention/synthesis vector corresponding to a canonical coordinate;
- whether that vector is a projection normal, dual vector, pseudoinverse/minimum-norm solution, or another mathematically justified construction;
- whether all 256 tokens are perturbed;
- whether only selected tokens are perturbed;
- whether spatial weighting is used;
- per-token perturbation amplitude;
- perturbation norm;
- the final \(\epsilon\) schedule;
- C6-B candidate-count / statistical thresholds;
- any Tex3D attack loss.

In particular, if:

\[
z=(x-\mu)V
\]

and:

\[
h=zW,
\]

then \(VW\) describes the linear readout/projection geometry of a canonical coordinate.

It must **not automatically be treated as the unique native-space synthesis/intervention direction**.

That mathematical choice belongs to C6-B scientific design.

---

## 7. Intervention Interface Requirement

The read-only audit established that neither model currently exposes a native continuation/replacement API at the scientific representation node:

```text
OpenVLA:
no public O2 continuation API

π0.5:
no public P2 replacement API
```

Therefore a real intervention smoke test cannot occur until an explicit engineering interface is implemented.

### 7.1 Required separate interface-coding stage

After mapping materialization, the project must create a **separate intervention-interface coding contract**.

That coding contract may authorize implementation of:

- an OpenVLA O2 replacement / continuation interface;
- a π0.5 P2 replacement / prefix-continuation interface;
- minimal validation utilities required to verify tensor replacement and downstream continuation.

### 7.2 What the interface contract must not decide

The interface-coding stage must be explicitly **O2/P2 node-specific**, while remaining **direction- and metric-agnostic**.

It must **not** decide:

- which canonical direction to inject;
- the native intervention vector;
- token scope;
- \(\epsilon\);
- sensitivity metric;
- candidate selection;
- C6-B thresholds;
- Tex3D loss.

The engineering interface should be capable of accepting a controlled replacement/perturbation tensor without embedding a scientific intervention policy into the API.

### 7.3 Validation order

Interface validation should proceed as:

```text
implementation
→ static/unit-level correctness checks where feasible
→ independent code review
→ real-device / accelerator smoke validation
```

No CPU-only requirement is imposed.

The real smoke must confirm that:

- the intended O2/P2 node is replaced;
- other upstream components are not accidentally recomputed or bypassed in an invalid way;
- normal downstream policy computation continues;
- output shapes and action semantics remain valid;
- fixed-noise π0.5 evaluation is reproducible under the declared setup.

---

## 8. C6-A Closure Criteria

C6-A can be considered scientifically closed once the following statements are accepted:

1. Controlled representation-space intervention is technically feasible for both O2 and P2, although neither model currently exposes a native continuation API.
2. A fully differentiable deployed-action path is not required for the scientific definition of policy sensitivity.
3. Gradient/JVP may be used for screening, while controlled directional intervention provides stronger evidence of policy relevance.
4. The confirmed primary cross-model action object is first-step **translation** only.
5. Rotation dimensions may be added only after the deployed robosuite `OSC_POSE` rotation-command semantics are confirmed.
6. Gripper remains a separate analysis dimension.
7. Historical C5-B mappings cannot be claimed as exactly recoverable because they were not persisted.
8. A new, versioned, authoritative O2↔P2 99%-PCA true-TRAIN mapping artifact must be produced by an explicitly authorized re-fit before C6-B.
9. The new mapping must use deterministic sign canonicalization and formal provenance.
10. A separate O2/P2 intervention-interface implementation stage is required before real intervention smoke testing.
11. Native synthesis direction, token scope, \(\epsilon\), and final sensitivity metric remain deferred to C6-B.

**Frozen C6-A status:**

```text
INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES
```

This status does **not** imply:

```text
policy relevance PASS
causal relevance PASS
transferability PASS
```

---

## 9. Post-C6-A Workflow

After this contract receives final read-only approval:

```text
1. C6-A contract freeze
        ↓
2. update research-status documentation
        ↓
3. C5-BM — Authoritative Mapping Materialization Contract
        ↓
4. mapping implementation
        ↓
5. formal mapping materialization + consistency validation
        ↓
6. independent mapping review
        ↓
7. separate Intervention Interface Coding Contract
        ↓
8. O2/P2 interface implementation
        ↓
9. static/unit-level validation where feasible
        ↓
10. independent interface code review
        ↓
11. real-device intervention-interface smoke
        ↓
12. C6-B scientific contract design
```

The mapping-materialization stage and intervention-interface stage are conceptually distinct engineering prerequisites.

Their serial ordering here is a project workflow choice, not a claim that the interface architecture mathematically depends on the materialized CCA mappings.

---

## 10. Documentation Update After Final PASS

After final C6-A contract approval and freeze:

- update `AGENTS.md`;
- update `docs/research-map.md`;

to record C6-A as completed at the **interface-closure** level.

Do not describe C6-A as establishing policy/action relevance.

`docs/pilot-v0.2-spec.md` does not require modification for this closure.

---

## 10.1 Freeze Hygiene Record

Repository-level freeze hygiene has been verified with `git diff --check`.

This is a formatting / repository-hygiene requirement only and does not change the scientific contract.

## 11. Final Read-Only Review Record

The final read-only review was completed against:

- the completed C6-A audit;
- the current `shared-feature-tex3d` source;
- the audited local OpenVLA source;
- the audited local OpenPI source;
- the audited Tex3D rollout/action wrappers;
- LIBERO and the currently available controller evidence.

The review confirmed that:

1. the primary action comparison is now internally consistent;
2. the definitions of `a_open` and `A_pi` correctly refer to deployed, checkpoint-unnormalized action quantities;
3. the contract no longer claims recovery of exact historical C5-B matrices;
4. the authorized re-fit / new authoritative mapping distinction is scientifically and technically correct;
5. the minimum mapping scope
   `O2 ↔ P2 / 99%-PCA / true unpermuted TRAIN fit`
   is sufficient for the current C6 mainline;
6. the intentionally excluded mapping configurations are safe to leave outside the required artifact;
7. the required consistency-validation checks compare only historically persisted quantities and do not claim element-wise validation of the unsaved historical `sigma` vector;
8. the new authoritative mapping artifact correctly persists full `sigma`, its ordering, dtype/shape, and content hash;
9. sign canonicalization is correctly treated as a mapping/provenance issue rather than a native intervention-vector decision;
10. the workflow contains the intervention-interface implementation stage before smoke testing;
11. the interface coding stage is correctly O2/P2 node-specific while remaining direction- and metric-agnostic;
12. no remaining prerequisite blocks C6-A interface closure.

### Verdict

```text
C6-A CONTRACT REVIEW: PASS
```

This review did not:

- redesign C6-B;
- choose a final policy-sensitivity metric;
- define the native intervention vector;
- choose token scope;
- choose \(\epsilon\);
- implement mapping materialization;
- implement intervention interfaces;
- design the Tex3D attack loss.

---

*Frozen after the C6-A source audit, contract reviews, and human scientific review.*
