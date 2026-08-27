# C5-A Representation Geometry Test Contract — Pilot v0.2

Status:

```text
DRAFT v3 — FINAL READ-ONLY RE-AUDIT REQUIRED BEFORE IMPLEMENTATION
```

This contract defines the formal C5-A representation-geometry test for the frozen Pilot v0.2 paired-feature dataset.

It authorizes only:

```text
C5-A — cross-model representation geometry testing
```

It does not authorize C5-B SVCCA/CCA, explicit shared-space fitting, policy/action relevance analysis, adversarial loss construction, or Tex3D optimization.

---

## 1. Scientific Question

C5-A tests:

> Do paired observations induce non-randomly consistent representation geometry across OpenVLA and π0.5?

C5-A is only an existence-level geometry test.

A positive result supports proceeding to a stronger shared-space alignment test.

It does not establish:

```text
shared == transferable
shared == policy-relevant
shared == action-relevant
```

and it does not prove that an explicit shared coordinate system already exists.

---

## 2. Authoritative Input

The only valid C5-A scientific input is the completed formal C4 paired manifest:

```text
schema_version = "paired_features_v1"
num_samples = 200
```

Runtime CLI:

```text
--paired-manifest
--output-dir
```

C5-A must consume the exact 200 canonical paired records in the order serialized by the C4 manifest.

Do not reconstruct pairing by scanning feature directories.

Do not replace, supplement, filter, reorder, or cherry-pick paired observations.

Each paired record must resolve both:

```text
openvla_feature_path
pi05_feature_path
```

and the existing paired `source_image_hash` integrity semantics must remain valid.

### 2.1 Formal C4 identity validation

Because `paired_features_v1` alone does not encode the full Pilot v0.2 / C2 / C3 manifest identity, C5-A must re-validate the formal feature archive provenance when loading each paired sample.

This is provenance verification only.

C5-A must reuse the existing frozen C2/C3/C4 archive/schema semantics or validators where practical and must not redefine provenance rules.

For every OpenVLA archive, validate at minimum:

```text
logical checkpoint identity =
openvla/openvla-7b-finetuned-libero-spatial

expected source_model
expected feature_schema_version
sample_id
source_image_hash
required array keys
exact required shapes
saved dtype = float32
all required arrays finite
```

For every π0.5 archive, validate at minimum:

```text
checkpoint identity =
gs://openpi-assets/checkpoints/pi05_libero

expected source_model
expected feature_schema_version
sample_id
source_image_hash
required array keys
exact required shapes
saved dtype = float32
all required arrays finite
```

Cross-model validation must also confirm:

```text
paired sample_id matches both archives
paired source_image_hash matches both archives
```

Do not treat an arbitrary or historical `paired_features_v1` manifest as formal Pilot v0.2 input merely because it has 200 records.

If any required provenance or archive validation fails, stop before metric computation.

---

## 3. Output Directory Admission Rule

The path passed through:

```text
--output-dir
```

must satisfy exactly one of:

```text
path does not exist
```

or:

```text
path exists and is empty
```

Any non-empty existing output directory must be rejected.

Do not merge with, resume into, or partially reuse a previous formal C5-A output directory.

---

## 4. Frozen Representation Pairs

### Primary pair

```text
O2 ↔ P2
```

Archive mapping:

```text
OpenVLA O2  -> o2_projected   [256, 4096]
π0.5 P2     -> p2_projected   [256, 2048]
```

This is the only representation pair that determines formal C5-A Go / No-Go.

### Control pair

```text
O1-S ↔ P1
```

Archive mapping:

```text
OpenVLA O1-S -> o1_siglip   [256, 1152]
π0.5 P1      -> p1_siglip   [256, 1152]
```

For the control pair, compute all three C5-A metrics:

```text
debiased Linear CKA
biased Linear CKA
Spearman RSA
```

The control pair does not determine the formal gate.

### Supplementary OpenVLA node

```text
O1-F -> o1_fused [256, 2176]
```

O1-F remains supplementary and does not determine the formal gate.

Do not redefine node semantics from shape alone.

---

## 5. Statistical Unit

The statistical unit is:

```text
one PilotObservation
```

not one visual token.

Do not flatten the dataset into:

```text
200 × 256 token samples
```

and do not treat within-observation tokens as independent statistical observations.

This avoids pseudo-replication from strongly dependent tokens belonging to the same image / rollout state.

---

## 6. Per-Observation Representation

For every frozen node tensor:

```text
[256, D]
```

construct one observation vector using simple arithmetic mean pooling over all 256 tokens:

\[
\bar{x}
=
\frac{1}{256}
\sum_{t=1}^{256} x_t
\]

Thus:

```text
O2:    [256,4096] -> [4096]
P2:    [256,2048] -> [2048]
O1-S:  [256,1152] -> [1152]
P1:    [256,1152] -> [1152]
```

No other pooling variant is part of the formal C5-A protocol.

Do not perform:

```text
token selection
token weighting
attention pooling
max pooling
token reordering
z-score normalization
L2 normalization
PCA
CCA
SVCCA
dimensionality reduction
```

before the formal C5-A metrics.

Mean pooling intentionally tests global observation-level geometry and is not interpreted as token-level spatial correspondence.

---

## 7. Canonical Group Reconstruction and Progress-Slot Inheritance

The group identity is:

```text
(task_id, initial_state_id)
```

The full C4 dataset must reconstruct as:

```text
10 tasks
5 accepted groups per task
50 groups total
4 observations per group
200 observations total
```

The canonical group order for all later C5-A operations is:

> order of first appearance of each group in the canonical C4 paired-manifest record sequence.

### 7.1 Required group validation

For every reconstructed group, validate:

```text
exactly 4 records
the 4 group records are contiguous in canonical C4 order
step_id values are strictly increasing
```

### 7.2 Frozen progress-slot assignment

The C4 paired manifest does not contain an explicit `target_relative_progress` field.

Therefore C5-A must not claim to independently recover or prove the original target progress from C4 fields alone.

Instead, after the group validation above succeeds, inherit the already validated Pilot v0.2 / C4 canonical provenance as follows:

```text
1st record in the group's canonical C4 block -> slot 0.10
2nd record in the group's canonical C4 block -> slot 0.40
3rd record in the group's canonical C4 block -> slot 0.70
4th record in the group's canonical C4 block -> slot 0.90
```

This is:

```text
inherited slot assignment from validated upstream provenance
```

not:

```text
independent reconstruction of target_relative_progress from C4 fields
```

Do not infer another slot order from raw `step_id` spacing or relative magnitude.

---

## 8. Deterministic Dataset Split

The split unit is one full rollout group:

```text
(task_id, initial_state_id)
```

All four observations from one group must remain in the same split.

For each group, serialize exactly:

```text
pilot-v0.2-c5-split-v1|task_id={task_id}|initial_state_id={initial_state_id}
```

Compute SHA-256 of the UTF-8 encoded string.

Within each task, sort the five groups by:

```text
(sha256_hex_digest, initial_state_id)
```

in ascending lexicographic / numeric tuple order.

Freeze:

```text
smallest ranked group -> HELD-OUT
remaining four        -> TRAIN
```

Expected totals:

```text
TRAIN:
40 groups
160 observations

HELD-OUT:
10 groups
40 observations
```

Every task must contribute exactly:

```text
4 TRAIN groups
1 HELD-OUT group
```

Do not use random split assignment.

Concrete held-out state IDs in the current frozen dataset may be used as regression expectations, but they are derived outputs, not scientific rules.

The scientific rule is the SHA-256 ranking above.

---

## 9. Primary Similarity Estimator: Debiased Linear CKA

The formal primary estimator is:

```text
Debiased Linear CKA
```

C5-A operates in a low-sample / high-dimensional regime:

```text
N << D
```

so the primary estimator must use the frozen unbiased HSIC formulation below.

### 9.1 Internal dtype

All C5-A metric calculations must use:

```text
float64
```

internally after loading the saved float32 features.

Do not silently calculate formal metrics in float32.

### 9.2 Linear Gram matrices

For a mean-pooled representation matrix:

```text
X ∈ R^(n × d_x)
Y ∈ R^(n × d_y)
```

construct:

\[
K = XX^\top
\]

\[
L = YY^\top
\]

using the raw mean-pooled vectors, with no extra feature normalization.

### 9.3 Frozen unbiased HSIC formula

Let:

\[
\tilde{K}
\]

and:

\[
\tilde{L}
\]

be copies of `K` and `L` with their diagonals set exactly to zero.

For:

```text
n >= 4
```

define:

\[
\mathrm{HSIC}_u(K,L)
=
\frac{1}{n(n-3)}
\left[
\mathrm{tr}(\tilde{K}\tilde{L})
+
\frac{
(\mathbf{1}^{\top}\tilde{K}\mathbf{1})
(\mathbf{1}^{\top}\tilde{L}\mathbf{1})
}{
(n-1)(n-2)
}
-
\frac{2}{n-2}
\mathbf{1}^{\top}\tilde{K}\tilde{L}\mathbf{1}
\right]
\]

Use this exact formula for:

```text
HSIC_u(K,L)
HSIC_u(K,K)
HSIC_u(L,L)
```

Do not substitute another U-statistic, minibatch, centered-kernel, or library-specific HSIC variant.

### 9.4 Frozen debiased CKA formula

Define:

\[
\mathrm{CKA}_u
=
\frac{
\mathrm{HSIC}_u(K,L)
}{
\sqrt{
\mathrm{HSIC}_u(K,K)
\mathrm{HSIC}_u(L,L)
}
}
\]

Behavior:

- preserve negative `HSIC_u(K,L)` values;
- preserve negative finite debiased CKA values;
- do not clip results to `[0,1]`;
- if either self-HSIC term is non-positive, stop with a validation error;
- if the denominator is non-positive or non-finite, stop with a validation error;
- if the final result is non-finite, stop with a validation error.

Apply this exact estimator identically to:

```text
true pairing
all shuffled-null pairings
TRAIN
HELD-OUT
primary pair
control pair
```

---

## 10. Diagnostic Metric: Standard Biased Linear CKA

Also compute standard biased Linear CKA as:

```text
diagnostic / supporting only
```

It must not determine Go / No-Go.

Use centered mean-pooled representation matrices:

\[
X_c = X - \mathrm{mean}(X,\text{axis}=0)
\]

\[
Y_c = Y - \mathrm{mean}(Y,\text{axis}=0)
\]

and:

\[
\mathrm{CKA}_{biased}
=
\frac{
\|X_c^\top Y_c\|_F^2
}{
\|X_c^\top X_c\|_F
\|Y_c^\top Y_c\|_F
}
\]

Use float64.

Non-finite values or a non-positive denominator are validation errors.

Do not use biased CKA for the formal gate.

---

## 11. Robustness Metric: Spearman RSA

Compute one robustness metric:

```text
Spearman Representational Similarity Analysis (RSA)
```

### 11.1 Frozen RSA convention

For each split and representation pair:

1. use the same raw mean-pooled observation vectors used by CKA;
2. compute all pairwise squared Euclidean distances between observations within each model representation;
3. extract the strict upper triangle only;
4. exclude the diagonal;
5. compare the two upper-triangular vectors with:

```python
scipy.stats.spearmanr
```

The Spearman statistic is the RSA value.

Larger RSA means stronger agreement of pairwise geometry.

### 11.2 Tie and validation behavior

Use the standard SciPy Spearman rank convention, including average ranks for ties.

Treat the following as validation errors:

```text
constant distance vector
non-finite distance vector
non-finite Spearman statistic
```

Do not switch between similarity and dissimilarity conventions across true/null runs.

Apply the identical squared-Euclidean-distance convention to:

```text
true pairing
all shuffled-null pairings
TRAIN
HELD-OUT
primary pair
control pair
```

RSA is robustness evidence only and does not determine the formal gate.

---

## 12. Group-Block Shuffled Null

The null hypothesis asks whether correct cross-model observation correspondence produces greater geometry similarity than deliberately incorrect group correspondence.

The permutation unit is:

```text
(task_id, initial_state_id)
```

The four progress-slot observations in one group move together as one block.

Within every moved group, preserve inherited slot correspondence:

```text
0.10 -> 0.10
0.40 -> 0.40
0.70 -> 0.70
0.90 -> 0.90
```

Do not shuffle individual observations independently.

Do not shuffle tokens.

For every null repeat:

```text
OpenVLA group order remains fixed
π0.5 group blocks are permuted
```

---

## 13. Frozen C5-A Null RNG and Derangement Sampling

Freeze specifically for C5-A:

```text
C5-A null repeats = 50
C5-A root RNG seed = 7
```

Use NumPy's random API as follows:

```python
root = np.random.SeedSequence(7)
train_seed, heldout_seed = root.spawn(2)

train_rng = np.random.Generator(np.random.PCG64(train_seed))
heldout_rng = np.random.Generator(np.random.PCG64(heldout_seed))
```

This creates independent deterministic RNG streams for TRAIN and HELD-OUT.

### 13.1 Group order

Within each split, enumerate groups in the canonical group order defined by first appearance in the C4 paired-manifest sequence.

### 13.2 Derangement sampler

For each repeat and each split:

```python
perm = rng.permutation(n_groups)
```

Repeatedly draw until:

```text
perm[i] != i
```

for every group index `i`.

Accept the first permutation with no fixed points.

This rejection-sampling procedure is the frozen C5-A derangement sampler.

Do not use Sattolo's algorithm.

Do not force the permutation to be one cycle.

Store the accepted permutation indices for reproducibility.

### 13.3 Split independence

TRAIN and HELD-OUT must use their separate spawned generators.

Do not consume one shared RNG stream for both splits.

Do not generate one 50-group permutation and then split it.

### 13.4 Scope boundary

These RNG decisions are frozen only for C5-A.

They do not freeze:

```text
C5-B RNG
SVCCA null protocol
final joint C5 RNG convention
```

Those remain outside this contract.

---

## 14. HELD-OUT Null Limitation

HELD-OUT contains exactly one group per task.

Therefore a same-task wrong-state permutation is impossible within HELD-OUT.

The HELD-OUT null necessarily permits cross-task group mismatches.

It must therefore be interpreted as:

> a broad cross-group mismatch null

and not as:

> a task-conditioned state-level null.

Do not change the frozen Pilot v0.2 split merely to remove this limitation.

---

## 15. Metric Computation Under Null

For every accepted null permutation, construct the π0.5 observation sequence by moving whole group blocks according to the permutation while preserving the four inherited within-group slots.

For each null repeat, compute:

```text
debiased Linear CKA
biased Linear CKA
Spearman RSA
```

for:

```text
O2 ↔ P2
O1-S ↔ P1
```

on TRAIN and HELD-OUT independently.

Use the exact same preprocessing and metric conventions as the true pairing.

---

## 16. Primary C5-A Go / No-Go Statistic

For the primary pair:

```text
O2 ↔ P2
```

and primary estimator:

```text
Debiased Linear CKA
```

compute separately for TRAIN and HELD-OUT:

```text
one true value
50 shuffled-null values
```

Use the one-sided empirical permutation p-value:

\[
p_{\mathrm{emp}}
=
\frac{
1+
\sum_{r=1}^{R}
\mathbf{1}
[
m_{\mathrm{null}}^{(r)}
\ge
m_{\mathrm{true}}
]
}{
R+1
}
\]

with:

```text
R = 50
```

The inequality direction is frozen as:

```text
null >= true
```

because larger debiased CKA indicates stronger geometry agreement.

The smallest possible empirical p-value is:

```text
1 / 51 ≈ 0.0196078
```

---

## 17. C5-A Split-Level PASS Rule

For each split independently:

```text
PASS(split)
```

iff both conditions hold:

```text
true debiased Linear CKA > 0
```

and:

```text
p_emp <= 0.05
```

With 50 null repeats, this means at most one null value may be greater than or equal to the true value.

---

## 18. Formal C5-A Gate

Freeze only the geometry-stage gate:

```text
C5-A GO
iff
TRAIN PASS
AND
HELD-OUT PASS
```

Only the primary pair O2 ↔ P2 and primary estimator debiased Linear CKA determine this C5-A gate.

Possible C5-A outcomes:

```text
TRAIN PASS + HELD-OUT PASS
-> C5-A GO

otherwise
-> C5-A NO-GO
```

A C5-A NO-GO means only:

> the frozen C5-A analysis did not provide sufficient evidence for global observation-level shared geometry.

It must not be interpreted as proof that no token-level, local, nonlinear, action-conditioned, or policy-relevant shared structure exists.

A C5-A NO-GO does not authorize silently changing metrics, pooling, split, null construction, or thresholds.

### 18.1 Explicit stage boundary

This contract does not freeze:

```text
C5-B / SVCCA PASS rule
C5-B RNG
C5-B null protocol
final CKA + SVCCA joint C5 gate
final overall C5 PASS/FAIL rule
```

These remain:

```text
OPEN / NOT STARTED
```

until separately discussed and frozen.

---

## 19. Required Null Summaries

For every metric / pair / split null distribution, report:

```text
null mean
null standard deviation
null median
null 95th percentile
```

Freeze:

```python
null_std = np.std(null_values, ddof=0)
null_q95 = np.quantile(null_values, 0.95, method="linear")
```

Also report:

```text
true metric
true - null median
empirical p-value
```

The empirical p-value convention for RSA and diagnostic biased CKA is also:

```text
null >= true
```

but these supporting p-values do not determine the formal C5-A gate.

Do not introduce arbitrary absolute CKA thresholds such as 0.2, 0.3, or 0.5.

---

## 20. Interpretation of Control and Robustness Results

### O1-S ↔ P1

This pair is a shared-backbone control.

Examples:

```text
O1-S↔P1 strong
O2↔P2 strong
```

supports the interpretation that shared geometry survives VLA adaptation.

```text
O1-S↔P1 strong
O2↔P2 weak
```

supports the interpretation that geometry may be shared mainly at the visual-backbone level and diverge after VLA adaptation.

The control does not override the primary C5-A gate.

### RSA

If RSA agrees with debiased CKA:

```text
true >> null
```

this strengthens cross-metric robustness.

If RSA is weak while primary CKA passes, record the result as metric-sensitive evidence.

Do not convert RSA into a second mandatory gate without contract revision.

### Biased CKA

Use biased CKA only to diagnose the practical effect of finite-sample/high-dimensional bias.

It must not override debiased CKA.

---

## 21. Formal Output Layout

A successful formal C5-A run must create exactly the following core outputs inside `--output-dir`:

```text
split_manifest.json
metric_summary.json
null_metrics.npz
summary.md
```

Additional temporary files must not remain after successful completion.

### 21.1 `split_manifest.json`

Must record at minimum:

```text
schema/version identifier
source paired-manifest path
split rule identifier
canonical group order
per-group task_id
per-group initial_state_id
SHA-256 split digest
TRAIN / HELD-OUT assignment
canonical four sample_ids per group
inherited progress-slot assignment
held-out group per task
```

### 21.2 `metric_summary.json`

Must record at minimum:

```text
analysis schema/version
primary pair
control pair
metric conventions
C5-A RNG convention
null repeat count
TRAIN true metrics
HELD-OUT true metrics
null summaries
empirical p-values
split PASS statuses
formal C5-A GO / NO-GO
held-out-null limitation text
```

### 21.3 `null_metrics.npz`

Must contain sufficient machine-readable data to reproduce all null summaries, including at minimum:

```text
accepted TRAIN permutation indices
accepted HELD-OUT permutation indices
all 50 null metric values
for each metric / pair / split
```

Freeze saved dtypes:

```text
permutation index arrays = int64
metric arrays            = float64
```

### 21.4 `summary.md`

Must provide a concise human-readable report of:

```text
data counts
split
primary true/null results
control results
RSA robustness
biased-vs-debiased diagnostic
formal C5-A gate
held-out null limitation
```

Do not reinterpret a formal C5-A NO-GO as proof of no shared representation of any kind.

---

## 22. Output Safety

Do not overwrite an existing non-empty output directory.

If a run fails before all validation and outputs are complete:

```text
no final metric_summary.json claiming completion
no final summary.md claiming completion
```

The implementation may use temporary files/directories for safe-write behavior.

Do not modify:

```text
C2 feature archives
C3 feature archives
C4 paired manifest
PilotObservation files
```

---

## 23. Implementation Scope

Preferred new implementation files:

```text
scripts/c5a_representation_geometry.py
tests/test_c5a_representation_geometry.py
```

A small helper module may be added only if it materially improves testability.

Prefer not to modify:

```text
shared_feature/openvla_features.py
shared_feature/pi05_features.py
shared_feature/paired_features.py
scripts/c2_full_feature_extraction.py
scripts/c3_full_feature_extraction.py
scripts/c4_full_paired_features.py
```

If implementing the frozen estimator, archive provenance verification, or formal C4 loading requires a production-module change, stop and report the blocker before expanding scope.

Do not run model inference.

---

## 24. Required Tests

Tests must be CPU-only.

Do not run OpenVLA, π0.5, LIBERO, CUDA, or JAX workloads.

Test at least:

1. exact C4 paired-manifest loading and 200-pair validation;
2. formal OpenVLA checkpoint identity validation;
3. formal π0.5 checkpoint identity validation;
4. source_model and feature_schema_version validation;
5. paired/archive `sample_id` agreement;
6. paired/archive `source_image_hash` agreement;
7. required array keys/shapes/float32/finiteness validation;
8. deterministic group reconstruction;
9. every group has exactly four records;
10. group records are contiguous in canonical C4 order;
11. group `step_id` values are strictly increasing;
12. progress slots are inherited as canonical record positions `[0.10,0.40,0.70,0.90]`;
13. canonical group order equals first appearance in C4 order;
14. exact SHA-256 split serialization;
15. split sorting uses `(digest, initial_state_id)`;
16. one held-out group per task;
17. totals of 40 TRAIN / 10 HELD-OUT groups;
18. totals of 160 TRAIN / 40 HELD-OUT observations;
19. all four observations from a group remain together;
20. mean pooling maps `[256,D] -> [D]`;
21. no forbidden preprocessing is applied;
22. exact unbiased HSIC formula on controlled synthetic matrices;
23. debiased Linear CKA on controlled synthetic examples;
24. negative finite debiased CKA is preserved;
25. non-positive self-HSIC / denominator is rejected;
26. standard biased Linear CKA diagnostic path;
27. squared-Euclidean Spearman RSA on controlled synthetic examples;
28. RSA constant/non-finite inputs are rejected;
29. `SeedSequence(7).spawn(2)` convention is exact;
30. PCG64 TRAIN and HELD-OUT streams are deterministic and independent;
31. every accepted null permutation is a derangement;
32. rejection sampler does not impose one-cycle structure;
33. OpenVLA order remains fixed while π0.5 group blocks move;
34. progress-slot order is preserved inside moved groups;
35. exactly 50 accepted null permutations per split;
36. empirical p-value uses the frozen `+1` formula and `null >= true`;
37. null standard deviation uses `ddof=0`;
38. null q95 uses `method="linear"`;
39. control pair computes all three metrics;
40. split-level PASS rule is exact;
41. formal C5-A GO requires both TRAIN and HELD-OUT PASS;
42. `null_metrics.npz` permutation arrays are int64;
43. `null_metrics.npz` metric arrays are float64;
44. non-empty output directory is rejected;
45. existing inputs are never modified;
46. failed run cannot leave final outputs claiming completion.

Where possible, use synthetic examples with known expected high similarity, low similarity, and shuffled correspondence behavior.

---

## 25. Required Checks

After implementation run at minimum:

```bash
python -m pytest \
  tests/test_c5a_representation_geometry.py \
  -q
```

Also run relevant C4 regression tests and the full test suite if practical.

Run:

```text
Ruff check
Ruff format check
git diff --check
```

C5-A is CPU analysis over already extracted feature artifacts and must not trigger real model inference.

---

## 26. Documentation Synchronization Required Before Coding

The following decisions are now frozen specifically for C5-A:

```text
C5-A primary geometry contract
C5-A primary estimator
C5-A supporting metrics
C5-A null RNG seed / exact RNG convention
C5-A geometry-stage Go / No-Go rule
```

Before implementation authorization, synchronize:

```text
AGENTS.md
docs/research-map.md
```

The synchronized documents must explicitly distinguish C5-A from later C5 stages.

Required status:

```text
C5-A geometry contract:
FROZEN

C5-A null RNG:
FROZEN = 7

C5-A geometry-stage GO/NO-GO:
FROZEN

C5-B / SVCCA contract:
NOT STARTED

C5-B RNG:
OPEN

C5-B PASS/FAIL:
OPEN

final CKA + SVCCA joint C5 gate:
OPEN

final overall C5 PASS/FAIL rule:
OPEN
```

The synchronized documents must make clear that:

> this independent C5-A scientific contract is the frozen authority only for C5-A decisions.

Documentation synchronization must not change the scientific rules in this contract.

---

## 27. Explicitly Out of Scope

Do not implement or discuss as part of this coding task:

```text
SVCCA
CCA fitting
PCA for shared-space alignment
Procrustes alignment
PWCCA
GULP
kNN overlap
action relevance
policy relevance
shared-feature attack loss
Tex3D optimization
transfer attack evaluation
final joint C5 decision rule
```

Those belong to later stages.

---

## 28. Final Read-Only Re-Audit Before Coding

Before implementation, Codex must perform one final read-only re-audit against:

```text
AGENTS.md
docs/research-map.md
docs/pilot-v0.2-spec.md
current task.md
formal C4 paired manifest
current feature archive schemas
existing C2/C3/C4 implementation and tests
```

The re-audit must confirm:

1. documentation no longer conflicts with the frozen C5-A RNG or C5-A geometry-stage gate;
2. documentation still leaves C5-B and final joint C5 gate OPEN / NOT STARTED;
3. formal C4 identity is sufficiently revalidated without modifying the C4 schema;
4. OpenVLA and π0.5 checkpoint identities are exact;
5. source_model / feature schema / sample / hash / array validation is unambiguous;
6. progress-slot semantics are explicitly inherited rather than independently recovered;
7. the exact unbiased HSIC formula is unambiguous;
8. debiased CKA denominator/error behavior is unambiguous;
9. float64 internal computation is feasible;
10. the squared-Euclidean Spearman RSA convention is unambiguous;
11. the deterministic split can be reconstructed exactly;
12. canonical group reconstruction is unambiguous;
13. the `SeedSequence(7).spawn(2)` + PCG64 rejection-sampled derangement sequence is implementable exactly;
14. the held-out null limitation is preserved in reporting;
15. the C5-A Go / No-Go rule is unambiguous;
16. fixed output artifacts and dtypes are sufficient for reproducibility;
17. non-empty output directories are safely rejected;
18. no C5-B or downstream scientific semantics are introduced.

During this re-audit:

```text
DO NOT MODIFY CODE
DO NOT RUN FORMAL C5-A
```

If a concrete statistical or artifact incompatibility remains, stop and report it before implementation.

---

## 29. Current Gate

Current state:

```text
C5-D0 Formal Collection              PASS
C2 OpenVLA Full Feature Extraction   PASS
C3 π0.5 Full Feature Extraction      PASS
C4 Formal Paired Feature             PASS
C5-A Representation Geometry         DRAFT v3 — FINAL READ-ONLY RE-AUDIT REQUIRED
C5-B Explicit Shared-Space Alignment NOT STARTED
Final joint C5 gate                   OPEN
```

After documentation synchronization and a PASS final read-only re-audit:

```text
C5-A implementation may be separately authorized.
```

After formal C5-A execution:

```text
C5-A GO
-> proceed to C5-B explicit shared-space alignment discussion

C5-A NO-GO
-> stop and reassess the shared-representation hypothesis under the frozen C5-A result
```

No statement about final overall C5 PASS/FAIL is authorized by this contract.
