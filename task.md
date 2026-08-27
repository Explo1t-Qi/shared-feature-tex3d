# C5-B Explicit Shared-Space Alignment — Implementation Contract

**Status:** DRAFT v2.1 — FINAL READ-ONLY RE-AUDIT REQUIRED BEFORE IMPLEMENTATION

This document defines the frozen scientific and implementation contract for **C5-B only**.

It does **not** authorize implementation or formal execution by itself.

After this document is synchronized with repository governance state, Codex must perform a final read-only consistency audit. Only after that audit passes may C5-B implementation be explicitly authorized.

---

## 1. Scientific Question

C5-A has already established:

> Under the frozen Pilot v0.2 protocol, correctly paired OpenVLA and π0.5 observations induce non-randomly consistent global observation-level representation geometry.

C5-B asks the stronger question:

> Can paired OpenVLA and π0.5 representations be mapped, using TRAIN-only learned linear mappings, into an explicit candidate shared coordinate space whose canonical alignment generalizes to HELD-OUT observations beyond a frozen shuffled-null baseline?

C5-B is therefore an:

```text
explicit linear shared-space alignment test
```

A positive C5-B result supports:

```text
shared geometry
+
explicit held-out-generalizing linear alignment
```

It does **not** establish:

```text
policy relevance
action relevance
adversarial vulnerability
transferability
attack-time usefulness
```

The hierarchy remains:

```text
shared != transferable != policy-relevant
```

---

## 2. Upstream Preconditions

C5-B assumes the following upstream stages are complete and immutable:

```text
C5-D0 Formal Collection              PASS
C2 OpenVLA Feature Extraction        PASS
C3 π0.5 Feature Extraction           PASS
C4 Formal Paired Features            PASS
C5-A Representation Geometry         GO
```

C5-A is closed.

Formal C5-A primary result:

```text
O2 ↔ P2
TRAIN debiased Linear CKA     = 0.528420895275
HELD-OUT debiased Linear CKA  = 0.478168155531
TRAIN empirical p             = 1/51
HELD-OUT empirical p          = 1/51
C5-A gate                     = GO
```

C5-B must not modify, reinterpret, recompute, or replace the formal C5-A conclusion.

---

## 3. Authoritative Inputs

C5-B has **two required upstream inputs**.

### 3.1 Formal C4 paired manifest

Required CLI argument:

```text
--paired-manifest
```

Required schema:

```text
paired_features_v1
```

Required count:

```text
200 paired observations
```

This is the authoritative paired-feature identity input.

### 3.2 Formal C5-A output directory

Required CLI argument:

```text
--c5a-output-dir
```

The directory must contain the exact formal C5-A four-file artifact set:

```text
split_manifest.json
metric_summary.json
null_metrics.npz
summary.md
```

C5-B reads and semantically validates:

```text
split_manifest.json
metric_summary.json
```

The presence of:

```text
null_metrics.npz
summary.md
```

is validated only as part of formal C5-A artifact-set completeness.

No additional files are allowed in the formal C5-A output directory.

C5-B must validate:

```text
C5-A split schema == c5a_split_manifest_v1
C5-A metric schema == c5a_metric_summary_v1
C5-A run_status == COMPLETED
C5-A c5a_gate == GO
C5-A source_paired_manifest resolves to the same formal C4 manifest
C5-A split counts match 50 groups / 200 observations
C5-A split contains 40 TRAIN groups / 160 observations
C5-A split contains 10 HELD-OUT groups / 40 observations
```

C5-B must not proceed if any of these checks fail.

---

## 4. Authoritative Split Provenance

Pilot v0.2 defines the serialized C5 split manifest as the authoritative materialization of the frozen split rule.

Therefore:

> C5-B must inherit the formally materialized C5-A split.

C5-B must **not** create an independent scientific split authority from C4 alone.

It may independently recompute the deterministic split rule only as a **validation check**.

The recomputed split must match the C5-A serialized split exactly.

Mismatch is a validation error.

Frozen statistical group:

```text
(task_id, initial_state_id)
```

Frozen split rule:

```text
pilot-v0.2-c5-split-v1|task_id={task_id}|initial_state_id={initial_state_id}
```

Digest:

```text
SHA-256 over exact UTF-8 bytes
```

Within each task:

```text
sort by (digest, initial_state_id) ascending
lowest-ranked group -> HELD-OUT
remaining four -> TRAIN
```

Regression-only expected HELD-OUT state IDs:

```text
task 0 -> 3
task 1 -> 2
task 2 -> 1
task 3 -> 1
task 4 -> 4
task 5 -> 0
task 6 -> 3
task 7 -> 5
task 8 -> 7
task 9 -> 0
```

These expected IDs are not the rule.

---

## 5. Formal Archive Provenance Validation

For every paired observation, C5-B must revalidate the frozen C2/C3/C4 provenance semantics.

At minimum validate:

```text
sample_id
source_image_hash
paired sample_id equality
paired source_image_hash equality
archive existence
required feature keys
required feature shapes
saved dtype == float32
all required feature values finite
expected source_model
expected feature_schema_version
expected checkpoint/config identity
```

Required model identities:

```text
OpenVLA logical checkpoint:
openvla/openvla-7b-finetuned-libero-spatial

π0.5 config:
pi05_libero

π0.5 checkpoint:
gs://openpi-assets/checkpoints/pi05_libero
```

Required nodes:

```text
Primary:
O2 ↔ P2

Control:
O1-S ↔ P1
```

Required raw shapes per observation:

```text
O2   [256, 4096]
P2   [256, 2048]

O1-S [256, 1152]
P1   [256, 1152]
```

No upstream artifact may be modified.

---

## 6. Statistical Unit and Computational Rows

The statistical unit remains the Pilot observation / trajectory-group structure.

Token rows are:

```text
computational repeated measurements
```

and must **not** be treated as independent statistical observations.

For each node:

```text
[N, 256, D]
```

C5-B uses position-aligned token-wise rows.

TRAIN:

```text
[160, 256, D]
->
[40960, D]
```

HELD-OUT:

```text
[40, 256, D]
->
[10240, D]
```

Row correspondence:

```text
(i, t)_OpenVLA <-> (i, t)_π0.5
```

where:

```text
i = PilotObservation
t = native spatial token index
```

The 256 token positions inherit the frozen 16×16 native spatial-token order.

Token-index correspondence is an inherited positional anchor only.

It must not be interpreted as proof of pure local semantic equivalence.

Primary C5-B forbids:

```text
token reordering
nearest-neighbor token matching
learned token matching
token selection
token weighting
token pooling
```

Full token-channel flattening is forbidden:

```text
[N, 256D]
```

---

## 7. Numeric Precision

Saved upstream arrays remain:

```text
float32
```

After loading, all formal C5-B:

```text
centering
PCA/SVD
CCA
projection
correlation
null calculation
summary statistics
```

must use:

```text
float64
```

Any non-finite intermediate or final formal value is a validation error.

---

## 8. PCA — Exact Frozen Definition

C5-B uses model-specific TRAIN-only PCA.

For each node independently:

```text
1. construct TRAIN token-row matrix X
2. convert to float64
3. compute feature-wise TRAIN mean μ
4. center:
   Xc = X - μ
5. compute one economy SVD using exactly:
```

```python
U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
V = Vh.T
```

This NumPy API choice and `full_matrices=False` are frozen.

The mathematical identity is:

```text
Xc = U diag(S) V^T
```

No z-score, feature standardization, or L2 normalization is allowed.

### 8.1 PCA explained variance

Per-component variance weight is defined exactly as:

```text
e_j = S_j^2
```

The common multiplicative factor `1/(n-1)` is omitted because it cancels in the explained-variance ratio.

Total variance weight:

```text
E = sum_j S_j^2
```

If:

```text
E <= 0
```

or non-finite, validation fails.

For cutoff `tau`, define:

```text
d_tau =
minimum k such that
sum_{j=1..k} S_j^2 / E >= tau
```

Frozen cutoffs:

```text
primary:
tau = 0.99

robustness:
tau = 0.95
```

### 8.2 One SVD per node

Each node must compute its full TRAIN PCA/SVD exactly once.

The 95% and 99% configurations use prefixes of the **same ordered PCA basis**:

```text
V_95 = V[:, :d_95]
V_99 = V[:, :d_99]
```

No separate PCA refit is allowed for 95% vs 99%.

### 8.3 Numerical rank

Let:

```text
s_max = max(S)
eps = np.finfo(np.float64).eps
tol = max(Xc.shape) * eps * s_max
```

A singular value is numerically valid iff:

```text
S_j > tol
```

The cutoff-specific retained PCA dimension must not include numerically invalid components.

If the minimum explained-variance cutoff would require a numerically invalid component, validation fails.

If a cutoff-specific retained dimension is less than 10 on either side of a pair, validation fails.

### 8.4 HELD-OUT transform

HELD-OUT uses only frozen TRAIN parameters:

```text
μ
V[:, :d_tau]
```

Transform:

```text
Z_heldout = (X_heldout - μ) @ V[:, :d_tau]
```

No HELD-OUT refit or recentering is allowed.

---

## 9. Ordinary Linear CCA — Exact Frozen Definition

C5-B uses deterministic ordinary linear CCA implemented by covariance whitening plus SVD.

No:

```text
ridge
shrinkage
regularization
kernelization
Deep CCA
iterative neural optimization
```

is permitted.

For a cutoff-specific TRAIN pair:

```text
Z_A in R^(n × d_A)
Z_B in R^(n × d_B)
```

where both matrices are already derived from TRAIN-centered PCA projections.

Define:

```text
C_AA = (Z_A^T Z_A) / (n - 1)
C_BB = (Z_B^T Z_B) / (n - 1)
C_AB = (Z_A^T Z_B) / (n - 1)
```

### 9.1 Covariance eigendecomposition

Use exactly:

```python
lambda_A, Q_A = np.linalg.eigh(C_AA)
lambda_B, Q_B = np.linalg.eigh(C_BB)
```

`np.linalg.eigh` returns eigenvalues in ascending order.

Explicitly reorder each side into descending order using:

```python
order_A = np.argsort(lambda_A)[::-1]
lambda_A = lambda_A[order_A]
Q_A = Q_A[:, order_A]

order_B = np.argsort(lambda_B)[::-1]
lambda_B = lambda_B[order_B]
Q_B = Q_B[:, order_B]
```

The mathematical identities are:

```text
C_AA = Q_A diag(lambda_A) Q_A^T
C_BB = Q_B diag(lambda_B) Q_B^T
```

No alternate eigensolver is part of the formal contract.

For each side define covariance tolerance:

```text
lambda_tol =
max(d, 1) * eps * lambda_max
```

where:

```text
eps = np.finfo(np.float64).eps
```

A covariance eigenvalue is valid iff:

```text
lambda > lambda_tol
```

No tolerance-based ridge is allowed.

Invalid/zero directions are removed from the whitening basis.

If fewer than 10 valid covariance directions remain on either side, validation fails.

Whitening matrices:

```text
P_A =
Q_A_valid @ diag(lambda_A_valid^(-1/2))

P_B =
Q_B_valid @ diag(lambda_B_valid^(-1/2))
```

### 9.2 Whitened cross-covariance

Construct:

```text
M = P_A^T @ C_AB @ P_B
```

Compute exactly:

```python
U, sigma, Vh = np.linalg.svd(M, full_matrices=False)
V = Vh.T
```

The mathematical identity is:

```text
M = U diag(sigma) V^T
```

`np.linalg.svd` returns singular values in non-increasing order; this order is the frozen TRAIN canonical-component order.

Singular values are interpreted as TRAIN canonical correlations.

Canonical mappings:

```text
W_A = P_A @ U
W_B = P_B @ V
```

Canonical variables:

```text
H_A = Z_A @ W_A
H_B = Z_B @ W_B
```

### 9.3 Canonical-dimension count

The valid canonical dimension count is:

```text
k = min(number of valid A whitening directions,
        number of valid B whitening directions,
        len(sigma))
```

Formal execution requires:

```text
k >= 10
```

### 9.4 Repeated canonical singular values

If exact or numerically repeated canonical singular values occur, use the deterministic component order returned by the frozen SVD implementation.

No secondary data-dependent reordering or tie-breaking is allowed.

---

## 10. Candidate Shared-Space Mapping Semantics

The learned mappings are:

```text
candidate shared-space mappings
```

They must not be called:

```text
transferable mappings
policy-relevant mappings
attack mappings
```

CCA fitting is TRAIN-only.

HELD-OUT uses frozen TRAIN:

```text
PCA mean
PCA basis
CCA mappings
canonical component order
sign orientation
```

---

## 11. Held-Out Canonical Evaluation

For each TRAIN-ordered canonical component `j`, compute HELD-OUT Pearson correlation by the exact direct formula.

For vectors:

```text
x = H_A_heldout[:, j]
y = H_B_heldout[:, j]
```

define:

```text
xc = x - mean(x)
yc = y - mean(y)
```

and:

```text
rho_j_heldout =
dot(xc, yc) / (norm(xc) * norm(yc))
```

All operations use float64.

If either centered-vector norm is:

```text
<= 0
```

or non-finite, validation fails.

Do not call an alternate Pearson implementation whose edge-case behavior may differ.

No HELD-OUT:

```text
CCA refit
component reorder
best-component selection
absolute value
sign flipping
```

is allowed.

Correlation must be finite.

Constant canonical variates causing undefined Pearson correlation are validation errors.

---

## 12. Exact Evaluation Statistics

All component indices below refer to frozen TRAIN CCA order.

Define exactly:

```text
Top1 =
rho_1

Top5Mean =
mean(rho_1, rho_2, rho_3, rho_4, rho_5)

Top10Mean =
mean(rho_1, ..., rho_10)
```

Primary statistic:

```text
HELD-OUT Top5Mean
```

Supporting metrics:

```text
HELD-OUT Top1
HELD-OUT Top10Mean
```

Diagnostics:

```text
TRAIN Top5Mean
TRAIN→HELD-OUT gap
```

Define TRAIN Top5Mean exactly from the TRAIN CCA singular values:

```text
TRAIN Top5Mean =
mean(sigma_1, sigma_2, sigma_3, sigma_4, sigma_5)
```

Do not recompute TRAIN Pearson correlations from canonical variables.

Gap:

```text
TRAIN Top5Mean - HELD-OUT Top5Mean
```

Top1, Top10Mean, TRAIN Top5Mean, and the gap never determine the formal gate.

---

## 13. Primary and Control Pairs

Formal primary pair:

```text
O2 ↔ P2
```

Formal C5-B PASS/FAIL is determined only by:

```text
O2 ↔ P2
99% PCA
HELD-OUT Top5Mean
```

Control pair:

```text
O1-S ↔ P1
```

The control pair must compute the same:

```text
99% PCA analysis
95% PCA robustness
Top1
Top5Mean
Top10Mean
TRAIN Top5Mean
TRAIN→HELD-OUT gap
null summaries
```

The control pair never gates C5-B.

---

## 14. Primary C5-B Null Question

The formal null asks:

> If cross-model correspondence is deliberately wrong, how strong an apparent HELD-OUT shared alignment can ordinary linear CCA produce from its fitting capacity?

Therefore the null must:

```text
break TRAIN pairing
refit CCA from scratch
evaluate on independently broken HELD-OUT pairing
```

It is insufficient to fit CCA on the correct TRAIN pairing and shuffle only HELD-OUT.

---

## 15. PCA Behavior Under Null

PCA is pairing-independent.

Therefore:

```text
PCA is fit once per node on the true TRAIN row set
```

and reused for:

```text
true analysis
all 200 null repeats
99% cutoff
95% cutoff
```

The only cutoff-specific difference is the prefix length of the same PCA basis.

No null-repeat PCA refit is allowed.

---

## 16. Group-Block Fit-and-Evaluate Null

Null unit:

```text
(task_id, initial_state_id)
```

Within a moved group preserve:

```text
all four progress slots together
all 256 token positions in native order
```

For repeat `r`:

### 16.1 TRAIN null

Keep OpenVLA fixed.

Derange π0.5 TRAIN groups:

```text
g_i^A <-> g_{pi_train_r(i)}^B
```

with:

```text
pi_train_r(i) != i
```

for every TRAIN group.

Preserve:

```text
0.10 <-> 0.10
0.40 <-> 0.40
0.70 <-> 0.70
0.90 <-> 0.90

token 0 <-> token 0
...
token 255 <-> token 255
```

Using this wrong TRAIN pairing:

```text
refit ordinary linear CCA from scratch
```

to obtain repeat-specific null mappings.

### 16.2 HELD-OUT null

Independently derange π0.5 HELD-OUT groups.

Keep OpenVLA HELD-OUT fixed.

Apply:

```text
frozen TRAIN PCA
repeat-specific null CCA mappings
```

Then compute HELD-OUT:

```text
Top1
Top5Mean
Top10Mean
```

using frozen TRAIN canonical order and sign orientation.

---

## 17. HELD-OUT Null Limitation

HELD-OUT contains one group per task.

Therefore a fixed-point-free HELD-OUT derangement necessarily permits cross-task mismatch.

The HELD-OUT null is:

```text
a broad cross-group mismatch null
```

not:

```text
a task-conditioned state-level mismatch null
```

Formal summaries must surface this limitation.

C5-B must not be interpreted as proving:

> same-state shared alignment after conditioning on fixed task identity.

---

## 18. Null Repeats

Formal repeats:

```text
R = 200
```

Empirical p-value:

```text
p_emp =
(1 + # {null >= true}) / 201
```

Minimum possible p:

```text
1 / 201
```

Formal threshold:

```text
p_emp <= 0.05
```

Therefore at most 9 null values may satisfy:

```text
null >= true
```

for formal PASS.

---

## 19. C5-B RNG — Exact Frozen Convention

C5-B root seed:

```text
17
```

Exact initialization:

```python
root = np.random.SeedSequence(17)

train_seed, heldout_seed = root.spawn(2)

train_rng = np.random.Generator(
    np.random.PCG64(train_seed)
)

heldout_rng = np.random.Generator(
    np.random.PCG64(heldout_seed)
)
```

Within each split, group order is defined by first appearance in the canonical C4 paired-manifest sequence and must match the inherited C5-A split materialization.

For each repeat:

```python
perm = rng.permutation(n_groups)
```

Repeatedly draw until the first fixed-point-free permutation is obtained.

Accept that first derangement.

Forbidden:

```text
Sattolo
forced single-cycle construction
hand-generated derangements
```

Accepted permutation indices must be stored.

---

## 20. One Shared Permutation Bank for All Configurations

The same exact:

```text
200 TRAIN derangements
200 HELD-OUT derangements
```

must be reused across **all four configurations**:

```text
O2 ↔ P2, 99%
O2 ↔ P2, 95%
O1-S ↔ P1, 99%
O1-S ↔ P1, 95%
```

This isolates:

```text
pair choice
PCA cutoff
```

from random-null variation.

---

## 21. Null Metric Semantics

For every configuration and every null repeat, formal stored null metrics are HELD-OUT only:

```text
Top1
Top5Mean
Top10Mean
```

TRAIN null metrics are not part of the formal artifact.

For each HELD-OUT metric report:

```text
true
null mean
null std
null median
null q95
true - null median
empirical p
```

Use exactly:

```python
np.std(null_values, ddof=0)
```

and:

```python
np.quantile(null_values, 0.95, method="linear")
```

Empirical p-values are required for all three HELD-OUT metrics.

Only Top5Mean determines the formal C5-B gate.

---

## 22. C5-B Primary PASS / FAIL Rule

Formal primary configuration:

```text
O2 ↔ P2
99% TRAIN-only PCA
ordinary linear CCA
```

Formal primary statistic:

```text
TRAIN-order HELD-OUT Top5Mean
```

C5-B PASS iff:

```text
true HELD-OUT Top5Mean > 0
AND
empirical p <= 0.05
```

against the frozen fit-and-evaluate group-block null.

No absolute correlation threshold is part of the gate.

The following are non-gating:

```text
Top1
Top10Mean
TRAIN Top5Mean
TRAIN→HELD-OUT gap
95% robustness
O1-S↔P1 control
```

Formal code must not classify alignment as:

```text
strong
weak
```

using any unstated threshold.

Every formal result must instead include the fixed interpretation note:

> Statistical significance relative to the frozen null does not by itself imply strong practical alignment; raw HELD-OUT Top5Mean and null effect-size summaries must be interpreted directly.

---

## 23. 95% PCA Robustness

The 95% configuration is required.

It uses:

```text
same TRAIN/HELD-OUT split
same token-row semantics
same node-specific full PCA SVD
same permutation bank
same CCA definition
same metrics
```

It does not independently gate C5-B.

Qualitative interpretation only:

```text
99% and 95% both similar:
alignment is robust to PCA cutoff

99% stronger than 95%:
alignment may depend on lower-variance retained directions

95% stronger than 99%:
99% retained tail may introduce noise or conditioning burden
```

Do not switch the primary cutoff based on observed results.

---

## 24. C5-B Scientific Interpretation

If C5-B PASS:

> The frozen C5-B analysis provides evidence that TRAIN-only learned linear mappings produce an explicit candidate shared coordinate system whose primary canonical alignment generalizes to HELD-OUT paired observations beyond the frozen fit-and-evaluate mismatch null.

If C5-B FAIL:

> The frozen C5-B analysis does not provide sufficient evidence that the current linear SVCCA formulation yields a stable explicit shared coordinate system that generalizes to HELD-OUT observations.

C5-B FAIL does not prove absence of:

```text
nonlinear shared structure
local/token-specific shared structure
alternative alignment formulations
policy-relevant shared structure
```

---

## 25. C5 Representation-Stage Joint Gate — Frozen

This gate is **not** the final overall research conclusion.

It is only the representation-analysis joint gate.

Define:

```text
C5 representation-stage PASS
iff
C5-A GO
AND
C5-B PASS
```

Since C5-A is already frozen as:

```text
GO
```

the remaining representation-stage condition is C5-B.

If representation-stage PASS:

> The shared-representation hypothesis is supported under frozen Pilot v0.2 at the level of both reproducible global observation geometry and an explicit HELD-OUT-generalizing linear alignment.

This still does **not** establish:

```text
policy relevance
action relevance
adversarial transferability
attack effectiveness
```

If:

```text
C5-A GO
C5-B FAIL
```

then:

> Global representation geometry is supported, but the current linear SVCCA formulation does not provide sufficient evidence for a stable explicit shared coordinate system.

### 25.1 Final overall research gate

The final overall research PASS/FAIL beyond representation analysis remains:

```text
OPEN / NOT DEFINED HERE
```

C5-B must not define or imply a final overall project success criterion.

---

## 26. Formal CLI

Formal CLI:

```bash
python scripts/c5b_explicit_shared_space.py \
  --paired-manifest <paired_features_manifest.json> \
  --c5a-output-dir <formal_c5a_output_dir> \
  --output-dir <new-or-empty-output-dir>
```

No formal scientific constant may be user-selectable.

Do not expose CLI arguments for:

```text
seed
R
PCA cutoff
pair
metric
threshold
CCA regularization
```

The output directory must:

```text
not exist
```

or:

```text
exist and be empty
```

Non-empty output is rejected.

No resume/merge behavior.

---

## 27. Formal Output Set — Exact

Successful formal execution must produce **exactly four files**:

```text
split_manifest.json
alignment_summary.json
null_alignment_metrics.npz
summary.md
```

No additional formal files are allowed.

---

## 28. split_manifest.json — Exact Schema

Schema version:

```text
c5b_split_manifest_v1
```

Exact top-level keys:

```text
schema_version
source_paired_manifest
source_c5a_split_manifest
split_rule_id
counts
heldout_state_by_task
canonical_group_order
groups
```

Path fields:

```text
source_paired_manifest
source_c5a_split_manifest
```

must be absolute resolved filesystem paths recorded as runtime provenance.

`groups` must preserve canonical C4/C5-A order and include exact keys:

```text
canonical_group_index
task_id
initial_state_id
assignment
sample_ids
step_ids
inherited_progress_slots
```

C5-B split output must exactly reproduce the validated C5-A split assignments.

---

## 29. alignment_summary.json — Exact Schema

Schema version:

```text
c5b_alignment_summary_v1
```

Exact top-level keys:

```text
schema_version
run_status
source_paired_manifest
source_c5a_metric_summary
source_c5a_split_manifest
c5a_gate
primary_pair
control_pair
pca
cca
null
results
c5b_result
representation_stage_result
heldout_null_limitation
interpretation_boundary
```

Allowed values:

```text
run_status:
COMPLETED

c5a_gate:
GO

c5b_result:
PASS | FAIL

representation_stage_result:
PASS | FAIL
```

All three source path fields:

```text
source_paired_manifest
source_c5a_metric_summary
source_c5a_split_manifest
```

must be absolute resolved filesystem paths.

`primary_pair` exact keys and exact values:

```text
name          = "o2_p2"
openvla_node  = "O2"
pi05_node     = "P2"
gates_c5b     = true
```

`control_pair` exact keys and exact values:

```text
name          = "o1s_p1"
openvla_node  = "O1-S"
pi05_node     = "P1"
gates_c5b     = false
```

`pca` must contain exactly these node keys:

```text
o2
p2
o1s
p1
```

Each node object must contain exactly:

```text
full_rank
d_95
d_99
explained_variance_95
explained_variance_99
```

Definitions:

```text
full_rank =
count(S_j > tol)

explained_variance_95 =
actual cumulative sum(S[:d_95]^2) / sum(S^2)

explained_variance_99 =
actual cumulative sum(S[:d_99]^2) / sum(S^2)
```

The explained-variance fields must contain the actual achieved ratios and must not be hard-coded to `0.95` or `0.99`.

`cca` must contain exactly:

```text
method
covariance_denominator
eigendecomposition_api
svd_api
regularization
min_valid_canonical_dims
```

with exact values:

```text
method =
"ordinary_linear_covariance_whitening_svd"

covariance_denominator =
"n_minus_1"

eigendecomposition_api =
"numpy.linalg.eigh"

svd_api =
"numpy.linalg.svd_full_matrices_false"

regularization =
"none"

min_valid_canonical_dims =
10
```

`null` must contain exactly:

```text
repeats
root_seed
rng
sampler
train_group_count
heldout_group_count
shared_bank_across_configurations
```

with exact values:

```text
repeats = 200
root_seed = 17
rng = "SeedSequence.spawn(2)+PCG64"
sampler = "first_fixed_point_free_rng_permutation"
train_group_count = 40
heldout_group_count = 10
shared_bank_across_configurations = true
```

`heldout_null_limitation` must be the exact string:

```text
"HELD-OUT contains one group per task; the fixed-point-free HELD-OUT derangement is therefore a broad cross-group mismatch null, not a task-conditioned state-level mismatch null."
```

`interpretation_boundary` must be the exact string:

```text
"Statistical significance relative to the frozen null does not by itself imply strong practical alignment; C5 representation-stage PASS does not establish policy relevance, action relevance, adversarial transferability, or attack effectiveness."
```

`results` must contain exactly four configuration entries:

```text
o2_p2__99
o2_p2__95
o1s_p1__99
o1s_p1__95
```

Each configuration must contain:

```text
train_top5mean
heldout_top1
heldout_top5mean
heldout_top10mean
train_to_heldout_top5mean_gap
null_summary
```

`null_summary` must contain exactly:

```text
top1
top5mean
top10mean
```

Each metric summary must contain:

```text
true
null_mean
null_std
null_median
null_q95
true_minus_null_median
empirical_p
```

---

## 30. null_alignment_metrics.npz — Exact Keys

The NPZ must contain exactly:

```text
train_permutations
heldout_permutations

o2_p2__99__heldout_top1
o2_p2__99__heldout_top5mean
o2_p2__99__heldout_top10mean

o2_p2__95__heldout_top1
o2_p2__95__heldout_top5mean
o2_p2__95__heldout_top10mean

o1s_p1__99__heldout_top1
o1s_p1__99__heldout_top5mean
o1s_p1__99__heldout_top10mean

o1s_p1__95__heldout_top1
o1s_p1__95__heldout_top5mean
o1s_p1__95__heldout_top10mean
```

Exact dtypes/shapes:

```text
train_permutations:
int64 [200, 40]

heldout_permutations:
int64 [200, 10]

all metric arrays:
float64 [200]
```

No extra arrays are allowed.

---

## 31. summary.md — Required Content

`summary.md` must contain:

```text
C5-B PASS/FAIL
C5 representation-stage PASS/FAIL
primary O2↔P2 99% true/null summary
95% robustness summary
control-pair summary
actual retained PCA dimensions
TRAIN→HELD-OUT diagnostic gap
held-out null limitation
fixed interpretation boundary
```

It must explicitly state:

> Statistical significance relative to the frozen null does not by itself imply strong practical alignment.

It must also state:

> C5 representation-stage PASS does not establish policy relevance, action relevance, adversarial transferability, or attack effectiveness.

---

## 32. Output Safety

Use staged output publication.

A failed run must not leave a final output directory that appears formally complete.

Input artifacts must remain immutable.

No partially written:

```text
PASS
FAIL
COMPLETED
```

formal artifact may remain after failure.

Successful publication must produce the exact four-file set and exact schemas above.

---

## 33. Memory / Compute Discipline

This section is an implementation constraint, not a scientific change.

Formal code should process one representation pair at a time:

```text
1. O2 ↔ P2
2. release unnecessary pair-specific arrays/workspace
3. O1-S ↔ P1
```

Do not require all four full float64 token-feature tensors to remain resident simultaneously.

Each node's full TRAIN PCA/SVD is computed once.

The implementation should reuse:

```text
fixed PCA transforms
cutoff-specific covariance/whitening structure where mathematically valid
shared null permutation banks
```

Do not implement the null as hundreds of full iterative CCA training jobs.

Use the frozen closed-form linear-algebra CCA.

---

## 34. Required Validation

Implementation must validate at least:

```text
formal C4 paired schema
exact 200 pairs
exact C5-A four-file artifact set
C5-A split schema
C5-A metric schema
C5-A run_status == COMPLETED
C5-A gate == GO
C5-A and C5-B source C4 identity match
C5-A serialized split matches recomputed frozen rule
canonical sample order
sample IDs
source hashes
archive paths
model identities
feature schema versions
required keys
required shapes
saved float32 dtype
finiteness
50 groups
5 groups/task
4 observations/group
contiguous group ordering
strictly increasing step IDs
position-aligned token-row construction
PCA total variance
PCA numerical rank
95%/99% prefix relationship
CCA covariance numerical rank
>=10 valid canonical dimensions
finite canonical correlations
finite null arrays
fixed-point-free derangements
RNG reproducibility
same permutation bank across all four configurations
exact output dtypes
exact output schemas
exact output file set
```

---

## 35. Required Unit Tests

Tests must be CPU-only and synthetic.

At minimum test:

```text
C4 provenance rejection
C5-A artifact provenance rejection
C5-A incomplete/extra-file artifact-set rejection
C5-A non-GO rejection
C5-A source-manifest mismatch rejection
C5-A split mismatch rejection
group reconstruction
exact SHA-256 split validation
position-aligned token flattening
TRAIN-only centering
single full PCA SVD per node
99% minimum-prefix selection
95% minimum-prefix selection
95% prefix contained in 99% prefix
zero-variance PCA rejection
PCA numerical-rank rejection
HELD-OUT transform without refit
ordinary covariance-whitening CCA
CCA rank-deficiency rejection
>=10 canonical-dimension requirement
TRAIN component order preserved on HELD-OUT
no abs(correlation)
Top1 exact definition
Top5Mean exact definition
Top10Mean exact definition
TRAIN→HELD-OUT gap
TRAIN group-block movement
HELD-OUT group-block movement
progress-slot preservation
token-position preservation
CCA refit for every null repeat
PCA not refit per null repeat
R=200
SeedSequence(17).spawn(2)
PCG64 deterministic replay
derangement rejection sampling
same bank reused by all four configurations
empirical p-value
ddof=0 null std
linear q95
supporting metric p-values produced
primary gate uses only O2↔P2 99% Top5Mean
control never gates
95% never gates
representation-stage result = C5-A GO AND C5-B PASS
no strong/weak automatic classification
non-empty output rejection
failed publication leaves no formal completed output
exact four-file output set
exact JSON top-level schemas
exact NPZ keys/dtypes/shapes
input artifacts remain unchanged
```

---

## 36. Required Checks After Implementation

Run:

```text
focused C5-B tests
relevant C4 regression tests
relevant C5-A regression tests
full unit suite if practical
Ruff check
Ruff format --check
git diff --check
```

Inspect final diff for scope creep.

---

## 37. Explicitly Out of Scope

C5-B must not implement:

```text
policy/action relevance
action gradients
Jacobian analysis
shared-policy direction selection
adversarial loss
Tex3D texture optimization
held-out attack evaluation
Deep CCA
Kernel CCA
Ridge CCA
PWCCA as a formal gate
token matching
token weighting
learned alignment networks
```

Do not automatically begin later roadmap stages after C5-B.

---

## 38. Formal Execution Boundary

Implementation and formal execution are separate authorizations.

After coding:

```text
1. run CPU unit/regression/static checks
2. stop
3. report implementation
4. perform independent read-only code audit
5. only then authorize formal C5-B execution
```

Formal C5-B execution must not begin during implementation authorization unless explicitly requested.

---

## 39. Documentation Synchronization Required Before Coding

Before implementation authorization, Codex may make the minimal state-only synchronization edits required in:

```text
AGENTS.md
docs/research-map.md
```

No separate scientific redesign is required for those edits.

They must reflect that the following are now frozen for C5-B:

```text
R = 200
root seed = 17
exact TRAIN/HELD-OUT derangement convention
fit-and-evaluate null
C5-B PASS/FAIL
C5 representation-stage joint gate
```

The documents must also explicitly preserve:

```text
final overall research gate:
OPEN / NOT DEFINED HERE
```

Do not modify:

```text
docs/pilot-v0.2-spec.md
```

unless a genuine scientific contradiction is discovered.

---

## 40. Final Read-Only Re-Audit Required

Before coding authorization, Codex must perform a read-only audit of:

```text
task.md
AGENTS.md
docs/research-map.md
docs/pilot-v0.2-spec.md
existing C2/C3/C4/C5-A implementation interfaces
formal C5-A artifact schemas
```

The audit must verify:

```text
1. C5-B scientific question is consistent
2. C5-A remains closed and unchanged
3. C5-A formal GO is explicitly validated from artifacts
4. C5-B inherits authoritative C5-A split materialization
5. token rows are computational repeated measurements
6. PCA exact SVD/cutoff/rank semantics are unambiguous
7. one full TRAIN PCA/SVD is computed per node
8. 95% and 99% are prefixes of the same PCA basis
9. ordinary CCA whitening/SVD definition is unambiguous
10. no ridge/regularization is introduced
11. held-out has no refit/reorder/abs/sign repair
12. Top1/Top5Mean/Top10Mean definitions are exact
13. null refits CCA after TRAIN derangement
14. PCA is not refit under null
15. TRAIN/HELD-OUT derangements are independent
16. all four configurations reuse the same permutation banks
17. R=200 and seed=17 are consistent across documents
18. only O2↔P2 99% Top5Mean gates C5-B
19. representation-stage joint gate is distinct from final overall research gate
20. output schemas and NPZ keys are exact
21. no policy/action/transferability conclusion is implied
22. no upstream artifact or scientific protocol is modified
```

If any conflict remains:

```text
STOP
```

and report exact locations.

Do not resolve a scientific inconsistency by silently choosing an implementation assumption.

---

## 41. Current Status

```text
C5-A Representation Geometry:
FORMAL GO / CLOSED

C5-B Scientific Contract:
DRAFT v2.1 — FINAL READ-ONLY RE-AUDIT REQUIRED

C5-B Implementation:
NOT AUTHORIZED

C5-B Formal Execution:
NOT AUTHORIZED

C5 Representation-Stage Joint Gate:
FROZEN
PASS iff C5-A GO AND C5-B PASS

Final Overall Research Gate:
OPEN / NOT DEFINED HERE

Policy/Action Relevance:
NOT STARTED

Transferability / Tex3D Optimization:
NOT STARTED
```
