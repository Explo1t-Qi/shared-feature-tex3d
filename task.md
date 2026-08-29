# C5-BM — Authoritative Mapping Materialization Contract

## Contract Status

```text
C5-BM CONTRACT REVIEW: PASS
C5-BM scientific / engineering contract: FROZEN
C5-BM implementation: NOT AUTHORIZED
C5-BM formal materialization: NOT AUTHORIZED
```

This contract defines the minimum scientific, numerical, provenance, and engineering requirements for materializing a new authoritative reusable PCA+CCA mapping artifact for the C6 mainline.

It does **not** authorize C6-B policy-sensitivity analysis, intervention-vector design, token intervention, or Tex3D optimization.

---

## 1. Purpose

C5-B established that OpenVLA O2 and π0.5 P2 admit a statistically non-random explicit shared space under TRAIN-only PCA + ordinary linear CCA.

However, the historical formal C5-B artifacts did **not** persist the fitted mapping objects required to reuse that shared coordinate system directly.

The purpose of C5-BM is therefore:

> perform one explicitly authorized re-fit under the frozen primary C5-B scientific configuration, validate that the re-fit reproduces the historically persisted C5-B scientific results, and publish a new versioned authoritative mapping artifact for all future C6 work.

C5-BM is an **engineering / provenance prerequisite**.

It is **not** a new representation-similarity experiment and does not change the historical C5-B PASS conclusion.

---

## 2. Historical Limitation

The historical C5-B run created, in memory, objects including:

- PCA TRAIN means;
- PCA bases;
- CCA whitening transforms;
- CCA mappings \(W_A\) and \(W_B\);
- canonical correlations \(\sigma\).

Those objects were not fully persisted as formal reusable artifacts.

Therefore:

- the exact historical in-memory matrices cannot be recovered or verified element-by-element;
- the project must not claim that the new materialized matrices are identical to the historical unsaved matrices;
- the historical scientific conclusion remains valid because the formal C5-B scalar/statistical results were already frozen.

The new C5-BM artifact is a **new authoritative reusable mapping**, not a recovered historical memory image.

---

## 3. Authorized Materialization Scope

The required C5-BM scope is intentionally minimal.

### Required configuration

```text
representation pair:
O2 ↔ P2

PCA:
99% explained variance

fit data:
TRAIN only

pairing:
true, unpermuted pairing

estimator:
the frozen C5-B PCA + ordinary linear CCA implementation
```

This configuration becomes the authoritative shared-coordinate reference for the C6 mainline.

### Intentionally excluded

The following are not required for this materialization:

```text
O2 ↔ P2 @95% PCA
O1-S ↔ P1 @99% PCA
O1-S ↔ P1 @95% PCA
all null / permuted fits
```

These configurations remain valid C5 robustness/control evidence, but they are intentionally excluded from the minimum C6 mapping artifact.

If future work needs them, they require a separately declared materialization scope.

---

## 4. Frozen Inputs and Configuration Identity

The implementation must use the same frozen scientific inputs/configuration as the primary historical C5-B fit.

At minimum, the contract must identify and verify:

- formal C4 paired manifest;
- frozen TRAIN / HELD-OUT split manifest;
- canonical observation ordering;
- authoritative token-row ordering defined procedurally as:
  - iterate observations in frozen split / C4-C5-B record order;
  - for each observation, consume the stored NPZ token rows in original row order `0..255`;
  - flatten observation-major, then token-major;
  - perform no token sorting, matching, spatial remapping, or reordering;
- O2 node identity;
- P2 node identity;
- PCA cutoff = `0.99`;
- true, unpermuted TRAIN pairing;
- current frozen C5-B estimator implementation;
- float64 fitting semantics where used by the existing estimator.

Any mismatch in required discrete/configuration identity is a blocking error.

### 4.1 Historical C5-B formal result as an explicit input

C5-BM must also take the historical formal C5-B output set as an explicit provenance/validation input.

The required historical formal files are:

```text
split_manifest.json
alignment_summary.json
null_alignment_metrics.npz
summary.md
```

Requirements:

- all four required files must exist;
- each required file must satisfy the expected historical schema/role;
- `alignment_summary.json` must report the completed historical C5-B run, including `run_status == COMPLETED` and `c5b_result == PASS`;
- the historical source paired-manifest identity recorded by C5-B must be consistent with the frozen formal input used by C5-BM;
- SHA-256 over exact file bytes must be computed for all four required historical files before materialization;
- the hashes must be recomputed after materialization and must remain identical;
- the new C5-BM metadata/validation must persist these historical file hashes, including the exact hash of `alignment_summary.json` used as the source of historical scalar metrics.

Additional unrelated files in the historical directory must not be interpreted as C5-BM inputs unless explicitly declared.

### 4.2 Portable identity and hash semantics

Use the following identity rules:

- JSON / NPZ / source feature archive / historical artifact file identity:
  `SHA-256` over exact file bytes;
- serialized format:
  `sha256:<64 lowercase hex>`;
- array content hash:
  `SHA-256` over contiguous array bytes, with dtype and shape stored separately;
- paired-manifest and split content hashes are the authoritative portable identities;
- resolved absolute filesystem paths are runtime provenance only and must not serve as portable scientific identity.

Before fitting, reuse the existing C2/C3/C4 validation logic to validate the complete formal paired input set and all 200 paired source feature archives.

Do not create a new incompatible C5-BM-specific feature-validation semantics when the existing validators already cover the required checks.

---

## 5. Authorized Re-Fit Semantics

Once implementation and formal materialization are separately authorized, C5-BM
authorizes exactly one new fit under the following frozen semantics.

The implementation must:

1. load the frozen formal inputs;
2. reproduce the primary C5-B O2↔P2 99%-PCA TRAIN fit;
3. compute the PCA and CCA mapping objects;
4. apply deterministic canonical ordering and sign canonicalization;
5. validate historically persisted scalar results;
6. publish the new artifact only if all required validation passes.

This re-fit must be recorded explicitly as a new materialization event.

It must not be described as:

```text
recovery of the exact historical C5-B mapping
```

---

## 6. Canonical Component Ordering

The fitted canonical components must preserve the canonical ordering defined by the CCA estimator.

The saved artifact must guarantee that, for each component index \(k\):

```text
W_A[:, k]
W_B[:, k]
sigma[k]
```

refer to the same canonical component.

The full fitted canonical-correlation vector:

\[
\sigma = [\sigma_1,\ldots,\sigma_K]
\]

must be persisted.

---

## 7. Deterministic Sign Canonicalization

CCA has a pairwise sign ambiguity:

\[
(d_A,d_B)
\equiv
(-d_A,-d_B).
\]

Because future C6 work may refer to `+d_k` and `-d_k`, C5-BM must impose a deterministic sign convention.

### Frozen sign rule

For each canonical component \(k\), compute the native-feature-space linear readout vectors:

\[
q_{A,k}=V_AW_A[:,k],
\]

\[
q_{B,k}=V_BW_B[:,k].
\]

Concatenate them in fixed order:

```text
q_k = concat(q_A,k, q_B,k)
```

using:

```text
OpenVLA O2 coordinates first
π0.5 P2 coordinates second
```

Before selecting the sign anchor, require:

- `q_k` is finite float64;
- `max(abs(q_k)) > 0`.

If `q_k` contains any non-finite value or is all-zero, C5-BM is **BLOCKED**.

Find:

\[
j^*=\arg\max_j |q_k[j]|.
\]

If multiple entries have exactly the same maximum absolute value, choose the **smallest concatenated index**.

Then:

- if \(q_k[j^*] > 0\), keep the pair unchanged;
- if \(q_k[j^*] < 0\), multiply both \(W_A[:,k]\) and \(W_B[:,k]\) by \(-1\).

Only paired sign flips are allowed: `W_A[:,k]` and `W_B[:,k]` must never be sign-flipped independently.

The corresponding \(\sigma_k\) is unchanged.

### Important boundary

This use of \(VW\) is **only a sign-canonicalization rule**.

It does **not** imply that \(VW[:,k]\) is the future native-space intervention/synthesis vector.

That scientific decision remains deferred to C6-B.

### Limitation

Sign canonicalization resolves only the ± ambiguity.

It does not guarantee recovery across independent re-fits in degenerate or near-degenerate canonical subspaces where component rotation/permutation may occur.

This does not block C6 because the authoritative mapping is fitted once, validated, frozen, and reused thereafter.

---

## 8. Required Mapping Artifact Contents

The new authoritative artifact must persist the complete forward shared-space mapping.

### 8.1 Core arrays

At minimum:

- OpenVLA PCA TRAIN mean \(\mu_A\);
- π0.5 PCA TRAIN mean \(\mu_B\);
- OpenVLA PCA basis \(V_A\);
- π0.5 PCA basis \(V_B\);
- any CCA whitening/projection transforms required by the existing estimator;
- CCA mapping \(W_A\);
- CCA mapping \(W_B\);
- full canonical-correlation vector `sigma`.

### 8.2 Mapping identity

Persist:

- retained PCA dimensions;
- PCA cutoff;
- canonical component count;
- canonical component ordering;
- sign-canonicalization rule/version;
- confirmation that `W_A[:,k]`, `W_B[:,k]`, and `sigma[k]` share the same order.

### 8.3 Array metadata and integrity

For every persisted array, store:

- array name;
- shape;
- dtype;
- content hash.

This includes `sigma`.

### 8.4 Provenance

Persist at least:

- artifact schema/version;
- materialization/run identifier;
- source paired-manifest identity/hash;
- TRAIN/HELD-OUT split identity/hash;
- source feature artifact identities/hashes;
- node identities;
- repository/code commit;
- Python version;
- NumPy version;
- BLAS/LAPACK implementation/version where available;
- platform information;
- SHA-256 identities of all four required historical C5-B formal files;
- hash of the exact `alignment_summary.json` used for historical scalar validation;
- validation result for the complete 200-pair source feature set.

---

## 9. Required Published Artifact Set

The authoritative published artifact must contain the following four logical files:

```text
c5bm_mapping/
├── mapping.npz
├── metadata.json
├── validation.json
└── summary.md
```

### `mapping.npz`

Expected to contain the fitted arrays, for example:

```text
mean_a
mean_b
basis_a
basis_b
whitening_a
whitening_b
w_a
w_b
sigma
```

Exact key names may follow the current codebase naming style, but the scientific content above is required.

### `metadata.json`

Contains:

- schema/version;
- fit configuration;
- provenance;
- array shapes/dtypes/hashes;
- canonical ordering;
- sign convention.

### `validation.json`

Contains:

- exact-match checks;
- historical scalar consistency checks;
- tolerances;
- PASS/BLOCKED result.

### `summary.md`

Human-readable publication summary.

Do not add speculative C6-B intervention artifacts.

---

## 9.1 Transactional publication semantics

Publication of the authoritative C5-BM artifact must be transactional.

Required behavior:

- the target formal output directory must not exist or must be empty;
- existing formal outputs must never be overwritten;
- all frozen-input validation, historical-file hashing, fitting, sign canonicalization, and consistency validation must complete before publication;
- all four required output files must be written and then reloaded/validated successfully before the artifact is considered published;
- after publication, the target formal output directory must contain exactly the four required output files and no additional files;
- any existing non-empty target directory must be rejected, regardless of whether it already contains an authoritative-publication marker;
- a BLOCKED run, exception, or interrupted run must not leave a directory that can be mistaken for a complete authoritative publication;
- C2/C3/C4/C5-A/C5-B input artifacts must remain unchanged.

A sibling staging directory followed by atomic rename is the recommended implementation pattern, but the scientific contract freezes the transactional behavior rather than one specific filesystem mechanism.

---

## 10. Consistency Validation

The new fit must be validated only against quantities that were actually persisted by historical C5-B.

### 10.1 Exact-match quantities

The following must match exactly:

- representation pair;
- PCA cutoff;
- TRAIN/HELD-OUT split identity;
- paired-manifest identity;
- observation ordering;
- authoritative token-row ordering procedure and the resulting observation-major/token-major row order;
- retained PCA dimensions;
- other required discrete configuration metadata.

### 10.2 Floating-point scalar validation

Historical persisted floating-point scalar metrics must satisfy:

\[
|m_{\text{new}}-m_{\text{historical}}|
\le 10^{-8}.
\]

Use **absolute tolerance = \(10^{-8}\)**.

At minimum validate:

- TRAIN Top5Mean;
- HELD-OUT Top1;
- HELD-OUT Top5Mean;
- HELD-OUT Top10Mean.

Where applicable:

\[
\text{TRAIN Top5Mean}
=
\frac{1}{5}\sum_{k=1}^{5}\sigma_k
\]

from the new fit must match the historical persisted TRAIN Top5Mean within the frozen tolerance.

HELD-OUT Top1 / Top5Mean / Top10Mean recomputed using the new fitted mapping must match the corresponding historical persisted metrics within the same tolerance.

### 10.3 Full sigma

The newly fitted full `sigma` vector must be saved.

However, the project must **not** claim element-wise validation of the full new `sigma` vector against the historical run because the historical full `sigma` vector was not persisted.

### 10.4 Validation failure

If any required validation fails:

```text
C5-BM = BLOCKED
```

The authoritative mapping must **not** be published.

The implementation must investigate the cause.

Do not respond to failure by silently:

- widening the tolerance;
- changing inputs;
- changing ordering;
- changing estimator behavior;
- rewriting historical artifacts.

Any scientific/configuration change requires a new explicit decision.

---

## 11. Historical Artifact Immutability

The existing formal C5-B artifacts must remain unchanged.

C5-BM must:

- publish a separate versioned mapping artifact;
- record the authorized re-fit as a new event;
- not overwrite or rewrite the historical formal C5-B outputs;
- not retroactively claim that the new mapping was part of the historical artifact set.

---

## 12. C5-BM PASS / BLOCKED Gate

C5-BM is **PASS** iff all of the following hold:

```text
correct frozen primary configuration
AND
authorized re-fit completed
AND
required mapping arrays persisted
AND
full sigma persisted
AND
canonical ordering preserved
AND
deterministic sign canonicalization applied
AND
required provenance/integrity metadata complete
AND
all four required historical C5-B file identities recorded and unchanged
AND
complete 200-pair source feature validation passes
AND
exact-match validation passes
AND
historical scalar metrics reproduced within abs tolerance 1e-8
AND
transactional publication validation passes
AND
historical C5-B artifacts remain unchanged
```

If any required condition fails:

```text
C5-BM = BLOCKED
```

and no authoritative mapping may be released.

---

## 13. Explicitly Deferred Questions

C5-BM must **not** decide or implement:

- native-space intervention/synthesis direction;
- whether \(VW[:,k]\) is the intervention direction;
- pseudoinverse / dual / minimum-norm intervention construction;
- token intervention scope;
- spatial token weighting;
- perturbation norm;
- perturbation scale \(\epsilon\);
- gradient/JVP screening design;
- final policy-sensitivity metric;
- C6-B candidate-selection thresholds;
- intervention-interface implementation;
- Tex3D loss design;
- adversarial texture optimization.

These remain future C6 decisions.

---

## 14. Expected Workflow

```text
C5-BM contract review
        ↓
contract freeze
        ↓
implementation
        ↓
unit/static validation
        ↓
independent code review
        ↓
formal materialization
        ↓
consistency validation
        ↓
C5-BM PASS / BLOCKED
        ↓
if PASS:
separate O2/P2 intervention-interface coding contract
```

C6-B scientific design must not begin by assuming the mapping artifact exists until C5-BM has formally passed.

---

## 15. Contract Review Record

The final read-only contract audit passed after the output-safety, mathematical
notation, and document-integrity corrections were applied.

This freezes the C5-BM scientific and engineering contract only. It does not
authorize implementation or formal materialization. It also does not authorize
the intervention interface, C6-B policy-sensitivity analysis, native intervention
vectors, token scope, perturbation scale, or Tex3D loss design.
