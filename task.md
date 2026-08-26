# C2/C3 Full Feature Extraction Coding Contract — Pilot v0.2

Status:

```text
PASS — READY FOR SEPARATE IMPLEMENTATION AUTHORIZATION
```

This contract defines formal full-feature extraction for the frozen Pilot v0.2
observation dataset.

It covers only:

```text
C2 — OpenVLA full feature extraction
C3 — π0.5 full feature extraction
```

It does not authorize C4 pairing, C5 analysis, or adversarial optimization.

---

## 1. Goal

Use the exact 200 canonical observations from the completed Pilot v0.2 formal
collection and export the already validated C2/C3 representation nodes without
changing their scientific meaning or existing feature-artifact semantics.

Target outputs:

```text
C2 OpenVLA:
200 feature NPZ archives
600 node tensors total

C3 π0.5:
200 feature NPZ archives
400 node tensors total
```

A single feature NPZ is one observation-level bundle containing all frozen nodes
for that model.

Maximum successful statuses:

```text
C2 OpenVLA Full Feature Extraction — COMPLETE
C3 π0.5 Full Feature Extraction — COMPLETE
```

---

## 2. Authoritative Inputs

Read and follow:

1. `AGENTS.md`;
2. `docs/research-map.md`;
3. `docs/pilot-v0.2-spec.md`;
4. this `task.md`;
5. the validated C2/C3 smoke extraction code and tests;
6. the current C4 paired-feature schema when checking compatibility only.

If this contract conflicts with the frozen Pilot v0.2 protocol or already validated
C2/C3 feature-node semantics, stop and report the conflict.

Pilot v0.1 must not supply defaults.

---

## 3. Frozen Source Dataset

The only valid scientific source dataset is the completed formal Pilot v0.2
collection whose manifest records:

```text
schema_version         pilot_v0_2_collection_v1
run_status             COMPLETED
completeness_status    COMPLETE
task count             10
accepted groups        50
observations           200
```

Both C2 and C3 entrypoints must take the formal:

```text
--collection-manifest
```

as the source of truth.

Do not accept an arbitrary observations directory as the scientific dataset input.

The entrypoints must:

1. read the canonical records from the collection manifest;
2. preserve their canonical order;
3. read each source sample's `observation_path` field and resolve it relative to the
   directory containing the collection manifest;
4. copy that exact relative POSIX path into the feature-manifest record as
   `source_observation_path`;
5. use exactly those 200 observations;
6. require the direct-child `*.npz` set in the canonical source `observations/`
   directory to equal the manifest-referenced source archive set exactly;
7. reject missing, duplicate, malformed, or extra source identities.

Source PilotObservation files are read-only. C2/C3 must never write, rename,
regenerate, overwrite, or delete them.

No new source-observation file-hash provenance scheme is introduced by this
contract.

---

## 4. Frozen Feature Nodes

Do not re-select layers or change extraction locations.

### C2 — OpenVLA

Scientific nodes and existing archive-array mapping:

```text
O1-S    -> o1_siglip       [256, 1152]
O1-F    -> o1_fused        [256, 2176]
O2      -> o2_projected    [256, 4096]
```

Scientific roles:

```text
O2 ↔ P2       primary cross-model pair
O1-S ↔ P1     shared-backbone control pair
O1-F          supplementary OpenVLA node
```

### C3 — π0.5

Scientific nodes and existing archive-array mapping:

```text
P1      -> p1_siglip       [256, 1152]
P2      -> p2_projected    [256, 2048]
```

The full extraction must reuse the already validated smoke-stage feature locations,
array keys, shapes, and semantics.

Do not redefine node meaning from tensor shape alone.

---

## 5. Existing Feature Archive Schema Is Authoritative

Formal full extraction must preserve the already validated observation-level bundle
format.

### C2 archive

Each:

```text
features/<sample_id>.npz
```

must contain the existing OpenVLA feature schema, including all three arrays:

```text
o1_siglip
o1_fused
o2_projected
```

and the existing metadata required by `openvla_features_v1`.

### C3 archive

Each:

```text
features/<sample_id>.npz
```

must contain the existing π0.5 feature schema, including both arrays:

```text
p1_siglip
p2_projected
```

and the existing metadata required by `pi05_features_v1`.

Do not change `shared_feature/openvla_features.py`,
`shared_feature/pi05_features.py`, or `shared_feature/paired_features.py` merely to
create a new full-run format.

---

## 6. Preserve Existing `source_image_hash`

The current C2/C3 feature archives already contain:

```text
source_image_hash
```

and the current C4 integrity path relies on its existing semantics.

Therefore:

- preserve the existing `source_image_hash` field exactly;
- validate it according to the existing feature-artifact schema;
- require it to equal the following value for the corresponding source observation:

  ```python
  image_bytes = np.ascontiguousarray(
      PilotObservation.base_rgb_raw
  ).tobytes()
  source_image_hash = (
      f"sha256:{hashlib.sha256(image_bytes).hexdigest()}"
  )
  ```

- do not remove or rename it;
- do not redefine what it hashes;
- do not introduce a second source hash scheme.

The new model-level feature manifest does not need to duplicate
`source_image_hash`; the authoritative copy remains in each feature NPZ unless the
existing implementation requires otherwise.

---

## 7. Extraction Semantics

For both C2 and C3:

```text
batch_size = 1
```

The scientific extraction unit remains one frozen PilotObservation.

Runtime precision must remain the already validated BF16 extraction path.

All saved feature arrays must be:

```text
dtype = float32
```

Runtime precision and saved dtype are intentionally separate.

Do not introduce alternate batch-dependent preprocessing.

---

## 8. Raw Token-Level Feature Preservation

Preserve each complete token-level representation:

```text
[256, D]
```

C2/C3 must not perform:

- pooling;
- token selection or dropping;
- token sorting or reordering;
- z-score normalization;
- L2 normalization;
- PCA;
- CCA;
- SVCCA;
- CKA preprocessing;
- dimensionality reduction;
- attack- or transfer-driven transformations.

C2/C3 only export frozen representations.

---

## 9. Token Layout Provenance

Preserve the model-native token order exactly as produced by the validated
extraction paths.

Record at model-manifest level:

```text
num_tokens = 256
grid_shape = [16, 16]
token_order = "model_native_spatial_flatten_order"
```

Do not insert CLS tokens, delete tokens, permute indices, or reconstruct token
ordering.

Whether a later C5-B analysis may interpret particular node pairs as
position-aligned is explicitly outside this contract and must not block C2/C3.

---

## 10. Frozen Model Identities

### C2 — OpenVLA

Logical feature checkpoint identity:

```text
openvla/openvla-7b-finetuned-libero-spatial
```

This identity must match the source collection checkpoint identity.

The machine-local checkpoint directory is a runtime location only.

The formal C2 entrypoint must separate loading location from artifact identity:

1. load `get_model()` / `get_processor()` using the resolved local
   `--pretrained-checkpoint`;
2. call the existing `extract_openvla_features()` using the frozen logical identity
   as its `pretrained_checkpoint` argument for prompt/metadata semantics.

This minimal identity separation is explicitly authorized because the existing
extractor does not load model weights itself.

The implementation must test that the resulting archive metadata stores the frozen
logical identity, not the machine-local checkpoint path.

Do not modify the extractor public API unless this behavior proves impossible.

### C3 — π0.5

Logical checkpoint identity:

```text
gs://openpi-assets/checkpoints/pi05_libero
```

Frozen config:

```text
pi05_libero
```

Formal C3 must reuse the validated smoke mechanism:

```text
download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
```

or its already validated equivalent.

The logical identity remains the metadata checkpoint identity even when runtime
loading resolves to a local cache.

Do not expose checkpoint identity or local checkpoint path as a scientific CLI
override.

---

## 11. User-Supplied Runtime Parameters

### C2 — OpenVLA

Required:

```text
--collection-manifest
--pretrained-checkpoint
--output-dir
```

Allowed optional integration path:

```text
--tex3d-openvla-root
```

with the already validated repository-relative default when available.

### C3 — π0.5

Required:

```text
--collection-manifest
--output-dir
```

Allowed optional integration path:

```text
--openpi-root
```

with the already validated repository-relative default when available.

C3 must not add:

```text
--checkpoint
```

unless a later audited blocker proves that the validated runtime can no longer load
the frozen checkpoint through the existing mechanism.

Do not expose CLI overrides for scientific node selection, node shapes, batch size,
saved dtype, token count, grid layout, normalization, pooling, or dimensionality
reduction.

---

## 12. Canonical Full-Extraction Layout

Use:

```text
<c2_output>/
    features/
        <sample_id>.npz
    openvla_feature_manifest.json

<c3_output>/
    features/
        <sample_id>.npz
    pi05_feature_manifest.json
```

Each model therefore has exactly:

```text
200 canonical feature NPZ archives
1 final feature manifest
```

when COMPLETE.

Feature archive paths recorded in the feature manifest must be relative POSIX paths
resolved relative to the directory containing that feature manifest.

Canonical feature path:

```text
features/<sample_id>.npz
```

The `source_observation_path` in each record must preserve the exact canonical
relative POSIX path serialized by the source collection manifest. It is interpreted
relative to the directory containing the source collection manifest, not relative
to the feature manifest.

---

## 13. Frozen Feature Manifest Schema

Both model manifests use:

```text
schema_version = "pilot_v0_2_full_feature_manifest_v1"
```

Exact top-level fields:

```text
schema_version
pilot_version
run_status
completeness_status
model_family
source_collection
extraction
feature_nodes
records
```

No additional top-level fields are required unless a concrete implementation
blocker is reported before coding.

Frozen common top-level values:

```text
pilot_version = "0.2"

C2 model_family = "openvla"
C3 model_family = "pi05"
```

### 13.1 Common status fields

A final manifest exists only for a fully validated run:

```text
run_status = "COMPLETED"
completeness_status = "COMPLETE"
```

There is no `USABLE_WITH_SHORTFALL`.

No final `FAILED` manifest is required.

### 13.2 `source_collection`

Exact fields:

```text
manifest_path
schema_version
checkpoint_identity
libero_revision
task_count
group_count
observation_count
```

Requirements:

```text
schema_version      = pilot_v0_2_collection_v1
task_count          = 10
group_count         = 50
observation_count   = 200
```

`manifest_path` is the resolved source collection manifest path used for this run.
It is runtime provenance, not a portable or unique collection identity by itself.

Two model manifests refer to the same frozen source collection when their frozen
source-collection metadata and ordered `(sample_id, source_observation_path)`
records agree, and C4 confirms equality of the existing per-archive
`source_image_hash` values. Different resolved `manifest_path` strings caused only
by copying the same frozen collection to another machine do not by themselves make
the collections scientifically different.

### 13.3 `extraction`

Common exact fields:

```text
feature_checkpoint_identity
runtime_precision
saved_dtype
batch_size
num_tokens
grid_shape
token_order
```

C3 additionally records:

```text
feature_config
```

Frozen values:

```text
runtime_precision = "bf16"
saved_dtype       = "float32"
batch_size        = 1
num_tokens        = 256
grid_shape        = [16, 16]
token_order       = "model_native_spatial_flatten_order"
```

For C2:

```text
feature_checkpoint_identity =
"openvla/openvla-7b-finetuned-libero-spatial"
```

For C3:

```text
feature_checkpoint_identity =
"gs://openpi-assets/checkpoints/pi05_libero"

feature_config = "pi05_libero"
```

### 13.4 `feature_nodes`

C2:

```json
{
  "O1-S": {"archive_key": "o1_siglip", "shape": [256, 1152]},
  "O1-F": {"archive_key": "o1_fused", "shape": [256, 2176]},
  "O2": {"archive_key": "o2_projected", "shape": [256, 4096]}
}
```

C3:

```json
{
  "P1": {"archive_key": "p1_siglip", "shape": [256, 1152]},
  "P2": {"archive_key": "p2_projected", "shape": [256, 2048]}
}
```

### 13.5 `records`

Exactly 200 records, in the canonical source collection order.

Each record has exactly:

```text
sample_id
source_observation_path
feature_path
```

Example:

```json
{
  "sample_id": "libero_spatial__task00__state00__step0008",
  "source_observation_path":
    "observations/libero_spatial__task00__state00__step0008.npz",
  "feature_path":
    "features/libero_spatial__task00__state00__step0008.npz"
}
```

The manifest does not duplicate per-archive `source_image_hash`.

---

## 14. Completeness Semantics

C2 and C3 are strict full-coverage gates.

C2 is COMPLETE only if:

```text
200 canonical archives exist
each archive contains O1-S/O1-F/O2
=> 600 valid node tensors total
```

C3 is COMPLETE only if:

```text
200 canonical archives exist
each archive contains P1/P2
=> 400 valid node tensors total
```

Missing any source observation or node tensor means the run is not COMPLETE.

Do not introduce shortfall semantics.

C4 must not consume incomplete C2 or C3 outputs.

---

## 15. Resume and Failure Semantics

Formal full extraction supports validated resume from a partial output directory
because the source dataset and extraction configuration are frozen.

### 15.1 First run

The output directory may be absent or empty.

The script creates:

```text
features/
```

and writes observation-level feature archives as extraction proceeds.

### 15.2 Interrupted/failed run

If extraction fails:

- preserve already completed feature archives;
- propagate the exception and return non-zero;
- do not silently skip the failed sample;
- do not create the final feature manifest;
- do not classify a partial directory as COMPLETE.

Partial archives are engineering recovery artifacts, not a scientific result.

### 15.3 Resume

When the output directory already contains `features/` but no final manifest:

1. enumerate the 200 canonical expected sample IDs from the source collection
   manifest;
2. validate each existing expected archive using the existing model-specific schema;
3. additionally verify sample ID, checkpoint identity, all frozen array keys,
   shapes, float32 dtype, and finite values;
4. recompute the existing `source_image_hash` from the corresponding source
   `PilotObservation.base_rgb_raw` and require exact equality with archive metadata;
5. reuse only archives that fully pass validation;
6. extract only missing canonical observations;
7. if an existing expected archive is malformed or metadata-inconsistent, stop and
   report it;
8. do not silently overwrite malformed artifacts;
9. reject unexpected canonical NPZ archives that are not part of the 200 expected
   sample IDs.

### 15.4 Existing final manifest

If the final model manifest already exists, treat that output as an immutable
completed result.

Do not overwrite or regenerate it in place.

A new scientific rerun requires a new output directory.

### 15.5 Final manifest commit

Create the final manifest only after all 200 archives pass complete validation.

Write it atomically so that interruption cannot leave a partially written final
manifest.

No final `FAILED` manifest is required.

---

## 16. Source Read-Only Guarantee

The implementation must contain no write path targeting source PilotObservation
files.

Do not add a new permanent source-file SHA-256 provenance field.

A pre/post full-file hash pass is not required by this contract.

The existing per-feature `source_image_hash` semantics remain authoritative for
C2/C3/C4 source-image integrity.

---

## 17. Required Final Validation

Before creating the final COMPLETE manifest, verify at least:

1. source collection schema is `pilot_v0_2_collection_v1`;
2. source collection is `COMPLETED + COMPLETE`;
3. source collection has exactly 10 tasks, 50 groups, and 200 observations;
4. the 200 source sample IDs and paths are unique and canonical;
5. the direct-child source `observations/*.npz` set exactly equals the
   manifest-referenced source archive set;
6. exactly 200 expected feature archives exist;
7. there are no unexpected canonical feature NPZ archives;
8. every archive loads with the existing model-specific feature schema;
9. archive `sample_id` matches the source record;
10. archive checkpoint metadata matches the frozen logical feature identity;
11. archive `source_image_hash` exactly matches the hash recomputed from the
    corresponding source observation's contiguous raw base-image bytes;
12. all expected node-array keys exist;
13. all node-array shapes match Section 4;
14. all saved node arrays are float32;
15. all node arrays contain finite values;
16. final records are ordered exactly like the source collection records;
17. every manifest `feature_path` is canonical relative POSIX;
18. every manifest `source_observation_path` exactly matches the source collection
    manifest;
19. manifest record count is exactly 200.

Do not duplicate deeper extractor unit tests when the existing schema validators can
be reused.

---

## 18. C2/C3 Independence

C2 and C3 have no scientific execution-order dependency.

Either may run first, and they may run independently if resources permit.

C4 may begin only after both manifests are COMPLETE and refer to the same frozen
source collection.

---

## 19. Allowed Implementation Files

Allowed new files:

```text
scripts/c2_full_feature_extraction.py
scripts/c3_full_feature_extraction.py
tests/test_c2_full_feature_extraction.py
tests/test_c3_full_feature_extraction.py
```

Optional shared script helper, only if it materially reduces duplicated manifest and
source-validation code:

```text
scripts/_full_feature_extraction_common.py
```

Do not modify:

```text
shared_feature/openvla_features.py
shared_feature/pi05_features.py
shared_feature/paired_features.py
shared_feature/pilot_observation.py
```

If implementation cannot satisfy this contract without changing one of those files,
stop and report the exact blocker before broadening scope.

Do not modify upstream OpenVLA/OpenPI repositories.

---

## 20. CPU-Only Tests

Tests must not execute real OpenVLA, π0.5, LIBERO, CUDA, or JAX accelerator
workloads.

Using fakes and small valid feature archives, test at least:

### Common

1. formal collection manifest validation;
2. exact 200-source requirement logic;
3. canonical source ordering;
4. output path rules;
5. no scientific CLI overrides;
6. no final manifest for partial/failed runs;
7. final manifest exact schema;
8. atomic final-manifest behavior;
9. valid partial archives are reused;
10. malformed partial archives stop without overwrite;
11. unexpected feature archives are rejected;
12. existing final manifest is not overwritten.

### C2

1. runtime model/processor loading uses the local checkpoint path;
2. extractor receives the frozen logical checkpoint identity;
3. `batch_size=1`;
4. expected bundled keys/shapes are validated;
5. archive metadata records the logical identity.

### C3

1. frozen `pi05_libero` config is used;
2. frozen `gs://openpi-assets/checkpoints/pi05_libero` identity is used;
3. existing validated checkpoint-download/cache mechanism is used;
4. no `--checkpoint` CLI is exposed;
5. `batch_size=1`;
6. expected bundled keys/shapes are validated.

---

## 21. Required Checks After Implementation

Run focused tests plus relevant existing regressions.

At minimum:

```bash
python -m pytest \
  tests/test_c2_full_feature_extraction.py \
  tests/test_c3_full_feature_extraction.py \
  tests/test_openvla_features.py \
  tests/test_pi05_features.py \
  tests/test_paired_features.py \
  -q
```

Run the full test suite if practical.

Also run Ruff check and Ruff format check on changed Python files and:

```bash
git diff --check
```

Do not run the real full C2/C3 workloads during implementation.

---

## 22. Scope Boundary

Do not:

- recollect Pilot observations;
- modify the frozen source observations;
- re-select feature nodes;
- change existing C2/C3 feature archive schemas;
- change C4 pairing semantics;
- perform pooling or normalization;
- run CKA;
- run PCA;
- run CCA/SVCCA;
- materialize the C5 train/held-out split;
- build the C4 full paired manifest;
- inspect cross-model similarity results;
- choose C5 thresholds;
- choose the null RNG seed;
- implement shared-feature adversarial losses;
- optimize Tex3D textures.

The exact C5 Go/No-Go rule and null-permutation RNG seed remain independent OPEN
decisions and do not block C2/C3 extraction.

---

## 23. Documentation Sync Result

`docs/research-map.md` is synchronized with the already verified current state:

```text
C5-D0 Pilot v0.2 Formal Collection    COMPLETE
C2/C3 Full Extraction Contract        PASS
```

This is a documentation-state correction only and must not change scientific
protocol.

---

## 24. Read-Only Re-Audit Result

The v2 contract was re-audited against the existing C2/C3/C4 code and tests.

The re-audit confirmed:

1. the observation-level bundled NPZ layout is compatible;
2. `source_image_hash` semantics are preserved;
3. C2 local-path/logical-identity separation is implementable without extractor API
   changes;
4. C3 requires no user checkpoint argument;
5. the exact feature manifest schema is sufficient for later C4 work;
6. validated resume is implementable without modifying existing extractor schemas;
7. allowed implementation files are sufficient;
8. no remaining contract conflict requires a scientific decision.

The re-audit observed these constraints:

```text
DO NOT MODIFY CODE
DO NOT RUN REAL MODEL WORKLOADS
```

Result:

```text
PASS
```

---

## 25. Current Gate

```text
C5-D0 Pilot v0.2 Formal Collection    COMPLETE
C2/C3 Full Extraction Contract v2     PASS — READY FOR SEPARATE IMPLEMENTATION AUTHORIZATION
```

A separate implementation authorization is still required before coding begins.

Formal C5 PASS/FAIL evaluation remains blocked by its independent unresolved
decision rule and null-permutation RNG seed.
