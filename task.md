# C4 Formal Paired Feature Materialization Contract — Pilot v0.2

Status:

```text
DRAFT — READ-ONLY AUDIT REQUIRED BEFORE IMPLEMENTATION
```

This contract defines the formal C4 pairing stage for the frozen Pilot v0.2 feature dataset.

It authorizes only:

```text
C4 — formal paired-feature manifest materialization
```

It does not authorize C5 Linear CKA, PCA, CCA/SVCCA, split analysis, adversarial optimization, or any modification of C2/C3 feature artifacts.

---

## 1. Goal

Construct one formal paired-feature manifest from the already completed:

```text
C2 OpenVLA Full Feature Extraction — PASS
C3 π0.5 Full Feature Extraction — PASS
```

The formal C4 result must contain exactly 200 cross-model pairs corresponding to the same 200 canonical Pilot v0.2 observations.

C4 is a lightweight manifest-materialization stage only.

It must not copy, rewrite, transform, or duplicate feature tensors.

Maximum successful status:

```text
C4 Formal Paired Feature Materialization — COMPLETE
```

---

## 2. Authoritative Inputs

The scientific inputs are the two completed model-level feature manifests:

```text
--openvla-feature-manifest
--pi05-feature-manifest
```

The feature directories alone are not sufficient formal inputs.

The entrypoint must read and validate both feature manifests before constructing the paired manifest.

Expected source manifests:

```text
schema_version = "pilot_v0_2_full_feature_manifest_v1"
run_status = "COMPLETED"
completeness_status = "COMPLETE"
records = exactly 200
```

The OpenVLA feature manifest must describe:

```text
model_family = "openvla"
feature checkpoint =
openvla/openvla-7b-finetuned-libero-spatial
```

The π0.5 feature manifest must describe:

```text
model_family = "pi05"
feature checkpoint =
gs://openpi-assets/checkpoints/pi05_libero
feature config =
pi05_libero
```

---

## 3. Source Collection Identity

Both feature manifests must refer to the same frozen Pilot v0.2 source collection.

At minimum, require equality of the formal source-collection provenance carried by both manifests, including:

```text
source collection schema version
source collection checkpoint identity
LIBERO revision
task count
group count
observation count
```

Expected scientific source identity:

```text
schema_version       pilot_v0_2_collection_v1
checkpoint_identity  openvla/openvla-7b-finetuned-libero-spatial
task_count           10
group_count          50
observation_count    200
```

Machine-local or server-local source manifest paths may differ after artifact synchronization and must not by themselves cause pairing failure.

Runtime-path provenance is not the scientific pairing identity.

---

## 4. Canonical Record Order

The ordered `records` sequence in the completed model feature manifests is authoritative.

C4 must require:

```text
len(OpenVLA records) = 200
len(π0.5 records)    = 200
```

and for every canonical index `i`:

```text
OpenVLA.records[i].sample_id
==
π0.5.records[i].sample_id
```

Also require:

```text
OpenVLA.records[i].source_observation_path
==
π0.5.records[i].source_observation_path
```

The formal pair order must be exactly this shared feature-manifest record order.

Do not derive formal order using:

```text
sorted(sample_id)
filesystem enumeration order
feature filename order
```

even if those orders happen to match the current dataset.

---

## 5. Feature Path Resolution

Each feature-manifest record contains one bundled feature archive path:

```text
feature_path
```

Resolve that path relative to the directory containing its own feature manifest.

For every record:

- the referenced feature archive must exist;
- it must resolve to a regular file;
- it must remain within the intended feature output boundary;
- the resolved archive must correspond to the record `sample_id`.

Do not discover formal C4 inputs by scanning arbitrary directories when the manifest already supplies the authoritative paths.

---

## 6. Reuse Existing Archive Validation

Reuse the already validated archive semantics in:

```text
shared_feature/paired_features.py
```

as much as possible.

For each OpenVLA archive, preserve validation of the existing schema:

```text
openvla_features_v1

o1_siglip       [256, 1152]
o1_fused        [256, 2176]
o2_projected    [256, 4096]
```

For each π0.5 archive:

```text
pi05_features_v1

p1_siglip       [256, 1152]
p2_projected    [256, 2048]
```

All feature arrays must remain:

```text
dtype = float32
finite values only
```

Existing archive metadata semantics must be preserved.

Do not redefine the C2/C3 archive schemas in C4.

---

## 7. Pair Integrity

A formal pair is valid only if both archives describe the same canonical observation.

For each ordered pair require:

```text
OpenVLA sample_id == π0.5 sample_id
```

and:

```text
OpenVLA source_image_hash == π0.5 source_image_hash
```

The existing `source_image_hash` field remains the cross-model image-content integrity check.

Reject:

- missing records;
- duplicate records;
- malformed feature manifests;
- malformed feature archives;
- unresolved feature paths;
- mismatched sample IDs;
- mismatched source observation paths;
- mismatched source collection provenance;
- mismatched `source_image_hash`;
- wrong feature checkpoint identity;
- wrong feature schema;
- wrong tensor key;
- wrong tensor shape;
- wrong dtype;
- non-finite feature values;
- any record count other than 200.

No shortfall semantics are allowed.

---

## 8. Formal Output

The C4 output remains the existing lightweight paired-feature schema:

```text
schema_version = "paired_features_v1"
```

Do not create a new paired-feature tensor archive format.

Do not copy C2/C3 NPZ files.

The output is one JSON manifest containing exactly 200 paired records.

Each pair must preserve the existing `paired_features_v1` semantics:

```json
{
  "sample_id": "...",
  "source_image_hash": "sha256:...",
  "openvla_feature_path": "...",
  "pi05_feature_path": "..."
}
```

Top level:

```json
{
  "schema_version": "paired_features_v1",
  "num_samples": 200,
  "pairs": [...]
}
```

The order of `pairs` must equal the shared canonical feature-manifest record order.

Feature paths in the paired manifest must remain valid relative POSIX paths according to the existing `paired_features_v1` path semantics.

---

## 9. Output Path and Overwrite Semantics

Runtime interface:

```text
--openvla-feature-manifest
--pi05-feature-manifest
--output-path
```

Do not freeze a machine-specific absolute output path into the scientific contract.

Requirements:

```text
output_path must not already exist
```

Refuse overwrite.

If writing fails, no incomplete paired manifest may remain.

A successful run creates exactly one formal paired manifest and no new feature archives.

---

## 10. C4 Does Not Materialize C5 Split

C4 must pair the full 200-observation canonical dataset only.

Do not in C4:

- create TRAIN / HELD-OUT splits;
- apply the SHA-256 split ranking rule;
- select held-out groups;
- calculate CKA;
- mean-pool tokens;
- shuffle groups;
- calculate null distributions;
- run PCA;
- run CCA/SVCCA.

Those belong to C5.

---

## 11. Implementation Scope

Preferred new files:

```text
scripts/c4_full_paired_features.py
tests/test_c4_full_paired_features.py
```

Prefer to keep unchanged:

```text
shared_feature/paired_features.py
shared_feature/openvla_features.py
shared_feature/pi05_features.py
scripts/c2_full_feature_extraction.py
scripts/c3_full_feature_extraction.py
```

The new formal C4 entrypoint should act as an orchestration/validation layer around the existing paired-feature implementation.

If preserving canonical manifest-driven order is impossible without modifying the public API of `shared_feature/paired_features.py`, stop and report the exact blocker before changing it.

Do not modify upstream OpenVLA or OpenPI code.

---

## 12. Required Tests

Tests must be CPU-only and must not run real OpenVLA, π0.5, LIBERO, CUDA, or JAX workloads.

Test at least:

1. valid C2/C3 COMPLETE feature manifests produce exactly 200 pairs;
2. output order exactly matches canonical feature-manifest record order;
3. deliberately shuffled feature-manifest order is respected and not replaced by `sorted(sample_id)`;
4. differing C2/C3 record order is rejected;
5. differing sample sets are rejected;
6. differing `source_observation_path` is rejected;
7. differing source-collection provenance is rejected;
8. non-COMPLETE feature manifest is rejected;
9. record count other than 200 is rejected;
10. unresolved feature path is rejected;
11. malformed archive is rejected;
12. wrong checkpoint or feature schema is rejected;
13. `source_image_hash` mismatch is rejected;
14. output overwrite is rejected;
15. write failure leaves no incomplete output;
16. no feature NPZ files are copied or created.

Run relevant existing paired-feature regressions as well.

---

## 13. Required Checks

After implementation run at minimum:

```bash
python -m pytest   tests/test_c4_full_paired_features.py   tests/test_paired_features.py   -q
```

Also run the relevant full regression suite if practical.

Run:

```text
Ruff check
Ruff format check
git diff --check
```

No real model workload is required for C4.

Formal C4 materialization itself may be run against the already synchronized C2/C3 feature outputs because it is CPU-only read/validation work.

---

## 14. Formal Runtime Inputs for Current Data

Current formal inputs are:

```text
experiment_inbox/c5-c2-output/openvla_feature_manifest.json
experiment_inbox/c5-c3-output/pi05_feature_manifest.json
```

The actual `--output-path` is a runtime choice and must point to a new, non-existing file.

Do not encode these repository-local paths as universal scientific constants.

---

## 15. Scope Boundary

This contract authorizes only formal C4 paired-manifest materialization.

Do not:

- modify C2/C3 feature archives;
- regenerate features;
- copy feature tensors;
- change node definitions;
- modify source observations;
- create the C5 split;
- mean-pool features;
- run Linear CKA;
- calculate shuffled nulls;
- run PCA;
- run CCA/SVCCA;
- choose C5 Go/No-Go thresholds;
- choose the C5 null RNG seed;
- implement shared-feature losses;
- optimize Tex3D textures.

---

## 16. Read-Only Audit Before Coding

Before implementation, Codex must re-audit this C4 contract against:

```text
shared_feature/paired_features.py
existing paired-feature tests
C2 formal feature manifest
C3 formal feature manifest
```

The audit must confirm:

1. the existing `paired_features_v1` output schema can remain unchanged;
2. archive validation can be reused;
3. manifest-driven canonical order can be enforced by the new formal wrapper;
4. no feature copying is required;
5. no scientific incompatibility remains.

During this audit:

```text
DO NOT MODIFY CODE
DO NOT RUN C5 ANALYSIS
```

If a concrete incompatibility exists, stop and report it before implementation.

---

## 17. Current Gate

Current state:

```text
C5-D0 Formal Collection              PASS
C2 OpenVLA Full Feature Extraction   PASS
C3 π0.5 Full Feature Extraction      PASS
C4 Formal Paired Feature Contract    DRAFT — READ-ONLY AUDIT REQUIRED
C5-A Linear CKA                      NOT STARTED
```

After C4 reaches COMPLETE, C2/C3/C4 are closed and the next scientific discussion moves to C5-A Linear CKA.
