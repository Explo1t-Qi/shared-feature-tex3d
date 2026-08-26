# C5-D0-FULL Coding Contract — Pilot v0.2 Formal Collection Entrypoint

Status:

```text
DRAFT — READ-ONLY AUDIT REQUIRED BEFORE IMPLEMENTATION
```

This contract prepares the formal Pilot v0.2 collection entrypoint. It does not
authorize executing the real full collection.

## 1. Goal

Implement one reviewable server-side script that loads the frozen OpenVLA policy
through the validated C1 integration path and invokes the existing formal Pilot
v0.2 collector. The script must independently validate the returned result and
canonical artifacts without changing scientific or collector behavior.

Maximum successful implementation status:

```text
C5-D0 Pilot v0.2 full-collection entrypoint — READY, NOT EXECUTED
```

## 2. Authoritative Inputs

Read and follow:

1. `AGENTS.md`;
2. `docs/research-map.md`;
3. `docs/pilot-v0.2-spec.md`;
4. this `task.md`;
5. the validated C5-D0 collector and reduced-smoke script.

If this contract conflicts with the frozen Pilot v0.2 specification, stop and
report the conflict. Pilot v0.1 is historical and must not supply defaults.

## 3. Current Verified State

```text
C5-D0 Pilot v0.2 collector              UNIT-LEVEL PASS
C5-D0 reduced real integration smoke    PASS
```

The reduced smoke validated real OpenVLA/LIBERO integration on tasks `0` and `1`,
one successful group per task, four observations per group, and
`SMOKE_COMPLETED + null` status semantics.

The formal collector already exists as:

```python
from shared_feature import collect_pilot_v02_observations
```

Do not call the private reduced-plan helper from the formal entrypoint.

## 4. Frozen Checkpoint Identity

### DECISION

The exact portable OpenVLA checkpoint identity is:

```text
openvla/openvla-7b-finetuned-libero-spatial
```

It intentionally contains no immutable-revision suffix. The entrypoint must define
this identity as a fixed scientific constant and pass it unchanged to
`collect_pilot_v02_observations()`.

Do not expose a CLI option that permits replacing it. The machine-local checkpoint
directory remains separate runtime provenance.

## 5. Frozen Formal Collection Plan

The script must use the public formal collector without plan overrides. Effective
scientific scope is:

```text
suite                         libero_spatial
task IDs                      0..9
target accepted groups/task   5
observations/group            4
target relative progress      [0.10, 0.40, 0.70, 0.90]
candidate state order         ascending official index
resume                        disabled
```

The script must not expose CLI options for task IDs, group quota, progress targets,
dummy steps, action budget, camera resolution, seed, success semantics, or resume.

## 6. Target Files

Allowed implementation file:

```text
scripts/c5_d0_pilot_v02_full_collection.py
```

Allowed focused test file:

```text
tests/test_c5_d0_full_collection.py
```

Do not modify:

```text
shared_feature/pilot_v02_collector.py
shared_feature/libero_collector.py
shared_feature/pilot_observation.py
shared_feature/openvla_features.py
shared_feature/pi05_features.py
shared_feature/paired_features.py
scripts/c5_d0_pilot_v02_smoke_integration.py
```

If the existing formal public collector cannot satisfy this contract, stop and
report the exact blocker before changing production code.

## 7. Required CLI

The script must accept exactly these required runtime inputs:

```text
--pretrained-checkpoint
--libero-revision
--output-dir
```

It may additionally accept:

```text
--tex3d-openvla-root
```

using the repository-relative default validated by the C1/reduced-smoke scripts.

It must not accept `--checkpoint-identity`; the identity is frozen by Section 4.
The server checkpoint path must not be hardcoded.

## 8. Pre-Model-Load Validation

Before importing/loading real model weights, validate:

1. `pretrained_checkpoint` resolves to an existing local directory;
2. `<checkpoint>/dataset_statistics.json` exists;
3. the statistics contain `libero_spatial_no_noops`;
4. `libero_revision` is a non-empty string;
5. the official Tex3D OpenVLA source root exists;
6. `output_dir`, if present, is an empty directory.

Do not create `output_dir` during script preflight. The production collector owns
the canonical output-lifecycle boundary. Do not create a log or report inside
`output_dir` before collection.

## 9. OpenVLA Loading Boundary

Use the validated C1/OpenVLA source path:

```python
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_model, set_seed_everywhere
```

Before `get_model()` or `get_processor()`, call exactly:

```python
set_seed_everywhere(7)
```

Use a model configuration equivalent to:

```text
model_family          openvla
pretrained_checkpoint resolved local checkpoint path
load_in_8bit          false
load_in_4bit          false
unnorm_key            libero_spatial_no_noops
center_crop           true
```

Require CUDA before model loading. Preserve the existing official runtime behavior:

```text
camera resolution                 512
dummy/stabilization steps         10
maximum valid-policy actions      300
LIBERO environment seed           0
deterministic decoding            do_sample = false
```

Do not duplicate or replace action preprocessing in the script.

## 10. Formal Collector Invocation

Call only:

```python
collect_pilot_v02_observations(
    model=model,
    processor=processor,
    pretrained_checkpoint=resolved_checkpoint,
    checkpoint_identity="openvla/openvla-7b-finetuned-libero-spatial",
    libero_revision=supplied_libero_revision,
    output_dir=resolved_output_dir,
)
```

Do not call `_collect_pilot_v02_with_plan()` and do not reimplement traversal,
rollout, sampling, rejection, transaction, manifest, or completeness logic.

## 11. Output and Failure Semantics

The canonical output remains:

```text
<output_dir>/
    observations/
        <sample_id>.npz
    collection_manifest.json
```

Do not add permanent helper files, reports, or logs to this directory.

Allow the production collector to enforce fresh/empty output, no resume, no
overwrite, group transactions, fatal audit manifests, and exception chaining.

The script must not catch a collection error merely to continue, retry, patch, or
change classification. An uncaught failure must preserve the complete traceback and
non-zero process exit. Do not automatically delete or alter fatal/completed output.

## 12. Completed-Run Validation

After a non-fatal return, independently read the manifest and accepted
`PilotObservation` archives. Require:

1. `run_status == "COMPLETED"` in result and manifest;
2. result and manifest completeness statuses are equal;
3. completeness is `COMPLETE`, `USABLE_WITH_SHORTFALL`, or `BLOCKED`;
4. schema version is `pilot_v0_2_collection_v1`;
5. frozen checkpoint identity is exact;
6. supplied LIBERO revision is serialized exactly;
7. runtime discovers tasks `0..9`;
8. task results are ordered `0..9`;
9. coverage counts agree with task/group/sample records;
10. every accepted group contains exactly four samples;
11. target progress is `[0.10, 0.40, 0.70, 0.90]` in order;
12. observation paths are relative POSIX and resolve under `observations/`;
13. returned sample paths equal manifest paths in canonical order;
14. direct-child NPZ files exactly equal referenced archives;
15. every archive loads with `PilotObservation.load()`;
16. loaded metadata agrees with task/group/sample manifest records;
17. sample IDs are globally unique;
18. every accepted record has `episode_success = True`.

Do not reinterpret completeness. `COMPLETED + BLOCKED` is a validly completed
collection execution with insufficient scientific coverage, not a runtime failure
and not a PASS for downstream C5 admission.

## 13. Console Summary

Print one concise deterministic JSON summary containing at least:

```text
run_status
completeness_status
checkpoint_identity
libero_revision
manifest_path
target_total_groups
actual_total_groups
target_total_observations
actual_total_observations
accepted_groups_per_task
attempted_states_per_task
rejected_states_per_task
observation_file_count
```

The summary is console output only. Do not write it into the canonical output.

## 14. CPU-Only Tests

Tests must not import or execute real OpenVLA, LIBERO, CUDA, or model weights.
Using fakes and small real NPZ files, verify at least:

1. the frozen checkpoint identity is not a CLI override;
2. input validation occurs before runtime/model loading;
3. `set_seed_everywhere(7)` precedes model and processor loading;
4. quantization, unnorm key, and center-crop configuration are exact;
5. the public formal collector is called with no plan overrides;
6. COMPLETE result validation and summary;
7. USABLE_WITH_SHORTFALL result validation and summary;
8. BLOCKED remains a completed result with `BLOCKED` status;
9. formal/smoke/fatal status confusion is rejected;
10. malformed counts, paths, IDs, metadata, or archives are rejected;
11. non-empty output fails before model loading.

Do not duplicate production collector unit tests in this script-test file.

## 15. Checks

After implementation, run:

```bash
python -m pytest \
  tests/test_c5_d0_full_collection.py \
  tests/test_pilot_v02_collector.py \
  tests/test_pilot_observation.py \
  tests/test_libero_collector.py \
  -q
```

Run the full suite if practical. Run Ruff check and format check on the two changed
Python files. Run `git diff --check`.

## 16. Forbidden Scope

Do not:

- execute the real full collection during implementation;
- call the private reduced-smoke collector;
- change the frozen checkpoint identity;
- expose scientific plan overrides;
- implement resume/retry;
- modify production collectors or observation schemas;
- start C2/C3 full extraction;
- build the C4 full manifest;
- materialize the C5 split;
- implement CKA, SVCCA, shuffled null, or C5 PASS/FAIL logic;
- inspect results to choose C5 thresholds;
- modify external reference repositories.

## 17. Stop Conditions

Stop and report before broadening scope if:

1. the public formal collector cannot express the frozen plan;
2. the script would need production collector changes;
3. real OpenVLA loading requires semantics different from validated C1;
4. the frozen checkpoint identity cannot be passed unchanged;
5. canonical output would need additional files;
6. result validation conflicts with the manifest schema;
7. implementation requires resolving a remaining `OPEN` C5 decision.

## 18. Completion and Execution Boundary

Implementation is complete only when the entrypoint and CPU-only tests pass, Ruff
passes, and the final diff has no scope creep. Then stop with maximum status:

```text
C5-D0 Pilot v0.2 full-collection entrypoint — READY, NOT EXECUTED
```

Do not run the real model or LIBERO workload during implementation.

Later real full-collection execution requires explicit user authorization after this
contract and implementation have been reviewed.

Formal C5 PASS/FAIL evaluation remains separately blocked by the unresolved exact
decision rule and null-permutation RNG seed.
