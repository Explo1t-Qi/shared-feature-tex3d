# C5-D0 Coding Contract — Pilot v0.2 Dataset Collector

## 1. Goal

Implement the Pilot v0.2 observation-collection layer used to prepare the canonical clean observation dataset for later C2/C3/C4/C5 work.

This coding milestone implements:

```text
Pilot v0.2 multi-task collection orchestration
+
collection manifest
+
quota / rejection / completeness logic
+
transactional group writes
+
focused unit tests
+
a private reduced-real-smoke entry
```

This milestone does **not** perform:

- formal Pilot v0.2 collection;
- OpenVLA feature extraction;
- π0.5 feature extraction;
- C4 paired-feature construction;
- C5 split materialization;
- CKA;
- SVCCA;
- shuffled-null analysis;
- C5 PASS/FAIL evaluation;
- policy-relevance analysis;
- Tex3D attack optimization.

Maximum successful status for this coding task:

```text
C5-D0 Pilot v0.2 collector — UNIT-LEVEL PASS
```

Formal collection remains blocked by the `OPEN` portable OpenVLA checkpoint identity.

---

## 2. Authoritative Documents

Before implementation, follow:

1. `AGENTS.md`
2. `docs/research-map.md`
3. `docs/pilot-v0.2-spec.md`
4. this `task.md`

Pilot v0.1 is historical.

Do not import scientific defaults from `docs/pilot-v0.1-spec.md` into this task.

If this contract conflicts with `docs/pilot-v0.2-spec.md`, stop and report the conflict rather than choosing an implementation.

---

## 3. Current Project State

The following earlier milestones are already closed:

```text
C0 — PilotObservation schema                  PASS
C1 — LIBERO/OpenVLA observation collector     PASS
C1 — real OpenVLA/LIBERO smoke                PASS
C2 — OpenVLA feature extraction               PASS
C2 — real OpenVLA/GPU smoke                   PASS
C3 — π0.5 feature extraction                  PASS
C3 — real π0.5 integration smoke              PASS
C4 — paired-feature manifest                  PASS
C4 — real-artifact smoke                      PASS
```

Existing two-sample real artifacts prove pipeline closure only.

They must not be used for formal C5 scientific conclusions.

---

## 4. Current OPEN Blocker

### OPEN — Portable OpenVLA Checkpoint Identity

The portable logical identity of the OpenVLA LIBERO-Spatial checkpoint has not yet been frozen.

Therefore:

```text
formal Pilot v0.2 collection MUST NOT run
```

during this coding task.

Do not infer a portable checkpoint identity from a machine-local directory name.

This OPEN does not block:

- implementing the collector;
- unit tests;
- fake-runtime integration tests;
- implementing the private reduced-smoke path;
- static validation of the smoke script.

A real smoke may only be executed later when an explicitly approved checkpoint identity and resolved local checkpoint are supplied.

---

## 5. Frozen Scientific Inputs

The implementation must operationalize, not redesign, these Pilot v0.2 decisions:

```text
suite = libero_spatial
tasks = all 10 official LIBERO-Spatial tasks
target accepted groups/task = 5
observations/group = 4
target progress = [0.10, 0.40, 0.70, 0.90]
group identity = (task_id, initial_state_id)
```

Candidate initial states are the zero-based indices returned by:

```text
get_task_init_states(task_id)
```

and are traversed in ascending index order.

Under the audited runtime, each task exposes 50 states indexed `0..49`.

The runtime must still validate actual availability instead of silently assuming that unavailable indices exist.

---

## 6. Frozen Rollout Configuration

Pilot v0.2 must preserve the validated C1 OpenVLA/LIBERO inference path and use:

```text
model_family = "openvla"
unnorm_key = "libero_spatial_no_noops"
center_crop = True
camera_resolution = 512
dummy_steps = 10
max_valid_policy_actions = 300
load_in_4bit = False
load_in_8bit = False
global OpenVLA evaluation seed = 7
LIBERO environment seed = 0
action decoding do_sample = False
```

The global seed must be applied through the existing official-equivalent:

```text
set_seed_everywhere(7)
```

before model loading and rollout in a real integration entry.

The LIBERO environment seed must remain the value applied by the official `get_libero_env()` path.

Do not introduce an alternative preprocessing or action-decoding path.

---

## 7. Valid-Policy Trajectory Semantics

The 10 dummy/stabilization steps are not part of the valid-policy trajectory.

For valid-policy step `t`:

```text
capture observation x_t
→ query OpenVLA
→ obtain action a_t
→ apply action
→ inspect done / success
```

`T` is:

```text
the number of observations actually submitted to the OpenVLA policy query
```

Therefore:

- dummy observations are excluded;
- the query observation that produces a successful or terminating action is included;
- the observation returned after that successful/terminal action is not appended;
- `step_id` is the zero-based valid-policy query index;
- `step_id` is not a simulator-global timestep.

Dummy-phase `done == True` or success is not `policy_failure`.

It is a collection-level fatal error.

---

## 8. Success and Rejection Semantics

Success is authoritative only when:

```text
env.check_success() == True
```

Ordinary rejected candidate outcomes use this precedence:

```text
1. unsuccessful rollout
   → policy_failure

2. successful rollout with T < 4
   → trajectory_too_short

3. successful rollout with T >= 4 but non-unique target indices
   → sampling_index_collision
```

`policy_failure` includes:

- `done == True` while success is false;
- exhaustion of 300 valid-policy actions without success.

Do not automatically retry a failed candidate state.

Do not extend the action budget.

Do not change rollout settings for a failed state.

Infrastructure errors are fatal and must not be converted to ordinary rejection records.

Examples:

- environment creation/reset error;
- dummy-phase early termination/success;
- model/action runtime error;
- malformed observation;
- invalid action;
- schema failure;
- serialization failure;
- filesystem failure.

---

## 9. Sampling Rule

For:

```text
Q = [0.10, 0.40, 0.70, 0.90]
```

and trajectory length `T`, compute:

```text
t_q = floor(q * (T - 1) + 0.5)
```

The four indices must be unique.

Do not:

- duplicate frames;
- move to neighboring indices;
- modify `Q`;
- accept fewer than four observations.

For each sampled observation:

```text
normalized_episode_progress = step_id / (T - 1)
```

The manifest must preserve both:

```text
target_relative_progress
actual_normalized_episode_progress
```

They must not be conflated.

---

## 10. Existing C1 Boundary

The existing low-level collector is:

```text
shared_feature/libero_collector.py
```

Its public API:

```text
collect_pilot_observations(...)
```

must remain behaviorally backward-compatible.

Do not redesign or duplicate validated low-level behavior for:

- environment setup;
- reset semantics;
- dummy actions;
- OpenVLA preprocessing;
- action generation;
- gripper normalization;
- gripper inversion;
- raw observation capture;
- canonical 8D state construction;
- official success checking.

A minimal internal refactor of `libero_collector.py` is allowed only when required for code reuse.

Any such refactor must preserve:

```text
existing public API
existing unit tests
existing C1 smoke semantics
```

If safe reuse requires breaking C1 behavior, stop and report the blocker.

---

## 11. Production Module

Implement a new high-level module:

```text
shared_feature/pilot_v02_collector.py
```

Use:

```python
class PilotV02CollectionError(RuntimeError):
    ...
```

The public formal collection API must be:

```python
def collect_pilot_v02_observations(
    *,
    model,
    processor,
    pretrained_checkpoint: str | Path,
    checkpoint_identity: str,
    libero_revision: str,
    output_dir: str | Path,
) -> "PilotV02CollectionResult":
    ...
```

The public API must always use the frozen formal Pilot v0.2 plan.

Preflight requirements for the public real-collection path:

- `checkpoint_identity` must be a non-empty string;
- `libero_revision` must be a non-empty string;
- `pretrained_checkpoint` must resolve to an existing local directory;
- `output_dir`, if it already exists, must be an empty directory;
- the caller must have applied `set_seed_everywhere(7)` before loading `model` / `processor` and before invoking the collector.

The collector cannot prove that the seed was applied before an already-loaded model. This is a real-integration caller precondition and must be enforced by the real smoke / formal collection entrypoint.

It must not accept public overrides for:

- task IDs;
- target groups per task;
- target progress values;
- dummy steps;
- action budget;
- seed;
- `unnorm_key`;
- `center_crop`;
- camera resolution;
- quantization;
- success semantics.

Those values are scientific protocol constants, not user-tunable collection parameters.

---

## 12. Public Result Type

Define one small immutable result dataclass:

```python
@dataclass(frozen=True)
class PilotV02CollectionResult:
    sample_paths: tuple[Path, ...]
    manifest_path: Path
    run_status: str
    completeness_status: str | None
```

Do not expose additional public orchestration types unless required by a concrete implementation constraint.

Return-path semantics are frozen:

- `manifest_path` is the resolved filesystem path to the written manifest;
- `sample_paths` are resolved filesystem paths to accepted observation artifacts;
- `sample_paths` ordering must match canonical manifest ordering:
  `task_id` ascending → accepted-group traversal order → target progress `0.10, 0.40, 0.70, 0.90`.

Portable paths stored inside the manifest remain relative POSIX paths.

Export only the intended public Pilot v0.2 symbols from `shared_feature/__init__.py`.

---

## 13. Private Reduced-Smoke Entry

The formal public API must never reduce the frozen collection plan.

For later real integration smoke, provide one private/internal entry capable of running a reduced plan.

Recommended internal shape:

```python
def _collect_pilot_v02_with_plan(
    *,
    model,
    processor,
    pretrained_checkpoint: str | Path,
    checkpoint_identity: str,
    libero_revision: str,
    output_dir: str | Path,
    task_ids: Sequence[int],
    target_groups_per_task: int,
) -> PilotV02CollectionResult:
    ...
```

This helper is not part of the public scientific API.

It exists only so a dedicated smoke script can validate:

- task switching;
- official state traversal;
- successful rollout;
- four-point sampling;
- manifest generation;
- group-transaction behavior.

The smoke helper must still preserve:

- four observations/group;
- frozen progress targets;
- frozen rollout semantics;
- deterministic state ordering.

Do not expose arbitrary progress or rollout overrides even in the reduced-smoke path.

Reduced-smoke result status is frozen:

```text
run_status = "SMOKE_COMPLETED"
completeness_status = null
```

A reduced smoke must never emit formal completeness labels.

---

## 14. Preflight and Output-Lifecycle Boundary

The canonical `output_dir` for one collection invocation must be a dedicated fresh location.

### Preflight Phase

Preflight occurs before creating any new output directory or canonical artifact.

Preflight must validate at least:

1. non-empty `checkpoint_identity`;
2. non-empty `libero_revision`;
3. `pretrained_checkpoint` resolves to an existing local directory;
4. `output_dir`, if it exists, is an empty directory;
5. required runtime dependencies/configuration can be initialized sufficiently to verify the frozen suite/task availability contract;
6. the formal plan is internally valid.

If preflight fails:

```text
raise PilotV02CollectionError
do not create output_dir
do not write a fatal manifest
```

A pre-existing empty `output_dir` may remain empty.

### Active Output Phase

Only after preflight succeeds may the collector create the canonical layout.

After that point, failures are governed by the fatal-run contract and must attempt to write a fatal audit manifest.

Do not:

- resume from previous runs;
- merge with previous runs;
- overwrite previous artifacts;
- infer which existing files are safe to reuse.

Pilot v0.2 first implementation has:

```text
no resume semantics
```

---

## 15. Canonical Output Layout

Use this exact layout:

```text
<output_dir>/
    observations/
        <sample_id>.npz
    collection_manifest.json
```

Only accepted canonical `PilotObservation` files belong in `observations/`.

Rejected trajectories must not be serialized as canonical observation NPZ files.

Temporary files used transactionally must not remain after successful completion.

---

## 16. PilotObservation Contract

Do not modify `PilotObservation`.

Every accepted sample must use the existing schema and preserve:

```text
sample_id
task_id
initial_state_id
episode_id
step_id
normalized_episode_progress
base_rgb_raw
wrist_rgb_raw
state
prompt
episode_success
```

For backward compatibility:

```text
episode_id = initial_state_id
```

Sample ID grammar remains:

```text
libero_spatial__taskXX__stateYY__stepZZZZ
```

where:

- `XX` is the real task ID;
- `YY` is the official initial-state sequence index;
- `ZZZZ` is the valid-policy `step_id`.

`sample_id` must be globally unique within the collection.

Exactly four samples in one accepted group intentionally share:

```text
(task_id, initial_state_id)
```

---

## 17. Group Identity and Acceptance

Scientific group identity is:

```text
(task_id, initial_state_id)
```

Each group identity may correspond to exactly one accepted trajectory-group record.

One accepted group contains exactly four observations.

A candidate group is accepted only if:

1. runtime initialization succeeds;
2. one valid clean OpenVLA rollout is produced;
3. official success is true;
4. `T >= 4`;
5. the four frozen indices are unique;
6. all four PilotObservations validate;
7. all four sample IDs are unique;
8. all destination paths are unused;
9. all four artifacts can be committed transactionally.

No partial accepted group is allowed.

---

## 18. Candidate Traversal

For each task:

```text
accepted_count = 0
```

Visit official initial states in ascending index order.

For each state:

1. run exactly one clean rollout;
2. classify the outcome;
3. if rejected, record the reason and continue to the next state;
4. if accepted, commit exactly four observations and record the group;
5. stop the task after 5 accepted groups.

Rejected states may only be replaced by later states from the same task.

Do not use:

- random shuffling;
- visual selection;
- downstream-result-driven selection.

---

## 19. Transactional Group Write

One accepted group is the minimum write transaction.

Before committing:

1. construct all four `PilotObservation` objects;
2. validate all four objects;
3. compute all four final paths;
4. verify sample-ID uniqueness;
5. verify no destination path exists.

Write through temporary paths.

Only after all four writes succeed should the group be committed to its final canonical state.

If any write in the current group fails:

```text
rollback files created for the current incomplete group
```

Do not delete previously completed groups.

Do not leave a partial group in `observations/`.

---

## 20. Fatal-Run Semantics

A fatal error is not an ordinary candidate rejection.

On fatal error after the active output phase has begun:

1. rollback the current incomplete group, if any;
2. preserve all previously completed accepted groups;
3. attempt to write a fatal audit manifest to the canonical manifest path;
4. mark run status as `FATAL`;
5. record fatal error category and context;
6. raise `PilotV02CollectionError` with exception chaining.

Do not continue to later candidate states after a fatal error.

If writing the fatal manifest itself fails:

- perform best-effort cleanup of temporary files;
- preserve the original fatal exception;
- preserve the manifest-write exception as additional chained/contextual information;
- do not claim that a fatal manifest exists.

Do not silently convert the fatal run into `BLOCKED`.

State semantics are frozen:

```text
COMPLETED + COMPLETE
COMPLETED + USABLE_WITH_SHORTFALL
COMPLETED + BLOCKED
SMOKE_COMPLETED + null
FATAL + null
```

`BLOCKED` means a validly completed formal collection with insufficient scientific coverage only.

`FATAL` means execution, schema, filesystem, serialization, or dataset-integrity failure.

Internal invariant failures such as duplicate sample IDs, duplicate accepted group records, malformed accepted-group structure, schema inconsistency, or write failure are always `FATAL` when detected during collection.

---

## 21. Exact Manifest Artifact

Write:

```text
<output_dir>/collection_manifest.json
```

Encoding:

```text
UTF-8
```

JSON must be human-readable and deterministic:

```text
indent = 2
ensure_ascii = False
newline = LF
```

Schema version:

```text
pilot_v0_2_collection_v1
```

Do not include feature tensors.

The collection manifest is distinct from the C4 paired-feature manifest.

---

## 22. Exact Top-Level Manifest Schema

Canonical formal manifests use a closed schema.

Unless this contract explicitly says otherwise, every manifest object must contain exactly the fields listed for that object and no unspecified canonical fields.

Schema expansion requires a future schema-version change.

The successful/completed manifest must contain exactly these top-level fields:

```text
schema_version
pilot_version
suite
run_status
completeness_status
rollout
protocol
runtime
coverage
task_results
```

Required values:

```text
schema_version = "pilot_v0_2_collection_v1"
pilot_version = "0.2"
suite = "libero_spatial"
run_status = "COMPLETED"
```

`completeness_status` must be one of:

```text
COMPLETE
USABLE_WITH_SHORTFALL
BLOCKED
```

For fatal manifests:

```text
run_status = "FATAL"
completeness_status = null
```

and one additional required top-level field is present:

```text
fatal_error
```

---

## 23. Manifest `rollout` Object

Exact fields:

```text
policy_family
checkpoint_identity
resolved_checkpoint_path
unnorm_key
center_crop
camera_resolution
dummy_steps
max_valid_policy_actions
load_in_4bit
load_in_8bit
global_seed
environment_seed
do_sample
```

Frozen values:

```text
policy_family = "openvla"
unnorm_key = "libero_spatial_no_noops"
center_crop = true
camera_resolution = 512
dummy_steps = 10
max_valid_policy_actions = 300
load_in_4bit = false
load_in_8bit = false
global_seed = 7
environment_seed = 0
do_sample = false
```

`checkpoint_identity` is the supplied portable identity.

`resolved_checkpoint_path` is the resolved runtime path string.

The public formal collection path must reject an empty checkpoint identity.

This field does not authorize inventing an identity while the research item remains `OPEN`.

---

## 24. Manifest `protocol` Object

Exact fields:

```text
target_task_count
target_groups_per_task
observations_per_group
target_relative_progress
sampling_rounding
group_identity_fields
candidate_state_order
resume_enabled
```

Frozen values:

```text
target_task_count = 10
target_groups_per_task = 5
observations_per_group = 4
target_relative_progress = [0.10, 0.40, 0.70, 0.90]
sampling_rounding = "floor(q*(T-1)+0.5)"
group_identity_fields = ["task_id", "initial_state_id"]
candidate_state_order = "ascending_official_index"
resume_enabled = false
```

---

## 25. Manifest `runtime` Object

Exact fields:

```text
libero_revision
discovered_task_count
task_ids
official_initial_state_counts
```

Rules:

- `libero_revision` is the exact non-empty provenance string supplied by the caller;
- `task_ids` are sorted ascending;
- `official_initial_state_counts` maps each task ID string to the discovered count;
- the real runtime is authoritative;
- if required availability cannot support the frozen protocol, stop before canonical collection begins where possible.

Do not report fake/mock availability as real runtime provenance.

---

## 26. Manifest `coverage` Object

Exact fields:

```text
target_total_groups
target_total_observations
actual_total_groups
actual_total_observations
accepted_groups_per_task
```

Frozen targets:

```text
target_total_groups = 50
target_total_observations = 200
```

`accepted_groups_per_task` maps each task ID string to its accepted group count.

---

## 27. Per-Task Manifest Record

Each `task_results` entry must contain exactly:

```text
task_id
task_language
available_initial_state_count
target_accepted_groups
actual_accepted_groups
attempted_state_ids
accepted_state_ids
rejected_states
accepted_groups
```

Ordering:

```text
task_results by task_id ascending
attempted_state_ids in actual traversal order
accepted_state_ids in actual traversal order
accepted_groups in accepted state traversal order
```

---

## 28. Rejected-State Manifest Record

Each rejected-state record must contain exactly:

```text
initial_state_id
reason
trajectory_length
```

`reason` must be one of:

```text
policy_failure
trajectory_too_short
sampling_index_collision
```

`trajectory_length`:

- integer when a valid-policy trajectory exists;
- `null` only when the rejection semantics do not produce a valid trajectory length.

Infrastructure failures must never appear here.

---

## 29. Accepted-Group Manifest Record

Each accepted-group record must contain exactly:

```text
task_id
initial_state_id
trajectory_length
episode_success
samples
```

Required:

```text
episode_success = true
len(samples) = 4
```

Samples are ordered by:

```text
0.10, 0.40, 0.70, 0.90
```

---

## 30. Per-Sample Manifest Record

Each sample record must contain exactly:

```text
sample_id
step_id
target_relative_progress
actual_normalized_episode_progress
observation_path
```

`observation_path` must be a relative POSIX path from `output_dir`:

```text
observations/<sample_id>.npz
```

Do not store an absolute observation path.

---

## 31. Fatal Manifest Record

When `run_status = "FATAL"`, `fatal_error` must contain exactly:

```text
category
message
task_id
initial_state_id
```

`category` must be one of:

```text
RUNTIME_ERROR
DUMMY_PHASE_ERROR
OBSERVATION_ERROR
ACTION_ERROR
INTEGRITY_ERROR
SERIALIZATION_ERROR
FILESYSTEM_ERROR
MANIFEST_WRITE_ERROR
```

Generation rule:

- environment/model/runtime initialization or execution failure after preflight → `RUNTIME_ERROR`;
- dummy-phase `done` or success → `DUMMY_PHASE_ERROR`;
- malformed or invalid observation/state/PilotObservation input → `OBSERVATION_ERROR`;
- invalid action or action-generation contract failure → `ACTION_ERROR`;
- duplicate IDs, duplicate group records, malformed accepted-group structure, or internal invariant violation → `INTEGRITY_ERROR`;
- NPZ/JSON serialization failure before filesystem commit → `SERIALIZATION_ERROR`;
- directory/file creation, rename, replace, unlink, or other filesystem-operation failure → `FILESYSTEM_ERROR`;
- failure specifically while writing/replacing the canonical manifest → `MANIFEST_WRITE_ERROR`.

Preflight failures occur before fatal-manifest semantics and therefore do not receive a canonical `fatal_error` record.

`task_id` and `initial_state_id` may be `null` when failure occurs before a candidate state is active.

The fatal manifest must also preserve the completed task/group census accumulated before failure.

Do not serialize Python traceback text into the canonical scientific manifest.

The raised exception/log may preserve traceback context separately.

---

## 32. Completeness Classification

Classification is evaluated only for a non-fatal completed run.

### COMPLETE

All must hold:

```text
every task has exactly 5 accepted groups
total groups = 50
every accepted group has exactly 4 observations
total observations = 200
all structural integrity checks pass
```

### USABLE_WITH_SHORTFALL

Only if not COMPLETE, all must hold:

```text
every task has 4 or 5 accepted groups
at least one task has exactly 4
total groups is 40..49 inclusive
every accepted group has exactly 4 observations
all structural integrity checks pass
```

### BLOCKED

A completed formal collection is `BLOCKED` only when structural integrity is valid but scientific coverage is insufficient.

This includes:

- any task with fewer than 4 accepted groups; or
- fewer than 40 total accepted groups.

The following are not `BLOCKED`; when detected during collection they are `FATAL` integrity failures:

- malformed accepted-group structure;
- duplicate sample IDs;
- two accepted group records using the same scientific group identity;
- schema inconsistency;
- other internal structural invariant failure.

Do not classify a fatal execution as BLOCKED.

---

## 33. No C5 Split in This Task

Do not create TRAIN/HELD-OUT membership during collection.

The frozen split rule exists in the scientific documents, but split materialization is a later C5 data-preparation step.

Collection must remain independent of downstream C5 results.

---

## 34. Allowed Production Changes

Preferred production changes:

```text
shared_feature/pilot_v02_collector.py
shared_feature/__init__.py
```

Allowed only if necessary for safe low-level reuse:

```text
shared_feature/libero_collector.py
```

Do not modify more production files without reporting why the coding contract genuinely requires it.

---

## 35. Test Files

Add:

```text
tests/test_pilot_v02_collector.py
```

If `shared_feature/libero_collector.py` changes, existing tests for it must remain green.

A later real-smoke script may be added as:

```text
scripts/c5_d0_pilot_v02_smoke_integration.py
```

The smoke script must call only the private reduced-plan path and must not expose a second public scientific API.

---

## 36. Required Unit-Test Coverage

Tests must use fakes/mocks and must not require GPU, OpenVLA weights, or a real LIBERO installation.

Collectively verify at least:

1. deterministic task traversal;
2. deterministic state traversal;
3. stop after 5 accepted groups per formal task;
4. successful trajectory acceptance;
5. `done` without success → `policy_failure`;
6. action-budget exhaustion → `policy_failure`;
7. failed states replaced only by later states in the same task;
8. dummy-phase success/done → fatal error;
9. `T` counts policy-query observations only;
10. terminal returned observation is excluded;
11. exact target progress values;
12. exact rounding formula;
13. unique sampling indices required;
14. `T < 4` rejection;
15. exactly four samples per accepted group;
16. `step_id` semantics preserved;
17. target and actual progress remain distinct;
18. global `sample_id` uniqueness;
19. four samples intentionally share one group identity;
20. two accepted group records cannot share one group identity;
21. output directory must be fresh/empty;
22. canonical output layout;
23. no rejected observation NPZ artifacts;
24. group write rollback on partial failure;
25. prior completed groups survive a later fatal error;
26. fatal audit manifest is written;
27. fatal run is not classified BLOCKED;
28. deterministic manifest ordering;
29. exact manifest schema/version;
30. relative POSIX observation paths;
31. COMPLETE classification;
32. USABLE_WITH_SHORTFALL classification;
33. BLOCKED classification;
34. formal public API does not expose plan overrides;
35. reduced private plan preserves four-point sampling;
36. no feature extraction occurs;
37. no π0.5/OpenPI/JAX runtime dependency is introduced;
38. existing `collect_pilot_observations()` behavior remains compatible;
39. empty `checkpoint_identity` fails during preflight;
40. empty `libero_revision` fails during preflight;
41. nonexistent `pretrained_checkpoint` fails during preflight;
42. preflight failure creates no new output and no fatal manifest;
43. post-preflight fatal failure attempts fatal-manifest writing;
44. fatal-manifest write failure does not falsely claim a manifest exists;
45. smoke result is `SMOKE_COMPLETED` with `completeness_status = null`;
46. returned `sample_paths` follow canonical manifest order;
47. returned paths are resolved filesystem paths;
48. `runtime.libero_revision` is serialized exactly;
49. canonical nested manifest objects reject unspecified extra fields;
50. integrity failures are FATAL, never BLOCKED.

Do not write a test that expects `sampling_index_collision` to naturally occur for a normal valid `T`.

If collision handling needs direct coverage, test the defensive validation helper with a controlled synthetic input.

---

## 37. Regression Tests

Always run:

```text
tests/test_pilot_v02_collector.py
tests/test_pilot_observation.py
```

If `shared_feature/libero_collector.py` is modified, also run:

```text
tests/test_libero_collector.py
```

Run additional directly affected tests when necessary.

Do not broaden or refactor the full test infrastructure without need.

---

## 38. Static / Diff Checks

Run Ruff on all changed Python files.

Run:

```bash
git diff --check
```

All modified text files must use LF line endings.

Report exact commands and results.

---

## 39. Real Integration Smoke Boundary

Do not run the full formal Pilot v0.2 collection as part of this coding task.

After unit-level PASS and after the portable checkpoint identity is separately frozen, a later explicitly authorized real smoke should use a reduced internal plan.

Minimum useful smoke:

```text
at least 2 LIBERO-Spatial tasks
1 accepted successful group per selected task
4 observations per accepted group
```

The smoke must validate:

- task switching;
- official initial-state availability;
- frozen seeds/configuration;
- successful rollout;
- four-point sampling;
- sample identity;
- raw PilotObservation validity;
- collection manifest;
- group transaction behavior.

A reduced smoke result must use exactly:

```text
run_status = "SMOKE_COMPLETED"
completeness_status = null
```

and must never be labeled:

```text
COMPLETE
USABLE_WITH_SHORTFALL
BLOCKED
```

Those completeness labels belong to the formal frozen plan only.

The reduced smoke manifest/run representation is private/internal and must not redefine the canonical formal collection schema.

---

## 40. Documentation Boundary

Do not modify during this coding task:

```text
docs/pilot-v0.1-spec.md
docs/pilot-v0.2-spec.md
docs/research-map.md
AGENTS.md
```

Do not modify `task.md` during implementation unless explicitly requested.

If implementation reveals a scientific-contract conflict, stop and report it instead of editing scientific documents.

---

## 41. Forbidden Scope

Do not implement or modify:

- C2 OpenVLA feature extraction;
- C3 π0.5 feature extraction;
- C4 paired-feature construction;
- C5 split materialization;
- Linear CKA;
- PCA/SVD;
- CCA/SVCCA;
- shuffled null;
- null RNG seed;
- C5 decision thresholds;
- C5 PASS/FAIL logic;
- policy/action relevance;
- adversarial losses;
- Tex3D texture optimization;
- cross-VLA attack evaluation.

Do not modify external OpenVLA, OpenPI, or LIBERO repositories.

---

## 42. Stop Conditions

Stop and report instead of expanding scope if:

1. the implementation requires resolving the `OPEN` portable checkpoint identity;
2. the official runtime contradicts frozen task/state semantics;
3. safe low-level reuse requires breaking `collect_pilot_observations()`;
4. existing C1 behavior cannot remain valid;
5. implementation would require changing `PilotObservation`;
6. implementation would require modifying C2/C3/C4;
7. implementation begins materializing C5 split or statistical analysis;
8. fatal-run requirements cannot be satisfied without ambiguous dataset state;
9. exact manifest requirements conflict with the frozen scientific protocol;
10. more than the allowed production-file scope is genuinely required and has not been reported.

---

## 43. Completion Criteria

This coding task is complete only when:

```text
Pilot v0.2 high-level collector exists
formal public API is fixed-plan only
private reduced-smoke plan exists
existing C1 semantics remain intact
exact closed-schema collection manifest is implemented
LIBERO revision provenance is implemented
preflight/output lifecycle is implemented
transactional group writes are implemented
fatal audit behavior is implemented
execution-state semantics are implemented
quota/completeness logic is implemented
focused unit tests pass
required regressions pass
Ruff passes
git diff --check passes
```

Then report:

1. files changed;
2. behavior implemented;
3. behavior intentionally not implemented;
4. tests/checks run and results;
5. whether `libero_collector.py` changed;
6. unresolved risks or blockers.

Maximum claim:

```text
C5-D0 Pilot v0.2 collector — UNIT-LEVEL PASS
```

Do not claim formal Pilot v0.2 dataset collection success.

---

## 44. Follow-Up Sequence

After this task reaches UNIT-LEVEL PASS:

```text
freeze portable OpenVLA checkpoint identity
        ↓
explicitly authorize reduced real integration smoke
        ↓
audit smoke
        ↓
explicitly authorize full Pilot v0.2 collection
        ↓
C2 full OpenVLA extraction
        ↓
C3 full π0.5 extraction
        ↓
C4 full paired-feature manifest
        ↓
materialize frozen C5 group-aware split
        ↓
freeze independent C5 scientific decision contract
        ↓
run C5-A Linear CKA
        ↓
run C5-B SVCCA
```

Formal C5 PASS/FAIL evaluation remains blocked until the exact C5 decision rule and null RNG seed are frozen before inspecting formal C5 results.
