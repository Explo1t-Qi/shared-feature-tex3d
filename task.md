# C5-D0 Coding Contract — Pilot-200 Scientific Dataset Collection

## Goal

Implement the formal Pilot-200 observation dataset collector used to prepare
scientific data for C5 shared-representation analysis.

This milestone does NOT perform C5 representation analysis.

Its purpose is to construct a balanced, auditable, deterministic set of
real LIBERO-Spatial observations collected from successful clean OpenVLA
rollouts.

The target dataset is:

    LIBERO-Spatial
    all 10 tasks, subject to runtime verification
    × 5 accepted initial-state groups per task
    × 4 observations per accepted trajectory
    = target 200 PilotObservations

The four observations from each accepted trajectory correspond to
predefined relative trajectory progress targets:

    0.10
    0.40
    0.70
    0.90

The resulting PilotObservations will later be consumed independently by:

    C2 OpenVLA feature extraction
    C3 π0.5 feature extraction
    C4 paired-feature validation
    C5 statistical analysis

This milestone must not perform any of those later stages.

---

## Current Project Status

Closed:

    C0 PilotObservation schema                   PASS
    
    C1 LIBERO observation collector              PASS
    C1 real OpenVLA/LIBERO smoke                 PASS
    
    C2 OpenVLA feature extractor                 PASS
    C2 real OpenVLA/GPU smoke                    PASS
    
    C3 π0.5 feature extractor                    PASS
    C3 real π0.5 integration smoke               PASS
    
    C4 paired-feature manifest builder           UNIT-LEVEL PASS
    C4 real-artifact smoke validation            PASS

C5 scientific methodology has been designed, but the current real
C2/C3/C4 artifacts contain only two smoke observations and must NOT be
used for scientific C5 conclusions.

---

# Scientific Dataset Decisions

## DECISION 1 — Suite

Pilot-200 uses:

    LIBERO-Spatial only

Do not mix LIBERO-Object, LIBERO-Goal, LIBERO-10, or other suites into
this first Pilot dataset.

The first Pilot must keep the OpenVLA policy/checkpoint family fixed.

---

## DECISION 2 — Task Coverage

The intended design is to cover all LIBERO-Spatial tasks.

The currently expected count is:

    10 tasks

but task count MUST be verified from the real official LIBERO runtime
before hardcoding availability assumptions.

If the real runtime does not expose the expected task count:

    STOP

and report the exact discovered task availability.

Do not silently substitute another suite or task subset.

---

## DECISION 3 — Dataset Structure

Target:

    5 accepted initial-state groups per task

Each accepted group contributes:

    exactly 4 observations

Target total:

    10 tasks
    × 5 groups/task
    × 4 observations/group
    = 200 observations

Target groups:

    50

---

## DECISION 4 — Balanced Task Contribution

Target contribution per task:

    5 accepted groups
    20 observations

Do not compensate for a difficult task by:

- oversampling another task;
- taking more than four observations from another trajectory;
- duplicating observations;
- adding dense adjacent frames.

Rejected states may only be replaced by later candidate states from the
same task.

---

## DECISION 5 — Minimum Dataset Usability

Preferred dataset:

    5 accepted groups per task
    50 total groups
    200 observations

Minimum acceptable Pilot coverage:

    at least 4 accepted groups for EVERY task
    at least 40 accepted groups total
    exactly 4 observations per accepted group

If any task has fewer than four accepted groups after candidate states
are exhausted:

    Pilot-200 dataset preparation is BLOCKED.

Do not begin C5 scientific analysis.

A dataset with small quota shortfall may later contain fewer than
200 observations, but the shortfall must be explicit and auditable.

---

# Rollout Distribution

## DECISION 6 — Rollout Source

Formal Pilot observations come from:

    successful clean OpenVLA policy rollouts

The same frozen raw observations are later supplied independently to
OpenVLA and π0.5 feature extraction.

Do NOT generate separate observation datasets from OpenVLA and π0.5
rollouts.

The Pilot therefore samples the state distribution visited by successful
clean OpenVLA trajectories.

This is an explicit scientific limitation and must be preserved in
provenance.

---

## DECISION 7 — Successful Trajectories Only

Only trajectories for which the official LIBERO environment reports
success are accepted into Pilot-200.

The authoritative success mechanism remains the official/public
environment success check already used by the validated C1 collector.

A normal unsuccessful clean-policy rollout is:

    a rejected candidate group

not automatically an infrastructure exception.

Examples of rejected candidate outcomes:

    policy_failure
    trajectory_too_short
    sampling_index_collision

Infrastructure or schema failures remain errors.

Examples:

    environment initialization error
    malformed observation
    invalid OpenVLA action
    serialization error
    schema validation failure

Do not conflate policy failure with infrastructure failure.

---

# Observation Sampling

## DECISION 8 — Relative Progress Targets

For every accepted successful trajectory, collect exactly four
observations corresponding to:

    0.10
    0.40
    0.70
    0.90

of the valid policy trajectory.

Do not use the existing C1 uniform 20-frame sampling policy for
Pilot-200.

Do not use:

    trajectory start = 0.0

or:

    exact terminal = 1.0

as formal Pilot-200 target points.

---

## DECISION 9 — Full Trajectory First

The complete successful trajectory must be collected before formal
sampling indices are chosen.

Sampling therefore follows:

    collect complete trajectory
    determine T
    compute four target indices
    validate four unique indices
    construct four PilotObservations

Do not sample formal Pilot-200 observations online before the trajectory
length is known.

---

## DECISION 10 — Sampling Index Rule

For target relative progress:

    q in {0.10, 0.40, 0.70, 0.90}

choose the deterministic nearest simulator timestep to:

    q * (T - 1)

Use one explicitly implemented deterministic rounding convention.

Recommended:

    floor(q * (T - 1) + 0.5)

The implementation must use one frozen rule for every task/state.

Do not adapt sampling percentages by task.

---

## DECISION 11 — Unique Real Timesteps

The four target progress values must map to four distinct real
trajectory indices.

If they do not:

    reject the candidate group

Do not:

- duplicate frames;
- perturb sampling percentages;
- search arbitrary nearby steps;
- reduce the group to three observations.

Every accepted group contributes exactly four real observations.

---

# Sample Identity and PilotObservation

## DECISION 12 — Preserve Existing PilotObservation Semantics

Do NOT change the PilotObservation schema.

Preserve existing semantics for:

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

Raw image semantics must remain unchanged.

Do not preprocess raw images before serialization.

---

## DECISION 13 — Sample ID Grammar

Preserve the existing sample ID grammar:

    libero_spatial__taskXX__stateYY__stepZZZZ

where:

    XX = real task ID
    YY = real official initial-state ID
    ZZZZ = real policy-trajectory simulator step ID

Do not replace real step identity with:

    quantile01
    progress40
    sample3

or other synthetic aliases.

---

## DECISION 14 — Actual vs Target Progress

PilotObservation continues to store the actual:

    normalized_episode_progress

derived from the selected real step and actual trajectory length.

The new collection manifest separately records:

    target_relative_progress

Example:

    target_relative_progress = 0.40
    actual normalized progress = 0.3978

Do not overwrite actual progress with the target fraction.

---

# Existing C1 Collector Boundary

## DECISION 15 — Existing C1 Is Already Validated

The existing:

    shared_feature/libero_collector.py

contains validated low-level OpenVLA/LIBERO rollout behavior.

Do not redesign or duplicate:

- official LIBERO environment setup;
- initial-state reset semantics;
- dummy-step behavior;
- OpenVLA policy preprocessing;
- get_action semantics;
- gripper normalization;
- gripper inversion;
- raw observation capture;
- canonical 8D state generation;
- official success detection.

---

## DECISION 16 — Backward Compatibility

Existing public API:

    collect_pilot_observations(...)

must remain behaviorally backward-compatible.

Existing C1 tests and smoke semantics must remain valid.

Pilot-200 must not silently redefine the existing C1 collector from:

    fixed-task generic Pilot collection

into:

    multi-task success-only Pilot-200 collection.

---

## DECISION 17 — New High-Level Orchestration Layer

Pilot-200 must be implemented as a new higher-level dataset collection
layer.

Recommended production module:

    shared_feature/pilot200_collector.py

Recommended public API:

    class Pilot200CollectionError(RuntimeError):
        ...
    
    def collect_pilot200_observations(
        *,
        model,
        processor,
        pretrained_checkpoint: str | Path,
        output_dir: str | Path,
        manifest_path: str | Path,
        unnorm_key: str | None = None,
        center_crop: bool = True,
    ) -> tuple[Path, ...]:
        ...

A small return dataclass may be proposed only if clearly necessary.

Do not expose many additional public APIs.

---

## DECISION 18 — Reuse Low-Level Rollout Logic

Do not copy the low-level rollout implementation into
pilot200_collector.py.

A minimal refactor of:

    shared_feature/libero_collector.py

is allowed only if required to expose reusable internal functionality.

Any such refactor must:

1. preserve collect_pilot_observations() behavior;
2. preserve existing C1 public API;
3. preserve existing C1 tests;
4. preserve existing real-smoke semantics;
5. introduce no scientific-policy changes into the low-level primitive.

If code reuse would require a risky C1 redesign:

    STOP

and report the design blocker.

---

# Runtime Availability Audit

## Required Real-Runtime Audit Before Finalizing Availability Constants

Before freezing task/state iteration constants, inspect the official
LIBERO-Spatial runtime and determine:

    actual task count

and for every task:

    number of official initial states

Report these values.

Do not assume availability from historical documentation alone.

The high-level collector should preferably derive availability from
the official runtime rather than hardcoding arbitrary state limits such as:

    tuple(range(10))

if more official initial states exist.

---

# Candidate-State Traversal

## DECISION 19 — Deterministic State Order

For every task, candidate official initial states are visited in
deterministic ascending state-ID order:

    state00
    state01
    state02
    ...

unless the official runtime exposes a different frozen canonical
ordering.

Do not randomly shuffle candidate states.

Do not visually inspect states and select preferred examples.

---

## DECISION 20 — Quota Filling

For each task:

    accepted_groups = []

Iterate candidate states deterministically.

For each candidate:

1. collect one clean OpenVLA trajectory;
2. classify runtime outcome;
3. require official success;
4. derive four frozen progress samples;
5. validate the four PilotObservations;
6. write the group transactionally;
7. mark the state accepted;
8. stop once 5 groups have been accepted.

Rejected states remain part of the collection census.

---

# Group Acceptance Contract

A candidate `(task_id, initial_state_id)` is accepted only if:

1. official environment/task initialization succeeds;
2. valid clean OpenVLA rollout is produced;
3. official task success is true;
4. trajectory supports four unique frozen sampling indices;
5. all four raw observations satisfy existing C1/PilotObservation rules;
6. all four sample IDs are unique;
7. no destination collisions exist;
8. all four records can be serialized successfully.

Anything less does not form an accepted group.

---

# Transactional Group Writing

Before writing one accepted group:

1. construct all four PilotObservation records;
2. determine all four output paths;
3. validate all records;
4. verify no output path exists;
5. verify no sample ID collision exists.

Then write the four artifacts.

If writing fails partway through:

    remove only files created by the current group write

when safe to do so.

Do not leave a group partially represented in the canonical dataset.

Do not delete pre-existing artifacts.

---

# Output Directory

The observation output directory contains only canonical accepted
PilotObservation NPZ files for this collection run.

Do not store rejected trajectories as canonical PilotObservation files.

The collector may create necessary parent directories only after
configuration/runtime validation succeeds.

Do not silently overwrite an existing observation artifact.

---

# Collection Manifest

## Required Artifact

Pilot-200 must produce one human-readable UTF-8 JSON collection manifest.

Recommended schema version:

    pilot200_collection_v1

The manifest is the authoritative dataset-level source for:

    task identity
    initial-state grouping
    trajectory acceptance/rejection
    target relative progress
    coverage census

It complements PilotObservation metadata.

It does not replace C4 paired-feature manifest.

---

## Manifest Top-Level Information

The manifest should include at least:

    schema_version
    suite
    rollout_policy
    pretrained_checkpoint
    unnorm_key
    center_crop
    
    target_groups_per_task
    observations_per_group
    target_relative_progress
    
    discovered_task_count
    target_total_groups
    target_total_observations
    
    actual_total_groups
    actual_total_observations
    
    task_results

Do not include feature tensors.

---

## Per-Task Census

Each task entry must preserve at least:

    task_id
    task_language
    available_initial_state_count
    target_accepted_groups
    actual_accepted_groups
    attempted_state_ids
    accepted_state_ids
    rejected_states

---

## Rejected-State Record

Each rejected state record must contain at least:

    initial_state_id
    reason

Prefer stable rejection reason codes such as:

    policy_failure
    trajectory_too_short
    sampling_index_collision

Infrastructure exceptions should not be silently converted into ordinary
policy_failure records.

---

## Accepted-Group Record

Each accepted group must contain at least:

    task_id
    initial_state_id
    trajectory_length
    episode_success
    samples

Require:

    episode_success == true

---

## Per-Sample Collection Record

For each of the four observations record at least:

    sample_id
    step_id
    target_relative_progress
    actual_normalized_episode_progress
    observation_path

The manifest may use deterministic paths relative to the manifest parent.

If paths are stored, use one frozen relative POSIX-path policy.

---

# Collection Manifest Ordering

Manifest ordering must be deterministic.

Recommended:

    tasks sorted by task_id
    attempted states in actual deterministic traversal order
    accepted groups in state-ID traversal order
    samples ordered by target progress:
        0.10
        0.40
        0.70
        0.90

Do not rely on filesystem enumeration ordering.

---

# Completeness Status

The manifest must make it possible to derive one of:

    COMPLETE
    USABLE_WITH_SHORTFALL
    BLOCKED

Recommended semantics:

COMPLETE:

    every task has 5 accepted groups
    total groups = 50
    total observations = 200

USABLE_WITH_SHORTFALL:

    every task has >= 4 accepted groups
    total groups >= 40
    exactly 4 observations per accepted group
    but at least one task has fewer than 5 groups

BLOCKED:

    any task has fewer than 4 accepted groups
    or total groups < 40
    or accepted group integrity is violated

Do not call a shortfall dataset COMPLETE.

---

# No C5 Split During Collection

Do NOT decide train/held-out membership during collection.

The sequence is:

    collect and freeze canonical dataset
    THEN
    construct C2/C3/C4 artifacts
    THEN
    create C5 group-aware split

Collection decisions must not depend on downstream held-out results.

---

# Forbidden Scope

Do not implement:

- OpenVLA feature extraction;
- π0.5 feature extraction;
- C4 paired-feature construction;
- train/held-out splitting;
- PCA;
- SVD;
- CCA;
- SVCCA;
- CKA;
- regression;
- shuffled null;
- representation similarity;
- plotting;
- attack objectives;
- Tex3D integration;
- policy relevance;
- adversarial transfer testing.

Do not modify OpenVLA/OpenPI/LIBERO external repositories.

---

# Allowed Production Changes

Preferred:

    shared_feature/pilot200_collector.py
    shared_feature/__init__.py

A minimal compatibility-preserving modification to:

    shared_feature/libero_collector.py

is allowed only if needed for safe low-level rollout reuse.

Tests may add:

    tests/test_pilot200_collector.py

If libero_collector.py is changed, existing:

    tests/test_libero_collector.py

must remain green.

Avoid modifying unrelated files.

---

# Engineering Constraints

## Preserve Validated Observation Semantics

Formal accepted samples must continue to use the frozen:

    PilotObservation

implementation.

Do not create a parallel NPZ schema.

---

## No Model-Specific Feature Work

The collector uses OpenVLA only as the clean rollout policy.

It must not extract or save:

    O1-S
    O1-F
    O2
    P1
    P2

Feature extraction remains separate.

---

## No π0.5 Runtime Dependency

Pilot-200 collection must not require:

    OpenPI
    JAX
    π0.5 checkpoint

π0.5 only consumes the frozen observations later.

---

# Error Handling

Use:

    Pilot200CollectionError

for high-level Pilot-200 orchestration and dataset integrity failures.

Preserve lower-level collection error context through exception chaining.

Distinguish:

    rejected candidate state

from:

    fatal infrastructure error.

Do not silently skip malformed observations or write failures.

Do not silently continue after a failure that makes dataset provenance
ambiguous.

---

# Unit-Test Requirements

Tests should use fakes/mocks for LIBERO/OpenVLA runtime boundaries.

Do not require GPU, real OpenVLA, or real LIBERO for unit tests.

Collectively verify:

1. deterministic multi-task traversal;
2. deterministic initial-state traversal;
3. 5 accepted groups/task target;
4. successful rollout acceptance;
5. policy failure rejection;
6. failed states replaced only from same task;
7. stop after quota satisfied;
8. exact four samples per accepted group;
9. target progress = 0.10/0.40/0.70/0.90;
10. deterministic nearest-step mapping;
11. four unique indices required;
12. short trajectory rejection;
13. real step IDs retained in sample IDs;
14. target vs actual progress remain distinct;
15. PilotObservation schema unchanged;
16. no partial group survives write failure;
17. no source artifact overwrite;
18. duplicate sample IDs rejected;
19. deterministic manifest ordering;
20. exact dataset census;
21. COMPLETE status;
22. USABLE_WITH_SHORTFALL status;
23. BLOCKED status;
24. per-task minimum coverage enforcement;
25. rejected-state reason preservation;
26. non-NPZ unrelated files do not become observations;
27. no feature extraction occurs;
28. existing collect_pilot_observations API remains compatible.

---

# Required Existing-Test Regression Check

If shared_feature/libero_collector.py is modified, run at minimum:

    tests/test_libero_collector.py
    tests/test_pilot_observation.py

Also run:

    tests/test_pilot200_collector.py

and preferably the full suite.

---

# Ruff / Static Checks

Run Ruff on all changed Python files.

Run:

    git diff --check

Report both results.

---

# Real Integration Boundary

Unit implementation may establish only:

    Pilot-200 collector — UNIT-LEVEL PASS

Do not immediately collect all 200 observations during the coding task.

After unit-level acceptance, perform a separate narrow real integration
smoke first.

Recommended real smoke:

    multiple LIBERO-Spatial tasks
    at least one successful accepted group per selected task
    four formal observations/group

The smoke must verify:

    task switching
    real official state availability
    successful rollout
    10/40/70/90 sampling
    sample identity
    collection manifest
    raw PilotObservation validity

Only after that smoke is accepted should the full Pilot-200 collection run
begin.

---

# Required Runtime Audit Output

Before claiming implementation complete, report from the actual checked
LIBERO API or clearly identify that runtime verification remains for the
real smoke:

    discovered LIBERO-Spatial task count
    official initial-state availability semantics

Do not fabricate these values in unit tests and report them as real facts.

---

# Documentation Boundary

Do not update:

    docs/pilot-v0.1-spec.md
    docs/research-map.md

inside this coding task.

Documentation synchronization is a separate later milestone.

Do not modify task.md during implementation unless explicitly asked.

---

# Maximum Status After Coding

Maximum successful coding status:

    C5-D0 Pilot-200 collector — UNIT-LEVEL PASS

Do not claim:

    Pilot-200 Dataset — COMPLETE

until the full real collection has been executed and audited.

Do not claim:

    shared representation discovered
    transferable representation discovered
    policy-relevant representation discovered

---

# Follow-Up Sequence

After unit-level PASS:

    real multi-task collection smoke

then:

    full Pilot-200 observation collection

then:

    C2 full OpenVLA feature extraction

then:

    C3 full π0.5 feature extraction
        batch_size = 1

then:

    C4 full paired-feature manifest

then:

    freeze C5 group-aware train/held-out split

then:

    implement/run C5 SVCCA analysis

---

# Stop Conditions

STOP and report rather than expanding scope if:

1. real LIBERO-Spatial task availability contradicts the frozen design;
2. official initial-state availability cannot support the required coverage;
3. safe reuse of low-level C1 rollout logic requires breaking the existing
   collect_pilot_observations API;
4. existing C1 tests cannot remain behaviorally valid;
5. successful trajectories cannot provide the frozen four-point sampling
   protocol;
6. implementation would require changing PilotObservation semantics;
7. implementation would require modifying C2/C3/C4;
8. implementation begins introducing C5 statistical analysis.

The first coding task ends when:

    Pilot-200 high-level orchestration exists
    existing C1 semantics remain intact
    collection manifest is implemented
    quota/completeness logic is implemented
    focused tests pass
    regressions pass
    Ruff passes
