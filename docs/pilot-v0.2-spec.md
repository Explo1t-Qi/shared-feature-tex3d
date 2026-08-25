# Pilot v0.2 Scientific Specification — Revised

## 1. Purpose

Pilot v0.2 defines the scientific data-collection protocol used to support cross-VLA representation analysis between OpenVLA and π0.5.

Its purpose is to construct a paired clean-observation dataset with sufficient task and initial-state diversity to evaluate whether selected OpenVLA and π0.5 representation nodes exhibit cross-model shared structure.

Pilot v0.2 is a new scientific protocol. It does not replace or retroactively modify Pilot v0.1.

## 2. Observation Source

All observations are collected from successful clean OpenVLA rollouts in LIBERO-Spatial.

For every accepted raw observation, the same frozen observation is used for both OpenVLA and π0.5 feature extraction.

Therefore, for every paired sample:

\[
x_i^{\mathrm{OpenVLA}}
=
x_i^{\pi0.5}
=
x_i.
\]

Pilot v0.2 samples the successful clean OpenVLA rollout state distribution. It is not intended to represent:

- the π0.5 on-policy state distribution;
- a uniform LIBERO state distribution;
- an unbiased distribution over all successful LIBERO initial states.

Because the protocol accepts the first successful candidate states under deterministic canonical traversal, it introduces an additional deterministic selection bias beyond conditioning on OpenVLA success. No claim should be made that the resulting accepted states are representative of all successful initial states.

## 3. OpenVLA Rollout Identity

### DECISION — Frozen Rollout Configuration

Pilot v0.2 data collection must use one frozen OpenVLA LIBERO-Spatial policy identity and one frozen rollout configuration.

The following rollout settings are part of the scientific protocol:

- model family: OpenVLA;
- task suite: LIBERO-Spatial;
- `unnorm_key = "libero_spatial_no_noops"`;
- `center_crop = True`;
- LIBERO camera resolution: `512`;
- dummy/stabilization steps before policy execution: `10`;
- maximum valid-policy actions per rollout: `300`;
- 4-bit quantization: disabled;
- 8-bit quantization: disabled;
- global OpenVLA evaluation seed: `7`, applied through `set_seed_everywhere(7)` before model loading and rollout;
- LIBERO environment seed: `0`, applied by the official `get_libero_env()` path;
- action decoding: deterministic with `do_sample = False`;
- action generation and preprocessing must follow the already validated C1 OpenVLA/LIBERO inference path.

Both seed values and the deterministic action-decoding setting must be recorded in collection provenance.

### OPEN — Portable OpenVLA Checkpoint Identity

The portable logical identity of the OpenVLA checkpoint must be frozen before formal Pilot v0.2 collection begins.

The checkpoint identity must not be defined only by a machine-local filesystem path. A portable identity should use an appropriate stable identifier such as a repository/revision, artifact identifier, or checkpoint content digest.

The resolved local checkpoint path and other runtime provenance belong in the collection manifest rather than the scientific identity itself.

Formal Pilot v0.2 collection must not begin while this checkpoint identity remains `OPEN`.

## 4. Dataset Scope

Pilot v0.2 targets all 10 LIBERO-Spatial tasks.

For each task:

- traverse the official initial-state sequence in deterministic canonical order;
- accept the first 5 candidate initial states whose clean OpenVLA rollouts satisfy the acceptance criteria;
- sample exactly 4 observations from each accepted trajectory.

The target dataset is therefore:

- 10 tasks;
- 5 accepted trajectory groups per task;
- 4 observations per group;
- 50 groups total;
- 200 observations total.

The protocol prioritizes task diversity and initial-state diversity over dense frame sampling from a small number of trajectories.

## 5. Initial-State Traversal

### FACT — Audited LIBERO-Spatial State Availability

For each LIBERO-Spatial task, the candidate initial-state identifiers are the zero-based indices into the ordered sequence returned by:

`get_task_init_states(task_id)`.

Under the currently audited LIBERO-Spatial assets, all 10 LIBERO-Spatial tasks expose 50 official initial states, indexed:

`0, 1, ..., 49`.

The exact LIBERO revision used for formal collection must be recorded in provenance.

### DECISION — Candidate Traversal Rule

Candidate states are traversed in ascending index order.

Candidate states must not be randomly shuffled.

State selection must not use:

- visual cherry-picking;
- downstream representation-analysis results;
- CKA or SVCCA outcomes;
- attack-transfer outcomes;
- any other post-collection scientific result.

A rejected state may only be replaced by a later candidate state from the same task.

Collection for a task stops once either:

- 5 accepted trajectory groups have been obtained; or
- all available candidate initial states have been exhausted.

## 6. Valid-Policy Trajectory Definition

### DECISION — Dummy Phase and Trajectory Boundary

The dummy/stabilization phase is not part of the valid-policy trajectory.

If `done == True` or `env.check_success() == True` occurs during the 10-step dummy/stabilization phase, the event is not an ordinary policy outcome and must not be recorded as `policy_failure` or used as a normal candidate-state rejection. It is a collection-level protocol/runtime error; exact fatal-run handling belongs to the implementation contract.

After the 10 dummy steps, each valid-policy iteration proceeds conceptually as:

\[
x_t
\rightarrow
\text{policy query}
\rightarrow
a_t
\rightarrow
\text{environment step}.
\]

The observation \(x_t\) captured immediately before the policy query is part of the valid-policy trajectory.

Trajectory length \(T\) is defined as:

\[
T
=
\text{number of observations actually submitted to the OpenVLA policy query}.
\]

Therefore:

- dummy-phase observations are excluded from \(T\);
- the pre-action query observation is included;
- if an action causes success or environment termination, the query observation that produced that action remains included;
- the observation returned after that terminal or successful action is not appended to the valid-policy trajectory.

`step_id` is the zero-based index of the policy-query observation within this valid-policy trajectory.

It is not a simulator-global timestep.

## 7. Success, Failure, and Trajectory Acceptance

### DECISION

A trajectory is successful only if:

`env.check_success() == True`.

A successful trajectory is eligible for observation sampling only if all sampling requirements are satisfied.

The following outcomes are ordinary policy failures:

- `done == True` while `env.check_success() == False`;
- 300 valid-policy actions are exhausted without success.

An ordinary policy failure is recorded as:

`policy_failure`.

A failed policy rollout must not be automatically retried, extended beyond the frozen action budget, or replaced by an altered rollout configuration for the same candidate state.

Infrastructure or execution failures are not policy failures.

Examples include:

- model-loading failure;
- checkpoint corruption;
- GPU or memory failure;
- malformed environment observation;
- runtime exception;
- environment-construction failure;
- serialization or filesystem failure.

Such failures must not silently convert a candidate state into a normal rejected `policy_failure`.

Rejected trajectories follow this scientific precedence:

1. unsuccessful rollout → `policy_failure`;
2. successful rollout with \(T < 4\) → `trajectory_too_short`;
3. successful rollout with \(T \ge 4\) but non-unique target sampling indices → `sampling_index_collision`.

`sampling_index_collision` is retained as a defensive protocol condition. Under the frozen target progress values and normal valid \(T\), it is not expected to occur routinely.

No partially accepted trajectory group is permitted.

## 8. Observation Sampling

### DECISION

Each accepted trajectory contributes exactly 4 observations.

The target relative progress values are:

\[
Q
=
\{0.10, 0.40, 0.70, 0.90\}.
\]

For a valid-policy trajectory of length \(T\), the sampling index associated with target progress \(q\) is:

\[
t_q
=
\left\lfloor q(T-1)+0.5 \right\rfloor.
\]

The four resulting indices must be unique.

The protocol does not permit:

- duplicate sampled frames;
- task-specific changes to the target progress values;
- arbitrary neighboring-frame search to resolve a collision;
- partial acceptance of a trajectory group.

For any sampled observation with `step_id = t`:

\[
\mathrm{normalized\_episode\_progress}
=
\frac{t}{T-1}.
\]

The target progress \(q\) and the actual discrete normalized progress are conceptually distinct:

\[
q
\neq
\frac{t_q}{T-1}
\]

in general because \(t_q\) is integer-valued after rounding.

Both the target progress and the realized normalized progress should be preserved in collection provenance.

## 9. Observation and Group Identity

Each sampled observation retains the existing PilotObservation schema.

The sample identifier follows:

`libero_spatial__taskXX__stateYY__stepZZZZ`.

For backward compatibility:

`episode_id = initial_state_id`.

Because `episode_id` is not globally unique across multiple tasks, the scientific trajectory-group identity is:

`(task_id, initial_state_id)`.

All downstream group-aware analyses and splits must use this compound identity.

`sample_id` must be globally unique.

Each scientific group identity `(task_id, initial_state_id)` must identify exactly one accepted trajectory-group record. Exactly 4 sampled observations intentionally belong to that group and therefore share the same scientific group identity.

Two distinct accepted trajectory-group records must not use the same scientific group identity.

## 10. Pairing Requirement

The same accepted raw observation must be used to extract both OpenVLA and π0.5 representations.

Cross-model pairing must preserve stable observation identity exactly and must verify that both model-side feature records originate from the same frozen raw source observation.

The exact pairing-validation fields and hash schema belong in the C4 implementation contract rather than this scientific specification.

## 11. Dataset Completeness

### DECISION

Pilot v0.2 defines three mutually exclusive and collectively exhaustive completeness states.

The states are evaluated in the following order.

### COMPLETE

The dataset is `COMPLETE` if all of the following hold:

- every task has exactly 5 accepted groups;
- exactly 50 accepted groups are present in total;
- every accepted group contains exactly 4 observations;
- exactly 200 observations are present in total;
- all `sample_id` values are globally unique;
- each scientific group identity maps to exactly one accepted trajectory-group record containing exactly 4 observations.

A `COMPLETE` dataset may proceed to formal C5 analysis.

### USABLE_WITH_SHORTFALL

If the dataset is not `COMPLETE`, it is `USABLE_WITH_SHORTFALL` only if all of the following hold:

- every task has either 4 or 5 accepted groups;
- at least one task has exactly 4 accepted groups;
- the total number of accepted groups is between 40 and 49 inclusive;
- every accepted group contains exactly 4 observations;
- all `sample_id` values are globally unique;
- each scientific group identity maps to exactly one accepted trajectory-group record containing exactly 4 observations.

A `USABLE_WITH_SHORTFALL` dataset may proceed to formal C5 analysis, but the shortfall must be reported explicitly.

### BLOCKED

Any dataset that satisfies neither `COMPLETE` nor `USABLE_WITH_SHORTFALL` is `BLOCKED`.

This includes, but is not limited to:

- any task with fewer than 4 accepted groups;
- fewer than 40 accepted groups total;
- any accepted group containing other than exactly 4 observations;
- duplicate `sample_id` values;
- two distinct accepted trajectory-group records using the same scientific group identity;
- structural integrity failure.

A `BLOCKED` dataset must not be used for formal C5 analysis.

## 12. Downstream Statistical Grouping

The atomic statistical group for train/held-out partitioning is:

`(task_id, initial_state_id)`.

All 4 observations from one accepted trajectory group must be assigned together to either TRAIN or HELD-OUT.

Frame-level random splitting is prohibited.

## 13. Frozen Train / Held-Out Split Rule

### DECISION — Task-Stratified Group Split

The C5 split is task-stratified and group-aware.

For every LIBERO-Spatial task represented by an admissible Pilot v0.2 dataset:

- exactly 1 accepted trajectory group is assigned to HELD-OUT;
- all remaining accepted trajectory groups for that task are assigned to TRAIN.

Therefore:

- for a task with 5 accepted groups: 4 TRAIN + 1 HELD-OUT;
- for a task with 4 accepted groups: 3 TRAIN + 1 HELD-OUT.

Under a `COMPLETE` dataset, this yields:

- 40 TRAIN groups;
- 10 HELD-OUT groups;
- 160 TRAIN observations;
- 40 HELD-OUT observations.

Task coverage and group integrity take precedence over maintaining an exact global 80/20 ratio under shortfall.

The held-out group for each task must be chosen by a deterministic, representation-independent ranking rule that is frozen before formal C5 analysis.

The ranking rule must not depend on:

- trajectory length;
- success order;
- representation features;
- CKA or SVCCA results;
- downstream attack results.

### DECISION — Frozen Deterministic Held-Out Ranking

The frozen rule is:

1. for each accepted group, construct the canonical string

   `pilot-v0.2-c5-split-v1|task_id={task_id}|initial_state_id={initial_state_id}`

2. represent both `task_id` and `initial_state_id` as base-10 integers with no leading zeros;

3. encode the canonical string as UTF-8 with no trailing newline;

4. compute SHA-256 over those exact bytes;

5. represent the digest as the standard 64-character lowercase hexadecimal string;

6. rank accepted groups within each task by the tuple `(digest, initial_state_id)` in ascending order, comparing `digest` lexicographically and using `initial_state_id` as the deterministic tie-breaker;

7. assign the lowest-ranked group to HELD-OUT;

8. assign all remaining groups to TRAIN.

For example, task 2 and initial state 1 use the exact canonical string:

`pilot-v0.2-c5-split-v1|task_id=2|initial_state_id=1`

The serialized split manifest used by C5 is the authoritative materialization of this frozen split rule.

## 14. Scientific Limitations

Pilot v0.2 supports representation analysis only on observations generated by the frozen OpenVLA rollout protocol.

Results must not be interpreted as direct evidence that:

- shared representation is transferable;
- shared representation is policy-relevant;
- π0.5 would visit the same state distribution under its own policy;
- the selected successful states are an unbiased sample of LIBERO initial states;
- the accepted first-success states are representative of all successful states.

These questions require separate downstream analysis.

## 15. Scope Boundary

Pilot v0.2 defines the scientific collection and partitioning protocol required to construct the paired observation dataset for C5.

It does not define:

- exact manifest JSON schemas;
- exact checkpoint-path serialization;
- transactional write behavior;
- fatal-run recovery behavior;
- resume semantics;
- output-directory handling;
- reduced smoke-test APIs;
- exact hash-field serialization for C4 pairing verification;
- feature-extraction implementation details;
- CKA implementation details;
- SVCCA implementation details.

Those engineering decisions belong in `task.md` or the corresponding implementation contracts.
