# Pilot v0.2 Scientific Specification

## 1. Purpose

Pilot v0.2 defines the scientific data-collection protocol used to support the next-stage cross-VLA representation analysis between OpenVLA and π0.5.

Its purpose is to construct a paired clean-observation dataset with sufficient task and initial-state diversity to evaluate whether selected OpenVLA and π0.5 representation nodes exhibit cross-model shared structure.

Pilot v0.2 is a new scientific protocol and does not replace or retroactively modify Pilot v0.1.

## 2. Observation Source

All observations are collected from successful clean OpenVLA rollouts in LIBERO-Spatial.

For every accepted raw observation, the same frozen observation is used for both OpenVLA and π0.5 feature extraction.

Therefore, each cross-model feature pair must correspond to the same underlying environment observation.

Pilot v0.2 samples the successful clean OpenVLA rollout state distribution. It is not intended to represent the π0.5 on-policy state distribution or a uniform LIBERO state distribution.

## 3. Dataset Scope

Pilot v0.2 targets all 10 LIBERO-Spatial tasks.

For each task:

- traverse official initial states in deterministic canonical order;
- accept the first 5 initial states whose clean OpenVLA rollouts satisfy the acceptance criteria;
- sample exactly 4 observations from each accepted trajectory.

The target dataset is therefore:

- 10 tasks;
- 5 accepted trajectory groups per task;
- 4 observations per group;
- 50 groups total;
- 200 observations total.

The protocol prioritizes task diversity and initial-state diversity over dense frame sampling from a small number of trajectories.

## 4. Initial-State Traversal

For each LIBERO-Spatial task, candidate initial states are evaluated in deterministic canonical order:

`state00, state01, ..., state49`.

Candidate states must not be randomly shuffled.

State selection must not use visual cherry-picking or downstream representation-analysis results.

A rejected state may only be replaced by a later candidate state from the same task.

Collection for a task stops once 5 accepted trajectory groups have been obtained or all available candidate initial states have been exhausted.

## 5. Trajectory Acceptance

A trajectory is eligible only if the official LIBERO success signal is `True`.

Rejected trajectories are classified according to the following precedence:

1. unsuccessful rollout → `policy_failure`;
2. successful rollout with trajectory length `T < 4` → `trajectory_too_short`;
3. successful rollout with `T >= 4` but non-unique target sampling indices → `sampling_index_collision`.

No partially accepted trajectory group is permitted.

## 6. Observation Sampling

Each accepted trajectory contributes exactly 4 observations.

The target relative progress values are:

\[
Q = \{0.10, 0.40, 0.70, 0.90\}.
\]

For a valid-policy trajectory of length \(T\), the sampling index for progress value \(q\) is:

\[
t_q = \left\lfloor q(T-1) + 0.5 \right\rfloor.
\]

The four resulting indices must be unique.

The protocol does not permit:

- duplicate frames;
- task-specific changes to the target progress values;
- arbitrary neighboring-frame search to resolve collisions;
- partial acceptance of a trajectory group.

## 7. Observation and Group Identity

Each sampled observation retains the existing PilotObservation schema.

The sample identifier follows:

`libero_spatial__taskXX__stateYY__stepZZZZ`

`step_id` denotes the zero-based observation/action query index within the valid-policy rollout after the dummy/stabilization phase. It is not a simulator-global timestep.

For backward compatibility:

`episode_id = initial_state_id`.

Because `episode_id` is not globally unique across multiple tasks, the scientific trajectory-group identity is:

`(task_id, initial_state_id)`.

All downstream group-aware analyses and splits must use this compound identity.

## 8. Dataset Completeness

Pilot v0.2 defines three completeness states.

### COMPLETE

The dataset is `COMPLETE` if:

- every task has exactly 5 accepted groups;
- 50 accepted groups are present in total;
- every accepted group contains exactly 4 observations;
- 200 observations are present in total.

### USABLE_WITH_SHORTFALL

The dataset is `USABLE_WITH_SHORTFALL` only if all of the following hold:

- every task has at least 4 accepted groups;
- at least 40 accepted groups are present in total;
- every accepted group contains exactly 4 observations.

### BLOCKED

The dataset is `BLOCKED` if either:

- any task has fewer than 4 accepted groups; or
- fewer than 40 accepted groups are present in total.

A `BLOCKED` dataset must not be used for formal C5 analysis.

## 9. Pairing Requirement

The same accepted raw observation must be used to extract both model representations.

For each observation \(x_i\):

\[
x_i^{\mathrm{OpenVLA}} = x_i^{\pi0.5} = x_i.
\]

Paired representation analysis must preserve this observation identity exactly.

## 10. Downstream Statistical Grouping

The atomic statistical group for train/held-out partitioning is:

`(task_id, initial_state_id)`.

All 4 observations from the same accepted trajectory group must be assigned together to either TRAIN or HELD-OUT.

Frame-level random splitting is prohibited.

The target split is approximately 80% groups for TRAIN and 20% groups for HELD-OUT, while preserving group integrity.

The serialized split manifest used by C5 is the authoritative definition of the final split.

## 11. Scope Boundary

Pilot v0.2 defines only the scientific collection protocol required to construct the paired observation dataset.

It does not define:

- implementation-specific manifest schemas;
- transactional write behavior;
- fatal-run recovery behavior;
- resume semantics;
- output-directory handling;
- reduced smoke-test APIs;
- feature-extraction implementation details;
- CKA or SVCCA implementation details.

Those engineering decisions belong in `task.md` or the corresponding implementation contracts.
