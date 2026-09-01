# AGENTS.md

## Project Context

This repository is a research codebase for Shared-Feature Tex3D.

Before modifying code for research tasks, read:

1. `docs/research-map.md`
2. the Pilot specification applicable to the current task:
   - `docs/pilot-v0.1-spec.md` for explicitly historical Pilot v0.1 work;
   - `docs/pilot-v0.2-spec.md` for current Pilot v0.2 / C5 data-preparation work;
3. `task.md` when the task is an authorized implementation contract.

These documents have different roles:

- `docs/research-map.md` defines the current research rationale, stage, roadmap, and unresolved research questions;
- `docs/pilot-v0.x-spec.md` defines the frozen scientific protocol for that Pilot version;
- `task.md` defines the current implementation contract;
- `AGENTS.md` defines long-lived repository discipline and document routing.

Model-specific notes are read only when relevant:

- `openvla_insight.md` for OpenVLA-specific tasks;
- `pi0_insight.md` for π0 / π0.5 / openpi-specific tasks.

Do not read unrelated research notes unless the current task requires them.

If `task.md` conflicts with the applicable frozen scientific specification, do not guess which one should win. Stop and report the inconsistency.

`task.md` may operationalize a frozen scientific decision, but it must not silently override one.

---

## Research Discipline

Treat the following labels strictly:

- `FACT`: supported by source code, experiment, or cited paper;
- `HYPOTHESIS`: proposed explanation that is not yet established;
- `DECISION`: currently frozen project choice;
- `OPEN`: unresolved question.

Never silently convert a `HYPOTHESIS` or `OPEN` item into an implementation assumption.

If implementation requires resolving an `OPEN` question, stop and report it instead of guessing.

Different `OPEN` items may block different stages. Do not overgeneralize one blocker to unrelated work.

---

## Scope Control

The roadmap and Pilot specifications describe future work, but they do not authorize implementing future stages.

For every coding task:

- implement only the explicitly requested coding contract;
- do not proactively implement later roadmap stages;
- do not refactor unrelated code;
- do not change public interfaces unless explicitly required;
- do not add speculative abstractions for anticipated future requirements;
- leave unrelated imperfections untouched.

Default limit per coding task:

- at most 3 production files modified;
- tests may use additional test files when necessary.

If the task genuinely requires exceeding this limit, explain why before making the broader change.

---

## Reviewability

Optimize for human reviewability rather than maximum code volume.

Prefer:

- small diffs;
- explicit interfaces;
- simple data flow;
- deterministic behavior;
- clear validation;
- focused tests.

Avoid:

- large multi-module rewrites;
- broad cleanup;
- architecture changes not required by the task;
- "while we are here" fixes.

A task is too large if its diff cannot reasonably be understood as one conceptual change.

---

## Planning Before Writing

For non-trivial tasks:

1. inspect the relevant existing code;
2. identify the minimum files and interfaces required;
3. state the proposed implementation plan;
4. implement only the approved/current contract.

When explicitly asked for read-only analysis or review, do not modify files.

---

## Testing

Every behavior change must have appropriate validation.

Prefer the smallest relevant test set first.

Do not broaden tests or refactor test infrastructure unless required by the current contract.

After implementation, report:

1. files changed;
2. behavior added or changed;
3. behavior intentionally left unchanged;
4. tests run and results;
5. assumptions made;
6. unresolved risks or questions.

---

## Git Discipline

Keep each coding contract as one conceptual change.

Do not mix unrelated changes into the same commit.

Do not rewrite or amend existing history unless explicitly requested.

Before finishing, inspect the final diff and confirm there is no scope creep.

Run the smallest relevant formatting or diff-integrity checks for modified text files. For Markdown changes, ensure line endings and whitespace do not introduce avoidable `git diff --check` failures.

---

## Shared-Feature Pilot Routing

### Pilot v0.1

Pilot v0.1 is historical and frozen.

Use `docs/pilot-v0.1-spec.md` only when the current task explicitly concerns Pilot v0.1 or historical v0.1 behavior.

Do not retroactively apply Pilot v0.2 scientific decisions to Pilot v0.1.

Do not rewrite Pilot v0.1 to match the current C5 design.

---

### Pilot v0.2

Pilot v0.2 is the current scientific protocol for C5 data preparation, collection, grouping, and downstream split semantics.

For Pilot v0.2 work, follow:

1. `docs/research-map.md`;
2. `docs/pilot-v0.2-spec.md`;
3. the current authorized `task.md`, if implementation has been explicitly requested.

Do not import scientific defaults from Pilot v0.1 when working on Pilot v0.2.

Do not combine scientific decisions from different Pilot versions.

When a task declares a Pilot version, that version's scientific specification is authoritative for its scientific protocol.

---

## Current Research Constraints

The current project stage remains representation analysis and validation.

Do not implement shared-feature adversarial texture optimization unless explicitly authorized after the relevant research gates are satisfied.

OpenVLA and π0.5 feature extraction are offline and model-specific.

Paired cross-model features must preserve the same frozen raw observation identity. Use the applicable C4 / current implementation contract for the exact pairing-validation mechanism.

Frozen statistical, sampling, grouping, split, CKA, SVCCA, and null-design decisions must not be replaced with alternatives without explicit scientific approval.

### DECISION — Vulnerability-First Main Route

The current default scientific route is vulnerability-first:

1. independently identify model-specific adversarial/action-relevant vulnerable
   features within each VLA;
2. only then analyze cross-model alignment or fusion of those vulnerable
   structures;
3. use any supported shared vulnerable structure to design a future
   single-surrogate attack loss;
4. evaluate transfer only on held-out VLA models.

The previous clean-shared-first route (`clean CCA → action-relevant shared
direction → attack`) is retained as complementary analysis, ablation, and a
possible alternative route. It is deferred from the default main line, not rejected
or classified as a failed route.

Multi-model clean/adversarial representation analysis is allowed during discovery.
Future formal texture optimization must remain single-surrogate and must not become
an ensemble objective such as `L_OpenVLA + L_pi0.5`.

Completed C5/C5-BM results remain evidence that heterogeneous VLA representations
contain stable alignable clean shared structure. Completed C6 intervention closure
remains engineering evidence that native O2/P2 features can be explicitly
intervened and propagated downstream. Neither establishes vulnerable features,
shared vulnerability, policy relevance, transferability, or a Tex3D attack.
Maintain the boundary `shared != vulnerable != policy-relevant != transferable`.

The next vulnerability-first cross-model feature study is `NOT YET CONTRACTED /
NOT AUTHORIZED`. Do not implement vulnerability discovery, adversarial feature
extraction, cross-model vulnerable-feature fusion, or any attack loss without a new
explicit contract.

---

## Current Research Decisions and OPEN Blockers

### DECISION — Portable OpenVLA Checkpoint Identity

The portable OpenVLA checkpoint identity for formal Pilot v0.2 collection is frozen as:

```text
openvla/openvla-7b-finetuned-libero-spatial
```

This identity intentionally has no immutable-revision suffix. A machine-local
checkpoint path is runtime provenance only and must not replace this identity.
The identity decision removes the checkpoint-specific scientific blocker, but
formal collection execution still requires an explicit coding/execution contract.

---

### DECISION / FACT — C5-A Representation Geometry

The completed C5-A contract remains the frozen scientific authority for C5-A.

The following C5-A decisions are frozen:

- primary estimator: debiased Linear CKA on `O2 ↔ P2`;
- robustness metric: Spearman RSA;
- diagnostic metric: biased Linear CKA;
- representation input: observation-level mean pooling;
- null: 50 group-block derangements;
- C5-A null RNG seed and exact convention: seed `7` under the contract-defined
  independent TRAIN / HELD-OUT streams;
- C5-A geometry-stage gate: `TRAIN PASS AND HELD-OUT PASS`.

These decisions authorize only the C5-A geometry-stage contract. They do not
constitute a final C5 PASS/FAIL rule and must not be generalized to C5-B.

Formal C5-A execution on the completed 200-record C4 paired dataset is complete:

- TRAIN `O2 ↔ P2` debiased Linear CKA: `0.528420895275`, empirical
  `p = 0.0196078431373`, `PASS`;
- HELD-OUT `O2 ↔ P2` debiased Linear CKA: `0.478168155531`, empirical
  `p = 0.0196078431373`, `PASS`;
- frozen C5-A geometry-stage result: `GO`.

This `GO` authorized proceeding to a separately contracted C5-B discussion only.
It was not by itself a final C5 PASS.

### DECISION / FACT — C5-B and C5 Representation-Stage Joint Gate

The completed C5-B contract remains the frozen scientific authority for C5-B.
It freezes:

- C5-B null repeats: `200`;
- C5-B root RNG seed: `17`, using contract-defined independent TRAIN and
  HELD-OUT `SeedSequence.spawn(2)` / PCG64 streams;
- one shared fixed-point-free group-derangement bank across both representation
  pairs and both PCA cutoffs;
- the fit-and-evaluate null: refit ordinary CCA after TRAIN derangement and
  evaluate it on an independently deranged HELD-OUT pairing;
- C5-B PASS iff the true `O2 ↔ P2`, 99%-PCA, TRAIN-ordered HELD-OUT Top5Mean is
  positive and its one-sided empirical `p <= 0.05`;
- C5 representation-stage PASS iff `C5-A GO AND C5-B PASS`.

Formal C5-B execution on the completed 200-record C4 paired dataset is complete:

- primary `O2 ↔ P2`, 99%-PCA, TRAIN-ordered HELD-OUT Top5Mean:
  `0.970583518852`;
- one-sided empirical `p = 0.00497512437811`;
- frozen C5-B result: `PASS`;
- frozen C5 representation-stage result: `PASS`.

This representation-stage result is complete, but it is not a final overall
research PASS and does not establish policy/action relevance or transferability.

### DECISION / FACT — C6-A Policy-Sensitivity Interface Closure

The completed C6-A contract remains the frozen scientific and engineering authority
for the completed C6-A interface-closure stage.

The C6-A source audit, scientific review, and final contract review are complete.
The frozen C6-A status is:

```text
INTERFACE FEASIBLE WITH EXPLICIT PREREQUISITES
```

C6-A freezes only the following interface-level conclusions:

- controlled intervention at OpenVLA O2 and π0.5 P2 is technically implementable,
  although neither model currently exposes a native continuation/replacement API;
- the confirmed primary cross-model action object is first-step translation only;
- rotation remains conditional on deployed robosuite `OSC_POSE` semantics, and
  gripper remains separate;
- the historical C5-B matrices were not persisted and cannot be claimed as exactly
  recoverable;
- a new versioned `O2 ↔ P2` 99%-PCA true-TRAIN mapping artifact must be produced by
  an explicitly authorized re-fit before C6-B;
- a separate O2/P2 intervention-interface coding stage is required before a real
  intervention smoke test.

This closure does not establish policy relevance, causal relevance, action
relevance, or transferability.

### DECISION / FACT — C5-BM Authoritative Mapping Materialization

The completed C5-BM contract remains the frozen scientific and engineering
authority for C5-BM. Its final read-only contract audit is `PASS`.

C5-BM freezes the materialization semantics for the new
authoritative reusable `O2 ↔ P2`, 99%-PCA, true-TRAIN PCA+CCA mapping. It does not
claim recovery of the historical unsaved in-memory C5-B matrices and does not
change the historical `C5-B PASS` result.

C5-BM implementation is `UNIT-LEVEL PASS`, and formal C5-BM materialization is
`FORMAL COMPLETE / PASS` under the current `task.md`. The authoritative mapping
contains `262` canonical components; complete source-feature validation passed for
all `200 / 200` pairs, all four historical C5-B file identities remained unchanged,
all nine mapping-array hashes verified, and all frozen historical scalar checks
were reproduced with absolute difference `0.0`.

### DECISION / FACT — C6 Intervention-Interface and Real-Smoke Closure

The O2/P2 intervention-interface contract is `FROZEN`. Its implementation is
`UNIT-LEVEL PASS`, and unit validation is `PASS`. Subsequent separately authorized
real-checkpoint validation established clean equivalence for OpenVLA `2 / 2` and
pi0.5 `2 / 2` frozen observations.

The original intervention-smoke results remain historical facts:

```text
OpenVLA: BLOCKED under the frozen translation-response gate
pi0.5: PASS
```

The OpenVLA follow-up token/logit diagnostic is `PASS`. Under the same frozen
observations, directions, and `alpha`, modified O2 measurably changed downstream
action-token logits while greedy action-token IDs remained unchanged; decoded
translation therefore remained unchanged. This supports a discrete
argmax/token-boundary explanation. It validates the OpenVLA intervention path and
downstream response, but it does not retroactively change the original smoke result
to `PASS`.

The C6 intervention-interface feasibility / intervention closure is `COMPLETE`.
The completed outputs and provenance are:

```text
real-smoke project commit:
eefc0e652f801c20f3de29c5d53e821dd65aa978

OpenVLA diagnostic artifact/project commit:
fffea7571fcde7922b0d0abc1a56d1e88439c011

experiment_inbox/c6-real-smoke-output/
experiment_inbox/c6-openvla-logit-diagnostic/
```

This closure establishes only real-checkpoint continuation equivalence and that
native O2/P2 interventions enter downstream policy computation. It does not show
that shared CCA directions are action-relevant, jointly policy-sensitive across
models, adversarially transferable, or sufficient for a Tex3D attack. C6-B is
`DEFERRED / RETAINED AS COMPLEMENTARY ROUTE` and remains unauthorized. The next
default scientific stage is the vulnerability-first cross-model feature study,
which is `NOT YET CONTRACTED / NOT AUTHORIZED`; Tex3D optimization remains
`NOT AUTHORIZED`.

### OPEN — Final Overall Research Gate and Later Stages

The final overall research gate remains `OPEN / NOT DEFINED HERE`. A C5
representation-stage PASS would not establish policy relevance, action relevance,
adversarial transferability, or attack effectiveness.

The following scientific stages remain `NOT STARTED`:

- policy/action relevance;
- transferability evaluation;
- Tex3D optimization.

Do not rerun or overwrite the completed formal C5-BM materialization, C6 real-smoke
artifacts, or OpenVLA diagnostic artifacts without a new explicit contract. Do not
begin C6-B formal policy/action analysis, transferability evaluation, or Tex3D
optimization without their required explicit authorization. Do not begin
vulnerability discovery, adversarial feature extraction, or cross-model vulnerable
feature alignment/fusion without a new contract. Do not infer a final overall
research PASS/FAIL from the completed representation-stage, C6-A interface result,
C5-BM PASS, or C6 intervention closure.

---

## Version and Document Conflict Rule

If any of the following occur:

- v0.1 and v0.2 scientific decisions appear to be mixed;
- `task.md` contradicts the applicable Pilot specification;
- implementation requires resolving an `OPEN` research decision;
- the current task would cross a documented research gate without authorization;

stop and report the conflict before modifying behavior.

Do not resolve scientific ambiguity by choosing the most convenient implementation.

---

## Stop Condition

When the requested coding contract is complete and its tests pass:

- stop;
- summarize the result;
- do not continue implementing the next roadmap item.
