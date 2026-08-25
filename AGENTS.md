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

---

## Current OPEN Blockers

### Portable OpenVLA Checkpoint Identity

The portable OpenVLA checkpoint identity for formal Pilot v0.2 collection is currently `OPEN`.

A machine-local filesystem path is not sufficient as the scientific checkpoint identity.

While this item remains `OPEN`:

```text
formal Pilot v0.2 collection execution MUST NOT start
```

This blocker does not by itself prohibit:

- documentation updates;
- AGENTS routing updates;
- preparation or review of implementation contracts;
- other work that does not require selecting or executing the unresolved formal checkpoint identity.

Do not invent or infer a portable checkpoint identity from a local directory name.

---

### Exact C5 Go/No-Go Decision Rule

The exact C5 PASS/FAIL decision rule and null-permutation RNG seed are currently `OPEN`.

While these items remain `OPEN`:

```text
formal C5 PASS/FAIL evaluation MUST NOT begin
```

This blocker does not prevent:

- Pilot v0.2 collection, once its own checkpoint blocker is resolved;
- C2/C3/C4 feature extraction and pairing;
- implementation or validation work that does not inspect formal C5 results to choose thresholds.

Do not choose significance thresholds, effect-size cutoffs, null seeds, or CKA/SVCCA joint decision logic during implementation.

These decisions must be frozen in an independent C5 scientific contract before formal C5 results are evaluated.

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