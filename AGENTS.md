# AGENTS.md

## Project Context

This repository is a research codebase for Shared-Feature Tex3D.

Before modifying code for research tasks, read:

1. `docs/research-map.md`
2. `docs/pilot-v0.1-spec.md`

These documents define the current research goal, threat model, frozen Pilot design, and research gates.

Model-specific notes are read only when relevant:

- `openvla_insight.md` for OpenVLA-specific tasks
- `pi0_insight.md` for π0/openpi-specific tasks

Do not read unrelated research notes unless the current task requires them.

---

## Research Discipline

Treat the following labels strictly:

- `FACT`: supported by source code, experiment, or cited paper
- `HYPOTHESIS`: proposed explanation that is not yet established
- `DECISION`: currently frozen project choice
- `OPEN`: unresolved question

Never silently convert a `HYPOTHESIS` or `OPEN` item into an implementation assumption.

If implementation requires resolving an `OPEN` question, stop and report it instead of guessing.

---

## Scope Control

The roadmap and Pilot specification describe future work, but they do not authorize implementing future stages.

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

---

## Shared-Feature Pilot Constraints

For Pilot v0.1, follow `docs/pilot-v0.1-spec.md` exactly.

In particular:

- Pilot is currently representation analysis only;
- do not implement shared-feature attack optimization unless explicitly requested;
- OpenVLA and π0 feature extraction are offline and model-specific;
- paired samples must be aligned by stable `sample_id`;
- do not replace frozen statistical or data-split decisions with alternatives without explicit approval.

---

## Stop Condition

When the requested coding contract is complete and its tests pass:

- stop;
- summarize the result;
- do not continue implementing the next roadmap item.