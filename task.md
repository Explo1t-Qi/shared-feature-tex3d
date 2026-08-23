Read first:

- AGENTS.md
- docs/research-map.md
- docs/pilot-v0.1-spec.md

This coding task implements only the first Pilot v0.1 data contract.

Goal
-----
Implement the persistent observation sample schema used by the Pilot dataset.

This task must define how one saved LIBERO observation is represented and serialized.

Allowed files
-------------
You may create or modify only:

- one production module for the observation schema / serialization
- one corresponding test module

If an existing package structure already provides a clearly appropriate location, use it.
Do not reorganize unrelated packages.

Forbidden scope
---------------
Do NOT implement:

- LIBERO rollout collection
- OpenVLA inference or feature extraction
- pi0/openpi inference or feature extraction
- train / held-out splitting logic
- CCA / PCA / SVD / CKA / regression
- feature serialization
- Tex3D attack logic
- renderer changes
- model preprocessing
- CLI entrypoints
- unrelated refactors

Do not introduce abstractions for future stages unless they are strictly necessary for this data contract.

Required behavior
-----------------
Define one explicit observation record representing a single Pilot sample.

The schema must support at least:

- sample_id
- task_id
- initial_state_id
- episode_id
- step_id
- normalized_episode_progress
- base_rgb_raw
- wrist_rgb_raw
- state
- prompt
- episode_success

The implementation must preserve the raw observation arrays without applying model-specific preprocessing.

Expected semantic constraints:

- base_rgb_raw and wrist_rgb_raw are raw image arrays
- image dtype must be preserved during save/load
- image shape must be preserved during save/load
- state dtype and shape must be preserved
- prompt must round-trip exactly
- identifiers must round-trip exactly
- episode_success must round-trip exactly
- normalized_episode_progress must be validated to lie in [0, 1]

Use a deterministic, simple on-disk representation suitable for approximately hundreds of Pilot samples.

Prefer straightforward formats such as:
- metadata + NumPy arrays
or another equally simple representation already consistent with the repository.

Do not introduce a database, HDF5 dependency, or complex dataset framework.

Sample identity
---------------
The schema must expose a stable sample_id.

Do not rely on directory ordering, file ordering, or array index as sample identity.

The implementation does not need to generate sample_id automatically unless the existing repository already has an obvious convention.
It is acceptable for sample_id to be provided explicitly by the caller.

Validation
----------
Add explicit validation for structural errors that would make later paired-feature analysis unsafe.

At minimum validate:

- required identifiers are present / non-empty where appropriate
- normalized_episode_progress is within [0, 1]
- base_rgb_raw and wrist_rgb_raw are image-like arrays with matching expected rank
- state is an array with a valid non-empty shape

Do not add speculative validation for model-specific image size or normalization.

Serialization
-------------
Implement save/load round-trip support.

A loaded record must preserve the semantically relevant content of the original record.

If metadata and arrays are stored separately, their association must be explicit and deterministic.

Tests
-----
Add focused tests covering at least:

1. successful save/load round trip
2. preservation of image shape and dtype
3. preservation of state shape and dtype
4. preservation of metadata and prompt
5. invalid normalized_episode_progress
6. malformed image rank
7. two records with different sample_id values remain distinguishable

Tests must not require:

- GPU
- LIBERO
- OpenVLA
- pi0/openpi
- network access

Review constraints
------------------
Keep the implementation small enough to review as one conceptual change.

Default maximum:
- 1 production file
- 1 test file

If the existing repository layout genuinely requires more, stop before broadening the change and explain why.

Stop condition
--------------
Once the schema, serialization, and focused tests are complete:

1. run the smallest relevant test command
2. inspect the final diff
3. stop

Do not start implementing the rollout collector.

Final response must report only:

- files changed
- public interface introduced
- serialization format chosen
- validation implemented
- tests run and result
- assumptions made
- unresolved questions, if any
- behavior intentionally left unchanged