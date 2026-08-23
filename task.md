- # Current Coding Contract — C1 LIBERO Observation Collector

  This file defines the only implementation task authorized for the current Codex session.

  Read first:

  1. `AGENTS.md`
  2. `docs/research-map.md`
  3. `docs/pilot-v0.1-spec.md`
  4. `shared_feature/pilot_observation.py`
  5. this file

  Reference repositories:

  - Current writable project: `.`
  - Official Tex3D reference: `../tex3d`
  - Historical modified Tex3D reference: `../modified-tex3d`

  For implementation behavior, prefer the frozen decisions in this contract.

  Do not broadly copy code from `modified-tex3d`.
  It may only be consulted for already identified collector-relevant correctness behavior.

  ---

  ## Goal

  Implement a minimal LIBERO rollout observation collector for Pilot v0.1.

  The collector must:

  1. execute OpenVLA-driven LIBERO rollouts for the frozen Pilot task;
  2. use fixed official LIBERO initial states;
  3. capture raw observations before model-specific image preprocessing;
  4. retain only valid-policy observations after the stabilization phase;
  5. complete the episode before choosing samples;
  6. uniformly select exactly 20 valid-policy frames per valid episode;
  7. serialize those frames as `PilotObservation` records.

  This task is only about observation collection.

  ---

  ## Frozen Pilot Configuration

  Use:

  ```text
  task_suite = libero_spatial
  task_id = 2
  ```

  The task is:

  ```text
  pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
  ```

  Use official LIBERO initial states with indices:

  ```text
  0 ... 9
  ```

  One initial state corresponds to one episode.

  Do not substitute a different task or initial state automatically.

  ---

  ## Allowed Scope

  Implement only the collector and focused tests required for this contract.

  You may:

  - add one collector production module;
  - add or modify focused collector tests;
  - minimally export the collector API from `shared_feature/__init__.py` if needed.

  Default maximum:

  - 1 new production module;
  - 1 collector test module;
  - minimal `__init__.py` update if required.

  If more production files are genuinely necessary, stop before broadening the implementation and explain why.

  ---

  ## Forbidden Scope

  Do NOT implement:

  - OpenVLA feature extraction;
  - pi0/openpi feature extraction;
  - train / held-out split generation;
  - feature serialization;
  - PCA / SVD / CCA / CKA;
  - regression analysis;
  - manifest/index infrastructure;
  - Tex3D attack logic;
  - differentiable renderer integration;
  - shared-feature losses;
  - object-region sampling;
  - token-level analysis;
  - model preprocessing abstractions;
  - unrelated refactors.

  Do not copy abandoned spectral code from `modified-tex3d`.

  ---

  ## Official Rollout Semantics

  Use the official Tex3D/OpenVLA LIBERO rollout path as the behavioral reference for:

  - LIBERO task loading;
  - official initial-state retrieval;
  - environment construction;
  - OpenVLA policy execution;
  - action postprocessing;
  - environment stepping.

  Relevant reference files include:

  ```text
  ../tex3d/openvla/experiments/robot/libero/attack_openvla.py
  ../tex3d/openvla/experiments/robot/libero/libero_utils.py
  ../tex3d/openvla/experiments/robot/robot_utils.py
  ../tex3d/openvla/experiments/robot/openvla_utils.py
  ```

  Do not import the official attack pipeline wholesale.

  Extract only the minimum behavior needed for a clean observation collector.

  ---

  ## Raw Observation Boundary

  The collector must save observations directly from real LIBERO / MuJoCo environment outputs.

  Raw observations originate from:

  ```text
  env.set_init_state(...)
  env.step(...)
  ```

  Images must be copied before any call equivalent to:

  ```text
  get_libero_image(...)
  ```

  or any OpenVLA-specific:

  - rotation;
  - resize;
  - JPEG encode/decode;
  - crop;
  - PIL conversion;
  - processor transform;
  - normalization.

  Required raw observation keys:

  ```text
  agentview_image
  robot0_eye_in_hand_image
  robot0_eef_pos
  robot0_eef_quat
  robot0_gripper_qpos
  ```

  Missing required keys must cause an explicit collection error.

  Do not use differentiable-renderer or composited attack images as public Pilot observations.

  ---

  ## Canonical State

  `PilotObservation.state` must store the canonical 8D LIBERO policy state:

  ```text
  [
      robot0_eef_pos,
      quat_to_axis_angle(robot0_eef_quat),
      robot0_gripper_qpos,
  ]
  ```

  Conceptually:

  ```text
  3D end-effector position
  +
  3D axis-angle orientation
  +
  2D gripper state
  =
  8D policy state
  ```

  This is a deterministic policy-level representation derived from raw LIBERO proprioception.

  It is intentionally shared by OpenVLA and pi0 LIBERO inference.

  Do not store an arbitrary simulator state vector in this field.

  ---

  ## Stabilization / Dummy Phase

  Before policy execution, run exactly:

  ```text
  10 dummy environment steps
  ```

  with action:

  ```text
  [0, 0, 0, 0, 0, 0, -1]
  ```

  Dummy observations are not Pilot samples.

  The observation returned by the 10th dummy step becomes:

  ```text
  valid-policy step_id = 0
  ```

  If the environment terminates or reports task success during the dummy phase:

  - treat the episode as invalid;
  - report a collection error;
  - do not serialize it as an ordinary success or failure;
  - do not automatically replace the initial state.

  ---

  ## Valid-Policy Step Semantics

  `step_id` is:

  ```text
  0-based valid-policy observation index
  ```

  Dummy/stabilization steps do not count toward `step_id`.

  Example:

  ```text
  10 dummy steps
  ↓
  obs after final dummy step -> step_id 0
  ↓
  policy action
  ↓
  next obs -> step_id 1
  ```

  Do not use the absolute environment-step count as `PilotObservation.step_id`.

  ---

  ## Episode Success Semantics

  `PilotObservation.episode_success` means:

  ```text
  LIBERO task success condition
  ```

  Use:

  ```text
  env.check_success()
  ```

  or the equivalent public LIBERO success-check interface.

  Do not use `done` as the sole definition of task success.

  `done` may be used to control rollout termination.

  Normal outcomes:

  ```text
  task success detected
  → episode_success = True

  normal rollout reaches its allowed limit without task success
  → episode_success = False
  ```

  A runtime/environment/policy exception is not a normal failure:

  ```text
  exception
  → collection error
  → do not silently serialize the episode as failure
  ```

  ---

  ## Full Episode First, Sampling Second

  Do not decide the 20 Pilot frames online.

  For each episode:

  ```text
  run full valid-policy trajectory
  ↓
  retain valid-policy raw observations in memory
  ↓
  episode finishes
  ↓
  know trajectory length T
  ↓
  select 20 observations deterministically
  ↓
  serialize selected observations
  ```

  This Pilot is intentionally small, so a simple in-memory episode buffer is preferred over streaming infrastructure.

  ---

  ## Uniform Sampling

  For a valid-policy trajectory with length:

  ```text
  T
  ```

  select exactly 20 deterministic indices spanning:

  ```text
  0 ... T - 1
  ```

  Use behavior equivalent to:

  ```python
  np.linspace(0, T - 1, num=20)
  ```

  followed by a deterministic integer conversion that yields 20 valid monotonically ordered indices.

  The implementation must ensure the selected indices are unique.

  Do not randomly sample frames.

  Do not duplicate observations to reach 20 samples.

  If:

  ```text
  T < 20
  ```

  treat the episode as invalid for Pilot collection and report an explicit error.

  ---

  ## Normalized Episode Progress

  For every selected observation with original valid-policy `step_id = i`:

  ```text
  normalized_episode_progress = i / (T - 1)
  ```

  where:

  ```text
  T
  ```

  is the full number of valid-policy observations in that episode.

  Do not renumber the selected 20 frames to `0...19`.

  The saved `step_id` must remain the original valid-policy trajectory index.

  ---

  ## Sample ID

  Generate `sample_id` deterministically as:

  ```text
  {suite}__task{task_id:02d}__state{initial_state_id:02d}__step{step_id:04d}
  ```

  Example:

  ```text
  libero_spatial__task02__state03__step0047
  ```

  The same:

  ```text
  suite
  task_id
  initial_state_id
  step_id
  ```

  must always generate the same `sample_id`.

  Do not use:

  - filesystem order;
  - selected-frame ordinal;
  - random UUID;
  - timestamp.

  ---

  ## Observation Fields

  Each selected record must populate:

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

  Use:

  ```text
  base_rgb_raw  = raw obs["agentview_image"]
  wrist_rgb_raw = raw obs["robot0_eye_in_hand_image"]
  prompt        = task.language
  ```

  For Pilot v0.1:

  ```text
  episode_id == initial_state_id
  ```

  because each frozen initial state defines one episode.

  Arrays stored in the episode buffer must be copied so later environment mutation cannot alter already captured observations.

  ---

  ## Output Behavior

  Use the existing:

  ```python
  PilotObservation.save(...)
  ```

  serialization API.

  The collector should write one `.npz` file per selected sample.

  Do not implement:

  ```text
  manifest.jsonl
  dataset database
  global index
  feature files
  ```

  in this task.

  Use the deterministic `sample_id` as the basis for the filename.

  Example:

  ```text
  libero_spatial__task02__state03__step0047.npz
  ```

  The output directory may be supplied by the caller.

  Do not silently overwrite an existing sample unless the existing project already has an explicit overwrite convention.

  If no such convention exists, prefer explicit failure on collision.

  ---

  ## Smoke Mode

  The collector interface must make it possible to run the previously frozen smoke test:

  ```text
  1 task
  2 initial states
  5–10 sampled frames per episode
  ```

  without changing core rollout semantics.

  However:

  - do not build a generalized experiment framework;
  - do not implement multiple task-suite orchestration;
  - do not add broad configuration infrastructure.

  A small parameter controlling:

  ```text
  initial_state_ids
  num_samples_per_episode
  ```

  is sufficient if needed.

  The scientific Pilot defaults remain:

  ```text
  initial_state_ids = 0...9
  num_samples_per_episode = 20
  ```

  ---

  ## Error Handling

  Fail explicitly on:

  - missing LIBERO observation keys;
  - invalid dummy-phase termination/success;
  - episode trajectory shorter than requested sample count;
  - runtime/policy/environment exceptions;
  - duplicate generated sample IDs;
  - output-file collision;
  - malformed canonical state.

  Always close the LIBERO environment with reliable cleanup behavior, including when an exception occurs.

  Do not convert runtime exceptions into ordinary failed episodes.

  ---

  ## Test Requirements

  Tests should remain CPU-only and should not require:

  - real LIBERO;
  - MuJoCo rendering;
  - GPU;
  - OpenVLA checkpoint;
  - network access.

  Use small fake/stub objects only where necessary.

  At minimum test:

  1. deterministic sample-ID generation;
  2. canonical 8D state construction;
  3. valid-policy `step_id` excludes dummy steps;
  4. deterministic uniform sampling returns the requested number of unique ordered indices;
  5. progress uses original trajectory index and full `T`;
  6. `T < requested_samples` is rejected;
  7. raw image arrays are copied rather than aliased;
  8. dummy-phase early termination/success is rejected;
  9. runtime rollout errors are not converted into ordinary episode failure;
  10. output collision is rejected;
  11. selected records are serialized through `PilotObservation`;
  12. a small fake episode produces the expected selected sample IDs and metadata.

  Do not reproduce or unit-test the entire external OpenVLA model stack.

  ---

  ## Reviewability

  Prefer direct procedural code over generalized frameworks.

  Avoid introducing:

  - collector base classes;
  - plugin systems;
  - dataset registries;
  - asynchronous workers;
  - multiprocessing;
  - callback frameworks;
  - generic robotics abstractions.

  The code should be understandable as one small Pilot-specific collector.

  ---

  ## Planning Requirement

  Before modifying files:

  1. inspect the relevant current-project code;
  2. inspect only the minimum official Tex3D rollout files needed;
  3. propose the exact production/test files;
  4. describe the minimal collector API;
  5. explain how OpenVLA policy execution will be invoked without copying unrelated attack logic;
  6. identify any remaining ambiguity.

  For the first response to this contract:

  ```text
  DO NOT MODIFY FILES.
  ```

  Return only the implementation plan.

  Implementation begins only after the plan is reviewed.

  ---

  ## Stop Condition

  After the approved implementation is complete:

  1. run the smallest relevant tests;
  2. inspect the final diff for scope creep;
  3. stop.

  Do not start C2 OpenVLA representation extraction.

  Final implementation report must include only:

  - files changed;
  - public API introduced;
  - official reference code used;
  - rollout semantics implemented;
  - sampling semantics implemented;
  - error handling implemented;
  - tests run and results;
  - assumptions made;
  - unresolved questions;
  - behavior intentionally left unchanged.
