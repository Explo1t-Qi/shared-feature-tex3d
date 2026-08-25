from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pilot_observation import PilotObservation


PILOT_SUITE = "libero_spatial"
PILOT_TASK_ID = 2
PILOT_INITIAL_STATE_IDS = tuple(range(10))
PILOT_NUM_SAMPLES_PER_EPISODE = 20

_CAMERA_RESOLUTION = 512
_NUM_DUMMY_STEPS = 10
_MAX_POLICY_ACTIONS = 300
_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
_REQUIRED_OBSERVATION_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


class PilotCollectionError(RuntimeError):
    """Raised when an episode cannot produce a valid Pilot sample set."""


class _EpisodeCollectionError(PilotCollectionError):
    """Private structured episode failure used by higher-level collectors."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class _BufferedObservation:
    step_id: int
    base_rgb_raw: np.ndarray
    wrist_rgb_raw: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class _OpenVLAActionConfig:
    pretrained_checkpoint: str
    unnorm_key: str | None
    center_crop: bool
    model_family: str = "openvla"


@dataclass(frozen=True)
class _OfficialRuntime:
    benchmark: Any
    get_libero_env: Any
    get_libero_image: Any
    get_image_resize_size: Any
    get_action: Any
    normalize_gripper_action: Any
    invert_gripper_action: Any


def collect_pilot_observations(
    *,
    model: Any,
    processor: Any,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    unnorm_key: str | None = None,
    center_crop: bool = True,
    initial_state_ids: Sequence[int] = PILOT_INITIAL_STATE_IDS,
    num_samples_per_episode: int = PILOT_NUM_SAMPLES_PER_EPISODE,
) -> tuple[Path, ...]:
    """Collect and serialize the frozen Pilot's OpenVLA-driven observations."""
    selected_state_ids = _validate_collection_parameters(
        initial_state_ids,
        num_samples_per_episode,
    )
    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise PilotCollectionError(f"output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    runtime = _load_official_runtime()
    action_config = _OpenVLAActionConfig(
        pretrained_checkpoint=str(pretrained_checkpoint),
        unnorm_key=unnorm_key,
        center_crop=center_crop,
    )

    try:
        benchmark_factories = runtime.benchmark.get_benchmark_dict()
        task_suite = benchmark_factories[PILOT_SUITE]()
        task = task_suite.get_task(PILOT_TASK_ID)
        initial_states = task_suite.get_task_init_states(PILOT_TASK_ID)
    except Exception as error:
        raise PilotCollectionError(
            f"failed to load {PILOT_SUITE} task {PILOT_TASK_ID}"
        ) from error

    if not isinstance(task.language, str) or not task.language:
        raise PilotCollectionError("LIBERO task language must be a non-empty string")
    if any(state_id >= len(initial_states) for state_id in selected_state_ids):
        raise PilotCollectionError(
            "requested official initial state is unavailable: "
            f"requested={selected_state_ids}, available={len(initial_states)}"
        )

    written_paths: list[Path] = []
    generated_sample_ids: set[str] = set()
    for initial_state_id in selected_state_ids:
        trajectory, episode_success = _collect_episode(
            runtime=runtime,
            action_config=action_config,
            model=model,
            processor=processor,
            task=task,
            initial_state=initial_states[initial_state_id],
            initial_state_id=initial_state_id,
        )
        selected_indices = _uniform_sample_indices(
            len(trajectory),
            num_samples_per_episode,
        )
        records_and_paths: list[tuple[PilotObservation, Path]] = []
        trajectory_length = len(trajectory)
        for selected_index in selected_indices:
            buffered = trajectory[selected_index]
            sample_id = _generate_sample_id(initial_state_id, buffered.step_id)
            if sample_id in generated_sample_ids:
                raise PilotCollectionError(f"duplicate sample_id generated: {sample_id}")
            generated_sample_ids.add(sample_id)

            output_path = destination / f"{sample_id}.npz"
            if output_path.exists():
                raise PilotCollectionError(
                    f"refusing to overwrite existing sample: {output_path}"
                )
            record = PilotObservation(
                sample_id=sample_id,
                task_id=str(PILOT_TASK_ID),
                initial_state_id=initial_state_id,
                episode_id=initial_state_id,
                step_id=buffered.step_id,
                normalized_episode_progress=(
                    buffered.step_id / (trajectory_length - 1)
                ),
                base_rgb_raw=buffered.base_rgb_raw,
                wrist_rgb_raw=buffered.wrist_rgb_raw,
                state=buffered.state,
                prompt=task.language,
                episode_success=episode_success,
            )
            records_and_paths.append((record, output_path))

        for record, output_path in records_and_paths:
            try:
                record.save(output_path)
            except Exception as error:
                raise PilotCollectionError(
                    f"failed to serialize sample {record.sample_id}"
                ) from error
            written_paths.append(output_path)

    return tuple(written_paths)


def _collect_episode(
    *,
    runtime: _OfficialRuntime,
    action_config: _OpenVLAActionConfig,
    model: Any,
    processor: Any,
    task: Any,
    initial_state: Any,
    initial_state_id: int,
    task_id: int = PILOT_TASK_ID,
) -> tuple[list[_BufferedObservation], bool]:
    env: Any = None
    try:
        env, task_description = runtime.get_libero_env(
            task,
            "openvla",
            resolution=_CAMERA_RESOLUTION,
        )
        env.reset()
        observation = env.set_init_state(initial_state)
        env.env.sim.forward()

        for dummy_step in range(_NUM_DUMMY_STEPS):
            observation, _, done, _ = env.step(list(_DUMMY_ACTION))
            success = _check_success(env)
            if done or success:
                raise _EpisodeCollectionError(
                    "DUMMY_PHASE_ERROR",
                    "episode ended during dummy phase: "
                    f"task={task_id}, state={initial_state_id}, "
                    f"dummy_step={dummy_step}",
                )

        trajectory: list[_BufferedObservation] = []
        episode_success = False
        for step_id in range(_MAX_POLICY_ACTIONS):
            try:
                buffered = _capture_observation(observation, step_id)
            except PilotCollectionError as error:
                raise _EpisodeCollectionError(
                    "OBSERVATION_ERROR",
                    str(error),
                ) from error
            trajectory.append(buffered)

            try:
                policy_observation = _build_policy_observation(
                    runtime,
                    action_config,
                    observation,
                    buffered.state,
                )
            except Exception as error:
                raise _EpisodeCollectionError(
                    "OBSERVATION_ERROR",
                    "failed to build the OpenVLA policy observation",
                ) from error
            try:
                action = runtime.get_action(
                    action_config,
                    model,
                    policy_observation,
                    task_description,
                    processor=processor,
                )
            except Exception as error:
                raise _EpisodeCollectionError(
                    "ACTION_ERROR",
                    "OpenVLA action generation failed",
                ) from error
            if not isinstance(action, np.ndarray) or action.shape != (7,):
                raise _EpisodeCollectionError(
                    "ACTION_ERROR",
                    f"OpenVLA action must be a NumPy array with shape (7,), got "
                    f"{type(action).__name__} {getattr(action, 'shape', None)}",
                )
            try:
                action = runtime.normalize_gripper_action(action, binarize=True)
                action = runtime.invert_gripper_action(action)
            except Exception as error:
                raise _EpisodeCollectionError(
                    "ACTION_ERROR",
                    "OpenVLA action post-processing failed",
                ) from error

            observation, _, done, _ = env.step(action.tolist())
            episode_success = _check_success(env)
            if episode_success or done:
                break

        return trajectory, episode_success
    except _EpisodeCollectionError:
        raise
    except Exception as error:
        raise _EpisodeCollectionError(
            "RUNTIME_ERROR",
            "LIBERO collection failed: "
            f"suite={PILOT_SUITE}, task={task_id}, "
            f"state={initial_state_id}",
        ) from error
    finally:
        if env is not None:
            env.close()


def _build_policy_observation(
    runtime: _OfficialRuntime,
    action_config: _OpenVLAActionConfig,
    observation: Mapping[str, Any],
    state: np.ndarray,
) -> dict[str, np.ndarray]:
    from PIL import Image

    policy_source = runtime.get_libero_image(
        observation,
        _CAMERA_RESOLUTION,
    )
    model_input_size = runtime.get_image_resize_size(action_config)
    policy_image = np.array(
        Image.fromarray(policy_source).resize(
            (model_input_size, model_input_size)
        )
    )
    return {"full_image": policy_image, "state": state.copy()}


def _capture_observation(
    observation: Mapping[str, Any],
    step_id: int,
) -> _BufferedObservation:
    if not isinstance(observation, Mapping):
        raise PilotCollectionError("LIBERO observation must be a mapping")
    missing_keys = [
        key for key in _REQUIRED_OBSERVATION_KEYS if key not in observation
    ]
    if missing_keys:
        raise PilotCollectionError(
            f"LIBERO observation is missing required keys: {missing_keys}"
        )

    base_rgb_raw = _copy_raw_image(
        "agentview_image",
        observation["agentview_image"],
    )
    wrist_rgb_raw = _copy_raw_image(
        "robot0_eye_in_hand_image",
        observation["robot0_eye_in_hand_image"],
    )
    state = _canonical_state(
        observation["robot0_eef_pos"],
        observation["robot0_eef_quat"],
        observation["robot0_gripper_qpos"],
    )
    return _BufferedObservation(
        step_id=step_id,
        base_rgb_raw=base_rgb_raw,
        wrist_rgb_raw=wrist_rgb_raw,
        state=state,
    )


def _copy_raw_image(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PilotCollectionError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.size == 0 or value.dtype.hasobject:
        raise PilotCollectionError(
            f"{name} must be a non-empty rank-3 non-object array"
        )
    return value.copy()


def _canonical_state(
    eef_position: Any,
    eef_quaternion: Any,
    gripper_qpos: Any,
) -> np.ndarray:
    position = _real_numeric_component("robot0_eef_pos", eef_position, (3,))
    quaternion = _real_numeric_component(
        "robot0_eef_quat",
        eef_quaternion,
        (4,),
    )
    gripper = _real_numeric_component(
        "robot0_gripper_qpos",
        gripper_qpos,
        (2,),
    )
    axis_angle = _quat_to_axis_angle(quaternion.copy())
    state = np.concatenate((position, axis_angle, gripper))
    if (
        state.shape != (8,)
        or state.dtype.hasobject
        or not np.issubdtype(state.dtype, np.number)
        or not np.isrealobj(state)
        or not np.all(np.isfinite(state))
    ):
        raise PilotCollectionError(
            "canonical LIBERO state must be a finite real numeric array "
            "with shape (8,)"
        )
    return state


def _real_numeric_component(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PilotCollectionError(f"{name} must be a NumPy array")
    if (
        value.shape != expected_shape
        or value.dtype.hasobject
        or not np.issubdtype(value.dtype, np.number)
        or not np.isrealobj(value)
        or not np.all(np.isfinite(value))
    ):
        raise PilotCollectionError(
            f"{name} must be a finite real numeric array with shape "
            f"{expected_shape}"
        )
    return value


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    if quaternion[3] > 1.0:
        quaternion[3] = 1.0
    elif quaternion[3] < -1.0:
        quaternion[3] = -1.0
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (
        quaternion[:3]
        * 2.0
        * math.acos(float(quaternion[3]))
        / denominator
    )


def _uniform_sample_indices(
    trajectory_length: int,
    num_samples: int,
) -> tuple[int, ...]:
    if trajectory_length < num_samples:
        raise PilotCollectionError(
            "valid-policy trajectory is shorter than requested sample count: "
            f"T={trajectory_length}, requested={num_samples}"
        )
    indices_array = np.rint(
        np.linspace(0, trajectory_length - 1, num=num_samples)
    ).astype(np.int64)
    indices = tuple(int(index) for index in indices_array)
    if (
        len(indices) != num_samples
        or len(set(indices)) != num_samples
        or any(left >= right for left, right in zip(indices, indices[1:]))
        or indices[0] != 0
        or indices[-1] != trajectory_length - 1
    ):
        raise PilotCollectionError(
            f"uniform sampling did not produce {num_samples} unique ordered indices"
        )
    return indices


def _generate_sample_id(initial_state_id: int, step_id: int) -> str:
    return (
        f"{PILOT_SUITE}__task{PILOT_TASK_ID:02d}__"
        f"state{initial_state_id:02d}__step{step_id:04d}"
    )


def _validate_collection_parameters(
    initial_state_ids: Sequence[int],
    num_samples_per_episode: int,
) -> tuple[int, ...]:
    state_ids = tuple(initial_state_ids)
    if not state_ids:
        raise PilotCollectionError("initial_state_ids must be non-empty")
    if any(type(state_id) is not int for state_id in state_ids):
        raise PilotCollectionError("initial_state_ids must contain integers")
    if len(set(state_ids)) != len(state_ids):
        raise PilotCollectionError("initial_state_ids must be unique")
    if any(state_id not in PILOT_INITIAL_STATE_IDS for state_id in state_ids):
        raise PilotCollectionError(
            f"initial_state_ids must be selected from {PILOT_INITIAL_STATE_IDS}"
        )
    if type(num_samples_per_episode) is not int or num_samples_per_episode < 2:
        raise PilotCollectionError(
            "num_samples_per_episode must be an integer greater than or equal to 2"
        )
    return state_ids


def _check_success(env: Any) -> bool:
    checker = getattr(env, "check_success", None)
    if not callable(checker):
        raise PilotCollectionError(
            "LIBERO environment does not expose public check_success()"
        )
    return bool(checker())


def _load_official_runtime() -> _OfficialRuntime:
    try:
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import (
            get_libero_env,
            get_libero_image,
        )
        from experiments.robot.robot_utils import (
            get_action,
            get_image_resize_size,
            invert_gripper_action,
            normalize_gripper_action,
        )
    except ImportError as error:
        raise PilotCollectionError(
            "official OpenVLA and LIBERO modules must be importable"
        ) from error

    return _OfficialRuntime(
        benchmark=benchmark,
        get_libero_env=get_libero_env,
        get_libero_image=get_libero_image,
        get_image_resize_size=get_image_resize_size,
        get_action=get_action,
        normalize_gripper_action=normalize_gripper_action,
        invert_gripper_action=invert_gripper_action,
    )
