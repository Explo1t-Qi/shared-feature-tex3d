from types import SimpleNamespace

import numpy as np
import pytest

import shared_feature.libero_collector as collector
from shared_feature import PilotCollectionError, PilotObservation


def make_observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full(
            (3, 4, 3), value + 50, dtype=np.uint8
        ),
        "robot0_eef_pos": np.array([value, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }


class FakeEnv:
    def __init__(
        self,
        *,
        success_after_policy_actions: int | None = 21,
        done_after_policy_actions: int | None = None,
        dummy_done_at: int | None = None,
        dummy_success_at: int | None = None,
        missing_key: str | None = None,
    ) -> None:
        self.env = SimpleNamespace(sim=SimpleNamespace(forward=self._forward))
        self.observation = make_observation()
        if missing_key is not None:
            del self.observation[missing_key]
        self.success_after_policy_actions = success_after_policy_actions
        self.done_after_policy_actions = done_after_policy_actions
        self.dummy_done_at = dummy_done_at
        self.dummy_success_at = dummy_success_at
        self.step_count = 0
        self.policy_action_count = 0
        self.current_success = False
        self.closed = False
        self.forwarded = False
        self.actions: list[list[float]] = []

    def _forward(self) -> None:
        self.forwarded = True

    def reset(self) -> None:
        return None

    def set_init_state(self, initial_state):
        return self.observation

    def step(self, action):
        self.actions.append(list(action))
        self.step_count += 1
        in_dummy_phase = self.step_count <= collector._NUM_DUMMY_STEPS
        if not in_dummy_phase:
            self.policy_action_count += 1

        for value in self.observation.values():
            if value.dtype == np.uint8:
                value[...] = self.step_count % 256
            else:
                value[...] = self.step_count
        self.observation["robot0_eef_quat"][...] = [0.0, 0.0, 0.0, 1.0]

        done = False
        self.current_success = False
        if in_dummy_phase:
            done = self.dummy_done_at == self.step_count
            self.current_success = self.dummy_success_at == self.step_count
        else:
            done = self.done_after_policy_actions == self.policy_action_count
            self.current_success = (
                self.success_after_policy_actions == self.policy_action_count
            )
        return self.observation, 0.0, done, {}

    def check_success(self) -> bool:
        return self.current_success

    def close(self) -> None:
        self.closed = True


class FakeRuntimeFactory:
    def __init__(self, envs: list[FakeEnv], *, action_error: Exception | None = None):
        self.envs = envs
        self.action_error = action_error
        self.environment_calls: list[tuple[str, int]] = []
        self.policy_observations: list[dict[str, np.ndarray]] = []

        task = SimpleNamespace(language="pick up the black bowl")
        suite = SimpleNamespace(
            get_task=lambda task_id: task,
            get_task_init_states=lambda task_id: [object() for _ in range(10)],
        )
        benchmark = SimpleNamespace(
            get_benchmark_dict=lambda: {collector.PILOT_SUITE: lambda: suite}
        )
        self.runtime = collector._OfficialRuntime(
            benchmark=benchmark,
            get_libero_env=self.get_libero_env,
            get_libero_image=self.get_libero_image,
            get_image_resize_size=lambda cfg: 2,
            get_action=self.get_action,
            normalize_gripper_action=self.normalize_gripper_action,
            invert_gripper_action=self.invert_gripper_action,
        )

    def get_libero_env(self, task, model_family, resolution):
        self.environment_calls.append((model_family, resolution))
        return self.envs.pop(0), task.language

    @staticmethod
    def get_libero_image(observation, resize_size):
        return observation["agentview_image"]

    def get_action(self, cfg, model, observation, task_description, processor=None):
        if self.action_error is not None:
            raise self.action_error
        self.policy_observations.append(
            {key: value.copy() for key, value in observation.items()}
        )
        return np.zeros(7, dtype=np.float32)

    @staticmethod
    def normalize_gripper_action(action, binarize=True):
        action[-1] = -1.0
        return action

    @staticmethod
    def invert_gripper_action(action):
        action[-1] *= -1.0
        return action


def install_fake_runtime(monkeypatch, *envs: FakeEnv, action_error=None):
    factory = FakeRuntimeFactory(list(envs), action_error=action_error)
    monkeypatch.setattr(collector, "_load_official_runtime", lambda: factory.runtime)
    return factory


def test_sample_id_is_deterministic_and_exact() -> None:
    assert collector._generate_sample_id(3, 47) == (
        "libero_spatial__task02__state03__step0047"
    )
    assert collector._generate_sample_id(3, 47) == collector._generate_sample_id(
        3, 47
    )


def test_canonical_state_is_8d_and_preserves_concatenation_dtype() -> None:
    state = collector._canonical_state(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.1, 0.2], dtype=np.float32),
    )

    assert state.shape == (8,)
    assert state.dtype == np.result_type(np.float32, np.float64)
    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 0.2])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position", np.zeros(2, dtype=np.float32)),
        ("quaternion", np.zeros(3, dtype=np.float32)),
        ("gripper", np.zeros(1, dtype=np.float32)),
    ],
)
def test_canonical_state_rejects_malformed_components(field, value) -> None:
    components = {
        "position": np.zeros(3, dtype=np.float32),
        "quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper": np.zeros(2, dtype=np.float32),
    }
    components[field] = value

    with pytest.raises(PilotCollectionError, match="shape"):
        collector._canonical_state(
            components["position"],
            components["quaternion"],
            components["gripper"],
        )


def test_uniform_sampling_is_unique_ordered_and_spans_trajectory() -> None:
    assert collector._uniform_sample_indices(21, 5) == (0, 5, 10, 15, 20)


def test_uniform_sampling_rejects_short_trajectory() -> None:
    with pytest.raises(PilotCollectionError, match="shorter"):
        collector._uniform_sample_indices(4, 5)


@pytest.mark.parametrize(
    "env",
    [FakeEnv(dummy_done_at=3), FakeEnv(dummy_success_at=3)],
)
def test_dummy_phase_termination_or_success_is_rejected(
    monkeypatch, tmp_path, env
) -> None:
    install_fake_runtime(monkeypatch, env)

    with pytest.raises(PilotCollectionError, match="dummy phase"):
        collector.collect_pilot_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint="checkpoint",
            output_dir=tmp_path,
            initial_state_ids=(0,),
            num_samples_per_episode=5,
        )

    assert env.closed
    assert not list(tmp_path.glob("*.npz"))


def test_runtime_error_is_not_converted_to_episode_failure(
    monkeypatch, tmp_path
) -> None:
    env = FakeEnv()
    install_fake_runtime(monkeypatch, env, action_error=RuntimeError("policy failed"))

    with pytest.raises(PilotCollectionError) as error_info:
        collector.collect_pilot_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint="checkpoint",
            output_dir=tmp_path,
            initial_state_ids=(0,),
            num_samples_per_episode=5,
        )

    assert isinstance(error_info.value.__cause__, RuntimeError)
    assert env.closed
    assert not list(tmp_path.glob("*.npz"))


def test_missing_raw_key_is_rejected(monkeypatch, tmp_path) -> None:
    env = FakeEnv(missing_key="robot0_eye_in_hand_image")
    install_fake_runtime(monkeypatch, env)

    with pytest.raises(PilotCollectionError, match="missing required keys"):
        collector.collect_pilot_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint="checkpoint",
            output_dir=tmp_path,
            initial_state_ids=(0,),
            num_samples_per_episode=5,
        )

    assert env.closed


def test_output_collision_is_rejected_without_overwrite(
    monkeypatch, tmp_path
) -> None:
    env = FakeEnv()
    install_fake_runtime(monkeypatch, env)
    collision = tmp_path / (
        "libero_spatial__task02__state00__step0000.npz"
    )
    collision.write_bytes(b"existing")

    with pytest.raises(PilotCollectionError, match="overwrite"):
        collector.collect_pilot_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint="checkpoint",
            output_dir=tmp_path,
            initial_state_ids=(0,),
            num_samples_per_episode=5,
        )

    assert collision.read_bytes() == b"existing"
    assert env.closed


def test_fake_episode_serializes_selected_raw_records(
    monkeypatch, tmp_path
) -> None:
    env = FakeEnv(success_after_policy_actions=21)
    factory = install_fake_runtime(monkeypatch, env)

    paths = collector.collect_pilot_observations(
        model=object(),
        processor=object(),
        pretrained_checkpoint="checkpoint",
        output_dir=tmp_path,
        initial_state_ids=(0,),
        num_samples_per_episode=5,
    )

    assert [path.stem for path in paths] == [
        "libero_spatial__task02__state00__step0000",
        "libero_spatial__task02__state00__step0005",
        "libero_spatial__task02__state00__step0010",
        "libero_spatial__task02__state00__step0015",
        "libero_spatial__task02__state00__step0020",
    ]
    assert factory.environment_calls == [("openvla", 512)]
    assert env.actions[:10] == [collector._DUMMY_ACTION] * 10
    assert env.forwarded and env.closed
    assert len(factory.policy_observations) == 21

    records = [PilotObservation.load(path) for path in paths]
    assert [record.step_id for record in records] == [0, 5, 10, 15, 20]
    assert [record.normalized_episode_progress for record in records] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert all(record.task_id == "2" for record in records)
    assert all(record.initial_state_id == 0 for record in records)
    assert all(record.episode_id == 0 for record in records)
    assert all(record.prompt == "pick up the black bowl" for record in records)
    assert all(record.episode_success for record in records)

    # The fake environment mutates one observation object in place. Copies must
    # retain each policy-query input, and the terminal value 31 is not included.
    assert [int(record.base_rgb_raw[0, 0, 0]) for record in records] == [
        10,
        15,
        20,
        25,
        30,
    ]
    assert [int(record.wrist_rgb_raw[0, 0, 0]) for record in records] == [
        10,
        15,
        20,
        25,
        30,
    ]


def test_policy_action_limit_is_a_normal_failed_episode(
    monkeypatch, tmp_path
) -> None:
    env = FakeEnv(success_after_policy_actions=None)
    install_fake_runtime(monkeypatch, env)

    paths = collector.collect_pilot_observations(
        model=object(),
        processor=object(),
        pretrained_checkpoint="checkpoint",
        output_dir=tmp_path,
        initial_state_ids=(0,),
        num_samples_per_episode=2,
    )

    records = [PilotObservation.load(path) for path in paths]
    assert env.policy_action_count == 300
    assert [record.step_id for record in records] == [0, 299]
    assert [record.normalized_episode_progress for record in records] == [0.0, 1.0]
    assert all(not record.episode_success for record in records)
    assert int(records[-1].base_rgb_raw[0, 0, 0]) == 309 % 256


def test_duplicate_initial_state_ids_are_rejected_before_rollout(
    monkeypatch, tmp_path
) -> None:
    load_called = False

    def unexpected_load():
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(collector, "_load_official_runtime", unexpected_load)

    with pytest.raises(PilotCollectionError, match="unique"):
        collector.collect_pilot_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint="checkpoint",
            output_dir=tmp_path,
            initial_state_ids=(0, 0),
            num_samples_per_episode=5,
        )

    assert not load_called
