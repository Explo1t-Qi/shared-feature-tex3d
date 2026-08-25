from __future__ import annotations

import inspect
import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

import shared_feature.libero_collector as c1
import shared_feature.pilot_v02_collector as collector
from shared_feature import (
    PilotObservation,
    PilotV02CollectionError,
    PilotV02CollectionResult,
    collect_pilot_v02_observations,
)


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((3, 4, 3), value, dtype=np.uint8),
        "robot0_eef_pos": np.array([value, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }


class FakeEnv:
    def __init__(
        self,
        initial_state_id: int,
        *,
        success_after: int | None = 4,
        done_after: int | None = None,
        dummy_end: bool = False,
    ) -> None:
        self.env = SimpleNamespace(sim=SimpleNamespace(forward=lambda: None))
        self.initial_state_id = initial_state_id
        self.success_after = success_after
        self.done_after = done_after
        self.dummy_end = dummy_end
        self.observation = _observation()
        self.step_count = 0
        self.policy_action_count = 0
        self.current_success = False
        self.closed = False

    def reset(self) -> None:
        return None

    def set_init_state(self, initial_state: int) -> dict[str, np.ndarray]:
        assert initial_state == self.initial_state_id
        return self.observation

    def step(self, action: list[float]):
        del action
        self.step_count += 1
        in_dummy_phase = self.step_count <= c1._NUM_DUMMY_STEPS
        if not in_dummy_phase:
            self.policy_action_count += 1
        for value in self.observation.values():
            if value.dtype == np.uint8:
                value[...] = self.step_count % 256
            else:
                value[...] = self.step_count
        self.observation["robot0_eef_quat"][...] = [0.0, 0.0, 0.0, 1.0]
        self.current_success = (
            self.dummy_end
            if in_dummy_phase
            else self.success_after == self.policy_action_count
        )
        done = (
            self.dummy_end
            if in_dummy_phase
            else (self.done_after == self.policy_action_count)
        )
        return self.observation, 0.0, done, {}

    def check_success(self) -> bool:
        return self.current_success

    def close(self) -> None:
        self.closed = True


class FakeRuntimeFactory:
    def __init__(
        self,
        outcomes: dict[tuple[int, int], dict[str, object]] | None = None,
        *,
        state_count: int = 6,
        task_count: int = 10,
        action_error: Exception | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.state_count = state_count
        self.action_error = action_error
        self.calls: list[tuple[int, int]] = []
        self.envs: list[FakeEnv] = []
        self.next_state: defaultdict[int, int] = defaultdict(int)
        tasks = [
            SimpleNamespace(task_id=task_id, language=f"task language {task_id}")
            for task_id in range(task_count)
        ]
        suite = SimpleNamespace(
            get_num_tasks=lambda: task_count,
            get_task=lambda task_id: tasks[task_id],
            get_task_init_states=lambda task_id: list(range(state_count)),
        )
        benchmark = SimpleNamespace(
            get_benchmark_dict=lambda: {c1.PILOT_SUITE: lambda: suite}
        )
        self.runtime = c1._OfficialRuntime(
            benchmark=benchmark,
            get_libero_env=self.get_libero_env,
            get_libero_image=lambda observation, size: observation["agentview_image"],
            get_image_resize_size=lambda config: 2,
            get_action=self.get_action,
            normalize_gripper_action=lambda action, binarize: action,
            invert_gripper_action=lambda action: action,
        )

    def get_libero_env(self, task, model_family, resolution):
        assert model_family == "openvla"
        assert resolution == 512
        initial_state_id = self.next_state[task.task_id]
        self.next_state[task.task_id] += 1
        self.calls.append((task.task_id, initial_state_id))
        env = FakeEnv(
            initial_state_id,
            **self.outcomes.get((task.task_id, initial_state_id), {}),
        )
        self.envs.append(env)
        return env, task.language

    def get_action(self, *args, **kwargs):
        del args, kwargs
        if self.action_error is not None:
            raise self.action_error
        return np.zeros(7, dtype=np.float32)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    return path


def _install_runtime(monkeypatch, factory: FakeRuntimeFactory) -> None:
    monkeypatch.setattr(c1, "_load_official_runtime", lambda: factory.runtime)


def _run_smoke(monkeypatch, checkpoint, output_dir, factory, **overrides):
    _install_runtime(monkeypatch, factory)
    arguments = {
        "model": object(),
        "processor": object(),
        "pretrained_checkpoint": checkpoint,
        "checkpoint_identity": "portable-openvla-id",
        "libero_revision": "libero-test-revision",
        "output_dir": output_dir,
        "task_ids": (0,),
        "target_groups_per_task": 1,
    }
    arguments.update(overrides)
    return collector._collect_pilot_v02_with_plan(**arguments)


def _load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_collection_is_deterministic_complete_and_closed_schema(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory()
    _install_runtime(monkeypatch, factory)
    output_dir = tmp_path / "formal"

    result = collect_pilot_v02_observations(
        model=object(),
        processor=object(),
        pretrained_checkpoint=checkpoint,
        checkpoint_identity="portable-openvla-id",
        libero_revision="libero-test-revision",
        output_dir=output_dir,
    )

    assert isinstance(result, PilotV02CollectionResult)
    assert result.run_status == "COMPLETED"
    assert result.completeness_status == "COMPLETE"
    assert result.manifest_path == (output_dir / "collection_manifest.json").resolve()
    assert len(result.sample_paths) == 200
    assert all(path.is_absolute() and path.is_file() for path in result.sample_paths)
    assert factory.calls == [
        (task_id, state_id) for task_id in range(10) for state_id in range(5)
    ]
    assert all(env.closed for env in factory.envs)
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "collection_manifest.json",
        "observations",
    ]
    assert len(list((output_dir / "observations").glob("*.npz"))) == 200
    assert not list(output_dir.rglob("*.tmp"))

    manifest = _load_manifest(result.manifest_path)
    assert set(manifest) == collector._TOP_LEVEL_KEYS
    assert manifest["schema_version"] == "pilot_v0_2_collection_v1"
    assert manifest["pilot_version"] == "0.2"
    assert manifest["suite"] == "libero_spatial"
    assert manifest["runtime"] == {
        "libero_revision": "libero-test-revision",
        "discovered_task_count": 10,
        "task_ids": list(range(10)),
        "official_initial_state_counts": {str(i): 6 for i in range(10)},
    }
    assert manifest["coverage"] == {
        "target_total_groups": 50,
        "target_total_observations": 200,
        "actual_total_groups": 50,
        "actual_total_observations": 200,
        "accepted_groups_per_task": {str(i): 5 for i in range(10)},
    }
    assert [entry["task_id"] for entry in manifest["task_results"]] == list(range(10))
    first_group = manifest["task_results"][0]["accepted_groups"][0]
    assert [
        sample["target_relative_progress"] for sample in first_group["samples"]
    ] == [
        0.1,
        0.4,
        0.7,
        0.9,
    ]
    assert [sample["step_id"] for sample in first_group["samples"]] == [0, 1, 2, 3]
    assert [
        sample["actual_normalized_episode_progress"]
        for sample in first_group["samples"]
    ] == [
        0.0,
        1 / 3,
        2 / 3,
        1.0,
    ]
    assert all(
        sample["observation_path"].startswith("observations/")
        and "\\" not in sample["observation_path"]
        for task in manifest["task_results"]
        for group in task["accepted_groups"]
        for sample in group["samples"]
    )
    manifest_order = [
        output_dir / sample["observation_path"]
        for task in manifest["task_results"]
        for group in task["accepted_groups"]
        for sample in group["samples"]
    ]
    assert list(result.sample_paths) == [path.resolve() for path in manifest_order]

    first_record = PilotObservation.load(result.sample_paths[0])
    assert first_record.task_id == "0"
    assert first_record.episode_id == first_record.initial_state_id == 0
    assert first_record.episode_success
    assert int(first_record.base_rgb_raw[0, 0, 0]) == 10

    malformed = dict(manifest)
    malformed["rollout"] = {**manifest["rollout"], "extra": True}
    with pytest.raises(collector._ActiveFailure, match="invalid rollout fields"):
        collector._validate_manifest_structure(malformed, fatal=False)


def test_reduced_plan_replaces_only_later_states_and_preserves_sampling(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory(
        {
            (0, 0): {"success_after": None, "done_after": 2},
            (0, 1): {"success_after": 3},
            (0, 2): {"success_after": 11},
            (2, 0): {"success_after": None},
        }
    )
    result = _run_smoke(
        monkeypatch,
        checkpoint,
        tmp_path / "smoke",
        factory,
        task_ids=(0, 2),
    )

    assert result.run_status == "SMOKE_COMPLETED"
    assert result.completeness_status is None
    assert factory.calls == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)]
    assert [env.policy_action_count for env in factory.envs] == [2, 3, 11, 300, 4]
    assert all(env.closed for env in factory.envs)
    manifest = _load_manifest(result.manifest_path)
    assert manifest["run_status"] == "SMOKE_COMPLETED"
    assert manifest["completeness_status"] is None
    assert manifest["protocol"]["target_task_count"] == 2
    assert manifest["protocol"]["target_groups_per_task"] == 1
    assert manifest["coverage"]["actual_total_observations"] == 8
    first_task, second_task = manifest["task_results"]
    assert first_task["attempted_state_ids"] == [0, 1, 2]
    assert first_task["accepted_state_ids"] == [2]
    assert first_task["rejected_states"] == [
        {"initial_state_id": 0, "reason": "policy_failure", "trajectory_length": 2},
        {
            "initial_state_id": 1,
            "reason": "trajectory_too_short",
            "trajectory_length": 3,
        },
    ]
    assert second_task["rejected_states"] == [
        {
            "initial_state_id": 0,
            "reason": "policy_failure",
            "trajectory_length": 300,
        }
    ]
    assert [
        sample["step_id"] for sample in first_task["accepted_groups"][0]["samples"]
    ] == [1, 4, 7, 9]
    assert [
        sample["actual_normalized_episode_progress"]
        for sample in first_task["accepted_groups"][0]["samples"]
    ] == [0.1, 0.4, 0.7, 0.9]
    assert len(result.sample_paths) == 8
    assert not any(
        "state00" in path.name or "state01" in path.name
        for path in result.sample_paths[:4]
    )
    terminal_env = factory.envs[2]
    last_record = PilotObservation.load(result.sample_paths[3])
    assert int(last_record.base_rgb_raw[0, 0, 0]) == 19
    assert int(terminal_env.observation["agentview_image"][0, 0, 0]) == 21


def test_dummy_phase_end_is_fatal_and_writes_audit_manifest(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory({(0, 0): {"dummy_end": True}})
    output_dir = tmp_path / "fatal"

    with pytest.raises(PilotV02CollectionError, match="dummy phase") as error_info:
        _run_smoke(monkeypatch, checkpoint, output_dir, factory)

    assert isinstance(error_info.value.__cause__, collector._ActiveFailure)
    assert factory.envs[0].closed
    assert not list((output_dir / "observations").glob("*.npz"))
    manifest = _load_manifest(output_dir / "collection_manifest.json")
    assert set(manifest) == collector._TOP_LEVEL_KEYS | {"fatal_error"}
    assert manifest["run_status"] == "FATAL"
    assert manifest["completeness_status"] is None
    assert manifest["fatal_error"] == {
        "category": "DUMMY_PHASE_ERROR",
        "message": "episode ended during dummy phase: task=0, state=0, dummy_step=0",
        "task_id": 0,
        "initial_state_id": 0,
    }


def test_action_failure_is_fatal_not_an_ordinary_rejection(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory(action_error=RuntimeError("model failed"))
    output_dir = tmp_path / "action-fatal"

    with pytest.raises(PilotV02CollectionError, match="action generation"):
        _run_smoke(monkeypatch, checkpoint, output_dir, factory)

    manifest = _load_manifest(output_dir / "collection_manifest.json")
    assert manifest["fatal_error"]["category"] == "ACTION_ERROR"
    assert manifest["task_results"][0]["rejected_states"] == []
    assert not list((output_dir / "observations").glob("*.npz"))


def test_partial_group_rolls_back_and_prior_group_survives_fatal_error(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory()
    output_dir = tmp_path / "transaction"
    original_save = PilotObservation.save

    def failing_save(self, path):
        if "task01" in self.sample_id and self.step_id == 1:
            raise OSError("injected write failure")
        return original_save(self, path)

    monkeypatch.setattr(PilotObservation, "save", failing_save)

    with pytest.raises(PilotV02CollectionError, match="failed to write sample"):
        _run_smoke(
            monkeypatch,
            checkpoint,
            output_dir,
            factory,
            task_ids=(0, 1),
        )

    archives = sorted((output_dir / "observations").glob("*.npz"))
    assert len(archives) == 4
    assert all("task00" in path.name for path in archives)
    assert not list(output_dir.rglob("*.tmp"))
    manifest = _load_manifest(output_dir / "collection_manifest.json")
    assert manifest["coverage"]["accepted_groups_per_task"] == {"0": 1, "1": 0}
    assert manifest["coverage"]["actual_total_groups"] == 1
    assert manifest["fatal_error"]["category"] == "FILESYSTEM_ERROR"
    assert manifest["fatal_error"]["task_id"] == 1
    assert manifest["fatal_error"]["initial_state_id"] == 0


def test_fatal_manifest_write_failure_does_not_claim_manifest_exists(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory({(0, 0): {"dummy_end": True}})
    output_dir = tmp_path / "manifest-failure"

    def fail_manifest(path, manifest):
        del path, manifest
        raise collector._ActiveFailure(
            "MANIFEST_WRITE_ERROR", "injected manifest failure"
        )

    monkeypatch.setattr(collector, "_write_manifest", fail_manifest)
    with pytest.raises(PilotV02CollectionError) as error_info:
        _run_smoke(monkeypatch, checkpoint, output_dir, factory)

    assert not (output_dir / "collection_manifest.json").exists()
    active_failure = error_info.value.__cause__
    assert isinstance(active_failure, collector._ActiveFailure)
    assert any(
        "fatal manifest could not be written" in note
        for note in active_failure.__notes__
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checkpoint_identity": ""}, "checkpoint_identity"),
        ({"libero_revision": "  "}, "libero_revision"),
        ({"pretrained_checkpoint": "does-not-exist"}, "pretrained_checkpoint"),
    ],
)
def test_preflight_validation_creates_no_output(
    monkeypatch, tmp_path, checkpoint, overrides, message
) -> None:
    factory = FakeRuntimeFactory()
    output_dir = tmp_path / "not-created"

    with pytest.raises(PilotV02CollectionError, match=message):
        _run_smoke(
            monkeypatch,
            checkpoint,
            output_dir,
            factory,
            **overrides,
        )

    assert not output_dir.exists()
    assert not factory.calls


def test_nonempty_output_is_rejected_but_preexisting_empty_output_remains(
    monkeypatch, tmp_path, checkpoint
) -> None:
    factory = FakeRuntimeFactory()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(PilotV02CollectionError, match="must be empty"):
        _run_smoke(monkeypatch, checkpoint, output_dir, factory)
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert not factory.calls

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    broken_factory = FakeRuntimeFactory(task_count=9)
    with pytest.raises(PilotV02CollectionError, match="frozen LIBERO"):
        _run_smoke(monkeypatch, checkpoint, empty_dir, broken_factory)
    assert list(empty_dir.iterdir()) == []


def _task_result(task_id: int, count: int) -> dict[str, object]:
    groups = []
    for state_id in range(count):
        groups.append(
            {
                "task_id": task_id,
                "initial_state_id": state_id,
                "trajectory_length": 4,
                "episode_success": True,
                "samples": [
                    {
                        "sample_id": f"sample-{task_id}-{state_id}-{step_id}",
                        "step_id": step_id,
                        "target_relative_progress": collector._TARGET_PROGRESS[step_id],
                        "actual_normalized_episode_progress": step_id / 3,
                        "observation_path": f"observations/sample-{step_id}.npz",
                    }
                    for step_id in range(4)
                ],
            }
        )
    return {
        "task_id": task_id,
        "task_language": f"task {task_id}",
        "available_initial_state_count": 6,
        "target_accepted_groups": 5,
        "actual_accepted_groups": count,
        "attempted_state_ids": list(range(count)),
        "accepted_state_ids": list(range(count)),
        "rejected_states": [],
        "accepted_groups": groups,
    }


def test_completeness_classification_and_integrity_failures() -> None:
    assert (
        collector._classify_completeness(
            [_task_result(task_id, 5) for task_id in range(10)]
        )
        == "COMPLETE"
    )
    assert (
        collector._classify_completeness(
            [_task_result(0, 4)]
            + [_task_result(task_id, 5) for task_id in range(1, 10)]
        )
        == "USABLE_WITH_SHORTFALL"
    )
    assert (
        collector._classify_completeness(
            [_task_result(0, 3)]
            + [_task_result(task_id, 5) for task_id in range(1, 10)]
        )
        == "BLOCKED"
    )

    malformed = [_task_result(task_id, 5) for task_id in range(10)]
    malformed[1]["accepted_groups"][0]["initial_state_id"] = 0
    malformed[1]["accepted_groups"][0]["task_id"] = 0
    with pytest.raises(collector._ActiveFailure) as error_info:
        collector._classify_completeness(malformed)
    assert error_info.value.category == "INTEGRITY_ERROR"


def test_rounding_formula_and_defensive_collision_detection() -> None:
    assert collector._sample_indices(11) == (1, 4, 7, 9)
    assert collector._sample_indices(4) == (0, 1, 2, 3)
    collision = collector._sample_indices(4, (0.10, 0.11, 0.70, 0.90))
    assert len(set(collision)) < 4


def test_public_api_has_no_protocol_overrides() -> None:
    assert tuple(inspect.signature(collect_pilot_v02_observations).parameters) == (
        "model",
        "processor",
        "pretrained_checkpoint",
        "checkpoint_identity",
        "libero_revision",
        "output_dir",
    )
    assert "_collect_pilot_v02_with_plan" not in __import__("shared_feature").__all__
