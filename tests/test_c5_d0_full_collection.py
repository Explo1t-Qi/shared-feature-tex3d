from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.c5_d0_pilot_v02_full_collection as full
from shared_feature import PilotObservation, PilotV02CollectionResult


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "dataset_statistics.json").write_text(
        json.dumps({full.UNNORM_KEY: {}}),
        encoding="utf-8",
    )
    tex3d_root = tmp_path / "tex3d-openvla"
    tex3d_root.mkdir()
    return checkpoint, tex3d_root


def _arguments(checkpoint: Path, output_dir: Path, tex3d_root: Path):
    return SimpleNamespace(
        pretrained_checkpoint=checkpoint,
        libero_revision="libero-test-revision",
        output_dir=output_dir,
        tex3d_openvla_root=tex3d_root,
    )


def _write_collection(
    output_dir: Path,
    accepted_counts: list[int],
    completeness_status: str,
) -> PilotV02CollectionResult:
    observations_dir = output_dir / "observations"
    observations_dir.mkdir(parents=True)
    sample_paths: list[Path] = []
    task_results = []

    for task_id, accepted_count in enumerate(accepted_counts):
        accepted_groups = []
        for state_id in range(accepted_count):
            samples = []
            for step_id, progress in enumerate(full.TARGET_PROGRESS):
                sample_id = (
                    f"libero_spatial__task{task_id:02d}__state{state_id:02d}"
                    f"__step{step_id:04d}"
                )
                path = observations_dir / f"{sample_id}.npz"
                PilotObservation(
                    sample_id=sample_id,
                    task_id=str(task_id),
                    initial_state_id=state_id,
                    episode_id=state_id,
                    step_id=step_id,
                    normalized_episode_progress=step_id / 3,
                    base_rgb_raw=np.zeros((1, 1, 3), dtype=np.uint8),
                    wrist_rgb_raw=np.zeros((1, 1, 3), dtype=np.uint8),
                    state=np.zeros(8, dtype=np.float32),
                    prompt=f"task language {task_id}",
                    episode_success=True,
                ).save(path)
                sample_paths.append(path.resolve())
                samples.append(
                    {
                        "sample_id": sample_id,
                        "step_id": step_id,
                        "target_relative_progress": progress,
                        "actual_normalized_episode_progress": step_id / 3,
                        "observation_path": f"observations/{sample_id}.npz",
                    }
                )
            accepted_groups.append(
                {
                    "task_id": task_id,
                    "initial_state_id": state_id,
                    "trajectory_length": 4,
                    "episode_success": True,
                    "samples": samples,
                }
            )
        task_results.append(
            {
                "task_id": task_id,
                "task_language": f"task language {task_id}",
                "available_initial_state_count": 50,
                "target_accepted_groups": 5,
                "actual_accepted_groups": accepted_count,
                "attempted_state_ids": list(range(accepted_count)),
                "accepted_state_ids": list(range(accepted_count)),
                "rejected_states": [],
                "accepted_groups": accepted_groups,
            }
        )

    actual_groups = sum(accepted_counts)
    manifest = {
        "schema_version": "pilot_v0_2_collection_v1",
        "pilot_version": "0.2",
        "suite": "libero_spatial",
        "run_status": "COMPLETED",
        "completeness_status": completeness_status,
        "rollout": {"checkpoint_identity": full.CHECKPOINT_IDENTITY},
        "protocol": {},
        "runtime": {
            "libero_revision": "libero-test-revision",
            "discovered_task_count": 10,
            "task_ids": list(range(10)),
        },
        "coverage": {
            "target_total_groups": 50,
            "target_total_observations": 200,
            "actual_total_groups": actual_groups,
            "actual_total_observations": actual_groups * 4,
            "accepted_groups_per_task": {
                str(task_id): count for task_id, count in enumerate(accepted_counts)
            },
        },
        "task_results": task_results,
    }
    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return PilotV02CollectionResult(
        sample_paths=tuple(sample_paths),
        manifest_path=manifest_path.resolve(),
        run_status="COMPLETED",
        completeness_status=completeness_status,
    )


def _read_manifest(output_dir: Path) -> dict:
    return json.loads(
        (output_dir / "collection_manifest.json").read_text(encoding="utf-8")
    )


def _write_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / "collection_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_run_collection_uses_frozen_loading_and_public_formal_api(
    monkeypatch, tmp_path
) -> None:
    checkpoint, tex3d_root = _prepare_inputs(tmp_path)
    output_dir = tmp_path / "formal-output"
    events: list[object] = []
    model = object()
    processor = object()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            events.append("cuda")
            return True

    def set_seed(seed: int) -> None:
        events.append(("seed", seed))

    def get_model(config):
        assert events[-1] == ("seed", 7)
        assert vars(config) == {
            "model_family": "openvla",
            "pretrained_checkpoint": str(checkpoint.resolve()),
            "load_in_8bit": False,
            "load_in_4bit": False,
            "unnorm_key": "libero_spatial_no_noops",
            "center_crop": True,
        }
        events.append("model")
        return model

    def get_processor(config):
        assert events[-1] == "model"
        assert config.unnorm_key == full.UNNORM_KEY
        events.append("processor")
        return processor

    def collect_formal(**kwargs):
        assert events[-1] == "processor"
        assert kwargs == {
            "model": model,
            "processor": processor,
            "pretrained_checkpoint": checkpoint.resolve(),
            "checkpoint_identity": full.CHECKPOINT_IDENTITY,
            "libero_revision": "libero-test-revision",
            "output_dir": output_dir.resolve(),
        }
        events.append("collector")
        return _write_collection(
            output_dir.resolve(),
            [1] + [0] * 9,
            "BLOCKED",
        )

    monkeypatch.setattr(
        full,
        "_load_runtime_components",
        lambda root: (
            SimpleNamespace(cuda=FakeCuda()),
            get_model,
            get_processor,
            set_seed,
            collect_formal,
        ),
    )

    summary = full._run_collection(_arguments(checkpoint, output_dir, tex3d_root))

    assert events == ["cuda", ("seed", 7), "model", "processor", "collector"]
    assert summary["run_status"] == "COMPLETED"
    assert summary["completeness_status"] == "BLOCKED"
    assert summary["checkpoint_identity"] == full.CHECKPOINT_IDENTITY
    assert summary["actual_total_groups"] == 1
    assert summary["observation_file_count"] == 4


@pytest.mark.parametrize(
    ("counts", "status", "expected_groups", "expected_observations"),
    [
        ([5] * 10, "COMPLETE", 50, 200),
        ([4] * 10, "USABLE_WITH_SHORTFALL", 40, 160),
        ([0] * 10, "BLOCKED", 0, 0),
    ],
)
def test_completed_statuses_are_preserved_and_validated(
    tmp_path, counts, status, expected_groups, expected_observations
) -> None:
    output_dir = tmp_path / status.lower()
    result = _write_collection(output_dir, counts, status)

    summary = full._validate_result(
        result,
        output_dir.resolve(),
        "libero-test-revision",
    )

    assert summary["run_status"] == "COMPLETED"
    assert summary["completeness_status"] == status
    assert summary["actual_total_groups"] == expected_groups
    assert summary["actual_total_observations"] == expected_observations


@pytest.mark.parametrize("run_status", ["SMOKE_COMPLETED", "FATAL"])
def test_nonformal_run_status_is_rejected(tmp_path, run_status) -> None:
    result = SimpleNamespace(
        run_status=run_status,
        completeness_status=None,
        manifest_path=tmp_path / "collection_manifest.json",
        sample_paths=(),
    )

    with pytest.raises(AssertionError, match="run_status"):
        full._validate_result(result, tmp_path, "libero-test-revision")


def test_nonempty_output_fails_before_runtime_loading(monkeypatch, tmp_path) -> None:
    checkpoint, tex3d_root = _prepare_inputs(tmp_path)
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    marker = output_dir / "existing"
    marker.write_text("keep", encoding="utf-8")
    load_called = False

    def unexpected_load(root):
        del root
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(full, "_load_runtime_components", unexpected_load)

    with pytest.raises(FileExistsError, match="not empty"):
        full._run_collection(_arguments(checkpoint, output_dir, tex3d_root))

    assert not load_called
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_has_no_checkpoint_identity_or_protocol_override() -> None:
    with pytest.raises(SystemExit):
        full._parse_args(
            [
                "--pretrained-checkpoint",
                "checkpoint",
                "--libero-revision",
                "revision",
                "--output-dir",
                "output",
                "--checkpoint-identity",
                "replacement",
            ]
        )


def test_main_prints_only_deterministic_json(monkeypatch, capsys) -> None:
    expected = {
        "run_status": "COMPLETED",
        "completeness_status": "BLOCKED",
        "observation_file_count": 0,
    }
    monkeypatch.setattr(full, "_run_collection", lambda args: expected)

    exit_code = full.main(
        [
            "--pretrained-checkpoint",
            "checkpoint",
            "--libero-revision",
            "revision",
            "--output-dir",
            "output",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_malformed_coverage_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "coverage"
    result = _write_collection(output_dir, [1] + [0] * 9, "BLOCKED")
    manifest = _read_manifest(output_dir)
    manifest["coverage"]["actual_total_observations"] = 5
    _write_manifest(output_dir, manifest)

    with pytest.raises(AssertionError, match="coverage counts"):
        full._validate_result(result, output_dir, "libero-test-revision")


def test_noncanonical_path_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "path"
    result = _write_collection(output_dir, [1] + [0] * 9, "BLOCKED")
    manifest = _read_manifest(output_dir)
    manifest["task_results"][0]["accepted_groups"][0]["samples"][0][
        "observation_path"
    ] = str(result.sample_paths[0])
    _write_manifest(output_dir, manifest)

    with pytest.raises(AssertionError, match="relative POSIX"):
        full._validate_result(result, output_dir, "libero-test-revision")


def test_duplicate_sample_id_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "duplicate"
    result = _write_collection(output_dir, [1] + [0] * 9, "BLOCKED")
    manifest = _read_manifest(output_dir)
    samples = manifest["task_results"][0]["accepted_groups"][0]["samples"]
    samples[1]["sample_id"] = samples[0]["sample_id"]
    samples[1]["observation_path"] = samples[0]["observation_path"]
    _write_manifest(output_dir, manifest)

    with pytest.raises(AssertionError, match="duplicate manifest sample ID"):
        full._validate_result(result, output_dir, "libero-test-revision")


def test_archive_metadata_mismatch_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "metadata"
    result = _write_collection(output_dir, [1] + [0] * 9, "BLOCKED")
    first_path = result.sample_paths[0]
    record = PilotObservation.load(first_path)
    record.task_id = "9"
    record.save(first_path)

    with pytest.raises(AssertionError, match="metadata differs"):
        full._validate_result(result, output_dir, "libero-test-revision")


def test_missing_archive_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "missing"
    result = _write_collection(output_dir, [1] + [0] * 9, "BLOCKED")
    result.sample_paths[0].unlink()

    with pytest.raises(FileNotFoundError):
        full._validate_result(result, output_dir, "libero-test-revision")
