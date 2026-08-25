from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.c5_d0_pilot_v02_smoke_integration as smoke
from shared_feature import PilotObservation, PilotV02CollectionResult


def _arguments(checkpoint: Path, output_dir: Path, tex3d_root: Path):
    return SimpleNamespace(
        pretrained_checkpoint=checkpoint,
        checkpoint_identity="openvla/openvla-7b-finetuned-libero-spatial",
        libero_revision="libero-test-revision",
        output_dir=output_dir,
        tex3d_openvla_root=tex3d_root,
    )


def _prepare_inputs(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "dataset_statistics.json").write_text(
        json.dumps({smoke.UNNORM_KEY: {}}),
        encoding="utf-8",
    )
    tex3d_root = tmp_path / "tex3d-openvla"
    tex3d_root.mkdir()
    return checkpoint, tex3d_root


def _write_fake_result(output_dir: Path) -> PilotV02CollectionResult:
    observations_dir = output_dir / "observations"
    observations_dir.mkdir(parents=True)
    sample_paths: list[Path] = []
    task_results = []
    for task_id in smoke.TASK_IDS:
        samples = []
        for step_id, progress in enumerate(smoke.TARGET_PROGRESS):
            sample_id = f"libero_spatial__task{task_id:02d}__state00__step{step_id:04d}"
            path = observations_dir / f"{sample_id}.npz"
            PilotObservation(
                sample_id=sample_id,
                task_id=str(task_id),
                initial_state_id=0,
                episode_id=0,
                step_id=step_id,
                normalized_episode_progress=step_id / 3,
                base_rgb_raw=np.zeros((4, 5, 3), dtype=np.uint8),
                wrist_rgb_raw=np.zeros((3, 4, 3), dtype=np.uint8),
                state=np.zeros(8, dtype=np.float32),
                prompt=f"task {task_id}",
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
        task_results.append(
            {
                "task_id": task_id,
                "actual_accepted_groups": 1,
                "accepted_groups": [{"samples": samples}],
            }
        )

    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "pilot_v0_2_collection_v1",
                "run_status": "SMOKE_COMPLETED",
                "completeness_status": None,
                "coverage": {
                    "actual_total_groups": 2,
                    "actual_total_observations": 8,
                },
                "task_results": task_results,
            }
        ),
        encoding="utf-8",
    )
    return PilotV02CollectionResult(
        sample_paths=tuple(sample_paths),
        manifest_path=manifest_path.resolve(),
        run_status="SMOKE_COMPLETED",
        completeness_status=None,
    )


def test_run_smoke_uses_frozen_loading_and_reduced_plan(monkeypatch, tmp_path) -> None:
    checkpoint, tex3d_root = _prepare_inputs(tmp_path)
    output_dir = tmp_path / "smoke-output"
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
        assert config.unnorm_key == smoke.UNNORM_KEY
        events.append("processor")
        return processor

    def collect_with_plan(**kwargs):
        assert events[-1] == "processor"
        assert kwargs == {
            "model": model,
            "processor": processor,
            "pretrained_checkpoint": checkpoint.resolve(),
            "checkpoint_identity": ("openvla/openvla-7b-finetuned-libero-spatial"),
            "libero_revision": "libero-test-revision",
            "output_dir": output_dir.resolve(),
            "task_ids": (0, 1),
            "target_groups_per_task": 1,
        }
        events.append("collector")
        return _write_fake_result(output_dir.resolve())

    monkeypatch.setattr(
        smoke,
        "_load_runtime_components",
        lambda root: (
            SimpleNamespace(cuda=FakeCuda()),
            get_model,
            get_processor,
            set_seed,
            collect_with_plan,
        ),
    )

    summary = smoke._run_smoke(_arguments(checkpoint, output_dir, tex3d_root))

    assert events == ["cuda", ("seed", 7), "model", "processor", "collector"]
    assert summary["run_status"] == "SMOKE_COMPLETED"
    assert summary["completeness_status"] is None
    assert summary["accepted_groups"] == 2
    assert summary["observation_file_count"] == 8
    assert len(summary["sample_ids"]) == 8


def test_input_failure_occurs_before_runtime_loading(monkeypatch, tmp_path) -> None:
    checkpoint, tex3d_root = _prepare_inputs(tmp_path)
    output_dir = tmp_path / "nonempty"
    output_dir.mkdir()
    (output_dir / "existing").write_text("keep", encoding="utf-8")
    load_called = False

    def unexpected_load(root):
        del root
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(smoke, "_load_runtime_components", unexpected_load)

    with pytest.raises(FileExistsError, match="not empty"):
        smoke._run_smoke(_arguments(checkpoint, output_dir, tex3d_root))

    assert not load_called
    assert (output_dir / "existing").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("run_status", "completeness_status", "message"),
    [
        ("COMPLETED", None, "run_status"),
        ("SMOKE_COMPLETED", "COMPLETE", "completeness"),
    ],
)
def test_validation_rejects_formal_status_labels(
    tmp_path, run_status, completeness_status, message
) -> None:
    output_dir = tmp_path / "invalid-result"
    result = SimpleNamespace(
        run_status=run_status,
        completeness_status=completeness_status,
        manifest_path=output_dir / "collection_manifest.json",
        sample_paths=(),
    )

    with pytest.raises(AssertionError, match=message):
        smoke._validate_result(result, output_dir)


def test_cli_requires_provenance_arguments() -> None:
    with pytest.raises(SystemExit):
        smoke._parse_args(
            [
                "--pretrained-checkpoint",
                "checkpoint",
                "--output-dir",
                "output",
            ]
        )
