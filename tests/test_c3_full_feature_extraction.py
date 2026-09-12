from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import scripts._full_feature_extraction_common as common
import scripts.c3_full_feature_extraction as c3
from shared_feature import PilotObservation


def _small_spec() -> common.FeatureSpec:
    return common.FeatureSpec(
        model_family="pi05",
        source_model="pi05",
        checkpoint_identity=c3.CHECKPOINT_IDENTITY,
        feature_schema_version="pi05_torch_features_v1",
        manifest_filename="pi05_feature_manifest.json",
        nodes=(
            common.FeatureNode("P1", "p1_siglip", (2, 3)),
            common.FeatureNode("P2", "p2_projected", (2, 4)),
        ),
        feature_config=c3.EXTRACTION_IDENTITY,
    )


def _prepare_collection(tmp_path: Path) -> Path:
    collection_dir = tmp_path / "collection"
    observations_dir = collection_dir / "observations"
    observations_dir.mkdir(parents=True)
    task_results = []

    for task_id in range(10):
        groups = []
        for state_id in range(5):
            samples = []
            for step_id, progress in enumerate(common.EXPECTED_TARGET_PROGRESS):
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
                    base_rgb_raw=np.asarray(
                        [[[task_id, state_id, step_id]]],
                        dtype=np.uint8,
                    ),
                    wrist_rgb_raw=np.zeros((1, 1, 3), dtype=np.uint8),
                    state=np.zeros(8, dtype=np.float32),
                    prompt=f"task language {task_id}",
                    episode_success=True,
                ).save(path)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "step_id": step_id,
                        "target_relative_progress": progress,
                        "observation_path": f"observations/{sample_id}.npz",
                    }
                )
            groups.append(
                {
                    "task_id": task_id,
                    "initial_state_id": state_id,
                    "episode_success": True,
                    "samples": samples,
                }
            )
        task_results.append(
            {
                "task_id": task_id,
                "actual_accepted_groups": 5,
                "accepted_state_ids": list(range(5)),
                "accepted_groups": groups,
            }
        )

    manifest = {
        "schema_version": common.COLLECTION_SCHEMA_VERSION,
        "pilot_version": common.PILOT_VERSION,
        "suite": "libero_spatial",
        "run_status": "COMPLETED",
        "completeness_status": "COMPLETE",
        "rollout": {
            "checkpoint_identity": common.COLLECTION_CHECKPOINT_IDENTITY,
        },
        "runtime": {
            "libero_revision": "libero-test-revision",
            "discovered_task_count": 10,
            "task_ids": list(range(10)),
        },
        "coverage": {
            "target_total_groups": 50,
            "target_total_observations": 200,
            "actual_total_groups": 50,
            "actual_total_observations": 200,
            "accepted_groups_per_task": {str(index): 5 for index in range(10)},
        },
        "task_results": task_results,
    }
    manifest_path = collection_dir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _prepare_openpi_root(tmp_path: Path) -> Path:
    openpi_root = tmp_path / "openpi"
    (openpi_root / "src" / "openpi").mkdir(parents=True)
    (openpi_root / "packages" / "openpi-client" / "src" / "openpi_client").mkdir(
        parents=True
    )
    return openpi_root


def _prepare_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "pi05-checkpoint"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    return checkpoint


def _write_feature(
    path: Path,
    *,
    observation_path: Path,
    spec: common.FeatureSpec,
    wrong_shape: bool = False,
) -> None:
    observation = PilotObservation.load(observation_path)
    source_hash = (
        "sha256:"
        + hashlib.sha256(
            np.ascontiguousarray(observation.base_rgb_raw).tobytes()
        ).hexdigest()
    )
    metadata = {
        "sample_id": observation.sample_id,
        "source_model": spec.source_model,
        "checkpoint": spec.checkpoint_identity,
        "feature_schema_version": spec.feature_schema_version,
        "source_image_hash": source_hash,
    }
    arrays = {
        node.archive_key: np.zeros(
            (1, 1) if wrong_shape and index == 0 else node.shape,
            dtype=np.float32,
        )
        for index, node in enumerate(spec.nodes)
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _fake_runtime(
    spec: common.FeatureSpec,
    events: list[Any],
    *,
    malformed_first: bool = False,
) -> c3._Runtime:
    model = object()
    policy = object()

    def extractor(**kwargs: Any):
        events.append(kwargs)
        paths = []
        for index, observation_path in enumerate(kwargs["observation_paths"]):
            output_path = kwargs["output_dir"] / f"{Path(observation_path).stem}.npz"
            _write_feature(
                output_path,
                observation_path=Path(observation_path),
                spec=spec,
                wrong_shape=malformed_first and index == 0,
            )
            paths.append(output_path)
        return tuple(paths)

    return c3._Runtime(
        model=model,
        policy=policy,
        extractor=extractor,
    )


def test_full_c3_run_uses_frozen_config_identity_and_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(tmp_path)
    openpi_root = _prepare_openpi_root(tmp_path)
    checkpoint = _prepare_checkpoint(tmp_path)
    output_dir = tmp_path / "c3-output"
    spec = _small_spec()
    events: list[Any] = []
    monkeypatch.setattr(c3, "SPEC", spec)
    monkeypatch.setattr(
        c3,
        "_load_runtime",
        lambda root, checkpoint_path: _fake_runtime(spec, events),
    )
    args = SimpleNamespace(
        collection_manifest=manifest_path,
        output_dir=output_dir,
        openpi_root=openpi_root,
        checkpoint_dir=checkpoint,
    )

    summary = c3._run(args)

    assert summary == {
        "status": "Phase 1 PI0Pytorch Feature Extraction — COMPLETE",
        "manifest_path": str(output_dir.resolve() / spec.manifest_filename),
        "num_feature_archives": 200,
        "num_node_tensors": 400,
        "reused_archives": 0,
        "extracted_archives": 200,
        "backend": "PI0Pytorch",
        "checkpoint_path": str(checkpoint.resolve()),
    }
    assert len(events) == 1
    extractor_args = events[0]
    assert extractor_args["checkpoint"] == c3.CHECKPOINT_IDENTITY
    assert extractor_args["policy"] is not None
    assert extractor_args["batch_size"] == 1
    assert len(extractor_args["observation_paths"]) == 200
    manifest = json.loads(
        (output_dir / spec.manifest_filename).read_text(encoding="utf-8")
    )
    assert manifest["pilot_version"] == "0.2"
    assert manifest["model_family"] == "pi05"
    assert manifest["extraction"]["feature_config"] == c3.EXTRACTION_IDENTITY
    assert manifest["extraction"]["feature_checkpoint_identity"] == (
        c3.CHECKPOINT_IDENTITY
    )
    assert len(manifest["records"]) == 200


def test_c3_runtime_requires_current_pytorch_checkpoint_and_freezes_model(
    monkeypatch,
    tmp_path,
) -> None:
    openpi_root = _prepare_openpi_root(tmp_path)
    checkpoint_path = _prepare_checkpoint(tmp_path)
    events: list[Any] = []
    model_config = SimpleNamespace(
        pi05=True,
        action_horizon=10,
        action_dim=32,
        discrete_state_input=False,
        max_token_len=200,
    )
    train_config = SimpleNamespace(
        name=c3.CONFIG_NAME,
        model=model_config,
    )
    fake_model = torch.nn.Linear(2, 2)
    policy = SimpleNamespace(_is_pytorch_model=True, _model=fake_model)
    components = c3._OpenPIComponents(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            nn=torch.nn,
        ),
        model_type=torch.nn.Module,
        policy_config=SimpleNamespace(
            create_trained_policy=lambda config, checkpoint, pytorch_device: (
                events.append(("policy", config, checkpoint, pytorch_device))
                or policy
            )
        ),
        config=SimpleNamespace(
            get_config=lambda name: (events.append(("config", name)) or train_config)
        ),
        extractor=lambda **kwargs: (),
    )
    monkeypatch.setattr(c3, "_load_openpi_components", lambda root: components)

    runtime = c3._load_runtime(openpi_root, checkpoint_path)

    assert runtime.model is fake_model
    assert runtime.policy is policy
    assert fake_model.training is False
    assert not any(parameter.requires_grad for parameter in fake_model.parameters())
    assert ("config", c3.CONFIG_NAME) in events
    assert ("policy", train_config, checkpoint_path, "cuda") in events


def test_c3_rejects_malformed_extractor_output_without_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(tmp_path)
    openpi_root = _prepare_openpi_root(tmp_path)
    checkpoint = _prepare_checkpoint(tmp_path)
    output_dir = tmp_path / "malformed-output"
    spec = _small_spec()
    monkeypatch.setattr(c3, "SPEC", spec)
    monkeypatch.setattr(
        c3,
        "_load_runtime",
        lambda root, checkpoint_path: _fake_runtime(
            spec, [], malformed_first=True
        ),
    )

    with pytest.raises(common.FullFeatureExtractionError, match="shape"):
        c3._run(
            SimpleNamespace(
                collection_manifest=manifest_path,
                output_dir=output_dir,
                openpi_root=openpi_root,
                checkpoint_dir=checkpoint,
            )
        )
    assert not (output_dir / spec.manifest_filename).exists()
    assert len(tuple((output_dir / "features").glob("*.npz"))) == 200


def test_c3_frozen_schema_and_cli_expose_current_checkpoint_path(tmp_path) -> None:
    assert [(node.archive_key, node.shape) for node in c3.SPEC.nodes] == [
        ("p1_siglip", (256, 1152)),
        ("p2_projected", (256, 2048)),
    ]
    args = c3._parse_args(
        [
            "--collection-manifest",
            str(tmp_path / "collection.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert set(vars(args)) == {
        "collection_manifest",
        "output_dir",
        "openpi_root",
        "checkpoint_dir",
    }
    assert args.checkpoint_dir == c3.DEFAULT_CHECKPOINT_DIR
    assert c3.SPEC.feature_config == "pi05_libero:PI0Pytorch"
