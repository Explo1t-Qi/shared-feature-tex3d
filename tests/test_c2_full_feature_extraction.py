from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts._full_feature_extraction_common as common
import scripts.c2_full_feature_extraction as c2


def _small_spec() -> common.FeatureSpec:
    return common.FeatureSpec(
        model_family="openvla",
        source_model="openvla",
        checkpoint_identity=c2.CHECKPOINT_IDENTITY,
        feature_schema_version="openvla_features_v1",
        manifest_filename="openvla_feature_manifest.json",
        nodes=(
            common.FeatureNode("O1-S", "o1_siglip", (2, 3)),
            common.FeatureNode("O1-F", "o1_fused", (2, 4)),
            common.FeatureNode("O2", "o2_projected", (2, 5)),
        ),
    )


def _prepare_collection(monkeypatch, tmp_path: Path) -> Path:
    collection_dir = tmp_path / "collection"
    observations_dir = collection_dir / "observations"
    observations_dir.mkdir(parents=True)
    fake_observations: dict[Path, Any] = {}
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
                path.touch()
                image = np.asarray(
                    [[[task_id, state_id, step_id]]],
                    dtype=np.uint8,
                )
                fake_observations[path.resolve()] = SimpleNamespace(
                    sample_id=sample_id,
                    task_id=str(task_id),
                    initial_state_id=state_id,
                    episode_id=state_id,
                    step_id=step_id,
                    episode_success=True,
                    base_rgb_raw=image,
                )
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

    class FakePilotObservation:
        @staticmethod
        def load(path: str | Path):
            return fake_observations[Path(path).resolve()]

    monkeypatch.setattr(common, "PilotObservation", FakePilotObservation)
    return manifest_path


def _prepare_runtime_paths(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "dataset_statistics.json").write_text(
        json.dumps({c2.UNNORM_KEY: {}}),
        encoding="utf-8",
    )
    tex3d_root = tmp_path / "tex3d-openvla"
    tex3d_root.mkdir()
    return checkpoint, tex3d_root


def _args(
    manifest_path: Path,
    checkpoint: Path,
    tex3d_root: Path,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        collection_manifest=manifest_path,
        pretrained_checkpoint=checkpoint,
        output_dir=output_dir,
        tex3d_openvla_root=tex3d_root,
    )


def _write_feature(
    path: Path,
    *,
    sample_id: str,
    source_hash: str,
    spec: common.FeatureSpec,
    metadata_updates: dict[str, Any] | None = None,
    shape_updates: dict[str, tuple[int, ...]] | None = None,
) -> None:
    metadata = {
        "sample_id": sample_id,
        "source_model": spec.source_model,
        "checkpoint": spec.checkpoint_identity,
        "feature_schema_version": spec.feature_schema_version,
        "source_image_hash": source_hash,
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    arrays = {
        node.archive_key: np.zeros(
            (shape_updates or {}).get(node.archive_key, node.shape),
            dtype=np.float32,
        )
        for node in spec.nodes
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class _FakeTorch:
    bfloat16 = object()

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    @staticmethod
    def is_floating_point(value: Any) -> bool:
        return True


class _FakeModel:
    def __init__(self) -> None:
        parameter = SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            dtype=_FakeTorch.bfloat16,
        )
        self.vision_backbone = SimpleNamespace(parameters=lambda: iter([parameter]))


def _fake_runtime(spec: common.FeatureSpec, events: list[Any]) -> c2._Runtime:
    model = _FakeModel()
    processor = object()

    def get_model(config: Any) -> Any:
        events.append(("model_config", vars(config)))
        return model

    def get_processor(config: Any) -> Any:
        events.append(("processor_config", vars(config)))
        return processor

    def extractor(**kwargs: Any):
        events.append(("extractor", kwargs))
        paths = []
        for observation_path in kwargs["observation_paths"]:
            sample_id = Path(observation_path).stem
            record = common.PilotObservation.load(observation_path)
            image_bytes = np.ascontiguousarray(record.base_rgb_raw).tobytes()
            source_hash = "sha256:" + hashlib.sha256(image_bytes).hexdigest()
            output_path = kwargs["output_dir"] / f"{sample_id}.npz"
            _write_feature(
                output_path,
                sample_id=sample_id,
                source_hash=source_hash,
                spec=spec,
            )
            paths.append(output_path)
        return tuple(paths)

    return c2._Runtime(
        torch=_FakeTorch,
        get_model=get_model,
        get_processor=get_processor,
        set_seed_everywhere=lambda seed: events.append(("seed", seed)),
        extractor=extractor,
    )


def test_full_c2_run_uses_frozen_identity_and_writes_exact_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(monkeypatch, tmp_path)
    checkpoint, tex3d_root = _prepare_runtime_paths(tmp_path)
    output_dir = tmp_path / "c2-output"
    spec = _small_spec()
    events: list[Any] = []
    monkeypatch.setattr(c2, "SPEC", spec)
    monkeypatch.setattr(c2, "_load_runtime", lambda root: _fake_runtime(spec, events))

    summary = c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))

    assert summary == {
        "status": "C2 OpenVLA Full Feature Extraction — COMPLETE",
        "manifest_path": str(output_dir.resolve() / spec.manifest_filename),
        "num_feature_archives": 200,
        "num_node_tensors": 600,
        "reused_archives": 0,
        "extracted_archives": 200,
    }
    assert events[0] == ("seed", 7)
    model_config = events[1][1]
    assert model_config == {
        "model_family": "openvla",
        "pretrained_checkpoint": str(checkpoint.resolve()),
        "load_in_8bit": False,
        "load_in_4bit": False,
        "unnorm_key": c2.UNNORM_KEY,
        "center_crop": True,
    }
    extractor_args = events[3][1]
    assert extractor_args["pretrained_checkpoint"] == c2.CHECKPOINT_IDENTITY
    assert extractor_args["batch_size"] == 1
    assert extractor_args["center_crop"] is True
    assert len(extractor_args["observation_paths"]) == 200

    manifest = json.loads(
        (output_dir / spec.manifest_filename).read_text(encoding="utf-8")
    )
    assert set(manifest) == {
        "schema_version",
        "pilot_version",
        "run_status",
        "completeness_status",
        "model_family",
        "source_collection",
        "extraction",
        "feature_nodes",
        "records",
    }
    assert manifest["pilot_version"] == "0.2"
    assert manifest["model_family"] == "openvla"
    assert manifest["run_status"] == "COMPLETED"
    assert manifest["completeness_status"] == "COMPLETE"
    assert manifest["extraction"]["feature_checkpoint_identity"] == (
        c2.CHECKPOINT_IDENTITY
    )
    assert len(manifest["records"]) == 200
    assert manifest["records"][0]["sample_id"].endswith("task00__state00__step0000")
    assert manifest["records"][-1]["sample_id"].endswith("task09__state04__step0003")
    first_path = output_dir / manifest["records"][0]["feature_path"]
    with np.load(first_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["checkpoint"] == c2.CHECKPOINT_IDENTITY

    with pytest.raises(common.FullFeatureExtractionError, match="already exists"):
        c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))


def test_c2_resume_reuses_valid_archive_and_extracts_only_missing(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(monkeypatch, tmp_path)
    checkpoint, tex3d_root = _prepare_runtime_paths(tmp_path)
    output_dir = tmp_path / "resume-output"
    spec = _small_spec()
    monkeypatch.setattr(c2, "SPEC", spec)
    source = common.load_source_collection(manifest_path)
    first = source.records[0]
    _write_feature(
        output_dir / "features" / f"{first.sample_id}.npz",
        sample_id=first.sample_id,
        source_hash=first.source_image_hash,
        spec=spec,
    )
    events: list[Any] = []
    monkeypatch.setattr(c2, "_load_runtime", lambda root: _fake_runtime(spec, events))

    summary = c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))

    assert summary["reused_archives"] == 1
    assert summary["extracted_archives"] == 199
    extractor_args = next(value for name, value in events if name == "extractor")
    assert len(extractor_args["observation_paths"]) == 199
    assert first.resolved_observation_path not in extractor_args["observation_paths"]


def test_c2_rejects_malformed_or_unexpected_partial_archives_without_model_load(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(monkeypatch, tmp_path)
    checkpoint, tex3d_root = _prepare_runtime_paths(tmp_path)
    output_dir = tmp_path / "bad-partial"
    spec = _small_spec()
    monkeypatch.setattr(c2, "SPEC", spec)
    source = common.load_source_collection(manifest_path)
    first = source.records[0]
    malformed_path = output_dir / "features" / f"{first.sample_id}.npz"
    _write_feature(
        malformed_path,
        sample_id=first.sample_id,
        source_hash=f"sha256:{'f' * 64}",
        spec=spec,
    )
    original = malformed_path.read_bytes()
    monkeypatch.setattr(
        c2,
        "_load_runtime",
        lambda root: pytest.fail("model runtime must not load"),
    )

    with pytest.raises(common.FullFeatureExtractionError, match="metadata differs"):
        c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))
    assert malformed_path.read_bytes() == original
    assert not (output_dir / spec.manifest_filename).exists()

    malformed_path.unlink()
    unexpected = output_dir / "features" / "unexpected.npz"
    np.savez_compressed(unexpected, value=np.zeros(1))
    with pytest.raises(common.FullFeatureExtractionError, match="unexpected feature"):
        c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))


def test_c2_failure_preserves_partial_archive_without_final_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(monkeypatch, tmp_path)
    checkpoint, tex3d_root = _prepare_runtime_paths(tmp_path)
    output_dir = tmp_path / "failed-output"
    spec = _small_spec()
    events: list[Any] = []
    runtime = _fake_runtime(spec, events)
    base_extractor = runtime.extractor

    def failing_extractor(**kwargs: Any):
        first_kwargs = dict(kwargs)
        first_kwargs["observation_paths"] = kwargs["observation_paths"][:1]
        base_extractor(**first_kwargs)
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(c2, "SPEC", spec)
    monkeypatch.setattr(
        c2,
        "_load_runtime",
        lambda root: c2._Runtime(
            torch=runtime.torch,
            get_model=runtime.get_model,
            get_processor=runtime.get_processor,
            set_seed_everywhere=runtime.set_seed_everywhere,
            extractor=failing_extractor,
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        c2._run(_args(manifest_path, checkpoint, tex3d_root, output_dir))
    assert len(tuple((output_dir / "features").glob("*.npz"))) == 1
    assert not (output_dir / spec.manifest_filename).exists()


def test_common_rejects_non_200_source_set_and_cleans_atomic_write_failure(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = _prepare_collection(monkeypatch, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = manifest["task_results"][-1]["accepted_groups"][-1]["samples"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(common.FullFeatureExtractionError):
        common.load_source_collection(manifest_path)
    (manifest_path.parent / removed["observation_path"]).unlink()

    output_dir = tmp_path / "atomic"
    output_dir.mkdir()
    output_path = output_dir / "manifest.json"
    monkeypatch.setattr(
        common.os,
        "link",
        lambda source, target: (_ for _ in ()).throw(OSError("link failed")),
    )
    with pytest.raises(common.FullFeatureExtractionError, match="atomically write"):
        common.write_manifest_atomic(output_path, {"status": "COMPLETE"})
    assert not output_path.exists()
    assert not tuple(output_dir.glob(".*.tmp"))


def test_c2_frozen_schema_and_cli_have_no_scientific_overrides(tmp_path) -> None:
    assert [(node.archive_key, node.shape) for node in c2.SPEC.nodes] == [
        ("o1_siglip", (256, 1152)),
        ("o1_fused", (256, 2176)),
        ("o2_projected", (256, 4096)),
    ]
    args = c2._parse_args(
        [
            "--collection-manifest",
            str(tmp_path / "collection.json"),
            "--pretrained-checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert set(vars(args)) == {
        "collection_manifest",
        "pretrained_checkpoint",
        "output_dir",
        "tex3d_openvla_root",
    }
