from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts._full_feature_extraction_common as common
import scripts.c4_full_paired_features as c4
from shared_feature import paired_features as paired


def _small_specs(monkeypatch) -> tuple[common.FeatureSpec, common.FeatureSpec]:
    openvla_spec = common.FeatureSpec(
        model_family="openvla",
        source_model="openvla",
        checkpoint_identity="openvla/openvla-7b-finetuned-libero-spatial",
        feature_schema_version="openvla_features_v1",
        manifest_filename="openvla_feature_manifest.json",
        nodes=(
            common.FeatureNode("O1-S", "o1_siglip", (2, 3)),
            common.FeatureNode("O1-F", "o1_fused", (2, 4)),
            common.FeatureNode("O2", "o2_projected", (2, 5)),
        ),
    )
    pi05_spec = common.FeatureSpec(
        model_family="pi05",
        source_model="pi05",
        checkpoint_identity="gs://openpi-assets/checkpoints/pi05_libero",
        feature_schema_version="pi05_features_v1",
        manifest_filename="pi05_feature_manifest.json",
        nodes=(
            common.FeatureNode("P1", "p1_siglip", (2, 3)),
            common.FeatureNode("P2", "p2_projected", (2, 4)),
        ),
        feature_config="pi05_libero",
    )
    monkeypatch.setattr(c4.c2, "SPEC", openvla_spec)
    monkeypatch.setattr(c4.c3, "SPEC", pi05_spec)
    monkeypatch.setattr(
        paired,
        "_OPENVLA_SCHEMA",
        paired._ArchiveSchema(
            label="OpenVLA",
            feature_shapes={
                node.archive_key: node.shape for node in openvla_spec.nodes
            },
            source_model="openvla",
            feature_schema_version="openvla_features_v1",
        ),
    )
    monkeypatch.setattr(
        paired,
        "_PI05_SCHEMA",
        paired._ArchiveSchema(
            label="pi0.5",
            feature_shapes={node.archive_key: node.shape for node in pi05_spec.nodes},
            source_model="pi05",
            feature_schema_version="pi05_features_v1",
            checkpoint=pi05_spec.checkpoint_identity,
        ),
    )
    return openvla_spec, pi05_spec


def _source_hash(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode()).hexdigest()
    return f"sha256:{digest}"


def _write_archive(
    path: Path,
    *,
    sample_id: str,
    spec: common.FeatureSpec,
    source_hash: str | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "sample_id": sample_id,
        "source_model": spec.source_model,
        "checkpoint": spec.checkpoint_identity,
        "feature_schema_version": spec.feature_schema_version,
        "source_image_hash": source_hash or _source_hash(sample_id),
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    arrays = {
        node.archive_key: np.zeros(node.shape, dtype=np.float32) for node in spec.nodes
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _manifest(
    *,
    spec: common.FeatureSpec,
    records: list[dict[str, str]],
    source_updates: dict[str, Any] | None = None,
    top_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_collection = {
        "manifest_path": "/server/collection/collection_manifest.json",
        "schema_version": common.COLLECTION_SCHEMA_VERSION,
        "checkpoint_identity": common.COLLECTION_CHECKPOINT_IDENTITY,
        "libero_revision": "libero-test-revision",
        "task_count": 10,
        "group_count": 50,
        "observation_count": 200,
    }
    if source_updates:
        source_collection.update(source_updates)
    extraction = {
        "feature_checkpoint_identity": spec.checkpoint_identity,
        "runtime_precision": common.RUNTIME_PRECISION,
        "saved_dtype": common.SAVED_DTYPE,
        "batch_size": common.BATCH_SIZE,
        "num_tokens": common.NUM_TOKENS,
        "grid_shape": list(common.GRID_SHAPE),
        "token_order": common.TOKEN_ORDER,
    }
    if spec.feature_config is not None:
        extraction["feature_config"] = spec.feature_config
    value = {
        "schema_version": common.FEATURE_MANIFEST_SCHEMA_VERSION,
        "pilot_version": common.PILOT_VERSION,
        "run_status": "COMPLETED",
        "completeness_status": "COMPLETE",
        "model_family": spec.model_family,
        "source_collection": source_collection,
        "extraction": extraction,
        "feature_nodes": {
            node.scientific_name: {
                "archive_key": node.archive_key,
                "shape": list(node.shape),
            }
            for node in spec.nodes
        },
        "records": records,
    }
    if top_updates:
        value.update(top_updates)
    return value


def _prepare_inputs(
    monkeypatch,
    tmp_path: Path,
    *,
    count: int,
    order: list[str] | None = None,
    patch_count: bool = True,
) -> tuple[Path, Path, common.FeatureSpec, common.FeatureSpec]:
    openvla_spec, pi05_spec = _small_specs(monkeypatch)
    if patch_count:
        monkeypatch.setattr(c4, "EXPECTED_RECORD_COUNT", count)
    sample_ids = order or [f"sample-{index:03d}" for index in range(count)]
    roots = {
        "openvla": tmp_path / "c2",
        "pi05": tmp_path / "c3",
    }
    specs = {"openvla": openvla_spec, "pi05": pi05_spec}
    manifest_paths: dict[str, Path] = {}
    for model in ("openvla", "pi05"):
        records = []
        for sample_id in sample_ids:
            feature_path = roots[model] / "features" / f"{sample_id}.npz"
            _write_archive(feature_path, sample_id=sample_id, spec=specs[model])
            records.append(
                {
                    "sample_id": sample_id,
                    "source_observation_path": f"observations/{sample_id}.npz",
                    "feature_path": f"features/{sample_id}.npz",
                }
            )
        name = specs[model].manifest_filename
        manifest_path = roots[model] / name
        manifest_path.write_text(
            json.dumps(_manifest(spec=specs[model], records=records)),
            encoding="utf-8",
        )
        manifest_paths[model] = manifest_path
    return (
        manifest_paths["openvla"],
        manifest_paths["pi05"],
        openvla_spec,
        pi05_spec,
    )


def _args(openvla: Path, pi05: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        openvla_feature_manifest=openvla,
        pi05_feature_manifest=pi05,
        output_path=output,
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_materializes_exactly_200_pairs_without_copying_features(
    monkeypatch,
    tmp_path,
) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=200)
    output = tmp_path / "paired" / "paired_features_manifest.json"
    before_npz = {path.resolve() for path in tmp_path.rglob("*.npz")}

    summary = c4._run(_args(openvla, pi05, output))

    manifest = _read(output)
    assert summary == {
        "status": "C4 Formal Paired Feature Materialization — COMPLETE",
        "manifest_path": str(output.resolve()),
        "num_pairs": 200,
    }
    assert manifest["schema_version"] == "paired_features_v1"
    assert manifest["num_samples"] == 200 == len(manifest["pairs"])
    assert [pair["sample_id"] for pair in manifest["pairs"]] == [
        f"sample-{index:03d}" for index in range(200)
    ]
    assert {path.resolve() for path in tmp_path.rglob("*.npz")} == before_npz
    assert list(output.parent.iterdir()) == [output]
    for pair in manifest["pairs"]:
        assert not Path(pair["openvla_feature_path"]).is_absolute()
        assert not Path(pair["pi05_feature_path"]).is_absolute()
        assert (output.parent / pair["openvla_feature_path"]).resolve().is_file()
        assert (output.parent / pair["pi05_feature_path"]).resolve().is_file()


def test_preserves_shared_shuffled_manifest_order(monkeypatch, tmp_path) -> None:
    order = ["sample-c", "sample-a", "sample-b"]
    openvla, pi05, _, _ = _prepare_inputs(
        monkeypatch,
        tmp_path,
        count=3,
        order=order,
    )
    output = tmp_path / "paired.json"

    c4._run(_args(openvla, pi05, output))

    assert [pair["sample_id"] for pair in _read(output)["pairs"]] == order


def test_rejects_different_record_order(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    manifest = _read(pi05)
    manifest["records"].reverse()
    _write(pi05, manifest)

    with pytest.raises(
        paired.PairedFeatureValidationError,
        match="canonical record order differs",
    ):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_different_sample_sets(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, pi05_spec = _prepare_inputs(monkeypatch, tmp_path, count=3)
    manifest = _read(pi05)
    replacement = "sample-other"
    _write_archive(
        pi05.parent / "features" / f"{replacement}.npz",
        sample_id=replacement,
        spec=pi05_spec,
    )
    manifest["records"][-1] = {
        "sample_id": replacement,
        "source_observation_path": f"observations/{replacement}.npz",
        "feature_path": f"features/{replacement}.npz",
    }
    _write(pi05, manifest)

    with pytest.raises(paired.PairedFeatureValidationError, match="sample sets differ"):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_different_source_observation_path(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    manifest = _read(pi05)
    manifest["records"][0]["source_observation_path"] = "observations/wrong.npz"
    _write(pi05, manifest)

    with pytest.raises(
        paired.PairedFeatureValidationError,
        match="source_observation_path",
    ):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_different_source_collection_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    manifest = _read(pi05)
    manifest["source_collection"]["libero_revision"] = "different-revision"
    _write(pi05, manifest)

    with pytest.raises(
        paired.PairedFeatureValidationError,
        match="different source collections",
    ):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"completeness_status": "BLOCKED"}, "completeness_status"),
        ({"schema_version": "wrong"}, "schema_version"),
    ],
)
def test_rejects_invalid_feature_manifest_status_or_schema(
    monkeypatch,
    tmp_path,
    update,
    message,
) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    manifest = _read(openvla)
    manifest.update(update)
    _write(openvla, manifest)

    with pytest.raises(paired.PairedFeatureValidationError, match=message):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_record_count_other_than_200(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, _ = _prepare_inputs(
        monkeypatch,
        tmp_path,
        count=3,
        patch_count=False,
    )

    with pytest.raises(paired.PairedFeatureValidationError, match="exactly 200"):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_unresolved_feature_path(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    missing = pi05.parent / "features" / "sample-000.npz"
    missing.unlink()

    with pytest.raises(paired.PairedFeatureValidationError, match="unresolved"):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_malformed_archive(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    archive = openvla.parent / "features" / "sample-000.npz"
    archive.write_bytes(b"not-an-npz")

    with pytest.raises(paired.PairedFeatureValidationError, match="load or validate"):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


@pytest.mark.parametrize(
    ("target", "update", "message"),
    [
        ("manifest", "wrong-checkpoint", "extraction metadata"),
        ("archive", "wrong-schema", "feature_schema_version"),
    ],
)
def test_rejects_wrong_checkpoint_or_feature_schema(
    monkeypatch,
    tmp_path,
    target,
    update,
    message,
) -> None:
    openvla, pi05, openvla_spec, _ = _prepare_inputs(
        monkeypatch,
        tmp_path,
        count=3,
    )
    if target == "manifest":
        manifest = _read(openvla)
        manifest["extraction"]["feature_checkpoint_identity"] = update
        _write(openvla, manifest)
    else:
        _write_archive(
            openvla.parent / "features" / "sample-000.npz",
            sample_id="sample-000",
            spec=openvla_spec,
            metadata_updates={"feature_schema_version": update},
        )

    with pytest.raises(paired.PairedFeatureValidationError, match=message):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_rejects_source_image_hash_mismatch(monkeypatch, tmp_path) -> None:
    openvla, pi05, _, pi05_spec = _prepare_inputs(monkeypatch, tmp_path, count=3)
    _write_archive(
        pi05.parent / "features" / "sample-000.npz",
        sample_id="sample-000",
        spec=pi05_spec,
        source_hash=f"sha256:{'f' * 64}",
    )

    with pytest.raises(paired.PairedFeatureValidationError, match="hash mismatch"):
        c4._run(_args(openvla, pi05, tmp_path / "paired.json"))


def test_refuses_overwrite_without_changing_existing_output(
    monkeypatch,
    tmp_path,
) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    output = tmp_path / "paired.json"
    output.write_bytes(b"existing")

    with pytest.raises(paired.PairedFeatureValidationError, match="overwrite"):
        c4._run(_args(openvla, pi05, output))

    assert output.read_bytes() == b"existing"


def test_write_failure_leaves_no_incomplete_output(
    monkeypatch,
    tmp_path,
) -> None:
    openvla, pi05, _, _ = _prepare_inputs(monkeypatch, tmp_path, count=3)
    output = tmp_path / "output" / "paired.json"
    original_open = paired.Path.open

    class FailingOutput:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def write(self, value):
            raise OSError("forced write failure")

    def fail_target_write(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == output.resolve():
            return FailingOutput(stream)
        return stream

    monkeypatch.setattr(paired.Path, "open", fail_target_write)

    with pytest.raises(paired.PairedFeatureValidationError, match="failed to write"):
        c4._run(_args(openvla, pi05, output))

    assert not output.exists()
