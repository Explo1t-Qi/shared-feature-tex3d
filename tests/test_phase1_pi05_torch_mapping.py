import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import phase1_pi05_torch_mapping as phase1


def _mapping_arrays() -> dict[str, np.ndarray]:
    basis_a = np.zeros((12, 10), dtype=np.float64)
    basis_b = np.zeros((11, 10), dtype=np.float64)
    basis_a[:10] = np.eye(10)
    basis_b[:10] = np.eye(10)
    return {
        "mean_a": np.zeros(12, dtype=np.float64),
        "mean_b": np.zeros(11, dtype=np.float64),
        "basis_a": basis_a,
        "basis_b": basis_b,
        "whitening_a": np.eye(10, dtype=np.float64),
        "whitening_b": np.eye(10, dtype=np.float64),
        "w_a": np.eye(10, dtype=np.float64),
        "w_b": np.eye(10, dtype=np.float64),
        "sigma": np.linspace(1.0, 0.1, 10, dtype=np.float64),
    }


def test_pi05_manifest_must_preserve_paired_order_and_paths(
    monkeypatch, tmp_path
) -> None:
    manifest = tmp_path / "pi05_feature_manifest.json"
    manifest.write_text(
        json.dumps(
            {"extraction": {"feature_config": phase1.PI05_EXTRACTION_IDENTITY}}
        ),
        encoding="utf-8",
    )
    first = (tmp_path / "first.npz").resolve()
    second = (tmp_path / "second.npz").resolve()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    paired = (
        SimpleNamespace(sample_id="a", pi05_feature_path=first),
        SimpleNamespace(sample_id="b", pi05_feature_path=second),
    )
    loaded = SimpleNamespace(
        path=manifest,
        records=(
            SimpleNamespace(sample_id="a", feature_path=first),
            SimpleNamespace(sample_id="b", feature_path=second),
        ),
    )
    monkeypatch.setattr(
        phase1.c4, "_load_feature_manifest", lambda *args, **kwargs: loaded
    )

    phase1._validate_pi05_manifest(manifest, paired)

    loaded.records = tuple(reversed(loaded.records))
    with pytest.raises(phase1.Phase1MappingError, match="order or paths"):
        phase1._validate_pi05_manifest(manifest, paired)


def test_current_c5b_gate_is_frozen_to_positive_significant_heldout() -> None:
    primary = {
        "heldout_top5mean": 0.4,
        "null_summary": {"top5mean": {"empirical_p": 0.01}},
    }
    analysis = SimpleNamespace(
        alignment_summary={"c5b_result": "PASS", "results": {"o2_p2__99": primary}}
    )
    assert phase1._current_primary_result(analysis) is primary

    for result, score, p_value in (
        ("BLOCKED", 0.4, 0.01),
        ("PASS", 0.0, 0.01),
        ("PASS", 0.4, 0.06),
    ):
        analysis.alignment_summary = {
            "c5b_result": result,
            "results": {
                "o2_p2__99": {
                    "heldout_top5mean": score,
                    "null_summary": {"top5mean": {"empirical_p": p_value}},
                }
            },
        }
        with pytest.raises(phase1.Phase1MappingError, match="not scientifically"):
            phase1._current_primary_result(analysis)


def test_mapping_artifact_round_trip_validates_array_hashes(tmp_path) -> None:
    artifact = tmp_path / "phase1-map"
    artifact.mkdir()
    arrays = _mapping_arrays()
    np.savez(artifact / "mapping.npz", **arrays)
    metadata = {
        "schema_version": phase1.METADATA_SCHEMA_VERSION,
        "materialization_id": phase1.MATERIALIZATION_ID,
        "arrays": phase1._array_metadata(arrays),
    }
    (artifact / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (artifact / "validation.json").write_text(
        json.dumps(
            {
                "schema_version": phase1.VALIDATION_SCHEMA_VERSION,
                "phase1_result": "PASS",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "summary.md").write_text(
        "# Phase 1 PI0Pytorch O2/P2 Authoritative Mapping\n",
        encoding="utf-8",
    )

    phase1._validate_published_artifact(artifact)

    metadata["arrays"]["sigma"]["content_hash"] = "sha256:" + "0" * 64
    (artifact / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(phase1.Phase1MappingError, match="hashes differ"):
        phase1._validate_published_artifact(artifact)


def test_output_admission_never_overwrites_historical_artifact(tmp_path) -> None:
    historical = tmp_path / "c5bm-formal-output"
    historical.mkdir()
    for name in phase1.HISTORICAL_FILES:
        (historical / name).write_bytes(name.encode())
    before = phase1._hash_required_directory(
        historical,
        phase1.HISTORICAL_FILES,
        label="historical C5-BM artifact",
    )

    with pytest.raises(phase1.Phase1MappingError, match="historical"):
        phase1._validate_output_admission(historical, historical)
    nonempty = tmp_path / "new-map"
    nonempty.mkdir()
    (nonempty / "keep").write_bytes(b"keep")
    with pytest.raises(phase1.Phase1MappingError, match="must be empty"):
        phase1._validate_output_admission(nonempty, historical)

    assert phase1._hash_required_directory(
        historical,
        phase1.HISTORICAL_FILES,
        label="historical C5-BM artifact",
    ) == before


def test_transactional_publication_round_trips_and_cleans_failed_publish(
    monkeypatch, tmp_path
) -> None:
    arrays = _mapping_arrays()
    fit = SimpleNamespace(arrays=arrays)
    metadata = {
        "schema_version": phase1.METADATA_SCHEMA_VERSION,
        "materialization_id": phase1.MATERIALIZATION_ID,
        "arrays": phase1._array_metadata(arrays),
    }
    validation = {
        "schema_version": phase1.VALIDATION_SCHEMA_VERSION,
        "phase1_result": "PASS",
    }
    summary = "# Phase 1 PI0Pytorch O2/P2 Authoritative Mapping\n"
    output = tmp_path / "mapping-v1"

    phase1._publish(
        output,
        fit=fit,
        metadata=metadata,
        validation=validation,
        summary=summary,
    )
    assert {path.name for path in output.iterdir()} == phase1.OUTPUT_FILES
    phase1._validate_published_artifact(output)

    failing = tmp_path / "mapping-failed"
    original = phase1._validate_published_artifact
    calls = 0

    def fail_after_rename(path):
        nonlocal calls
        calls += 1
        original(path)
        if calls == 2:
            raise phase1.Phase1MappingError("post-rename failure")

    monkeypatch.setattr(phase1, "_validate_published_artifact", fail_after_rename)
    with pytest.raises(phase1.Phase1MappingError, match="post-rename"):
        phase1._publish(
            failing,
            fit=fit,
            metadata=metadata,
            validation=validation,
            summary=summary,
        )
    assert not failing.exists()
