from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.c5a_representation_geometry as c5a
import scripts.c5b_explicit_shared_space as c5b
import scripts.c5bm_authoritative_mapping as c5bm


def _mapping_arrays() -> dict[str, np.ndarray]:
    return {
        "mean_a": np.arange(12, dtype=np.float64),
        "mean_b": np.arange(11, dtype=np.float64),
        "basis_a": np.eye(12, 10, dtype=np.float64),
        "basis_b": np.eye(11, 10, dtype=np.float64),
        "whitening_a": np.eye(10, dtype=np.float64),
        "whitening_b": np.eye(10, dtype=np.float64),
        "w_a": np.eye(10, dtype=np.float64),
        "w_b": np.eye(10, dtype=np.float64),
        "sigma": np.linspace(1.0, 0.1, 10, dtype=np.float64),
    }


def _pca_model(mean_size: int) -> c5b.PCAModel:
    return c5b.PCAModel(
        mean=np.zeros(mean_size, dtype=np.float64),
        basis_99=np.eye(mean_size, 10, dtype=np.float64),
        full_rank=10,
        d_95=10,
        d_99=10,
        explained_variance_95=0.96,
        explained_variance_99=0.995,
    )


def _fit() -> c5bm.MappingFit:
    return c5bm.MappingFit(
        arrays=_mapping_arrays(),
        pca_a=_pca_model(12),
        pca_b=_pca_model(11),
        sign_anchors=tuple(
            {
                "component_index": index,
                "anchor_index": index,
                "anchor_side": "A",
                "anchor_coordinate": index,
                "precanonicalization_anchor_value": 1.0,
                "multiplier": 1,
            }
            for index in range(10)
        ),
        metrics={
            "train_top5mean": 0.9,
            "heldout_top1": 0.8,
            "heldout_top5mean": 0.7,
            "heldout_top10mean": 0.6,
        },
    )


def _historical(tmp_path: Path) -> c5bm.HistoricalC5B:
    directory = tmp_path / "historical"
    directory.mkdir()
    paths = {}
    for name in c5bm.HISTORICAL_FILES:
        path = directory / name
        path.write_bytes(f"historical:{name}".encode())
        paths[name] = path.resolve()
    return c5bm.HistoricalC5B(
        directory=directory.resolve(),
        paths=paths,
        hashes={name: c5bm._sha256_file(path) for name, path in paths.items()},
        split_manifest={},
        alignment_summary={},
    )


def _artifact_documents(
    fit: c5bm.MappingFit,
) -> tuple[dict, dict, str]:
    digest = "sha256:" + "0" * 64
    source_features = [
        {
            "sample_id": f"sample-{index:03d}",
            "source_image_hash": digest,
            "openvla_feature_path": f"/features/openvla/sample-{index:03d}.npz",
            "openvla_feature_hash": digest,
            "pi05_feature_path": f"/features/pi05/sample-{index:03d}.npz",
            "pi05_feature_hash": digest,
        }
        for index in range(c5bm.EXPECTED_PAIR_COUNT)
    ]
    metadata = {
        "schema_version": c5bm.METADATA_SCHEMA_VERSION,
        "mapping_schema_version": c5bm.MAPPING_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "materialization_id": c5bm.MATERIALIZATION_ID,
        "fit_configuration": dict(c5bm.FIT_CONFIGURATION),
        "mapping_identity": {
            "openvla_retained_pca_dimensions": 10,
            "pi05_retained_pca_dimensions": 10,
            "canonical_component_count": 10,
            "canonical_order": "descending_numpy_svd_order",
            "paired_component_arrays": ["w_a", "w_b", "sigma"],
            "sign_rule_version": (
                "c5bm_concat_native_readout_max_abs_smallest_index_v1"
            ),
            "sign_anchors": list(fit.sign_anchors),
        },
        "arrays": c5bm._array_metadata(fit.arrays),
        "provenance": {
            "source_paired_manifest": {
                "runtime_path": "/inputs/paired.json",
                "content_hash": digest,
            },
            "source_split_manifest": {
                "runtime_path": "/inputs/split.json",
                "content_hash": digest,
            },
            "historical_c5b_files": {
                name: {
                    "runtime_path": f"/inputs/c5b/{name}",
                    "content_hash": digest,
                }
                for name in c5bm.HISTORICAL_FILES
            },
            "historical_alignment_summary_hash": digest,
            "source_feature_artifacts": source_features,
            "source_feature_validation": {
                "validated_pair_count": c5bm.EXPECTED_PAIR_COUNT,
                "result": "PASS",
            },
            "repository_commit": "a" * 40,
            "python_version": "test",
            "numpy_version": np.__version__,
            "numpy_blas_lapack_configuration": "test",
            "platform": "test",
        },
    }
    scalar_checks = {
        name: {
            "new": 1.0,
            "historical": 1.0,
            "absolute_difference": 0.0,
            "passed": True,
        }
        for name in (
            "train_top5mean",
            "heldout_top1",
            "heldout_top5mean",
            "heldout_top10mean",
        )
    }
    validation = {
        "schema_version": c5bm.VALIDATION_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "c5bm_result": "PASS",
        "materialization_id": c5bm.MATERIALIZATION_ID,
        "exact_match_checks": {
            "representation_pair": True,
            "pca_cutoff": True,
            "split_identity": True,
            "paired_manifest_identity": True,
            "observation_order": True,
            "token_row_order": True,
            "openvla_retained_pca_dimensions": 10,
            "pi05_retained_pca_dimensions": 10,
        },
        "historical_scalar_tolerance": c5bm.SCALAR_TOLERANCE,
        "historical_scalar_checks": scalar_checks,
        "historical_file_hashes": {name: digest for name in c5bm.HISTORICAL_FILES},
        "historical_files_unchanged": True,
        "complete_source_feature_validation": "PASS",
        "transactional_publication_validation": "PASS",
    }
    summary = "# C5-BM Authoritative Mapping Materialization\n"
    return metadata, validation, summary


def _historical_summary(metrics: dict[str, float]) -> dict:
    return {"results": {"o2_p2__99": metrics}}


def test_sign_canonicalization_keeps_or_flips_both_sides() -> None:
    basis_a = np.eye(2, dtype=np.float64)
    basis_b = np.eye(2, dtype=np.float64)
    w_a = np.asarray([[3.0, 0.0], [0.0, -4.0]], dtype=np.float64)
    w_b = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    actual_a, actual_b, anchors = c5bm._canonicalize_signs(basis_a, basis_b, w_a, w_b)

    np.testing.assert_array_equal(actual_a[:, 0], w_a[:, 0])
    np.testing.assert_array_equal(actual_b[:, 0], w_b[:, 0])
    np.testing.assert_array_equal(actual_a[:, 1], -w_a[:, 1])
    np.testing.assert_array_equal(actual_b[:, 1], -w_b[:, 1])
    assert [anchor["multiplier"] for anchor in anchors] == [1, -1]
    np.testing.assert_array_equal(w_a, [[3.0, 0.0], [0.0, -4.0]])
    np.testing.assert_array_equal(w_b, [[2.0, 0.0], [0.0, 1.0]])


def test_sign_canonicalization_tie_uses_smallest_concatenated_index() -> None:
    basis = np.eye(2, dtype=np.float64)
    w_a = np.asarray([[-2.0], [0.0]], dtype=np.float64)
    w_b = np.asarray([[2.0], [0.0]], dtype=np.float64)

    actual_a, actual_b, anchors = c5bm._canonicalize_signs(basis, basis, w_a, w_b)

    np.testing.assert_array_equal(actual_a, -w_a)
    np.testing.assert_array_equal(actual_b, -w_b)
    assert anchors[0]["anchor_index"] == 0
    assert anchors[0]["multiplier"] == -1


@pytest.mark.parametrize("failure", ["nonfinite", "all_zero"])
def test_sign_canonicalization_rejects_invalid_readout(failure: str) -> None:
    basis_a = np.eye(2, dtype=np.float64)
    basis_b = np.eye(2, dtype=np.float64)
    w_a = np.ones((2, 1), dtype=np.float64)
    w_b = np.ones((2, 1), dtype=np.float64)
    if failure == "nonfinite":
        basis_a[0, 0] = np.nan
        match = "non-finite"
    else:
        w_a.fill(0.0)
        w_b.fill(0.0)
        match = "all-zero"

    with pytest.raises(c5bm.C5BMMaterializationError, match=match):
        c5bm._canonicalize_signs(basis_a, basis_b, w_a, w_b)


def test_historical_scalar_tolerance_boundary() -> None:
    names = (
        "train_top5mean",
        "heldout_top1",
        "heldout_top5mean",
        "heldout_top10mean",
    )
    historical = _historical_summary({name: 0.0 for name in names})
    at_tolerance = {name: c5bm.SCALAR_TOLERANCE for name in names}

    checks = c5bm._validate_historical_scalars(at_tolerance, historical)

    assert all(check["passed"] is True for check in checks.values())
    beyond = dict(at_tolerance)
    beyond["heldout_top5mean"] = np.nextafter(c5bm.SCALAR_TOLERANCE, np.inf)
    with pytest.raises(c5bm.C5BMMaterializationError, match="heldout_top5mean"):
        c5bm._validate_historical_scalars(beyond, historical)


def test_exact_split_and_summary_metadata_validation(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "paired.json"
    manifest.write_text("{}", encoding="utf-8")
    group = c5a.GroupRecord(
        canonical_index=0,
        task_id=0,
        initial_state_id=4,
        record_indices=(0, 1, 2, 3),
        sample_ids=("a", "b", "c", "d"),
        step_ids=(1, 2, 3, 4),
    )
    split = c5a.DatasetSplit(
        train_group_indices=(),
        heldout_group_indices=(0,),
        train_observation_indices=np.asarray([], dtype=np.int64),
        heldout_observation_indices=np.arange(4, dtype=np.int64),
        group_digests=("digest",),
        assignments=("HELD-OUT",),
        heldout_state_by_task={0: 4},
    )
    monkeypatch.setattr(c5a, "EXPECTED_TASK_IDS", (0,))
    split_value = c5bm._expected_split_content(manifest.resolve(), (group,), split)
    split_value["source_c5a_split_manifest"] = "/runtime/c5a/split.json"
    c5bm._validate_historical_split(
        split_value,
        manifest_path=manifest.resolve(),
        groups=(group,),
        split=split,
    )
    split_value["source_paired_manifest"] = "/unavailable/server/paired.json"
    c5bm._validate_historical_split(
        split_value,
        manifest_path=manifest.resolve(),
        groups=(group,),
        split=split,
    )
    invalid_split = json.loads(json.dumps(split_value))
    invalid_split["groups"][0]["assignment"] = "TRAIN"
    with pytest.raises(c5bm.C5BMMaterializationError, match="frozen split"):
        c5bm._validate_historical_split(
            invalid_split,
            manifest_path=manifest.resolve(),
            groups=(group,),
            split=split,
        )

    summary = {key: None for key in c5bm.HISTORICAL_SUMMARY_KEYS}
    pca_metadata = {
        "full_rank": 10,
        "d_95": 10,
        "d_99": 10,
        "explained_variance_95": 0.96,
        "explained_variance_99": 0.995,
    }
    result_metrics = {
        "train_top5mean": 0.9,
        "heldout_top1": 0.8,
        "heldout_top5mean": 0.7,
        "heldout_top10mean": 0.6,
    }
    summary.update(
        {
            "schema_version": c5b.SUMMARY_SCHEMA_VERSION,
            "run_status": "COMPLETED",
            "source_paired_manifest": str(manifest.resolve()),
            "c5a_gate": "GO",
            "c5b_result": "PASS",
            "representation_stage_result": "PASS",
            "primary_pair": {
                "name": "o2_p2",
                "openvla_node": "O2",
                "pi05_node": "P2",
                "gates_c5b": True,
            },
            "control_pair": {
                "name": "o1s_p1",
                "openvla_node": "O1-S",
                "pi05_node": "P1",
                "gates_c5b": False,
            },
            "cca": {
                "method": "ordinary_linear_covariance_whitening_svd",
                "covariance_denominator": "n_minus_1",
                "eigendecomposition_api": "numpy.linalg.eigh",
                "svd_api": "numpy.linalg.svd_full_matrices_false",
                "regularization": "none",
                "min_valid_canonical_dims": 10,
            },
            "pca": {name: dict(pca_metadata) for name in ("o2", "p2", "o1s", "p1")},
            "results": {
                name: dict(result_metrics)
                for name in (
                    "o2_p2__99",
                    "o2_p2__95",
                    "o1s_p1__99",
                    "o1s_p1__95",
                )
            },
        }
    )
    c5bm._validate_historical_summary(summary, manifest_path=manifest.resolve())
    summary["c5b_result"] = "FAIL"
    with pytest.raises(c5bm.C5BMMaterializationError, match="c5b_result"):
        c5bm._validate_historical_summary(summary, manifest_path=manifest.resolve())


def test_metadata_records_array_and_historical_hashes(tmp_path, monkeypatch) -> None:
    fit = _fit()
    historical = _historical(tmp_path)
    manifest = tmp_path / "paired.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c5bm, "_repository_commit", lambda: "a" * 40)
    monkeypatch.setattr(c5bm, "_numpy_build_configuration", lambda: "blas=test")

    metadata = c5bm._build_metadata(
        manifest_path=manifest.resolve(),
        manifest_hash=c5bm._sha256_file(manifest),
        historical=historical,
        source_features=[],
        fit=fit,
    )

    assert metadata["arrays"] == c5bm._array_metadata(fit.arrays)
    assert set(metadata["arrays"]) == c5bm.MAPPING_ARRAY_KEYS
    assert all(value["dtype"] == "float64" for value in metadata["arrays"].values())
    assert (
        metadata["provenance"]["historical_alignment_summary_hash"]
        == (historical.hashes["alignment_summary.json"])
    )
    assert {
        name: value["content_hash"]
        for name, value in metadata["provenance"]["historical_c5b_files"].items()
    } == historical.hashes


def test_historical_mutation_is_detected(tmp_path) -> None:
    historical = _historical(tmp_path)
    c5bm._assert_historical_unchanged(historical)
    historical.paths["alignment_summary.json"].write_text("changed", encoding="utf-8")

    with pytest.raises(c5bm.C5BMMaterializationError, match="changed"):
        c5bm._assert_historical_unchanged(historical)


def test_source_feature_mutation_is_detected(tmp_path) -> None:
    openvla = tmp_path / "openvla.npz"
    pi05 = tmp_path / "pi05.npz"
    openvla.write_bytes(b"openvla")
    pi05.write_bytes(b"pi05")
    source_features = [
        {
            "sample_id": "sample-000",
            "openvla_feature_path": str(openvla),
            "openvla_feature_hash": c5bm._sha256_file(openvla),
            "pi05_feature_path": str(pi05),
            "pi05_feature_hash": c5bm._sha256_file(pi05),
        }
    ]
    c5bm._assert_source_features_unchanged(source_features)
    pi05.write_bytes(b"changed")

    with pytest.raises(c5bm.C5BMMaterializationError, match="sample-000"):
        c5bm._assert_source_features_unchanged(source_features)


def test_transactional_publication_persists_exact_four_files(tmp_path) -> None:
    fit = _fit()
    historical = _historical(tmp_path)
    metadata, validation, summary = _artifact_documents(fit)
    output = tmp_path / "mapping"

    c5bm._publish(
        output,
        fit=fit,
        metadata=metadata,
        validation=validation,
        summary=summary,
        historical=historical,
    )

    assert {path.name for path in output.iterdir()} == c5bm.OUTPUT_FILES
    c5bm._validate_published_artifact(output)
    with np.load(output / "mapping.npz", allow_pickle=False) as archive:
        assert set(archive.files) == c5bm.MAPPING_ARRAY_KEYS
        np.testing.assert_array_equal(archive["sigma"], fit.arrays["sigma"])


def test_output_admission_and_failed_publication_are_safe(tmp_path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    existing = nonempty / "existing.txt"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(c5bm.C5BMMaterializationError, match="must be empty"):
        c5bm._validate_output_admission(nonempty)
    assert existing.read_text(encoding="utf-8") == "keep"

    fit = _fit()
    historical = _historical(tmp_path)
    metadata, validation, summary = _artifact_documents(fit)
    with pytest.raises(c5bm.C5BMMaterializationError, match="became non-empty"):
        c5bm._publish(
            nonempty,
            fit=fit,
            metadata=metadata,
            validation=validation,
            summary=summary,
            historical=historical,
        )
    assert existing.read_text(encoding="utf-8") == "keep"

    metadata["arrays"]["sigma"]["content_hash"] = "sha256:" + "0" * 64
    output = tmp_path / "failed"
    with pytest.raises(c5bm.C5BMMaterializationError, match="integrity"):
        c5bm._publish(
            output,
            fit=fit,
            metadata=metadata,
            validation=validation,
            summary=summary,
            historical=historical,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.*"))
