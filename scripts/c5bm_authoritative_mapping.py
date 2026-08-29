from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c5a_representation_geometry as c5a  # noqa: E402
from scripts import c5b_explicit_shared_space as c5b  # noqa: E402


MAPPING_SCHEMA_VERSION = "c5bm_mapping_v1"
METADATA_SCHEMA_VERSION = "c5bm_metadata_v1"
VALIDATION_SCHEMA_VERSION = "c5bm_validation_v1"
MATERIALIZATION_ID = "c5bm_o2_p2_99_true_train_v1"
SCALAR_TOLERANCE = 1e-8
EXPECTED_PAIR_COUNT = 200
OUTPUT_FILES = {"mapping.npz", "metadata.json", "validation.json", "summary.md"}
MAPPING_ARRAY_KEYS = {
    "mean_a",
    "mean_b",
    "basis_a",
    "basis_b",
    "whitening_a",
    "whitening_b",
    "w_a",
    "w_b",
    "sigma",
}
METADATA_KEYS = {
    "schema_version",
    "mapping_schema_version",
    "run_status",
    "materialization_id",
    "fit_configuration",
    "mapping_identity",
    "arrays",
    "provenance",
}
VALIDATION_KEYS = {
    "schema_version",
    "run_status",
    "c5bm_result",
    "materialization_id",
    "exact_match_checks",
    "historical_scalar_tolerance",
    "historical_scalar_checks",
    "historical_file_hashes",
    "historical_files_unchanged",
    "complete_source_feature_validation",
    "transactional_publication_validation",
}
EXACT_MATCH_KEYS = {
    "representation_pair",
    "pca_cutoff",
    "split_identity",
    "paired_manifest_identity",
    "observation_order",
    "token_row_order",
    "openvla_retained_pca_dimensions",
    "pi05_retained_pca_dimensions",
}
HISTORICAL_FILES = {
    "split_manifest.json",
    "alignment_summary.json",
    "null_alignment_metrics.npz",
    "summary.md",
}
HISTORICAL_SPLIT_KEYS = {
    "schema_version",
    "source_paired_manifest",
    "source_c5a_split_manifest",
    "split_rule_id",
    "counts",
    "heldout_state_by_task",
    "canonical_group_order",
    "groups",
}
HISTORICAL_SUMMARY_KEYS = {
    "schema_version",
    "run_status",
    "source_paired_manifest",
    "source_c5a_metric_summary",
    "source_c5a_split_manifest",
    "c5a_gate",
    "primary_pair",
    "control_pair",
    "pca",
    "cca",
    "null",
    "results",
    "c5b_result",
    "representation_stage_result",
    "heldout_null_limitation",
    "interpretation_boundary",
}
PRIMARY_SPEC = c5b.PAIR_SPECS[0]
FIT_CONFIGURATION = {
    "representation_pair": "O2_P2",
    "openvla_node": "O2",
    "pi05_node": "P2",
    "pca_cutoff": 0.99,
    "fit_split": "TRAIN",
    "pairing": "true_unpermuted",
    "estimator": "frozen_c5b_pca_ordinary_linear_cca",
    "numeric_dtype": "float64",
    "token_row_order": "observation_major_then_original_token_rows_0_to_255",
}


class C5BMMaterializationError(RuntimeError):
    """Raised when the frozen C5-BM mapping cannot be materialized safely."""


@dataclass(frozen=True)
class HistoricalC5B:
    directory: Path
    paths: dict[str, Path]
    hashes: dict[str, str]
    split_manifest: dict[str, Any]
    alignment_summary: dict[str, Any]


@dataclass(frozen=True)
class MappingFit:
    arrays: dict[str, np.ndarray]
    pca_a: c5b.PCAModel
    pca_b: c5b.PCAModel
    sign_anchors: tuple[dict[str, Any], ...]
    metrics: dict[str, float]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen C5-BM authoritative O2/P2 mapping."
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        required=True,
        help="Completed formal C4 paired_features_v1 manifest.",
    )
    parser.add_argument(
        "--c5b-output-dir",
        type=Path,
        required=True,
        help="Completed historical four-file C5-B output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Absent or empty destination for the four C5-BM artifacts.",
    )
    return parser.parse_args(argv)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise C5BMMaterializationError(f"failed to hash file: {path}") from error
    return f"sha256:{digest.hexdigest()}"


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise C5BMMaterializationError(f"failed to load {label}: {path}") from error
    if not isinstance(value, dict):
        raise C5BMMaterializationError(f"{label} must encode a JSON object")
    return value


def _resolve_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5BMMaterializationError(
            f"{label} does not exist: {candidate}"
        ) from error
    if not resolved.is_file():
        raise C5BMMaterializationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _validate_recorded_manifest_identity(
    value: Any,
    *,
    current_manifest: Path,
    label: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C5BMMaterializationError(f"invalid {label}: {value!r}")
    recorded = Path(value).expanduser()
    if recorded.exists():
        resolved = _resolve_file(recorded, label)
        if _sha256_file(resolved) != _sha256_file(current_manifest):
            raise C5BMMaterializationError(
                f"{label} content differs from the current paired manifest"
            )
    return value


def _validate_output_admission(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    if output.exists():
        if not output.is_dir():
            raise C5BMMaterializationError(
                f"C5-BM output path is not a directory: {output}"
            )
        try:
            entries = tuple(output.iterdir())
        except OSError as error:
            raise C5BMMaterializationError(
                f"failed to inspect C5-BM output directory: {output}"
            ) from error
        if entries:
            raise C5BMMaterializationError(
                f"C5-BM output directory must be empty: {output}"
            )
    return output


def _expected_split_content(
    manifest_path: Path,
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> dict[str, Any]:
    return {
        "schema_version": c5b.SPLIT_SCHEMA_VERSION,
        "source_paired_manifest": str(manifest_path),
        "split_rule_id": c5a.SPLIT_RULE_ID,
        "counts": {
            "groups": len(groups),
            "observations": sum(len(group.record_indices) for group in groups),
            "train_groups": len(split.train_group_indices),
            "heldout_groups": len(split.heldout_group_indices),
            "train_observations": int(split.train_observation_indices.size),
            "heldout_observations": int(split.heldout_observation_indices.size),
        },
        "heldout_state_by_task": {
            str(task_id): split.heldout_state_by_task[task_id]
            for task_id in c5a.EXPECTED_TASK_IDS
        },
        "canonical_group_order": [
            {"task_id": group.task_id, "initial_state_id": group.initial_state_id}
            for group in groups
        ],
        "groups": [
            {
                "canonical_group_index": group.canonical_index,
                "task_id": group.task_id,
                "initial_state_id": group.initial_state_id,
                "assignment": split.assignments[group.canonical_index],
                "sample_ids": list(group.sample_ids),
                "step_ids": list(group.step_ids),
                "inherited_progress_slots": list(c5a.PROGRESS_SLOTS),
            }
            for group in groups
        ],
    }


def _validate_historical_split(
    value: Mapping[str, Any],
    *,
    manifest_path: Path,
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> None:
    if set(value) != HISTORICAL_SPLIT_KEYS:
        raise C5BMMaterializationError("invalid historical C5-B split fields")
    _validate_recorded_manifest_identity(
        value.get("source_paired_manifest"),
        current_manifest=manifest_path,
        label="historical C5-B split source paired manifest",
    )
    source_c5a = value.get("source_c5a_split_manifest")
    if not isinstance(source_c5a, str) or not source_c5a.strip():
        raise C5BMMaterializationError(
            "historical C5-B split has invalid C5-A split provenance"
        )
    expected = _expected_split_content(manifest_path, groups, split)
    expected.pop("source_paired_manifest")
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise C5BMMaterializationError(
            "historical C5-B split differs from the recomputed frozen split"
        )


def _historical_primary_result(summary: Mapping[str, Any]) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise C5BMMaterializationError("historical C5-B results must be an object")
    primary = results.get("o2_p2__99")
    if not isinstance(primary, dict):
        raise C5BMMaterializationError("historical primary o2_p2__99 result is missing")
    return primary


def _validate_historical_summary(
    value: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> None:
    if set(value) != HISTORICAL_SUMMARY_KEYS:
        raise C5BMMaterializationError("invalid historical C5-B summary fields")
    expected_scalars = {
        "schema_version": c5b.SUMMARY_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "c5a_gate": "GO",
        "c5b_result": "PASS",
        "representation_stage_result": "PASS",
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise C5BMMaterializationError(
                f"invalid historical C5-B {key}: {value.get(key)!r}"
            )
    _validate_recorded_manifest_identity(
        value.get("source_paired_manifest"),
        current_manifest=manifest_path,
        label="historical C5-B summary source paired manifest",
    )
    if value.get("primary_pair") != {
        "name": "o2_p2",
        "openvla_node": "O2",
        "pi05_node": "P2",
        "gates_c5b": True,
    }:
        raise C5BMMaterializationError("historical C5-B primary pair is invalid")
    if value.get("control_pair") != {
        "name": "o1s_p1",
        "openvla_node": "O1-S",
        "pi05_node": "P1",
        "gates_c5b": False,
    }:
        raise C5BMMaterializationError("historical C5-B control pair is invalid")
    if value.get("cca") != {
        "method": "ordinary_linear_covariance_whitening_svd",
        "covariance_denominator": "n_minus_1",
        "eigendecomposition_api": "numpy.linalg.eigh",
        "svd_api": "numpy.linalg.svd_full_matrices_false",
        "regularization": "none",
        "min_valid_canonical_dims": 10,
    }:
        raise C5BMMaterializationError("historical C5-B CCA configuration is invalid")
    pca = value.get("pca")
    if not isinstance(pca, dict) or set(pca) != {"o2", "p2", "o1s", "p1"}:
        raise C5BMMaterializationError("historical C5-B PCA metadata is invalid")
    pca_fields = {
        "full_rank",
        "d_95",
        "d_99",
        "explained_variance_95",
        "explained_variance_99",
    }
    if any(
        not isinstance(pca[name], dict) or set(pca[name]) != pca_fields for name in pca
    ):
        raise C5BMMaterializationError("historical C5-B PCA metadata is invalid")
    primary = _historical_primary_result(value)
    results = value["results"]
    if set(results) != {"o2_p2__99", "o2_p2__95", "o1s_p1__99", "o1s_p1__95"}:
        raise C5BMMaterializationError(
            "historical C5-B result configurations are invalid"
        )
    for key in (
        "train_top5mean",
        "heldout_top1",
        "heldout_top5mean",
        "heldout_top10mean",
    ):
        metric = primary.get(key)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not np.isfinite(metric)
        ):
            raise C5BMMaterializationError(
                f"historical C5-B primary metric is invalid: {key}"
            )


def _load_historical_c5b(
    path: str | Path,
    *,
    manifest_path: Path,
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> HistoricalC5B:
    try:
        directory = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5BMMaterializationError(
            f"historical C5-B directory does not exist: {path}"
        ) from error
    if not directory.is_dir():
        raise C5BMMaterializationError(
            f"historical C5-B path is not a directory: {directory}"
        )
    paths = {
        name: _resolve_file(directory / name, f"historical C5-B {name}")
        for name in HISTORICAL_FILES
    }
    hashes = {name: _sha256_file(file_path) for name, file_path in paths.items()}
    split_manifest = _load_json(paths["split_manifest.json"], "historical C5-B split")
    alignment_summary = _load_json(
        paths["alignment_summary.json"], "historical C5-B summary"
    )
    _validate_historical_split(
        split_manifest,
        manifest_path=manifest_path,
        groups=groups,
        split=split,
    )
    _validate_historical_summary(alignment_summary, manifest_path=manifest_path)
    if alignment_summary.get("source_c5a_split_manifest") != split_manifest.get(
        "source_c5a_split_manifest"
    ):
        raise C5BMMaterializationError(
            "historical C5-B JSON files disagree on C5-A split provenance"
        )
    source_c5a_metric = alignment_summary.get("source_c5a_metric_summary")
    if not isinstance(source_c5a_metric, str) or not source_c5a_metric.strip():
        raise C5BMMaterializationError(
            "historical C5-B summary has invalid C5-A metric provenance"
        )
    try:
        with np.load(paths["null_alignment_metrics.npz"], allow_pickle=False) as data:
            c5b._validate_null_arrays({name: data[name] for name in data.files})
    except c5b.C5BValidationError as error:
        raise C5BMMaterializationError(str(error)) from error
    except Exception as error:
        raise C5BMMaterializationError(
            "failed to validate historical C5-B null archive"
        ) from error
    try:
        summary_text = paths["summary.md"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise C5BMMaterializationError(
            "failed to read historical C5-B summary.md"
        ) from error
    if not summary_text.startswith("# C5-B Explicit Shared-Space Alignment Summary"):
        raise C5BMMaterializationError("historical C5-B summary.md has invalid role")
    return HistoricalC5B(
        directory=directory,
        paths=paths,
        hashes=hashes,
        split_manifest=split_manifest,
        alignment_summary=alignment_summary,
    )


def _assert_historical_unchanged(historical: HistoricalC5B) -> None:
    current = {name: _sha256_file(path) for name, path in historical.paths.items()}
    if current != historical.hashes:
        changed = sorted(
            name
            for name in historical.hashes
            if current[name] != historical.hashes[name]
        )
        raise C5BMMaterializationError(
            f"historical C5-B artifacts changed during materialization: {changed}"
        )


def _canonicalize_signs(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    va = np.asarray(basis_a, dtype=np.float64)
    vb = np.asarray(basis_b, dtype=np.float64)
    canonical_a = np.ascontiguousarray(w_a, dtype=np.float64).copy()
    canonical_b = np.ascontiguousarray(w_b, dtype=np.float64).copy()
    if (
        va.ndim != 2
        or vb.ndim != 2
        or canonical_a.ndim != 2
        or canonical_b.ndim != 2
        or va.shape[1] != canonical_a.shape[0]
        or vb.shape[1] != canonical_b.shape[0]
        or canonical_a.shape[1] != canonical_b.shape[1]
    ):
        raise C5BMMaterializationError("sign canonicalization shapes are incompatible")
    anchors: list[dict[str, Any]] = []
    for component in range(canonical_a.shape[1]):
        q_a = va @ canonical_a[:, component]
        q_b = vb @ canonical_b[:, component]
        q = np.concatenate((q_a, q_b)).astype(np.float64, copy=False)
        if not np.all(np.isfinite(q)):
            raise C5BMMaterializationError(
                f"non-finite sign vector for canonical component {component}"
            )
        max_abs = float(np.max(np.abs(q)))
        if max_abs <= 0.0:
            raise C5BMMaterializationError(
                f"all-zero sign vector for canonical component {component}"
            )
        anchor_index = int(np.flatnonzero(np.abs(q) == max_abs)[0])
        anchor_value = float(q[anchor_index])
        multiplier = 1
        if anchor_value < 0.0:
            canonical_a[:, component] *= -1.0
            canonical_b[:, component] *= -1.0
            multiplier = -1
        anchors.append(
            {
                "component_index": component,
                "anchor_index": anchor_index,
                "anchor_side": "A" if anchor_index < q_a.size else "B",
                "anchor_coordinate": (
                    anchor_index if anchor_index < q_a.size else anchor_index - q_a.size
                ),
                "precanonicalization_anchor_value": anchor_value,
                "multiplier": multiplier,
            }
        )
    return canonical_a, canonical_b, tuple(anchors)


def _fit_mapping(
    records: Sequence[c5a.PairRecord],
    split: c5a.DatasetSplit,
) -> MappingFit:
    spec = PRIMARY_SPEC
    train_a_raw = c5b._load_node_rows(
        records,
        split.train_observation_indices,
        archive_attribute="openvla_feature_path",
        archive_key=spec.openvla_key,
        expected_shape=spec.openvla_shape,
    )
    pca_a, train_a = c5b._fit_pca(train_a_raw)
    del train_a_raw
    heldout_a_raw = c5b._load_node_rows(
        records,
        split.heldout_observation_indices,
        archive_attribute="openvla_feature_path",
        archive_key=spec.openvla_key,
        expected_shape=spec.openvla_shape,
    )
    heldout_a = c5b._transform_pca(pca_a, heldout_a_raw)
    del heldout_a_raw

    train_b_raw = c5b._load_node_rows(
        records,
        split.train_observation_indices,
        archive_attribute="pi05_feature_path",
        archive_key=spec.pi05_key,
        expected_shape=spec.pi05_shape,
    )
    pca_b, train_b = c5b._fit_pca(train_b_raw)
    del train_b_raw
    heldout_b_raw = c5b._load_node_rows(
        records,
        split.heldout_observation_indices,
        archive_attribute="pi05_feature_path",
        archive_key=spec.pi05_key,
        expected_shape=spec.pi05_shape,
    )
    heldout_b = c5b._transform_pca(pca_b, heldout_b_raw)
    del heldout_b_raw

    whitening = c5b._prepare_cca(train_a, train_b)
    fit = c5b._fit_cca(train_a, train_b, whitening)
    w_a, w_b, anchors = _canonicalize_signs(
        pca_a.basis_99,
        pca_b.basis_99,
        fit.w_a,
        fit.w_b,
    )
    canonical_fit = c5b.CCAFit(sigma=fit.sigma, w_a=w_a, w_b=w_b)
    heldout = c5b._heldout_metrics(heldout_a, heldout_b, canonical_fit)
    metrics = {
        "train_top5mean": float(np.mean(fit.sigma[:5], dtype=np.float64)),
        "heldout_top1": heldout["top1"],
        "heldout_top5mean": heldout["top5mean"],
        "heldout_top10mean": heldout["top10mean"],
    }
    arrays = {
        "mean_a": np.ascontiguousarray(pca_a.mean, dtype=np.float64),
        "mean_b": np.ascontiguousarray(pca_b.mean, dtype=np.float64),
        "basis_a": np.ascontiguousarray(pca_a.basis_99, dtype=np.float64),
        "basis_b": np.ascontiguousarray(pca_b.basis_99, dtype=np.float64),
        "whitening_a": np.ascontiguousarray(whitening.p_a, dtype=np.float64),
        "whitening_b": np.ascontiguousarray(whitening.p_b, dtype=np.float64),
        "w_a": w_a,
        "w_b": w_b,
        "sigma": np.ascontiguousarray(fit.sigma, dtype=np.float64),
    }
    return MappingFit(
        arrays=arrays,
        pca_a=pca_a,
        pca_b=pca_b,
        sign_anchors=anchors,
        metrics=metrics,
    )


def _validate_mapping_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != MAPPING_ARRAY_KEYS:
        raise C5BMMaterializationError("mapping arrays differ from the frozen schema")
    values = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in values.items():
        if value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise C5BMMaterializationError(f"invalid mapping array: {name}")
    if values["mean_a"].ndim != 1 or values["mean_b"].ndim != 1:
        raise C5BMMaterializationError("mapping means must be one-dimensional")
    for name in (
        "basis_a",
        "basis_b",
        "whitening_a",
        "whitening_b",
        "w_a",
        "w_b",
    ):
        if values[name].ndim != 2:
            raise C5BMMaterializationError(f"mapping matrix must be 2D: {name}")
    if values["basis_a"].shape[0] != values["mean_a"].size:
        raise C5BMMaterializationError("OpenVLA PCA basis/mean shape mismatch")
    if values["basis_b"].shape[0] != values["mean_b"].size:
        raise C5BMMaterializationError("pi0.5 PCA basis/mean shape mismatch")
    if values["basis_a"].shape[1] != values["w_a"].shape[0]:
        raise C5BMMaterializationError("OpenVLA PCA/CCA shape mismatch")
    if values["basis_b"].shape[1] != values["w_b"].shape[0]:
        raise C5BMMaterializationError("pi0.5 PCA/CCA shape mismatch")
    if values["whitening_a"].shape[0] != values["w_a"].shape[0]:
        raise C5BMMaterializationError("OpenVLA whitening/CCA shape mismatch")
    if values["whitening_b"].shape[0] != values["w_b"].shape[0]:
        raise C5BMMaterializationError("pi0.5 whitening/CCA shape mismatch")
    component_count = values["sigma"].size
    if (
        values["sigma"].ndim != 1
        or component_count < 10
        or values["w_a"].shape[1] != component_count
        or values["w_b"].shape[1] != component_count
        or values["whitening_a"].shape[1] < component_count
        or values["whitening_b"].shape[1] < component_count
    ):
        raise C5BMMaterializationError("canonical component shapes are invalid")
    sigma = values["sigma"]
    if np.any(sigma < 0.0) or np.any(np.diff(sigma) > 0.0):
        raise C5BMMaterializationError(
            "canonical correlations must preserve descending SVD order"
        )
    _validate_saved_sign_rule(values)


def _validate_saved_sign_rule(arrays: Mapping[str, np.ndarray]) -> None:
    basis_a = arrays["basis_a"]
    basis_b = arrays["basis_b"]
    w_a = arrays["w_a"]
    w_b = arrays["w_b"]
    for component in range(w_a.shape[1]):
        q = np.concatenate(
            (
                basis_a @ w_a[:, component],
                basis_b @ w_b[:, component],
            )
        )
        max_abs = float(np.max(np.abs(q)))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            raise C5BMMaterializationError(
                f"invalid saved sign vector for canonical component {component}"
            )
        anchor = int(np.flatnonzero(np.abs(q) == max_abs)[0])
        if q[anchor] <= 0.0:
            raise C5BMMaterializationError(
                f"saved canonical sign rule failed for component {component}"
            )


def _validate_historical_scalars(
    new_metrics: Mapping[str, float],
    historical_summary: Mapping[str, Any],
) -> dict[str, dict[str, float | bool]]:
    primary = _historical_primary_result(historical_summary)
    checks: dict[str, dict[str, float | bool]] = {}
    for name in (
        "train_top5mean",
        "heldout_top1",
        "heldout_top5mean",
        "heldout_top10mean",
    ):
        new = float(new_metrics[name])
        historical = float(primary[name])
        difference = abs(new - historical)
        passed = bool(difference <= SCALAR_TOLERANCE)
        checks[name] = {
            "new": new,
            "historical": historical,
            "absolute_difference": difference,
            "passed": passed,
        }
        if not passed:
            raise C5BMMaterializationError(
                f"historical scalar mismatch for {name}: "
                f"difference={difference}, tolerance={SCALAR_TOLERANCE}"
            )
    return checks


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_hash": _array_hash(value),
        }
        for name, value in sorted(arrays.items())
    }


def _source_feature_provenance(
    records: Sequence[c5a.PairRecord],
) -> list[dict[str, str]]:
    return [
        {
            "sample_id": record.sample_id,
            "source_image_hash": record.source_image_hash,
            "openvla_feature_path": str(record.openvla_feature_path),
            "openvla_feature_hash": _sha256_file(record.openvla_feature_path),
            "pi05_feature_path": str(record.pi05_feature_path),
            "pi05_feature_hash": _sha256_file(record.pi05_feature_path),
        }
        for record in records
    ]


def _assert_source_features_unchanged(
    source_features: Sequence[Mapping[str, str]],
) -> None:
    changed: list[str] = []
    for feature in source_features:
        if (
            _sha256_file(Path(feature["openvla_feature_path"]))
            != feature["openvla_feature_hash"]
            or _sha256_file(Path(feature["pi05_feature_path"]))
            != feature["pi05_feature_hash"]
        ):
            changed.append(feature["sample_id"])
    if changed:
        raise C5BMMaterializationError(
            f"source feature artifacts changed during materialization: {changed}"
        )


def _repository_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise C5BMMaterializationError(
            "failed to determine repository commit"
        ) from error
    commit = result.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise C5BMMaterializationError(f"invalid repository commit: {commit!r}")
    return commit


def _numpy_build_configuration() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.__config__.show()
    return buffer.getvalue().strip()


def _build_metadata(
    *,
    manifest_path: Path,
    manifest_hash: str,
    historical: HistoricalC5B,
    source_features: list[dict[str, str]],
    fit: MappingFit,
) -> dict[str, Any]:
    arrays = _array_metadata(fit.arrays)
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "mapping_schema_version": MAPPING_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "materialization_id": MATERIALIZATION_ID,
        "fit_configuration": dict(FIT_CONFIGURATION),
        "mapping_identity": {
            "openvla_retained_pca_dimensions": fit.pca_a.d_99,
            "pi05_retained_pca_dimensions": fit.pca_b.d_99,
            "canonical_component_count": int(fit.arrays["sigma"].size),
            "canonical_order": "descending_numpy_svd_order",
            "paired_component_arrays": ["w_a", "w_b", "sigma"],
            "sign_rule_version": "c5bm_concat_native_readout_max_abs_smallest_index_v1",
            "sign_anchors": list(fit.sign_anchors),
        },
        "arrays": arrays,
        "provenance": {
            "source_paired_manifest": {
                "runtime_path": str(manifest_path),
                "content_hash": manifest_hash,
            },
            "source_split_manifest": {
                "runtime_path": str(historical.paths["split_manifest.json"]),
                "content_hash": historical.hashes["split_manifest.json"],
            },
            "historical_c5b_files": {
                name: {
                    "runtime_path": str(historical.paths[name]),
                    "content_hash": historical.hashes[name],
                }
                for name in sorted(HISTORICAL_FILES)
            },
            "historical_alignment_summary_hash": historical.hashes[
                "alignment_summary.json"
            ],
            "source_feature_artifacts": source_features,
            "source_feature_validation": {
                "validated_pair_count": len(source_features),
                "result": "PASS",
            },
            "repository_commit": _repository_commit(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numpy_blas_lapack_configuration": _numpy_build_configuration(),
            "platform": platform.platform(),
        },
    }


def _build_validation(
    *,
    scalar_checks: Mapping[str, Any],
    historical: HistoricalC5B,
    fit: MappingFit,
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "c5bm_result": "PASS",
        "materialization_id": MATERIALIZATION_ID,
        "exact_match_checks": {
            "representation_pair": True,
            "pca_cutoff": True,
            "split_identity": True,
            "paired_manifest_identity": True,
            "observation_order": True,
            "token_row_order": True,
            "openvla_retained_pca_dimensions": fit.pca_a.d_99,
            "pi05_retained_pca_dimensions": fit.pca_b.d_99,
        },
        "historical_scalar_tolerance": SCALAR_TOLERANCE,
        "historical_scalar_checks": dict(scalar_checks),
        "historical_file_hashes": dict(sorted(historical.hashes.items())),
        "historical_files_unchanged": True,
        "complete_source_feature_validation": "PASS",
        "transactional_publication_validation": "PASS",
    }


def _build_summary(metadata: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    identity = metadata["mapping_identity"]
    lines = [
        "# C5-BM Authoritative Mapping Materialization",
        "",
        "C5-BM result: **PASS**",
        "",
        f"Materialization ID: `{metadata['materialization_id']}`",
        "",
        "## Frozen mapping",
        "",
        "- Representation pair: O2 ↔ P2",
        "- PCA cutoff: 99%",
        "- Fit: true, unpermuted TRAIN pairing",
        f"- O2 retained dimensions: {identity['openvla_retained_pca_dimensions']}",
        f"- P2 retained dimensions: {identity['pi05_retained_pca_dimensions']}",
        f"- Canonical components: {identity['canonical_component_count']}",
        "",
        "## Validation",
        "",
    ]
    for name, check in validation["historical_scalar_checks"].items():
        lines.append(f"- {name}: difference={check['absolute_difference']:.12g}, PASS")
    lines.extend(
        [
            "",
            "This is a new authoritative reusable mapping, not a recovery of the",
            "historical unsaved in-memory C5-B matrices.",
            "",
            "This artifact does not define a native-space intervention vector and",
            "does not establish policy/action relevance or transferability.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_metadata_document(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(metadata) != METADATA_KEYS:
        raise C5BMMaterializationError("staged C5-BM metadata fields are invalid")
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise C5BMMaterializationError("invalid staged C5-BM metadata schema")
    if metadata.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION:
        raise C5BMMaterializationError("invalid staged C5-BM mapping schema")
    if metadata.get("run_status") != "COMPLETED":
        raise C5BMMaterializationError("staged C5-BM metadata is not complete")
    if metadata.get("materialization_id") != MATERIALIZATION_ID:
        raise C5BMMaterializationError("invalid staged C5-BM materialization ID")
    if metadata.get("fit_configuration") != FIT_CONFIGURATION:
        raise C5BMMaterializationError("invalid staged C5-BM fit configuration")
    if metadata.get("arrays") != _array_metadata(arrays):
        raise C5BMMaterializationError("staged mapping integrity metadata mismatch")

    identity = metadata.get("mapping_identity")
    identity_keys = {
        "openvla_retained_pca_dimensions",
        "pi05_retained_pca_dimensions",
        "canonical_component_count",
        "canonical_order",
        "paired_component_arrays",
        "sign_rule_version",
        "sign_anchors",
    }
    if not isinstance(identity, dict) or set(identity) != identity_keys:
        raise C5BMMaterializationError("staged mapping identity is invalid")
    expected_identity = {
        "openvla_retained_pca_dimensions": arrays["basis_a"].shape[1],
        "pi05_retained_pca_dimensions": arrays["basis_b"].shape[1],
        "canonical_component_count": arrays["sigma"].size,
        "canonical_order": "descending_numpy_svd_order",
        "paired_component_arrays": ["w_a", "w_b", "sigma"],
        "sign_rule_version": "c5bm_concat_native_readout_max_abs_smallest_index_v1",
    }
    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise C5BMMaterializationError(
                f"invalid staged mapping identity field: {name}"
            )
    anchors = identity.get("sign_anchors")
    anchor_fields = {
        "component_index",
        "anchor_index",
        "anchor_side",
        "anchor_coordinate",
        "precanonicalization_anchor_value",
        "multiplier",
    }
    if not isinstance(anchors, list) or len(anchors) != arrays["sigma"].size:
        raise C5BMMaterializationError("staged sign-anchor metadata is invalid")
    native_a = arrays["basis_a"].shape[0]
    native_total = native_a + arrays["basis_b"].shape[0]
    for component, anchor in enumerate(anchors):
        if not isinstance(anchor, dict) or set(anchor) != anchor_fields:
            raise C5BMMaterializationError("staged sign-anchor fields are invalid")
        index = anchor.get("anchor_index")
        value = anchor.get("precanonicalization_anchor_value")
        multiplier = anchor.get("multiplier")
        if (
            anchor.get("component_index") != component
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < native_total
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value == 0.0
            or multiplier not in (-1, 1)
            or multiplier != (1 if value > 0.0 else -1)
        ):
            raise C5BMMaterializationError("staged sign-anchor value is invalid")
        expected_side = "A" if index < native_a else "B"
        expected_coordinate = index if index < native_a else index - native_a
        if (
            anchor.get("anchor_side") != expected_side
            or anchor.get("anchor_coordinate") != expected_coordinate
        ):
            raise C5BMMaterializationError("staged sign-anchor location is invalid")

    provenance = metadata.get("provenance")
    provenance_keys = {
        "source_paired_manifest",
        "source_split_manifest",
        "historical_c5b_files",
        "historical_alignment_summary_hash",
        "source_feature_artifacts",
        "source_feature_validation",
        "repository_commit",
        "python_version",
        "numpy_version",
        "numpy_blas_lapack_configuration",
        "platform",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_keys:
        raise C5BMMaterializationError("staged C5-BM provenance fields are invalid")
    for name in ("source_paired_manifest", "source_split_manifest"):
        identity_value = provenance.get(name)
        if (
            not isinstance(identity_value, dict)
            or set(identity_value) != {"runtime_path", "content_hash"}
            or not isinstance(identity_value.get("runtime_path"), str)
            or not Path(identity_value["runtime_path"]).is_absolute()
            or not _is_sha256(identity_value.get("content_hash"))
        ):
            raise C5BMMaterializationError(
                f"staged C5-BM provenance identity is invalid: {name}"
            )
    historical_files = provenance.get("historical_c5b_files")
    if (
        not isinstance(historical_files, dict)
        or set(historical_files) != HISTORICAL_FILES
    ):
        raise C5BMMaterializationError("staged historical provenance is invalid")
    for name, identity_value in historical_files.items():
        if (
            not isinstance(identity_value, dict)
            or set(identity_value) != {"runtime_path", "content_hash"}
            or not isinstance(identity_value.get("runtime_path"), str)
            or not Path(identity_value["runtime_path"]).is_absolute()
            or not _is_sha256(identity_value.get("content_hash"))
        ):
            raise C5BMMaterializationError(
                f"staged historical provenance is invalid: {name}"
            )
    if (
        provenance.get("historical_alignment_summary_hash")
        != historical_files["alignment_summary.json"]["content_hash"]
    ):
        raise C5BMMaterializationError(
            "staged historical alignment-summary identity is inconsistent"
        )
    feature_validation = provenance.get("source_feature_validation")
    features = provenance.get("source_feature_artifacts")
    if feature_validation != {
        "validated_pair_count": EXPECTED_PAIR_COUNT,
        "result": "PASS",
    } or not isinstance(features, list):
        raise C5BMMaterializationError("staged source-feature validation is invalid")
    feature_fields = {
        "sample_id",
        "source_image_hash",
        "openvla_feature_path",
        "openvla_feature_hash",
        "pi05_feature_path",
        "pi05_feature_hash",
    }
    if len(features) != EXPECTED_PAIR_COUNT:
        raise C5BMMaterializationError("staged source-feature count is invalid")
    sample_ids: set[str] = set()
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != feature_fields:
            raise C5BMMaterializationError("staged source-feature fields are invalid")
        sample_id = feature.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not _is_sha256(feature.get("source_image_hash"))
            or not _is_sha256(feature.get("openvla_feature_hash"))
            or not _is_sha256(feature.get("pi05_feature_hash"))
            or not isinstance(feature.get("openvla_feature_path"), str)
            or not isinstance(feature.get("pi05_feature_path"), str)
            or not Path(feature["openvla_feature_path"]).is_absolute()
            or not Path(feature["pi05_feature_path"]).is_absolute()
        ):
            raise C5BMMaterializationError(
                "staged source-feature provenance is invalid"
            )
        sample_ids.add(sample_id)
    commit = provenance.get("repository_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise C5BMMaterializationError("staged repository commit is invalid")


def _validate_validation_document(
    validation: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(validation) != VALIDATION_KEYS:
        raise C5BMMaterializationError("staged C5-BM validation fields are invalid")
    if validation.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise C5BMMaterializationError("invalid staged C5-BM validation schema")
    expected = {
        "run_status": "COMPLETED",
        "c5bm_result": "PASS",
        "materialization_id": MATERIALIZATION_ID,
        "historical_scalar_tolerance": SCALAR_TOLERANCE,
        "historical_files_unchanged": True,
        "complete_source_feature_validation": "PASS",
        "transactional_publication_validation": "PASS",
    }
    for name, expected_value in expected.items():
        if validation.get(name) != expected_value:
            raise C5BMMaterializationError(
                f"invalid staged C5-BM validation field: {name}"
            )
    exact = validation.get("exact_match_checks")
    if (
        not isinstance(exact, dict)
        or set(exact) != EXACT_MATCH_KEYS
        or any(
            value is not True
            for name, value in exact.items()
            if name
            not in {
                "openvla_retained_pca_dimensions",
                "pi05_retained_pca_dimensions",
            }
        )
    ):
        raise C5BMMaterializationError("staged exact-match checks are invalid")
    if (
        exact["openvla_retained_pca_dimensions"] != arrays["basis_a"].shape[1]
        or exact["pi05_retained_pca_dimensions"] != arrays["basis_b"].shape[1]
    ):
        raise C5BMMaterializationError("staged retained PCA dimensions are invalid")
    scalar_checks = validation.get("historical_scalar_checks")
    if not isinstance(scalar_checks, dict) or set(scalar_checks) != {
        "train_top5mean",
        "heldout_top1",
        "heldout_top5mean",
        "heldout_top10mean",
    }:
        raise C5BMMaterializationError("staged historical scalar checks are invalid")
    if any(
        not isinstance(check, dict)
        or set(check) != {"new", "historical", "absolute_difference", "passed"}
        or check.get("passed") is not True
        or not isinstance(check.get("new"), (int, float))
        or isinstance(check.get("new"), bool)
        or not np.isfinite(check["new"])
        or not isinstance(check.get("historical"), (int, float))
        or isinstance(check.get("historical"), bool)
        or not np.isfinite(check["historical"])
        or not isinstance(check.get("absolute_difference"), (int, float))
        or isinstance(check.get("absolute_difference"), bool)
        or not np.isfinite(check["absolute_difference"])
        or check["absolute_difference"] > SCALAR_TOLERANCE
        for check in scalar_checks.values()
    ):
        raise C5BMMaterializationError("staged historical scalar validation failed")
    hashes = validation.get("historical_file_hashes")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != HISTORICAL_FILES
        or not all(_is_sha256(value) for value in hashes.values())
    ):
        raise C5BMMaterializationError("staged historical file hashes are invalid")


def _validate_published_artifact(path: Path) -> None:
    try:
        entries = {candidate.name: candidate for candidate in path.iterdir()}
    except OSError as error:
        raise C5BMMaterializationError(
            f"failed to inspect staged C5-BM artifact: {path}"
        ) from error
    if set(entries) != OUTPUT_FILES or not all(
        item.is_file() for item in entries.values()
    ):
        raise C5BMMaterializationError(
            "C5-BM artifact must contain exactly the frozen four-file set"
        )
    metadata = _load_json(entries["metadata.json"], "staged C5-BM metadata")
    validation = _load_json(entries["validation.json"], "staged C5-BM validation")
    try:
        with np.load(entries["mapping.npz"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as error:
        raise C5BMMaterializationError(
            "failed to load staged mapping archive"
        ) from error
    _validate_mapping_arrays(arrays)
    _validate_metadata_document(metadata, arrays)
    _validate_validation_document(validation, arrays)
    metadata_historical_hashes = {
        name: value["content_hash"]
        for name, value in metadata["provenance"]["historical_c5b_files"].items()
    }
    if validation["historical_file_hashes"] != metadata_historical_hashes:
        raise C5BMMaterializationError(
            "staged historical identities disagree across metadata and validation"
        )
    try:
        summary = entries["summary.md"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise C5BMMaterializationError("failed to read staged summary.md") from error
    if not summary.startswith("# C5-BM Authoritative Mapping Materialization"):
        raise C5BMMaterializationError("staged summary.md has invalid role")


def _remove_staging_directory(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    except (FileNotFoundError, OSError):
        return


def _publish(
    output_dir: Path,
    *,
    fit: MappingFit,
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: str,
    historical: HistoricalC5B,
) -> None:
    _validate_mapping_arrays(fit.arrays)
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                dir=output_dir.parent,
                prefix=f".{output_dir.name}.",
            )
        )
    except OSError as error:
        raise C5BMMaterializationError(
            f"failed to create C5-BM staging directory near {output_dir}"
        ) from error
    removed_empty_target = False
    published = False
    try:
        np.savez_compressed(staging / "mapping.npz", **fit.arrays)
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "validation.json", validation)
        (staging / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
        _validate_published_artifact(staging)
        _assert_historical_unchanged(historical)
        if output_dir.exists():
            if not output_dir.is_dir() or tuple(output_dir.iterdir()):
                raise C5BMMaterializationError(
                    f"C5-BM output directory became non-empty: {output_dir}"
                )
            output_dir.rmdir()
            removed_empty_target = True
        staging.rename(output_dir)
        published = True
        _validate_published_artifact(output_dir)
    except C5BMMaterializationError:
        if published:
            _remove_staging_directory(output_dir)
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise
    except Exception as error:
        if published:
            _remove_staging_directory(output_dir)
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise C5BMMaterializationError(
            f"failed to publish complete C5-BM artifact: {output_dir}"
        ) from error
    finally:
        _remove_staging_directory(staging)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _validate_output_admission(args.output_dir)
    try:
        manifest_path, records, groups, split = c5b._load_dataset_structure(
            args.paired_manifest
        )
        c5b._validate_feature_archives(records)
    except c5b.C5BValidationError as error:
        raise C5BMMaterializationError(str(error)) from error
    if len(records) != EXPECTED_PAIR_COUNT:
        raise C5BMMaterializationError(
            f"C5-BM requires exactly {EXPECTED_PAIR_COUNT} paired records"
        )
    historical = _load_historical_c5b(
        args.c5b_output_dir,
        manifest_path=manifest_path,
        groups=groups,
        split=split,
    )
    manifest_hash = _sha256_file(manifest_path)
    source_features = _source_feature_provenance(records)
    try:
        fit = _fit_mapping(records, split)
    except c5b.C5BValidationError as error:
        raise C5BMMaterializationError(str(error)) from error
    _validate_mapping_arrays(fit.arrays)
    historical_pca = historical.alignment_summary["pca"]
    if fit.pca_a.d_99 != historical_pca["o2"].get(
        "d_99"
    ) or fit.pca_b.d_99 != historical_pca["p2"].get("d_99"):
        raise C5BMMaterializationError(
            "retained PCA dimensions differ from historical C5-B"
        )
    scalar_checks = _validate_historical_scalars(
        fit.metrics,
        historical.alignment_summary,
    )
    _assert_source_features_unchanged(source_features)
    _assert_historical_unchanged(historical)
    metadata = _build_metadata(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        historical=historical,
        source_features=source_features,
        fit=fit,
    )
    validation = _build_validation(
        scalar_checks=scalar_checks,
        historical=historical,
        fit=fit,
    )
    summary = _build_summary(metadata, validation)
    _publish(
        output_dir,
        fit=fit,
        metadata=metadata,
        validation=validation,
        summary=summary,
        historical=historical,
    )
    return {
        "status": "C5-BM Authoritative Mapping Materialization — COMPLETED",
        "c5bm_result": "PASS",
        "materialization_id": MATERIALIZATION_ID,
        "num_pairs": len(records),
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
