from __future__ import annotations

import argparse
import json
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


C5A_FILES = {
    "split_manifest.json",
    "metric_summary.json",
    "null_metrics.npz",
    "summary.md",
}
C5B_FILES = {
    "split_manifest.json",
    "alignment_summary.json",
    "null_alignment_metrics.npz",
    "summary.md",
}
SPLIT_SCHEMA_VERSION = "c5b_split_manifest_v1"
SUMMARY_SCHEMA_VERSION = "c5b_alignment_summary_v1"
NULL_REPEATS = 200
ROOT_RNG_SEED = 17
PCA_CUTOFFS = {"95": 0.95, "99": 0.99}
HELDOUT_NULL_LIMITATION = (
    "HELD-OUT contains one group per task; the fixed-point-free HELD-OUT "
    "derangement is therefore a broad cross-group mismatch null, not a "
    "task-conditioned state-level mismatch null."
)
INTERPRETATION_BOUNDARY = (
    "Statistical significance relative to the frozen null does not by itself "
    "imply strong practical alignment; C5 representation-stage PASS does not "
    "establish policy relevance, action relevance, adversarial transferability, "
    "or attack effectiveness."
)
EXPECTED_C5A_SPLIT_KEYS = {
    "schema_version",
    "source_paired_manifest",
    "split_rule_id",
    "canonical_group_order",
    "counts",
    "heldout_state_by_task",
    "groups",
}
EXPECTED_C5A_METRIC_KEYS = {
    "schema_version",
    "run_status",
    "source_paired_manifest",
    "primary_pair",
    "control_pair",
    "metric_conventions",
    "null",
    "splits",
    "c5a_gate",
    "heldout_null_limitation",
}


class C5BValidationError(RuntimeError):
    """Raised when the frozen C5-B contract cannot be executed exactly."""


@dataclass(frozen=True)
class PairSpec:
    name: str
    openvla_node: str
    pi05_node: str
    openvla_key: str
    pi05_key: str
    openvla_shape: tuple[int, int]
    pi05_shape: tuple[int, int]
    openvla_pca_name: str
    pi05_pca_name: str
    gates_c5b: bool


PAIR_SPECS = (
    PairSpec(
        name="o2_p2",
        openvla_node="O2",
        pi05_node="P2",
        openvla_key="o2_projected",
        pi05_key="p2_projected",
        openvla_shape=(256, 4096),
        pi05_shape=(256, 2048),
        openvla_pca_name="o2",
        pi05_pca_name="p2",
        gates_c5b=True,
    ),
    PairSpec(
        name="o1s_p1",
        openvla_node="O1-S",
        pi05_node="P1",
        openvla_key="o1_siglip",
        pi05_key="p1_siglip",
        openvla_shape=(256, 1152),
        pi05_shape=(256, 1152),
        openvla_pca_name="o1s",
        pi05_pca_name="p1",
        gates_c5b=False,
    ),
)


@dataclass(frozen=True)
class C5AProvenance:
    split_path: Path
    metric_path: Path
    split_manifest: dict[str, Any]
    metric_summary: dict[str, Any]


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray
    basis_99: np.ndarray
    full_rank: int
    d_95: int
    d_99: int
    explained_variance_95: float
    explained_variance_99: float

    def metadata(self) -> dict[str, int | float]:
        return {
            "full_rank": self.full_rank,
            "d_95": self.d_95,
            "d_99": self.d_99,
            "explained_variance_95": self.explained_variance_95,
            "explained_variance_99": self.explained_variance_99,
        }


@dataclass(frozen=True)
class CCAWhitening:
    p_a: np.ndarray
    p_b: np.ndarray
    denominator: float


@dataclass(frozen=True)
class CCAFit:
    sigma: np.ndarray
    w_a: np.ndarray
    w_b: np.ndarray


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Pilot v0.2 C5-B shared-space analysis."
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        required=True,
        help="Completed formal C4 paired_features_v1 manifest.",
    )
    parser.add_argument(
        "--c5a-output-dir",
        type=Path,
        required=True,
        help="Completed formal four-file C5-A output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Absent or empty destination for the four formal C5-B artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5BValidationError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise C5BValidationError(f"{label} is not a regular file: {resolved}")
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise C5BValidationError(f"failed to load {label}: {path}") from error
    if not isinstance(value, dict):
        raise C5BValidationError(f"{label} must encode a JSON object")
    return value


def _validate_output_admission(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve(strict=False)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise C5BValidationError(
                f"C5-B output path is not a directory: {output_dir}"
            )
        try:
            entries = tuple(output_dir.iterdir())
        except OSError as error:
            raise C5BValidationError(
                f"failed to inspect C5-B output directory: {output_dir}"
            ) from error
        if entries:
            raise C5BValidationError(
                f"C5-B output directory must be empty: {output_dir}"
            )
    return output_dir


def _resolve_recorded_manifest(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise C5BValidationError(f"invalid {label}: {value!r}")
    return _resolve_file(Path(value), label)


def _expected_c5a_split_manifest(
    manifest_path: Path,
    records: Sequence[c5a.PairRecord],
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> dict[str, Any]:
    dataset = c5a.LoadedDataset(
        manifest_path=manifest_path,
        records=tuple(records),
        groups=tuple(groups),
        features={},
    )
    return c5a._build_split_manifest(dataset, split)


def _load_c5a_provenance(
    path: str | Path,
    *,
    paired_manifest_path: Path,
    records: Sequence[c5a.PairRecord],
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> C5AProvenance:
    try:
        output_dir = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5BValidationError(
            f"C5-A output directory does not exist: {path}"
        ) from error
    if not output_dir.is_dir():
        raise C5BValidationError(f"C5-A output path is not a directory: {output_dir}")
    try:
        entries = {candidate.name: candidate for candidate in output_dir.iterdir()}
    except OSError as error:
        raise C5BValidationError(
            f"failed to inspect C5-A output directory: {output_dir}"
        ) from error
    if set(entries) != C5A_FILES or not all(
        candidate.is_file() for candidate in entries.values()
    ):
        raise C5BValidationError(
            "C5-A output directory must contain exactly the formal four-file set"
        )

    split_path = entries["split_manifest.json"].resolve(strict=True)
    metric_path = entries["metric_summary.json"].resolve(strict=True)
    split_manifest = _load_json(split_path, "C5-A split manifest")
    metric_summary = _load_json(metric_path, "C5-A metric summary")
    if set(split_manifest) != EXPECTED_C5A_SPLIT_KEYS:
        raise C5BValidationError("invalid C5-A split-manifest fields")
    if set(metric_summary) != EXPECTED_C5A_METRIC_KEYS:
        raise C5BValidationError("invalid C5-A metric-summary fields")
    if split_manifest.get("schema_version") != c5a.SPLIT_SCHEMA_VERSION:
        raise C5BValidationError("invalid C5-A split schema")
    if metric_summary.get("schema_version") != c5a.METRIC_SCHEMA_VERSION:
        raise C5BValidationError("invalid C5-A metric schema")
    if metric_summary.get("run_status") != "COMPLETED":
        raise C5BValidationError("formal C5-A run_status must be COMPLETED")
    if metric_summary.get("c5a_gate") != "GO":
        raise C5BValidationError("formal C5-A c5a_gate must be GO")

    for value, label in (
        (split_manifest.get("source_paired_manifest"), "C5-A split source manifest"),
        (metric_summary.get("source_paired_manifest"), "C5-A metric source manifest"),
    ):
        if _resolve_recorded_manifest(value, label) != paired_manifest_path:
            raise C5BValidationError(
                f"{label} does not resolve to the current C4 paired manifest"
            )

    expected_split = _expected_c5a_split_manifest(
        paired_manifest_path,
        records,
        groups,
        split,
    )
    if split_manifest != expected_split:
        raise C5BValidationError(
            "C5-A serialized split differs from the recomputed frozen split"
        )

    splits = metric_summary.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "heldout"}:
        raise C5BValidationError("invalid C5-A metric split results")
    expected_counts = {"train": (40, 160), "heldout": (10, 40)}
    for name, (groups_count, observations_count) in expected_counts.items():
        value = splits.get(name)
        if (
            not isinstance(value, dict)
            or value.get("pass") is not True
            or value.get("num_groups") != groups_count
            or value.get("num_observations") != observations_count
        ):
            raise C5BValidationError(f"invalid C5-A {name} metric result")

    return C5AProvenance(
        split_path=split_path,
        metric_path=metric_path,
        split_manifest=split_manifest,
        metric_summary=metric_summary,
    )


def _load_dataset_structure(
    path: str | Path,
) -> tuple[
    Path,
    tuple[c5a.PairRecord, ...],
    tuple[c5a.GroupRecord, ...],
    c5a.DatasetSplit,
]:
    try:
        manifest_path, records = c5a._load_paired_records(path)
        groups = c5a._reconstruct_groups(records)
        split = c5a._build_split(groups)
    except c5a.C5AValidationError as error:
        raise C5BValidationError(str(error)) from error
    return manifest_path, records, groups, split


def _validate_feature_archives(records: Sequence[c5a.PairRecord]) -> None:
    for record in records:
        try:
            c5a._validate_archive(
                record.openvla_feature_path,
                record=record,
                spec=c5a.c2.SPEC,
            )
            c5a._validate_archive(
                record.pi05_feature_path,
                record=record,
                spec=c5a.c3.SPEC,
            )
        except c5a.C5AValidationError as error:
            raise C5BValidationError(str(error)) from error


def _load_node_rows(
    records: Sequence[c5a.PairRecord],
    record_indices: np.ndarray,
    *,
    archive_attribute: str,
    archive_key: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    value = np.empty((record_indices.size, *expected_shape), dtype=np.float64)
    for output_index, record_index in enumerate(record_indices):
        record = records[int(record_index)]
        archive_path = getattr(record, archive_attribute)
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                feature = archive[archive_key]
                if feature.shape != expected_shape or feature.dtype != np.float32:
                    raise C5BValidationError(
                        f"invalid {archive_key} array for {record.sample_id}"
                    )
                if not np.all(np.isfinite(feature)):
                    raise C5BValidationError(
                        f"non-finite {archive_key} array for {record.sample_id}"
                    )
                value[output_index] = feature
        except C5BValidationError:
            raise
        except Exception as error:
            raise C5BValidationError(
                f"failed to load {archive_key} for {record.sample_id}"
            ) from error
    return value.reshape(record_indices.size * expected_shape[0], expected_shape[1])


def _fit_pca(train_rows: np.ndarray) -> tuple[PCAModel, np.ndarray]:
    value = np.asarray(train_rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1:
        raise C5BValidationError(f"PCA requires a non-empty 2D matrix: {value.shape}")
    if not np.all(np.isfinite(value)):
        raise C5BValidationError("PCA input contains non-finite values")
    mean = np.mean(value, axis=0, dtype=np.float64)
    centered = value - mean
    try:
        u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise C5BValidationError("PCA SVD did not converge") from error
    if not (
        np.all(np.isfinite(u))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(vh))
    ):
        raise C5BValidationError("PCA SVD produced non-finite values")
    variance = np.square(singular_values, dtype=np.float64)
    total_variance = float(np.sum(variance, dtype=np.float64))
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise C5BValidationError("PCA total variance must be positive and finite")
    cumulative = np.cumsum(variance, dtype=np.float64) / total_variance
    d_95 = int(np.searchsorted(cumulative, PCA_CUTOFFS["95"], side="left") + 1)
    d_99 = int(np.searchsorted(cumulative, PCA_CUTOFFS["99"], side="left") + 1)
    singular_max = float(np.max(singular_values))
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_max
    full_rank = int(np.count_nonzero(singular_values > tolerance))
    if d_95 > full_rank or d_99 > full_rank:
        raise C5BValidationError(
            "PCA cutoff requires a numerically invalid singular component"
        )
    if d_95 < 10 or d_99 < 10:
        raise C5BValidationError("PCA retained dimension must be at least 10")
    basis_99 = np.ascontiguousarray(vh[:d_99].T, dtype=np.float64)
    train_scores_99 = np.ascontiguousarray(
        u[:, :d_99] * singular_values[:d_99],
        dtype=np.float64,
    )
    model = PCAModel(
        mean=np.asarray(mean, dtype=np.float64),
        basis_99=basis_99,
        full_rank=full_rank,
        d_95=d_95,
        d_99=d_99,
        explained_variance_95=float(cumulative[d_95 - 1]),
        explained_variance_99=float(cumulative[d_99 - 1]),
    )
    return model, train_scores_99


def _transform_pca(model: PCAModel, rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != model.mean.size:
        raise C5BValidationError("PCA transform input has an invalid shape")
    transformed = (value - model.mean) @ model.basis_99
    if transformed.dtype != np.float64 or not np.all(np.isfinite(transformed)):
        raise C5BValidationError("PCA transform produced invalid values")
    return transformed


def _whitening_matrix(value: np.ndarray) -> np.ndarray:
    dimension = value.shape[1]
    covariance = (value.T @ value) / (value.shape[0] - 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as error:
        raise C5BValidationError("CCA covariance eigendecomposition failed") from error
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise C5BValidationError("CCA covariance eigendecomposition is non-finite")
    eigenvalue_max = float(eigenvalues[0])
    tolerance = max(dimension, 1) * np.finfo(np.float64).eps * eigenvalue_max
    valid = eigenvalues > tolerance
    if np.count_nonzero(valid) < 10:
        raise C5BValidationError("CCA requires at least 10 valid covariance directions")
    return np.asarray(
        eigenvectors[:, valid] * np.power(eigenvalues[valid], -0.5),
        dtype=np.float64,
    )


def _prepare_cca(z_a: np.ndarray, z_b: np.ndarray) -> CCAWhitening:
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise C5BValidationError("CCA TRAIN matrices have incompatible shapes")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise C5BValidationError("CCA TRAIN matrices contain non-finite values")
    return CCAWhitening(
        p_a=_whitening_matrix(a),
        p_b=_whitening_matrix(b),
        denominator=float(a.shape[0] - 1),
    )


def _validate_permutation(permutation: np.ndarray, n_groups: int) -> np.ndarray:
    value = np.asarray(permutation)
    if value.shape != (n_groups,) or value.dtype != np.int64:
        raise C5BValidationError("group permutation has an invalid shape or dtype")
    identity = np.arange(n_groups, dtype=np.int64)
    if not np.array_equal(np.sort(value), identity) or np.any(value == identity):
        raise C5BValidationError("group permutation must be fixed-point-free")
    return value


def _permuted_cross_covariance(
    z_a: np.ndarray,
    z_b: np.ndarray,
    permutation: np.ndarray,
    *,
    rows_per_group: int,
) -> np.ndarray:
    if rows_per_group <= 0 or z_a.shape[0] != z_b.shape[0]:
        raise C5BValidationError("invalid group-block cross-covariance inputs")
    if z_a.shape[0] % rows_per_group != 0:
        raise C5BValidationError("CCA rows do not form complete group blocks")
    n_groups = z_a.shape[0] // rows_per_group
    perm = _validate_permutation(permutation, n_groups)
    cross = np.zeros((z_a.shape[1], z_b.shape[1]), dtype=np.float64)
    for group_a, group_b in enumerate(perm):
        a_start = group_a * rows_per_group
        b_start = int(group_b) * rows_per_group
        cross += (
            z_a[a_start : a_start + rows_per_group].T
            @ z_b[b_start : b_start + rows_per_group]
        )
    return cross / (z_a.shape[0] - 1)


def _fit_cca(
    z_a: np.ndarray,
    z_b: np.ndarray,
    whitening: CCAWhitening,
    *,
    permutation: np.ndarray | None = None,
    rows_per_group: int | None = None,
) -> CCAFit:
    if permutation is None:
        cross_covariance = (z_a.T @ z_b) / whitening.denominator
    else:
        if rows_per_group is None:
            raise C5BValidationError("null CCA requires rows_per_group")
        cross_covariance = _permuted_cross_covariance(
            z_a,
            z_b,
            permutation,
            rows_per_group=rows_per_group,
        )
    whitened = whitening.p_a.T @ cross_covariance @ whitening.p_b
    try:
        u, sigma, vh = np.linalg.svd(whitened, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise C5BValidationError("CCA cross-covariance SVD did not converge") from error
    v = vh.T
    if not (
        np.all(np.isfinite(u)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(v))
    ):
        raise C5BValidationError("CCA SVD produced non-finite values")
    canonical_dimensions = min(
        whitening.p_a.shape[1],
        whitening.p_b.shape[1],
        sigma.size,
    )
    if canonical_dimensions < 10:
        raise C5BValidationError("CCA requires at least 10 canonical dimensions")
    return CCAFit(
        sigma=np.asarray(sigma[:canonical_dimensions], dtype=np.float64),
        w_a=np.asarray(whitening.p_a @ u[:, :canonical_dimensions], dtype=np.float64),
        w_b=np.asarray(whitening.p_b @ v[:, :canonical_dimensions], dtype=np.float64),
    )


def _block_row_permutation(
    permutation: np.ndarray,
    *,
    rows_per_group: int,
) -> np.ndarray:
    perm = _validate_permutation(permutation, permutation.size)
    offsets = np.arange(rows_per_group, dtype=np.int64)
    return (perm[:, None] * rows_per_group + offsets[None, :]).reshape(-1)


def _direct_pearson(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape or x.shape[1] < 10:
        raise C5BValidationError("Pearson inputs must have matching [N,K>=10] shapes")
    x_centered = x - np.mean(x, axis=0, dtype=np.float64)
    y_centered = y - np.mean(y, axis=0, dtype=np.float64)
    x_norm = np.linalg.norm(x_centered, axis=0)
    y_norm = np.linalg.norm(y_centered, axis=0)
    denominator = x_norm * y_norm
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise C5BValidationError("Pearson canonical variate has zero/invalid norm")
    correlations = np.sum(x_centered * y_centered, axis=0) / denominator
    if not np.all(np.isfinite(correlations)):
        raise C5BValidationError("Pearson correlation is non-finite")
    return np.asarray(correlations, dtype=np.float64)


def _heldout_metrics(
    z_a: np.ndarray,
    z_b: np.ndarray,
    fit: CCAFit,
    *,
    permutation: np.ndarray | None = None,
    rows_per_group: int | None = None,
) -> dict[str, float]:
    if permutation is not None:
        if rows_per_group is None:
            raise C5BValidationError("null HELD-OUT evaluation requires rows_per_group")
        order = _block_row_permutation(
            permutation,
            rows_per_group=rows_per_group,
        )
        z_b = z_b[order]
    h_a = z_a @ fit.w_a[:, :10]
    h_b = z_b @ fit.w_b[:, :10]
    correlations = _direct_pearson(h_a, h_b)
    return {
        "top1": float(correlations[0]),
        "top5mean": float(np.mean(correlations[:5], dtype=np.float64)),
        "top10mean": float(np.mean(correlations[:10], dtype=np.float64)),
    }


def _draw_derangements(
    rng: np.random.Generator,
    n_groups: int,
    repeats: int = NULL_REPEATS,
) -> np.ndarray:
    if n_groups < 2 or repeats <= 0:
        raise C5BValidationError("derangement sampling requires valid dimensions")
    identity = np.arange(n_groups, dtype=np.int64)
    values = []
    for _ in range(repeats):
        while True:
            permutation = np.asarray(rng.permutation(n_groups), dtype=np.int64)
            if np.all(permutation != identity):
                values.append(permutation)
                break
    return np.stack(values).astype(np.int64, copy=False)


def _generate_null_permutations(
    train_group_count: int = 40,
    heldout_group_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.random.SeedSequence(ROOT_RNG_SEED)
    train_seed, heldout_seed = root.spawn(2)
    train_rng = np.random.Generator(np.random.PCG64(train_seed))
    heldout_rng = np.random.Generator(np.random.PCG64(heldout_seed))
    return (
        _draw_derangements(train_rng, train_group_count, NULL_REPEATS),
        _draw_derangements(heldout_rng, heldout_group_count, NULL_REPEATS),
    )


def _summarize_metric(true_value: float, null_values: np.ndarray) -> dict[str, float]:
    null = np.asarray(null_values, dtype=np.float64)
    if null.shape != (NULL_REPEATS,) or not np.all(np.isfinite(null)):
        raise C5BValidationError("null metric array has an invalid shape or value")
    median = float(np.median(null))
    return {
        "true": float(true_value),
        "null_mean": float(np.mean(null, dtype=np.float64)),
        "null_std": float(np.std(null, ddof=0)),
        "null_median": median,
        "null_q95": float(np.quantile(null, 0.95, method="linear")),
        "true_minus_null_median": float(true_value - median),
        "empirical_p": float(
            (1 + np.count_nonzero(null >= true_value)) / (null.size + 1)
        ),
    }


def _analyze_configuration(
    z_a_train: np.ndarray,
    z_b_train: np.ndarray,
    z_a_heldout: np.ndarray,
    z_b_heldout: np.ndarray,
    train_permutations: np.ndarray,
    heldout_permutations: np.ndarray,
    *,
    rows_per_group: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    whitening = _prepare_cca(z_a_train, z_b_train)
    true_fit = _fit_cca(z_a_train, z_b_train, whitening)
    true_heldout = _heldout_metrics(z_a_heldout, z_b_heldout, true_fit)
    null_values = {
        name: np.empty(NULL_REPEATS, dtype=np.float64)
        for name in ("top1", "top5mean", "top10mean")
    }
    if train_permutations.shape != (NULL_REPEATS, 40):
        raise C5BValidationError("TRAIN permutation bank has an invalid shape")
    if heldout_permutations.shape != (NULL_REPEATS, 10):
        raise C5BValidationError("HELD-OUT permutation bank has an invalid shape")
    for repeat in range(NULL_REPEATS):
        null_fit = _fit_cca(
            z_a_train,
            z_b_train,
            whitening,
            permutation=train_permutations[repeat],
            rows_per_group=rows_per_group,
        )
        heldout = _heldout_metrics(
            z_a_heldout,
            z_b_heldout,
            null_fit,
            permutation=heldout_permutations[repeat],
            rows_per_group=rows_per_group,
        )
        for name in null_values:
            null_values[name][repeat] = heldout[name]

    train_top5mean = float(np.mean(true_fit.sigma[:5], dtype=np.float64))
    result = {
        "train_top5mean": train_top5mean,
        "heldout_top1": true_heldout["top1"],
        "heldout_top5mean": true_heldout["top5mean"],
        "heldout_top10mean": true_heldout["top10mean"],
        "train_to_heldout_top5mean_gap": (train_top5mean - true_heldout["top5mean"]),
        "null_summary": {
            name: _summarize_metric(true_heldout[name], null_values[name])
            for name in ("top1", "top5mean", "top10mean")
        },
    }
    return result, null_values


def _analyze_pair(
    records: Sequence[c5a.PairRecord],
    split: c5a.DatasetSplit,
    spec: PairSpec,
    train_permutations: np.ndarray,
    heldout_permutations: np.ndarray,
) -> tuple[dict[str, dict[str, int | float]], dict[str, Any], dict[str, np.ndarray]]:
    train_a_raw = _load_node_rows(
        records,
        split.train_observation_indices,
        archive_attribute="openvla_feature_path",
        archive_key=spec.openvla_key,
        expected_shape=spec.openvla_shape,
    )
    pca_a, train_a_99 = _fit_pca(train_a_raw)
    del train_a_raw
    heldout_a_raw = _load_node_rows(
        records,
        split.heldout_observation_indices,
        archive_attribute="openvla_feature_path",
        archive_key=spec.openvla_key,
        expected_shape=spec.openvla_shape,
    )
    heldout_a_99 = _transform_pca(pca_a, heldout_a_raw)
    del heldout_a_raw

    train_b_raw = _load_node_rows(
        records,
        split.train_observation_indices,
        archive_attribute="pi05_feature_path",
        archive_key=spec.pi05_key,
        expected_shape=spec.pi05_shape,
    )
    pca_b, train_b_99 = _fit_pca(train_b_raw)
    del train_b_raw
    heldout_b_raw = _load_node_rows(
        records,
        split.heldout_observation_indices,
        archive_attribute="pi05_feature_path",
        archive_key=spec.pi05_key,
        expected_shape=spec.pi05_shape,
    )
    heldout_b_99 = _transform_pca(pca_b, heldout_b_raw)
    del heldout_b_raw

    pca_metadata = {
        spec.openvla_pca_name: pca_a.metadata(),
        spec.pi05_pca_name: pca_b.metadata(),
    }
    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    rows_per_group = c5a.EXPECTED_OBSERVATIONS_PER_GROUP * spec.openvla_shape[0]
    for cutoff_name in ("99", "95"):
        d_a = pca_a.d_99 if cutoff_name == "99" else pca_a.d_95
        d_b = pca_b.d_99 if cutoff_name == "99" else pca_b.d_95
        configuration = f"{spec.name}__{cutoff_name}"
        result, null_values = _analyze_configuration(
            train_a_99[:, :d_a],
            train_b_99[:, :d_b],
            heldout_a_99[:, :d_a],
            heldout_b_99[:, :d_b],
            train_permutations,
            heldout_permutations,
            rows_per_group=rows_per_group,
        )
        results[configuration] = result
        for metric_name, values in null_values.items():
            arrays[f"{configuration}__heldout_{metric_name}"] = values
    return pca_metadata, results, arrays


def _build_split_manifest(
    manifest_path: Path,
    provenance: C5AProvenance,
    records: Sequence[c5a.PairRecord],
    groups: Sequence[c5a.GroupRecord],
    split: c5a.DatasetSplit,
) -> dict[str, Any]:
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_paired_manifest": str(manifest_path),
        "source_c5a_split_manifest": str(provenance.split_path),
        "split_rule_id": c5a.SPLIT_RULE_ID,
        "counts": {
            "groups": len(groups),
            "observations": len(records),
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


def _build_alignment_summary(
    manifest_path: Path,
    provenance: C5AProvenance,
    pca_metadata: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    primary = results["o2_p2__99"]
    primary_p = primary["null_summary"]["top5mean"]["empirical_p"]
    c5b_result = (
        "PASS" if primary["heldout_top5mean"] > 0.0 and primary_p <= 0.05 else "FAIL"
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "source_paired_manifest": str(manifest_path),
        "source_c5a_metric_summary": str(provenance.metric_path),
        "source_c5a_split_manifest": str(provenance.split_path),
        "c5a_gate": "GO",
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
        "pca": dict(pca_metadata),
        "cca": {
            "method": "ordinary_linear_covariance_whitening_svd",
            "covariance_denominator": "n_minus_1",
            "eigendecomposition_api": "numpy.linalg.eigh",
            "svd_api": "numpy.linalg.svd_full_matrices_false",
            "regularization": "none",
            "min_valid_canonical_dims": 10,
        },
        "null": {
            "repeats": NULL_REPEATS,
            "root_seed": ROOT_RNG_SEED,
            "rng": "SeedSequence.spawn(2)+PCG64",
            "sampler": "first_fixed_point_free_rng_permutation",
            "train_group_count": 40,
            "heldout_group_count": 10,
            "shared_bank_across_configurations": True,
        },
        "results": dict(results),
        "c5b_result": c5b_result,
        "representation_stage_result": c5b_result,
        "heldout_null_limitation": HELDOUT_NULL_LIMITATION,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# C5-B Explicit Shared-Space Alignment Summary",
        "",
        f"C5-B result: **{summary['c5b_result']}**",
        "",
        f"C5 representation-stage result: **{summary['representation_stage_result']}**",
        "",
        "## PCA dimensions",
        "",
    ]
    for node_name in ("o2", "p2", "o1s", "p1"):
        value = summary["pca"][node_name]
        lines.append(
            f"- {node_name}: full_rank={value['full_rank']}, "
            f"d_99={value['d_99']}, d_95={value['d_95']}"
        )
    lines.extend(["", "## Configurations", ""])
    for configuration in (
        "o2_p2__99",
        "o2_p2__95",
        "o1s_p1__99",
        "o1s_p1__95",
    ):
        value = summary["results"][configuration]
        lines.extend(
            [
                f"### {configuration}",
                "",
                f"- TRAIN Top5Mean: {value['train_top5mean']:.12g}",
                f"- HELD-OUT Top1: {value['heldout_top1']:.12g}",
                f"- HELD-OUT Top5Mean: {value['heldout_top5mean']:.12g}",
                f"- HELD-OUT Top10Mean: {value['heldout_top10mean']:.12g}",
                "- TRAIN→HELD-OUT Top5Mean gap: "
                f"{value['train_to_heldout_top5mean_gap']:.12g}",
            ]
        )
        for metric in ("top1", "top5mean", "top10mean"):
            null = value["null_summary"][metric]
            lines.append(
                f"- {metric} null: mean={null['null_mean']:.12g}, "
                f"median={null['null_median']:.12g}, "
                f"q95={null['null_q95']:.12g}, p={null['empirical_p']:.12g}"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            HELDOUT_NULL_LIMITATION,
            "",
            "Statistical significance relative to the frozen null does not by itself "
            "imply strong practical alignment.",
            "",
            "C5 representation-stage PASS does not establish policy relevance, action "
            "relevance, adversarial transferability, or attack effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_null_keys() -> set[str]:
    return {"train_permutations", "heldout_permutations"} | {
        f"{pair}__{cutoff}__heldout_{metric}"
        for pair in ("o2_p2", "o1s_p1")
        for cutoff in ("99", "95")
        for metric in ("top1", "top5mean", "top10mean")
    }


def _validate_null_arrays(values: Mapping[str, np.ndarray]) -> None:
    if set(values) != _expected_null_keys():
        raise C5BValidationError("C5-B null archive keys differ from the frozen schema")
    for name, expected_shape in (
        ("train_permutations", (NULL_REPEATS, 40)),
        ("heldout_permutations", (NULL_REPEATS, 10)),
    ):
        value = values[name]
        if value.shape != expected_shape or value.dtype != np.int64:
            raise C5BValidationError(f"invalid {name} array")
    for name in set(values) - {"train_permutations", "heldout_permutations"}:
        value = values[name]
        if (
            value.shape != (NULL_REPEATS,)
            or value.dtype != np.float64
            or not np.all(np.isfinite(value))
        ):
            raise C5BValidationError(f"invalid null metric array: {name}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _remove_staging_directory(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    except (FileNotFoundError, OSError):
        return


def _publish_outputs(
    output_dir: Path,
    *,
    split_manifest: Mapping[str, Any],
    alignment_summary: Mapping[str, Any],
    null_arrays: Mapping[str, np.ndarray],
    summary_markdown: str,
) -> None:
    _validate_null_arrays(null_arrays)
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output_dir.name}."))
    except OSError as error:
        raise C5BValidationError(
            f"failed to create C5-B staging directory near {output_dir}"
        ) from error
    removed_empty_target = False
    try:
        _write_json(staging / "split_manifest.json", split_manifest)
        np.savez_compressed(staging / "null_alignment_metrics.npz", **null_arrays)
        (staging / "summary.md").write_text(
            summary_markdown,
            encoding="utf-8",
            newline="\n",
        )
        _write_json(staging / "alignment_summary.json", alignment_summary)
        if {candidate.name for candidate in staging.iterdir()} != C5B_FILES:
            raise C5BValidationError("staged C5-B output set is incomplete")
        if output_dir.exists():
            if not output_dir.is_dir() or tuple(output_dir.iterdir()):
                raise C5BValidationError(
                    f"C5-B output directory became non-empty: {output_dir}"
                )
            output_dir.rmdir()
            removed_empty_target = True
        staging.rename(output_dir)
    except C5BValidationError:
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise
    except Exception as error:
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise C5BValidationError(
            f"failed to publish complete C5-B outputs: {output_dir}"
        ) from error
    finally:
        _remove_staging_directory(staging)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _validate_output_admission(args.output_dir)
    manifest_path, records, groups, split = _load_dataset_structure(
        args.paired_manifest
    )
    provenance = _load_c5a_provenance(
        args.c5a_output_dir,
        paired_manifest_path=manifest_path,
        records=records,
        groups=groups,
        split=split,
    )
    _validate_feature_archives(records)
    train_permutations, heldout_permutations = _generate_null_permutations()
    pca_metadata: dict[str, Any] = {}
    results: dict[str, Any] = {}
    null_arrays: dict[str, np.ndarray] = {
        "train_permutations": train_permutations,
        "heldout_permutations": heldout_permutations,
    }
    for pair_spec in PAIR_SPECS:
        pair_pca, pair_results, pair_arrays = _analyze_pair(
            records,
            split,
            pair_spec,
            train_permutations,
            heldout_permutations,
        )
        pca_metadata.update(pair_pca)
        results.update(pair_results)
        null_arrays.update(pair_arrays)

    split_manifest = _build_split_manifest(
        manifest_path,
        provenance,
        records,
        groups,
        split,
    )
    alignment_summary = _build_alignment_summary(
        manifest_path,
        provenance,
        pca_metadata,
        results,
    )
    summary_markdown = _build_summary_markdown(alignment_summary)
    _publish_outputs(
        output_dir,
        split_manifest=split_manifest,
        alignment_summary=alignment_summary,
        null_arrays=null_arrays,
        summary_markdown=summary_markdown,
    )
    return {
        "status": "C5-B Explicit Shared-Space Alignment — COMPLETED",
        "output_dir": str(output_dir),
        "c5b_result": alignment_summary["c5b_result"],
        "representation_stage_result": alignment_summary["representation_stage_result"],
        "num_pairs": len(records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
