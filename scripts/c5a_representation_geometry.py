from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _full_feature_extraction_common as common  # noqa: E402
from scripts import c2_full_feature_extraction as c2  # noqa: E402
from scripts import c3_full_feature_extraction as c3  # noqa: E402


PAIRED_SCHEMA_VERSION = "paired_features_v1"
SPLIT_SCHEMA_VERSION = "c5a_split_manifest_v1"
METRIC_SCHEMA_VERSION = "c5a_metric_summary_v1"
SPLIT_RULE_ID = "pilot-v0.2-c5-split-v1"
EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_GROUPS_PER_TASK = 5
EXPECTED_OBSERVATIONS_PER_GROUP = 4
EXPECTED_GROUP_COUNT = 50
EXPECTED_OBSERVATION_COUNT = 200
PROGRESS_SLOTS = (0.10, 0.40, 0.70, 0.90)
NULL_REPEATS = 50
ROOT_RNG_SEED = 7
PRIMARY_PAIR = "o2_p2"
CONTROL_PAIR = "o1s_p1"
PAIR_ARCHIVE_KEYS = {
    PRIMARY_PAIR: ("o2_projected", "p2_projected"),
    CONTROL_PAIR: ("o1_siglip", "p1_siglip"),
}
METRIC_NAMES = (
    "debiased_linear_cka",
    "biased_linear_cka",
    "spearman_rsa",
)
SPLIT_NAMES = ("train", "heldout")
HELDOUT_NULL_LIMITATION = (
    "HELD-OUT contains one group per task, so its derangement permits cross-task "
    "mismatches and is a broad cross-group mismatch null, not a task-conditioned "
    "state-level null."
)

_PAIR_FIELDS = {
    "sample_id",
    "source_image_hash",
    "openvla_feature_path",
    "pi05_feature_path",
}
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SAMPLE_ID_PATTERN = re.compile(
    r"libero_spatial__task(?P<task>\d{2})__state(?P<state>\d{2})"
    r"__step(?P<step>\d{4})"
)


class C5AValidationError(RuntimeError):
    """Raised when the frozen C5-A contract cannot be executed exactly."""


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    source_image_hash: str
    openvla_feature_path: Path
    pi05_feature_path: Path
    task_id: int
    initial_state_id: int
    step_id: int


@dataclass(frozen=True)
class GroupRecord:
    canonical_index: int
    task_id: int
    initial_state_id: int
    record_indices: tuple[int, int, int, int]
    sample_ids: tuple[str, str, str, str]
    step_ids: tuple[int, int, int, int]

    @property
    def identity(self) -> tuple[int, int]:
        return self.task_id, self.initial_state_id


@dataclass(frozen=True)
class DatasetSplit:
    train_group_indices: tuple[int, ...]
    heldout_group_indices: tuple[int, ...]
    train_observation_indices: np.ndarray
    heldout_observation_indices: np.ndarray
    group_digests: tuple[str, ...]
    assignments: tuple[str, ...]
    heldout_state_by_task: dict[int, int]


@dataclass(frozen=True)
class LoadedDataset:
    manifest_path: Path
    records: tuple[PairRecord, ...]
    groups: tuple[GroupRecord, ...]
    features: dict[str, np.ndarray]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Pilot v0.2 C5-A representation geometry test."
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        required=True,
        help="Completed formal C4 paired_features_v1 manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Absent or empty destination for the four formal C5-A artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5AValidationError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise C5AValidationError(f"{label} is not a regular file: {path}")
    return resolved


def _validate_output_admission(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve(strict=False)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise C5AValidationError(
                f"C5-A output path is not a directory: {output_dir}"
            )
        try:
            entries = tuple(output_dir.iterdir())
        except OSError as error:
            raise C5AValidationError(
                f"failed to inspect C5-A output directory: {output_dir}"
            ) from error
        if entries:
            raise C5AValidationError(
                f"C5-A output directory must be empty: {output_dir}"
            )
    return output_dir


def _resolve_relative_archive(
    manifest_parent: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise C5AValidationError(f"invalid relative POSIX {label}: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise C5AValidationError(f"absolute {label} is forbidden: {value!r}")
    try:
        resolved = (manifest_parent / Path(*relative.parts)).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise C5AValidationError(f"unresolved {label}: {value!r}") from error
    if not resolved.is_file():
        raise C5AValidationError(f"{label} is not a regular file: {value!r}")
    return resolved


def _load_paired_records(path: str | Path) -> tuple[Path, tuple[PairRecord, ...]]:
    manifest_path = _resolve_file(Path(path), "C4 paired manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise C5AValidationError(
            f"failed to load C4 paired manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "num_samples",
        "pairs",
    }:
        raise C5AValidationError("invalid paired_features_v1 top-level fields")
    if manifest.get("schema_version") != PAIRED_SCHEMA_VERSION:
        raise C5AValidationError(
            f"invalid C4 schema_version: {manifest.get('schema_version')!r}"
        )
    pairs = manifest.get("pairs")
    if (
        manifest.get("num_samples") != EXPECTED_OBSERVATION_COUNT
        or not isinstance(pairs, list)
        or len(pairs) != EXPECTED_OBSERVATION_COUNT
    ):
        raise C5AValidationError(
            "formal C4 manifest must contain exactly "
            f"{EXPECTED_OBSERVATION_COUNT} pairs"
        )

    records: list[PairRecord] = []
    sample_ids: set[str] = set()
    openvla_paths: set[Path] = set()
    pi05_paths: set[Path] = set()
    for index, value in enumerate(pairs):
        if not isinstance(value, dict) or set(value) != _PAIR_FIELDS:
            raise C5AValidationError(f"invalid C4 pair fields at index {index}")
        sample_id = value.get("sample_id")
        source_hash = value.get("source_image_hash")
        if not isinstance(sample_id, str):
            raise C5AValidationError(f"invalid sample_id at C4 index {index}")
        match = _SAMPLE_ID_PATTERN.fullmatch(sample_id)
        if match is None:
            raise C5AValidationError(
                f"non-canonical Pilot v0.2 sample_id: {sample_id!r}"
            )
        if sample_id in sample_ids:
            raise C5AValidationError(f"duplicate C4 sample_id: {sample_id!r}")
        if (
            not isinstance(source_hash, str)
            or _HASH_PATTERN.fullmatch(source_hash) is None
        ):
            raise C5AValidationError(
                f"invalid source_image_hash for sample_id={sample_id!r}"
            )
        openvla_path = _resolve_relative_archive(
            manifest_path.parent,
            value.get("openvla_feature_path"),
            label=f"OpenVLA feature path for {sample_id}",
        )
        pi05_path = _resolve_relative_archive(
            manifest_path.parent,
            value.get("pi05_feature_path"),
            label=f"pi0.5 feature path for {sample_id}",
        )
        if openvla_path in openvla_paths:
            raise C5AValidationError(f"duplicate OpenVLA archive path: {openvla_path}")
        if pi05_path in pi05_paths:
            raise C5AValidationError(f"duplicate pi0.5 archive path: {pi05_path}")
        sample_ids.add(sample_id)
        openvla_paths.add(openvla_path)
        pi05_paths.add(pi05_path)
        records.append(
            PairRecord(
                sample_id=sample_id,
                source_image_hash=source_hash,
                openvla_feature_path=openvla_path,
                pi05_feature_path=pi05_path,
                task_id=int(match.group("task")),
                initial_state_id=int(match.group("state")),
                step_id=int(match.group("step")),
            )
        )
    return manifest_path, tuple(records)


def _validate_archive(
    path: Path,
    *,
    record: PairRecord,
    spec: common.FeatureSpec,
) -> None:
    source_record = common.SourceRecord(
        sample_id=record.sample_id,
        source_observation_path="",
        resolved_observation_path=path,
        source_image_hash=record.source_image_hash,
    )
    try:
        common.validate_feature_archive(path, source_record, spec)
    except common.FullFeatureExtractionError as error:
        raise C5AValidationError(str(error)) from error


def _mean_pool(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2 or value.shape[0] != 256:
        raise C5AValidationError(
            f"mean pooling requires [256,D], received {value.shape}"
        )
    pooled = np.asarray(value, dtype=np.float64).mean(axis=0, dtype=np.float64)
    if pooled.ndim != 1 or not np.all(np.isfinite(pooled)):
        raise C5AValidationError("mean pooling produced an invalid representation")
    return pooled


def _load_and_pool_record(record: PairRecord) -> dict[str, np.ndarray]:
    _validate_archive(record.openvla_feature_path, record=record, spec=c2.SPEC)
    _validate_archive(record.pi05_feature_path, record=record, spec=c3.SPEC)
    try:
        with np.load(record.openvla_feature_path, allow_pickle=False) as archive:
            o2 = _mean_pool(archive["o2_projected"])
            o1s = _mean_pool(archive["o1_siglip"])
        with np.load(record.pi05_feature_path, allow_pickle=False) as archive:
            p2 = _mean_pool(archive["p2_projected"])
            p1 = _mean_pool(archive["p1_siglip"])
    except C5AValidationError:
        raise
    except Exception as error:
        raise C5AValidationError(
            f"failed to load paired features for {record.sample_id}"
        ) from error
    return {"o2": o2, "p2": p2, "o1s": o1s, "p1": p1}


def _reconstruct_groups(records: Sequence[PairRecord]) -> tuple[GroupRecord, ...]:
    groups: list[GroupRecord] = []
    seen: set[tuple[int, int]] = set()
    index = 0
    while index < len(records):
        first = records[index]
        identity = first.task_id, first.initial_state_id
        if identity in seen:
            raise C5AValidationError(
                f"group records are not contiguous in C4 order: {identity}"
            )
        end = index
        while end < len(records):
            current = records[end]
            if (current.task_id, current.initial_state_id) != identity:
                break
            end += 1
        block = records[index:end]
        if len(block) != EXPECTED_OBSERVATIONS_PER_GROUP:
            raise C5AValidationError(
                f"group {identity} must contain exactly "
                f"{EXPECTED_OBSERVATIONS_PER_GROUP} observations"
            )
        step_ids = tuple(record.step_id for record in block)
        if any(left >= right for left, right in zip(step_ids, step_ids[1:])):
            raise C5AValidationError(
                f"group {identity} step_ids must be strictly increasing: {step_ids}"
            )
        seen.add(identity)
        groups.append(
            GroupRecord(
                canonical_index=len(groups),
                task_id=identity[0],
                initial_state_id=identity[1],
                record_indices=tuple(range(index, end)),
                sample_ids=tuple(record.sample_id for record in block),
                step_ids=step_ids,
            )
        )
        index = end

    if len(groups) != EXPECTED_GROUP_COUNT:
        raise C5AValidationError(
            f"formal C5-A requires exactly {EXPECTED_GROUP_COUNT} groups"
        )
    task_ids = {group.task_id for group in groups}
    if task_ids != set(EXPECTED_TASK_IDS):
        raise C5AValidationError(
            f"formal C5-A task IDs differ: expected={EXPECTED_TASK_IDS}, "
            f"actual={sorted(task_ids)}"
        )
    for task_id in EXPECTED_TASK_IDS:
        task_groups = [group for group in groups if group.task_id == task_id]
        if len(task_groups) != EXPECTED_GROUPS_PER_TASK:
            raise C5AValidationError(
                f"task {task_id} must contain exactly {EXPECTED_GROUPS_PER_TASK} groups"
            )
        states = [group.initial_state_id for group in task_groups]
        if len(set(states)) != EXPECTED_GROUPS_PER_TASK:
            raise C5AValidationError(f"task {task_id} contains duplicate group states")
    return tuple(groups)


def _load_dataset(path: str | Path) -> LoadedDataset:
    manifest_path, records = _load_paired_records(path)
    groups = _reconstruct_groups(records)
    pooled: dict[str, list[np.ndarray]] = {
        "o2": [],
        "p2": [],
        "o1s": [],
        "p1": [],
    }
    for record in records:
        values = _load_and_pool_record(record)
        for name in pooled:
            pooled[name].append(values[name])
    features = {
        name: np.stack(values).astype(np.float64, copy=False)
        for name, values in pooled.items()
    }
    return LoadedDataset(
        manifest_path=manifest_path,
        records=records,
        groups=groups,
        features=features,
    )


def _split_digest(task_id: int, initial_state_id: int) -> str:
    canonical = f"{SPLIT_RULE_ID}|task_id={task_id}|initial_state_id={initial_state_id}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten_group_observations(
    groups: Sequence[GroupRecord],
    group_indices: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [
            record_index
            for group_index in group_indices
            for record_index in groups[group_index].record_indices
        ],
        dtype=np.int64,
    )


def _build_split(groups: Sequence[GroupRecord]) -> DatasetSplit:
    digests = tuple(
        _split_digest(group.task_id, group.initial_state_id) for group in groups
    )
    heldout_group_indices: list[int] = []
    heldout_state_by_task: dict[int, int] = {}
    for task_id in EXPECTED_TASK_IDS:
        candidates = [
            index for index, group in enumerate(groups) if group.task_id == task_id
        ]
        ranked = sorted(
            candidates,
            key=lambda index: (digests[index], groups[index].initial_state_id),
        )
        heldout_index = ranked[0]
        heldout_group_indices.append(heldout_index)
        heldout_state_by_task[task_id] = groups[heldout_index].initial_state_id

    heldout_set = set(heldout_group_indices)
    train_group_indices = tuple(
        index for index in range(len(groups)) if index not in heldout_set
    )
    heldout_group_indices_tuple = tuple(
        index for index in range(len(groups)) if index in heldout_set
    )
    if len(train_group_indices) != 40 or len(heldout_group_indices_tuple) != 10:
        raise C5AValidationError(
            "frozen split must contain 40 TRAIN and 10 HELD-OUT groups"
        )
    train_observations = _flatten_group_observations(groups, train_group_indices)
    heldout_observations = _flatten_group_observations(
        groups,
        heldout_group_indices_tuple,
    )
    if train_observations.shape != (160,) or heldout_observations.shape != (40,):
        raise C5AValidationError(
            "frozen split must contain 160 TRAIN and 40 HELD-OUT observations"
        )
    assignments = tuple(
        "HELD-OUT" if index in heldout_set else "TRAIN" for index in range(len(groups))
    )
    return DatasetSplit(
        train_group_indices=train_group_indices,
        heldout_group_indices=heldout_group_indices_tuple,
        train_observation_indices=train_observations,
        heldout_observation_indices=heldout_observations,
        group_digests=digests,
        assignments=assignments,
        heldout_state_by_task=heldout_state_by_task,
    )


def _unbiased_hsic(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    k = np.asarray(gram_x, dtype=np.float64)
    ell = np.asarray(gram_y, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or ell.shape != k.shape:
        raise C5AValidationError("unbiased HSIC requires equal square Gram matrices")
    n = k.shape[0]
    if n < 4:
        raise C5AValidationError("unbiased HSIC requires n >= 4")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(ell)):
        raise C5AValidationError("unbiased HSIC received non-finite Gram values")
    k_tilde = k.copy()
    l_tilde = ell.copy()
    np.fill_diagonal(k_tilde, 0.0)
    np.fill_diagonal(l_tilde, 0.0)
    trace_term = float(np.sum(k_tilde * l_tilde.T, dtype=np.float64))
    k_sum = float(np.sum(k_tilde, dtype=np.float64))
    l_sum = float(np.sum(l_tilde, dtype=np.float64))
    cross_sum = float(k_tilde.sum(axis=0) @ l_tilde.sum(axis=1))
    value = (
        trace_term + (k_sum * l_sum) / ((n - 1) * (n - 2)) - (2.0 * cross_sum) / (n - 2)
    ) / (n * (n - 3))
    if not math.isfinite(value):
        raise C5AValidationError("unbiased HSIC produced a non-finite value")
    return value


def _debiased_linear_cka_from_grams(
    gram_x: np.ndarray,
    gram_y: np.ndarray,
) -> float:
    cross_hsic = _unbiased_hsic(gram_x, gram_y)
    self_x = _unbiased_hsic(gram_x, gram_x)
    self_y = _unbiased_hsic(gram_y, gram_y)
    if self_x <= 0.0 or self_y <= 0.0:
        raise C5AValidationError(
            f"debiased CKA self-HSIC must be positive: X={self_x}, Y={self_y}"
        )
    denominator = math.sqrt(self_x * self_y)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise C5AValidationError("debiased CKA denominator is invalid")
    value = cross_hsic / denominator
    if not math.isfinite(value):
        raise C5AValidationError("debiased CKA produced a non-finite value")
    return float(value)


def _center_gram(gram: np.ndarray) -> np.ndarray:
    value = np.asarray(gram, dtype=np.float64)
    row_mean = value.mean(axis=1, keepdims=True)
    column_mean = value.mean(axis=0, keepdims=True)
    return value - row_mean - column_mean + value.mean(dtype=np.float64)


def _biased_linear_cka_from_grams(
    gram_x: np.ndarray,
    gram_y: np.ndarray,
) -> float:
    k_centered = _center_gram(gram_x)
    l_centered = _center_gram(gram_y)
    numerator = float(np.sum(k_centered * l_centered, dtype=np.float64))
    denominator = float(
        np.linalg.norm(k_centered, ord="fro") * np.linalg.norm(l_centered, ord="fro")
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise C5AValidationError("biased CKA denominator is invalid")
    value = numerator / denominator
    if not math.isfinite(value):
        raise C5AValidationError("biased CKA produced a non-finite value")
    return float(value)


def _distance_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise C5AValidationError("RSA requires a finite [N,D] representation matrix")
    distances = np.asarray(
        squareform(pdist(matrix, metric="sqeuclidean")), dtype=np.float64
    )
    if not np.all(np.isfinite(distances)):
        raise C5AValidationError("RSA produced non-finite squared distances")
    return distances


def _spearman_rsa_from_distances(
    distance_x: np.ndarray,
    distance_y: np.ndarray,
) -> float:
    x = np.asarray(distance_x, dtype=np.float64)
    y = np.asarray(distance_y, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or y.shape != x.shape:
        raise C5AValidationError("RSA requires equal square distance matrices")
    upper = np.triu_indices(x.shape[0], k=1)
    x_values = x[upper]
    y_values = y[upper]
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise C5AValidationError("RSA distance vector contains non-finite values")
    if np.ptp(x_values) == 0.0 or np.ptp(y_values) == 0.0:
        raise C5AValidationError("RSA distance vector must not be constant")
    statistic = float(spearmanr(x_values, y_values).statistic)
    if not math.isfinite(statistic):
        raise C5AValidationError("Spearman RSA produced a non-finite value")
    return statistic


def _metric_triplet(value_x: np.ndarray, value_y: np.ndarray) -> dict[str, float]:
    x = np.asarray(value_x, dtype=np.float64)
    y = np.asarray(value_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise C5AValidationError("paired metrics require [N,D] matrices with equal N")
    gram_x = x @ x.T
    gram_y = y @ y.T
    return _metric_triplet_from_matrices(
        gram_x,
        gram_y,
        _distance_matrix(x),
        _distance_matrix(y),
    )


def _metric_triplet_from_matrices(
    gram_x: np.ndarray,
    gram_y: np.ndarray,
    distance_x: np.ndarray,
    distance_y: np.ndarray,
) -> dict[str, float]:
    return {
        "debiased_linear_cka": _debiased_linear_cka_from_grams(gram_x, gram_y),
        "biased_linear_cka": _biased_linear_cka_from_grams(gram_x, gram_y),
        "spearman_rsa": _spearman_rsa_from_distances(distance_x, distance_y),
    }


def _draw_derangements(
    rng: np.random.Generator,
    n_groups: int,
    repeats: int = NULL_REPEATS,
) -> np.ndarray:
    if n_groups < 2 or repeats <= 0:
        raise C5AValidationError(
            "derangement sampling requires groups >= 2 and repeats > 0"
        )
    permutations: list[np.ndarray] = []
    identity = np.arange(n_groups)
    for _ in range(repeats):
        while True:
            permutation = np.asarray(rng.permutation(n_groups), dtype=np.int64)
            if permutation.shape != (n_groups,):
                raise C5AValidationError("RNG returned an invalid group permutation")
            if np.all(permutation != identity):
                permutations.append(permutation)
                break
    return np.stack(permutations).astype(np.int64, copy=False)


def _generate_null_permutations(
    train_group_count: int,
    heldout_group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.random.SeedSequence(ROOT_RNG_SEED)
    train_seed, heldout_seed = root.spawn(2)
    train_rng = np.random.Generator(np.random.PCG64(train_seed))
    heldout_rng = np.random.Generator(np.random.PCG64(heldout_seed))
    return (
        _draw_derangements(train_rng, train_group_count, NULL_REPEATS),
        _draw_derangements(heldout_rng, heldout_group_count, NULL_REPEATS),
    )


def _block_row_permutation(permutation: np.ndarray) -> np.ndarray:
    group_indices = np.asarray(permutation, dtype=np.int64)
    slots = np.arange(EXPECTED_OBSERVATIONS_PER_GROUP, dtype=np.int64)
    return (
        group_indices[:, None] * EXPECTED_OBSERVATIONS_PER_GROUP + slots[None, :]
    ).reshape(-1)


def _analyze_pair(
    value_x: np.ndarray,
    value_y: np.ndarray,
    permutations: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    x = np.asarray(value_x, dtype=np.float64)
    y = np.asarray(value_y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] != (
        permutations.shape[1] * EXPECTED_OBSERVATIONS_PER_GROUP
    ):
        raise C5AValidationError("pair analysis rows do not match group permutations")
    gram_x = x @ x.T
    gram_y = y @ y.T
    distance_x = _distance_matrix(x)
    distance_y = _distance_matrix(y)
    true_metrics = _metric_triplet_from_matrices(
        gram_x,
        gram_y,
        distance_x,
        distance_y,
    )
    null_metrics = {
        name: np.empty(permutations.shape[0], dtype=np.float64) for name in METRIC_NAMES
    }
    for repeat_index, permutation in enumerate(permutations):
        row_order = _block_row_permutation(permutation)
        permuted_gram_y = gram_y[np.ix_(row_order, row_order)]
        permuted_distance_y = distance_y[np.ix_(row_order, row_order)]
        metrics = _metric_triplet_from_matrices(
            gram_x,
            permuted_gram_y,
            distance_x,
            permuted_distance_y,
        )
        for name, value in metrics.items():
            null_metrics[name][repeat_index] = value
    return true_metrics, null_metrics


def _empirical_p_value(true_value: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.all(np.isfinite(null)):
        raise C5AValidationError("empirical p-value requires finite null values")
    return float((1 + np.count_nonzero(null >= true_value)) / (null.size + 1))


def _summarize_metric(
    true_value: float,
    null_values: np.ndarray,
) -> dict[str, float]:
    null = np.asarray(null_values, dtype=np.float64)
    median = float(np.median(null))
    return {
        "true": float(true_value),
        "null_mean": float(np.mean(null, dtype=np.float64)),
        "null_std": float(np.std(null, ddof=0)),
        "null_median": median,
        "null_q95": float(np.quantile(null, 0.95, method="linear")),
        "true_minus_null_median": float(true_value - median),
        "empirical_p_value": _empirical_p_value(true_value, null),
    }


def _compute_results(
    dataset: LoadedDataset,
    split: DatasetSplit,
    train_permutations: np.ndarray,
    heldout_permutations: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    split_inputs = {
        "train": (split.train_observation_indices, train_permutations),
        "heldout": (split.heldout_observation_indices, heldout_permutations),
    }
    pair_features = {
        PRIMARY_PAIR: ("o2", "p2"),
        CONTROL_PAIR: ("o1s", "p1"),
    }
    results: dict[str, Any] = {}
    null_arrays: dict[str, np.ndarray] = {
        "train_permutations": train_permutations.astype(np.int64, copy=False),
        "heldout_permutations": heldout_permutations.astype(np.int64, copy=False),
    }
    for split_name, (indices, permutations) in split_inputs.items():
        split_result: dict[str, Any] = {"pairs": {}}
        for pair_name, (x_name, y_name) in pair_features.items():
            true_metrics, null_metrics = _analyze_pair(
                dataset.features[x_name][indices],
                dataset.features[y_name][indices],
                permutations,
            )
            split_result["pairs"][pair_name] = {
                metric_name: _summarize_metric(
                    true_metrics[metric_name],
                    null_metrics[metric_name],
                )
                for metric_name in METRIC_NAMES
            }
            for metric_name in METRIC_NAMES:
                null_arrays[f"{split_name}__{pair_name}__{metric_name}"] = null_metrics[
                    metric_name
                ].astype(np.float64, copy=False)
        primary = split_result["pairs"][PRIMARY_PAIR]["debiased_linear_cka"]
        split_result["pass"] = bool(
            primary["true"] > 0.0 and primary["empirical_p_value"] <= 0.05
        )
        split_result["num_observations"] = int(indices.size)
        split_result["num_groups"] = int(permutations.shape[1])
        results[split_name] = split_result
    return results, null_arrays


def _build_split_manifest(
    dataset: LoadedDataset,
    split: DatasetSplit,
) -> dict[str, Any]:
    groups = []
    for group, digest, assignment in zip(
        dataset.groups,
        split.group_digests,
        split.assignments,
        strict=True,
    ):
        groups.append(
            {
                "canonical_group_index": group.canonical_index,
                "task_id": group.task_id,
                "initial_state_id": group.initial_state_id,
                "split_digest": digest,
                "assignment": assignment,
                "sample_ids": list(group.sample_ids),
                "step_ids": list(group.step_ids),
                "inherited_progress_slots": list(PROGRESS_SLOTS),
            }
        )
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_paired_manifest": str(dataset.manifest_path),
        "split_rule_id": SPLIT_RULE_ID,
        "canonical_group_order": [
            {
                "task_id": group.task_id,
                "initial_state_id": group.initial_state_id,
            }
            for group in dataset.groups
        ],
        "counts": {
            "groups": len(dataset.groups),
            "observations": len(dataset.records),
            "train_groups": len(split.train_group_indices),
            "heldout_groups": len(split.heldout_group_indices),
            "train_observations": int(split.train_observation_indices.size),
            "heldout_observations": int(split.heldout_observation_indices.size),
        },
        "heldout_state_by_task": {
            str(task_id): split.heldout_state_by_task[task_id]
            for task_id in EXPECTED_TASK_IDS
        },
        "groups": groups,
    }


def _build_metric_summary(
    dataset: LoadedDataset,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    gate = "GO" if results["train"]["pass"] and results["heldout"]["pass"] else "NO-GO"
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "source_paired_manifest": str(dataset.manifest_path),
        "primary_pair": {
            "name": PRIMARY_PAIR,
            "openvla_node": "O2",
            "pi05_node": "P2",
            "gate_metric": "debiased_linear_cka",
        },
        "control_pair": {
            "name": CONTROL_PAIR,
            "openvla_node": "O1-S",
            "pi05_node": "P1",
            "determines_gate": False,
        },
        "metric_conventions": {
            "representation": "arithmetic_mean_over_all_256_tokens",
            "internal_dtype": "float64",
            "debiased_linear_cka": "task.md frozen unbiased-HSIC formula",
            "biased_linear_cka": "column-centered standard linear CKA",
            "spearman_rsa": "strict-upper squared-Euclidean RDM Spearman correlation",
        },
        "null": {
            "repeats": NULL_REPEATS,
            "root_seed": ROOT_RNG_SEED,
            "rng": "SeedSequence(7).spawn(2) with independent PCG64 streams",
            "sampler": "first fixed-point-free rng.permutation draw",
            "unit": "four-observation trajectory group block",
        },
        "splits": dict(results),
        "c5a_gate": gate,
        "heldout_null_limitation": HELDOUT_NULL_LIMITATION,
    }


def _build_summary_markdown(metric_summary: Mapping[str, Any]) -> str:
    lines = [
        "# C5-A Representation Geometry Summary",
        "",
        f"Formal C5-A gate: **{metric_summary['c5a_gate']}**",
        "",
        "Data: 50 groups / 200 observations; TRAIN 40/160; HELD-OUT 10/40.",
        "",
    ]
    for split_name in SPLIT_NAMES:
        split = metric_summary["splits"][split_name]
        lines.extend(
            [
                f"## {split_name.upper()}",
                "",
                f"Primary split PASS: **{split['pass']}**",
                "",
            ]
        )
        for pair_name in (PRIMARY_PAIR, CONTROL_PAIR):
            lines.extend([f"### {pair_name}", ""])
            for metric_name in METRIC_NAMES:
                metric = split["pairs"][pair_name][metric_name]
                lines.append(
                    f"- {metric_name}: true={metric['true']:.12g}, "
                    f"null median={metric['null_median']:.12g}, "
                    f"p={metric['empirical_p_value']:.12g}"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Biased CKA is diagnostic and Spearman RSA is robustness evidence; neither "
            "changes the formal C5-A gate.",
            "",
            HELDOUT_NULL_LIMITATION,
            "",
            "A C5-A NO-GO is not proof that no shared representation of any kind exists.",
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


def _remove_staging_directory(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


def _publish_outputs(
    output_dir: Path,
    *,
    split_manifest: Mapping[str, Any],
    metric_summary: Mapping[str, Any],
    null_arrays: Mapping[str, np.ndarray],
    summary_markdown: str,
) -> None:
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output_dir.name}."))
    except OSError as error:
        raise C5AValidationError(
            f"failed to create C5-A staging directory near {output_dir}"
        ) from error

    removed_empty_target = False
    try:
        _write_json(staging / "split_manifest.json", split_manifest)
        np.savez_compressed(staging / "null_metrics.npz", **null_arrays)
        (staging / "summary.md").write_text(
            summary_markdown,
            encoding="utf-8",
            newline="\n",
        )
        _write_json(staging / "metric_summary.json", metric_summary)
        expected_names = {
            "split_manifest.json",
            "metric_summary.json",
            "null_metrics.npz",
            "summary.md",
        }
        if {path.name for path in staging.iterdir()} != expected_names:
            raise C5AValidationError("staged C5-A output set is incomplete")

        if output_dir.exists():
            if not output_dir.is_dir() or tuple(output_dir.iterdir()):
                raise C5AValidationError(
                    f"C5-A output directory became non-empty: {output_dir}"
                )
            output_dir.rmdir()
            removed_empty_target = True
        staging.rename(output_dir)
    except C5AValidationError:
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise
    except Exception as error:
        if removed_empty_target and not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        raise C5AValidationError(
            f"failed to publish complete C5-A outputs: {output_dir}"
        ) from error
    finally:
        _remove_staging_directory(staging)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _validate_output_admission(args.output_dir)
    dataset = _load_dataset(args.paired_manifest)
    split = _build_split(dataset.groups)
    train_permutations, heldout_permutations = _generate_null_permutations(
        len(split.train_group_indices),
        len(split.heldout_group_indices),
    )
    results, null_arrays = _compute_results(
        dataset,
        split,
        train_permutations,
        heldout_permutations,
    )
    split_manifest = _build_split_manifest(dataset, split)
    metric_summary = _build_metric_summary(dataset, results)
    summary_markdown = _build_summary_markdown(metric_summary)
    _publish_outputs(
        output_dir,
        split_manifest=split_manifest,
        metric_summary=metric_summary,
        null_arrays=null_arrays,
        summary_markdown=summary_markdown,
    )
    return {
        "status": "C5-A Representation Geometry — COMPLETED",
        "output_dir": str(output_dir),
        "c5a_gate": metric_summary["c5a_gate"],
        "num_pairs": len(dataset.records),
        "train_groups": len(split.train_group_indices),
        "heldout_groups": len(split.heldout_group_indices),
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
