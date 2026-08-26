from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from shared_feature import PilotObservation


COLLECTION_SCHEMA_VERSION = "pilot_v0_2_collection_v1"
FEATURE_MANIFEST_SCHEMA_VERSION = "pilot_v0_2_full_feature_manifest_v1"
PILOT_VERSION = "0.2"
COLLECTION_CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_GROUPS_PER_TASK = 5
EXPECTED_SAMPLES_PER_GROUP = 4
EXPECTED_GROUP_COUNT = 50
EXPECTED_OBSERVATION_COUNT = 200
EXPECTED_TARGET_PROGRESS = (0.10, 0.40, 0.70, 0.90)
RUNTIME_PRECISION = "bf16"
SAVED_DTYPE = "float32"
BATCH_SIZE = 1
NUM_TOKENS = 256
GRID_SHAPE = (16, 16)
TOKEN_ORDER = "model_native_spatial_flatten_order"

_METADATA_FIELDS = {
    "sample_id",
    "source_model",
    "checkpoint",
    "feature_schema_version",
    "source_image_hash",
}
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SAMPLE_ID_PATTERN = re.compile(
    r"libero_spatial__task(?P<task>\d{2})__state(?P<state>\d{2})"
    r"__step(?P<step>\d{4})"
)


class FullFeatureExtractionError(RuntimeError):
    """Raised when a formal C2/C3 extraction boundary is invalid."""


@dataclass(frozen=True)
class FeatureNode:
    scientific_name: str
    archive_key: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class FeatureSpec:
    model_family: str
    source_model: str
    checkpoint_identity: str
    feature_schema_version: str
    manifest_filename: str
    nodes: tuple[FeatureNode, ...]
    feature_config: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    source_observation_path: str
    resolved_observation_path: Path
    source_image_hash: str


@dataclass(frozen=True)
class SourceCollection:
    manifest_path: Path
    checkpoint_identity: str
    libero_revision: str
    records: tuple[SourceRecord, ...]


@dataclass(frozen=True)
class OutputPreparation:
    output_dir: Path
    features_dir: Path
    manifest_path: Path
    missing_records: tuple[SourceRecord, ...]
    reused_count: int


def load_source_collection(path: str | Path) -> SourceCollection:
    manifest_path = _resolve_file(Path(path), "collection manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FullFeatureExtractionError(
            f"failed to load collection manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise FullFeatureExtractionError(
            "collection manifest must encode a JSON object"
        )

    _require_equal(manifest, "schema_version", COLLECTION_SCHEMA_VERSION)
    _require_equal(manifest, "pilot_version", PILOT_VERSION)
    _require_equal(manifest, "suite", "libero_spatial")
    _require_equal(manifest, "run_status", "COMPLETED")
    _require_equal(manifest, "completeness_status", "COMPLETE")

    rollout = _require_mapping(manifest, "rollout")
    checkpoint_identity = rollout.get("checkpoint_identity")
    if checkpoint_identity != COLLECTION_CHECKPOINT_IDENTITY:
        raise FullFeatureExtractionError(
            "source collection checkpoint identity differs from the frozen "
            f"identity: {checkpoint_identity!r}"
        )

    runtime = _require_mapping(manifest, "runtime")
    libero_revision = runtime.get("libero_revision")
    if not isinstance(libero_revision, str) or not libero_revision.strip():
        raise FullFeatureExtractionError(
            "source collection LIBERO revision must be a non-empty string"
        )
    if runtime.get("discovered_task_count") != len(EXPECTED_TASK_IDS):
        raise FullFeatureExtractionError(
            "source collection discovered task count must equal 10"
        )
    if runtime.get("task_ids") != list(EXPECTED_TASK_IDS):
        raise FullFeatureExtractionError(
            "source collection runtime task IDs must equal 0..9"
        )

    coverage = _require_mapping(manifest, "coverage")
    expected_coverage = {
        "target_total_groups": EXPECTED_GROUP_COUNT,
        "target_total_observations": EXPECTED_OBSERVATION_COUNT,
        "actual_total_groups": EXPECTED_GROUP_COUNT,
        "actual_total_observations": EXPECTED_OBSERVATION_COUNT,
        "accepted_groups_per_task": {
            str(task_id): EXPECTED_GROUPS_PER_TASK for task_id in EXPECTED_TASK_IDS
        },
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise FullFeatureExtractionError(
                f"invalid source collection coverage {key}: "
                f"expected={expected!r}, actual={coverage.get(key)!r}"
            )

    task_results = manifest.get("task_results")
    if not isinstance(task_results, list) or len(task_results) != len(
        EXPECTED_TASK_IDS
    ):
        raise FullFeatureExtractionError(
            "source collection must contain exactly 10 task results"
        )
    if [task.get("task_id") for task in task_results if isinstance(task, dict)] != (
        list(EXPECTED_TASK_IDS)
    ):
        raise FullFeatureExtractionError(
            "source collection task results must be ordered 0..9"
        )

    records: list[SourceRecord] = []
    sample_ids: set[str] = set()
    resolved_paths: set[Path] = set()
    group_ids: set[tuple[int, int]] = set()
    manifest_parent = manifest_path.parent
    observations_dir = (manifest_parent / "observations").resolve()
    if not observations_dir.is_dir():
        raise FullFeatureExtractionError(
            f"source observations directory does not exist: {observations_dir}"
        )

    for task_id, task_result in zip(
        EXPECTED_TASK_IDS,
        task_results,
        strict=True,
    ):
        _load_task_records(
            task_id=task_id,
            task_result=task_result,
            manifest_parent=manifest_parent,
            observations_dir=observations_dir,
            records=records,
            sample_ids=sample_ids,
            resolved_paths=resolved_paths,
            group_ids=group_ids,
        )

    if len(records) != EXPECTED_OBSERVATION_COUNT:
        raise FullFeatureExtractionError(
            "source collection must contain exactly 200 canonical observations"
        )
    if len(group_ids) != EXPECTED_GROUP_COUNT:
        raise FullFeatureExtractionError(
            "source collection must contain exactly 50 unique groups"
        )

    try:
        direct_archives = {
            candidate.resolve(strict=True)
            for candidate in observations_dir.glob("*.npz")
        }
    except (OSError, RuntimeError) as error:
        raise FullFeatureExtractionError(
            f"failed to enumerate source observations: {observations_dir}"
        ) from error
    if direct_archives != resolved_paths:
        missing = sorted(str(path) for path in resolved_paths - direct_archives)
        unexpected = sorted(str(path) for path in direct_archives - resolved_paths)
        raise FullFeatureExtractionError(
            "source observation archives differ from collection manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return SourceCollection(
        manifest_path=manifest_path,
        checkpoint_identity=checkpoint_identity,
        libero_revision=libero_revision,
        records=tuple(records),
    )


def _load_task_records(
    *,
    task_id: int,
    task_result: Any,
    manifest_parent: Path,
    observations_dir: Path,
    records: list[SourceRecord],
    sample_ids: set[str],
    resolved_paths: set[Path],
    group_ids: set[tuple[int, int]],
) -> None:
    if not isinstance(task_result, dict):
        raise FullFeatureExtractionError(
            f"source task result {task_id} must be an object"
        )
    if task_result.get("actual_accepted_groups") != EXPECTED_GROUPS_PER_TASK:
        raise FullFeatureExtractionError(
            f"source task {task_id} must contain exactly five accepted groups"
        )
    groups = task_result.get("accepted_groups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_GROUPS_PER_TASK:
        raise FullFeatureExtractionError(
            f"source task {task_id} has malformed accepted groups"
        )
    if not all(isinstance(group, dict) for group in groups):
        raise FullFeatureExtractionError(
            f"source task {task_id} accepted groups must be objects"
        )
    state_ids = [group.get("initial_state_id") for group in groups]
    if (
        not all(type(state_id) is int and state_id >= 0 for state_id in state_ids)
        or state_ids != sorted(state_ids)
        or len(set(state_ids)) != EXPECTED_GROUPS_PER_TASK
        or task_result.get("accepted_state_ids") != state_ids
    ):
        raise FullFeatureExtractionError(
            f"source task {task_id} accepted state order is malformed"
        )

    for group in groups:
        state_id = group.get("initial_state_id")
        group_id = (task_id, state_id)
        if (
            group.get("task_id") != task_id
            or type(state_id) is not int
            or state_id < 0
            or group.get("episode_success") is not True
            or group_id in group_ids
        ):
            raise FullFeatureExtractionError(
                f"source task {task_id} contains malformed group metadata"
            )
        group_ids.add(group_id)
        samples = group.get("samples")
        if not isinstance(samples, list) or len(samples) != (
            EXPECTED_SAMPLES_PER_GROUP
        ):
            raise FullFeatureExtractionError(
                f"source group {group_id} must contain exactly four samples"
            )
        if not all(isinstance(sample, dict) for sample in samples):
            raise FullFeatureExtractionError(
                f"source group {group_id} samples must be objects"
            )
        if [sample.get("target_relative_progress") for sample in samples] != list(
            EXPECTED_TARGET_PROGRESS
        ):
            raise FullFeatureExtractionError(
                f"source group {group_id} has invalid target progress"
            )
        for sample in samples:
            record = _load_source_record(
                task_id=task_id,
                state_id=state_id,
                sample=sample,
                manifest_parent=manifest_parent,
                observations_dir=observations_dir,
            )
            if record.sample_id in sample_ids:
                raise FullFeatureExtractionError(
                    f"duplicate source sample_id: {record.sample_id}"
                )
            if record.resolved_observation_path in resolved_paths:
                raise FullFeatureExtractionError(
                    "duplicate source observation path: "
                    f"{record.source_observation_path}"
                )
            sample_ids.add(record.sample_id)
            resolved_paths.add(record.resolved_observation_path)
            records.append(record)


def _load_source_record(
    *,
    task_id: int,
    state_id: int,
    sample: Any,
    manifest_parent: Path,
    observations_dir: Path,
) -> SourceRecord:
    if not isinstance(sample, dict):
        raise FullFeatureExtractionError("source sample must be an object")
    sample_id = sample.get("sample_id")
    relative_string = sample.get("observation_path")
    if not isinstance(sample_id, str) or not sample_id:
        raise FullFeatureExtractionError("source sample_id must be a non-empty string")
    step_id = sample.get("step_id")
    if type(step_id) is not int or step_id < 0:
        raise FullFeatureExtractionError(
            f"source step_id is not a non-negative integer: {sample_id!r}"
        )
    match = _SAMPLE_ID_PATTERN.fullmatch(sample_id)
    if (
        match is None
        or int(match.group("task")) != task_id
        or int(match.group("state")) != state_id
        or int(match.group("step")) != step_id
    ):
        raise FullFeatureExtractionError(
            f"source sample_id is not canonical: {sample_id!r}"
        )
    if not isinstance(relative_string, str) or not relative_string:
        raise FullFeatureExtractionError(
            f"source observation_path is malformed: sample_id={sample_id}"
        )
    if "\\" in relative_string:
        raise FullFeatureExtractionError(
            f"source observation_path is not POSIX: {relative_string!r}"
        )
    relative_path = PurePosixPath(relative_string)
    expected_relative = PurePosixPath("observations", f"{sample_id}.npz")
    if relative_path.is_absolute() or relative_path != expected_relative:
        raise FullFeatureExtractionError(
            f"source observation_path is not canonical: {relative_string!r}"
        )
    try:
        resolved = (manifest_parent / Path(*relative_path.parts)).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullFeatureExtractionError(
            f"source observation does not exist: {relative_string}"
        ) from error
    if not resolved.is_file() or resolved.parent != observations_dir:
        raise FullFeatureExtractionError(
            f"source observation escapes canonical directory: {relative_string}"
        )

    try:
        observation = PilotObservation.load(resolved)
    except Exception as error:
        raise FullFeatureExtractionError(
            f"failed to load source PilotObservation: {resolved}"
        ) from error
    if (
        observation.sample_id != sample_id
        or observation.task_id != str(task_id)
        or observation.initial_state_id != state_id
        or observation.episode_id != state_id
        or observation.step_id != step_id
        or observation.episode_success is not True
    ):
        raise FullFeatureExtractionError(
            f"source PilotObservation metadata differs: {sample_id}"
        )
    image_bytes = np.ascontiguousarray(observation.base_rgb_raw).tobytes()
    source_image_hash = f"sha256:{hashlib.sha256(image_bytes).hexdigest()}"
    return SourceRecord(
        sample_id=sample_id,
        source_observation_path=relative_string,
        resolved_observation_path=resolved,
        source_image_hash=source_image_hash,
    )


def prepare_output(
    *,
    output_dir: str | Path,
    source: SourceCollection,
    spec: FeatureSpec,
) -> OutputPreparation:
    destination = Path(output_dir).expanduser().resolve(strict=False)
    manifest_path = destination / spec.manifest_filename
    features_dir = destination / "features"

    if manifest_path.exists():
        raise FullFeatureExtractionError(
            f"completed feature manifest already exists: {manifest_path}"
        )
    if destination.exists() and not destination.is_dir():
        raise FullFeatureExtractionError(
            f"output path is not a directory: {destination}"
        )
    if destination.exists():
        unexpected_entries = [
            path for path in destination.iterdir() if path.name != "features"
        ]
        if unexpected_entries:
            raise FullFeatureExtractionError(
                "output directory contains unexpected entries: "
                f"{sorted(path.name for path in unexpected_entries)}"
            )
    else:
        try:
            destination.mkdir(parents=True)
        except OSError as error:
            raise FullFeatureExtractionError(
                f"failed to create output directory: {destination}"
            ) from error

    if features_dir.exists() and not features_dir.is_dir():
        raise FullFeatureExtractionError(
            f"features path is not a directory: {features_dir}"
        )
    try:
        features_dir.mkdir(exist_ok=True)
    except OSError as error:
        raise FullFeatureExtractionError(
            f"failed to create features directory: {features_dir}"
        ) from error

    records_by_id = {record.sample_id: record for record in source.records}
    try:
        entries = tuple(features_dir.iterdir())
    except OSError as error:
        raise FullFeatureExtractionError(
            f"failed to enumerate features directory: {features_dir}"
        ) from error
    non_archives = [
        path.name for path in entries if not path.is_file() or path.suffix != ".npz"
    ]
    if non_archives:
        raise FullFeatureExtractionError(
            f"features directory contains unexpected entries: {sorted(non_archives)}"
        )

    existing_by_id: dict[str, Path] = {}
    for path in entries:
        sample_id = path.stem
        if sample_id not in records_by_id:
            raise FullFeatureExtractionError(f"unexpected feature archive: {path}")
        expected_path = features_dir / f"{sample_id}.npz"
        if path != expected_path or sample_id in existing_by_id:
            raise FullFeatureExtractionError(
                f"non-canonical or duplicate feature archive: {path}"
            )
        existing_by_id[sample_id] = path

    missing: list[SourceRecord] = []
    for record in source.records:
        existing_path = existing_by_id.get(record.sample_id)
        if existing_path is None:
            missing.append(record)
        else:
            validate_feature_archive(existing_path, record, spec)

    return OutputPreparation(
        output_dir=destination,
        features_dir=features_dir,
        manifest_path=manifest_path,
        missing_records=tuple(missing),
        reused_count=len(source.records) - len(missing),
    )


def validate_complete_output(
    *,
    preparation: OutputPreparation,
    source: SourceCollection,
    spec: FeatureSpec,
) -> tuple[Path, ...]:
    expected_paths = tuple(
        preparation.features_dir / f"{record.sample_id}.npz"
        for record in source.records
    )
    try:
        entries = tuple(preparation.features_dir.iterdir())
    except OSError as error:
        raise FullFeatureExtractionError(
            f"failed to enumerate completed features: {preparation.features_dir}"
        ) from error
    non_archives = [
        path.name for path in entries if not path.is_file() or path.suffix != ".npz"
    ]
    if non_archives:
        raise FullFeatureExtractionError(
            "completed features directory contains unexpected entries: "
            f"{sorted(non_archives)}"
        )
    actual_paths = tuple(sorted(entries, key=lambda path: path.name))
    if set(actual_paths) != set(expected_paths):
        missing = sorted(str(path) for path in set(expected_paths) - set(actual_paths))
        unexpected = sorted(
            str(path) for path in set(actual_paths) - set(expected_paths)
        )
        raise FullFeatureExtractionError(
            "completed feature archive set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for path, record in zip(expected_paths, source.records, strict=True):
        validate_feature_archive(path, record, spec)
    return expected_paths


def validate_feature_archive(
    path: Path,
    source_record: SourceRecord,
    spec: FeatureSpec,
) -> None:
    expected_fields = {"metadata_json", *(node.archive_key for node in spec.nodes)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields or len(archive.files) != len(
                expected_fields
            ):
                missing = sorted(expected_fields - set(archive.files))
                unexpected = sorted(set(archive.files) - expected_fields)
                raise FullFeatureExtractionError(
                    f"invalid feature archive fields in {path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            metadata_value = archive["metadata_json"]
            if metadata_value.ndim != 0 or metadata_value.dtype.kind != "U":
                raise FullFeatureExtractionError(
                    f"metadata_json must be a Unicode scalar: {path}"
                )
            metadata = json.loads(str(metadata_value.item()))
            if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
                raise FullFeatureExtractionError(
                    f"invalid feature metadata fields: {path}"
                )
            expected_metadata = {
                "sample_id": source_record.sample_id,
                "source_model": spec.source_model,
                "checkpoint": spec.checkpoint_identity,
                "feature_schema_version": spec.feature_schema_version,
                "source_image_hash": source_record.source_image_hash,
            }
            if metadata != expected_metadata:
                raise FullFeatureExtractionError(
                    f"feature metadata differs for {source_record.sample_id}: "
                    f"expected={expected_metadata!r}, actual={metadata!r}"
                )
            if not _HASH_PATTERN.fullmatch(metadata["source_image_hash"]):
                raise FullFeatureExtractionError(f"invalid source_image_hash in {path}")
            for node in spec.nodes:
                value = archive[node.archive_key]
                if value.shape != node.shape:
                    raise FullFeatureExtractionError(
                        f"invalid {node.archive_key} shape in {path}: "
                        f"expected={node.shape}, actual={value.shape}"
                    )
                if value.dtype != np.float32:
                    raise FullFeatureExtractionError(
                        f"invalid {node.archive_key} dtype in {path}: {value.dtype}"
                    )
                if not np.all(np.isfinite(value)):
                    raise FullFeatureExtractionError(
                        f"non-finite {node.archive_key} values in {path}"
                    )
    except FullFeatureExtractionError:
        raise
    except Exception as error:
        raise FullFeatureExtractionError(
            f"failed to load or validate feature archive: {path}"
        ) from error


def validate_extractor_paths(
    *,
    returned_paths: Sequence[str | Path],
    missing_records: Sequence[SourceRecord],
    features_dir: Path,
) -> None:
    expected = tuple(
        features_dir / f"{record.sample_id}.npz" for record in missing_records
    )
    actual = tuple(Path(path) for path in returned_paths)
    if actual != expected:
        raise FullFeatureExtractionError(
            f"extractor returned unexpected paths: expected={expected}, actual={actual}"
        )


def build_feature_manifest(
    *,
    source: SourceCollection,
    spec: FeatureSpec,
) -> dict[str, Any]:
    extraction: dict[str, Any] = {
        "feature_checkpoint_identity": spec.checkpoint_identity,
        "runtime_precision": RUNTIME_PRECISION,
        "saved_dtype": SAVED_DTYPE,
        "batch_size": BATCH_SIZE,
        "num_tokens": NUM_TOKENS,
        "grid_shape": list(GRID_SHAPE),
        "token_order": TOKEN_ORDER,
    }
    if spec.feature_config is not None:
        extraction["feature_config"] = spec.feature_config
    return {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "pilot_version": PILOT_VERSION,
        "run_status": "COMPLETED",
        "completeness_status": "COMPLETE",
        "model_family": spec.model_family,
        "source_collection": {
            "manifest_path": str(source.manifest_path),
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "checkpoint_identity": source.checkpoint_identity,
            "libero_revision": source.libero_revision,
            "task_count": len(EXPECTED_TASK_IDS),
            "group_count": EXPECTED_GROUP_COUNT,
            "observation_count": len(source.records),
        },
        "extraction": extraction,
        "feature_nodes": {
            node.scientific_name: {
                "archive_key": node.archive_key,
                "shape": list(node.shape),
            }
            for node in spec.nodes
        },
        "records": [
            {
                "sample_id": record.sample_id,
                "source_observation_path": record.source_observation_path,
                "feature_path": f"features/{record.sample_id}.npz",
            }
            for record in source.records
        ],
    }


def write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    try:
        payload = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise FullFeatureExtractionError(
            "failed to serialize full feature manifest"
        ) from error
    if path.exists():
        raise FullFeatureExtractionError(
            f"refusing to overwrite existing feature manifest: {path}"
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FullFeatureExtractionError(
            f"refusing to overwrite existing feature manifest: {path}"
        ) from error
    except Exception as error:
        raise FullFeatureExtractionError(
            f"failed to atomically write feature manifest: {path}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullFeatureExtractionError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise FullFeatureExtractionError(f"{label} is not a file: {path}")
    return resolved


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping):
        raise FullFeatureExtractionError(f"collection manifest {key} must be an object")
    return candidate


def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = value.get(key)
    if actual != expected:
        raise FullFeatureExtractionError(
            f"invalid collection manifest {key}: "
            f"expected={expected!r}, actual={actual!r}"
        )
