from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_MANIFEST_SCHEMA_VERSION = "paired_features_v1"
_PI05_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
_METADATA_FIELDS = {
    "sample_id",
    "source_model",
    "checkpoint",
    "feature_schema_version",
    "source_image_hash",
}
_SOURCE_IMAGE_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_OPENVLA_FEATURE_SHAPES = {
    "o1_siglip": (256, 1152),
    "o1_fused": (256, 2176),
    "o2_projected": (256, 4096),
}
_PI05_FEATURE_SHAPES = {
    "p1_siglip": (256, 1152),
    "p2_projected": (256, 2048),
}


class PairedFeatureValidationError(RuntimeError):
    """Raised when C4 cannot construct a valid paired-feature manifest."""


@dataclass(frozen=True)
class _ArchiveSchema:
    label: str
    feature_shapes: dict[str, tuple[int, int]]
    source_model: str
    feature_schema_version: str
    checkpoint: str | None = None


@dataclass(frozen=True)
class _ArchiveRecord:
    path: Path
    sample_id: str
    source_image_hash: str


_OPENVLA_SCHEMA = _ArchiveSchema(
    label="OpenVLA",
    feature_shapes=_OPENVLA_FEATURE_SHAPES,
    source_model="openvla",
    feature_schema_version="openvla_features_v1",
)
_PI05_SCHEMA = _ArchiveSchema(
    label="pi0.5",
    feature_shapes=_PI05_FEATURE_SHAPES,
    source_model="pi05",
    feature_schema_version="pi05_features_v1",
    checkpoint=_PI05_CHECKPOINT,
)


def build_paired_feature_manifest(
    *,
    openvla_feature_dir: str | Path,
    pi05_feature_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Validate paired C2/C3 archives and write their deterministic manifest."""
    output = Path(output_path)
    try:
        resolved_manifest_parent = output.resolve(strict=False).parent
    except (OSError, RuntimeError) as error:
        raise PairedFeatureValidationError(
            f"failed to resolve manifest path: {output}"
        ) from error

    openvla_records = _discover_archives(
        Path(openvla_feature_dir),
        _OPENVLA_SCHEMA,
    )
    pi05_records = _discover_archives(
        Path(pi05_feature_dir),
        _PI05_SCHEMA,
    )
    sample_ids = _validate_sample_sets(openvla_records, pi05_records)
    manifest = _build_manifest(
        sample_ids,
        openvla_records,
        pi05_records,
        resolved_manifest_parent,
    )
    try:
        manifest_json = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as error:
        raise PairedFeatureValidationError(
            "failed to serialize paired-feature manifest"
        ) from error

    if output.exists():
        raise PairedFeatureValidationError(
            f"refusing to overwrite existing manifest: {output}"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PairedFeatureValidationError(
            f"failed to create manifest parent directory: {output.parent}"
        ) from error

    _write_manifest(output, manifest_json)
    return output


def _discover_archives(
    directory: Path,
    schema: _ArchiveSchema,
) -> dict[str, _ArchiveRecord]:
    try:
        resolved_directory = directory.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PairedFeatureValidationError(
            f"{schema.label} feature directory does not exist: {directory}"
        ) from error
    if not resolved_directory.is_dir():
        raise PairedFeatureValidationError(
            f"{schema.label} feature path is not a directory: {directory}"
        )

    try:
        archive_paths = tuple(
            sorted(resolved_directory.glob("*.npz"), key=lambda path: path.name)
        )
    except OSError as error:
        raise PairedFeatureValidationError(
            f"failed to enumerate {schema.label} feature directory: {directory}"
        ) from error
    if not archive_paths:
        raise PairedFeatureValidationError(
            f"{schema.label} feature directory contains no NPZ archives: {directory}"
        )

    records: dict[str, _ArchiveRecord] = {}
    for archive_path in archive_paths:
        try:
            resolved_archive = archive_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PairedFeatureValidationError(
                f"failed to resolve {schema.label} archive: {archive_path}"
            ) from error
        if not resolved_archive.is_file():
            raise PairedFeatureValidationError(
                f"{schema.label} archive path is not a file: {archive_path}"
            )
        record = _validate_archive(resolved_archive, schema)
        if record.sample_id in records:
            raise PairedFeatureValidationError(
                f"duplicate {schema.label} sample_id: {record.sample_id!r}"
            )
        records[record.sample_id] = record
    return records


def _validate_archive(path: Path, schema: _ArchiveSchema) -> _ArchiveRecord:
    expected_fields = {"metadata_json", *schema.feature_shapes}
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_fields = set(archive.files)
            if (
                actual_fields != expected_fields
                or len(archive.files) != len(expected_fields)
            ):
                missing = sorted(expected_fields - actual_fields)
                unexpected = sorted(actual_fields - expected_fields)
                raise PairedFeatureValidationError(
                    f"invalid {schema.label} archive fields in {path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )

            metadata = _parse_metadata(
                archive["metadata_json"],
                path,
                schema,
            )
            for name, expected_shape in schema.feature_shapes.items():
                _validate_feature_array(
                    archive[name],
                    name,
                    expected_shape,
                    path,
                    schema,
                )
    except PairedFeatureValidationError:
        raise
    except Exception as error:
        raise PairedFeatureValidationError(
            f"failed to load or validate {schema.label} archive: {path}"
        ) from error

    return _ArchiveRecord(
        path=path,
        sample_id=metadata["sample_id"],
        source_image_hash=metadata["source_image_hash"],
    )


def _parse_metadata(
    value: np.ndarray,
    path: Path,
    schema: _ArchiveSchema,
) -> dict[str, Any]:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise PairedFeatureValidationError(
            f"{schema.label} metadata_json must be a Unicode scalar: {path}"
        )
    try:
        metadata = json.loads(str(value.item()))
    except (TypeError, ValueError) as error:
        raise PairedFeatureValidationError(
            f"malformed {schema.label} metadata_json: {path}"
        ) from error
    if not isinstance(metadata, dict):
        raise PairedFeatureValidationError(
            f"{schema.label} metadata_json must encode an object: {path}"
        )
    actual_fields = set(metadata)
    if actual_fields != _METADATA_FIELDS:
        missing = sorted(_METADATA_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _METADATA_FIELDS)
        raise PairedFeatureValidationError(
            f"invalid {schema.label} metadata fields in {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    sample_id = metadata["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise PairedFeatureValidationError(
            f"{schema.label} sample_id must be a non-empty string: {path}"
        )
    if metadata["source_model"] != schema.source_model:
        raise PairedFeatureValidationError(
            f"invalid {schema.label} source_model in {path}: "
            f"{metadata['source_model']!r}"
        )
    if metadata["feature_schema_version"] != schema.feature_schema_version:
        raise PairedFeatureValidationError(
            f"invalid {schema.label} feature_schema_version in {path}: "
            f"{metadata['feature_schema_version']!r}"
        )

    checkpoint = metadata["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise PairedFeatureValidationError(
            f"{schema.label} checkpoint must be a non-empty string: {path}"
        )
    if schema.checkpoint is not None and checkpoint != schema.checkpoint:
        raise PairedFeatureValidationError(
            f"invalid {schema.label} checkpoint in {path}: {checkpoint!r}; "
            f"expected {schema.checkpoint!r}"
        )

    source_image_hash = metadata["source_image_hash"]
    if not isinstance(source_image_hash, str) or not (
        _SOURCE_IMAGE_HASH_PATTERN.fullmatch(source_image_hash)
    ):
        raise PairedFeatureValidationError(
            f"invalid {schema.label} source_image_hash in {path}: "
            f"{source_image_hash!r}"
        )
    return metadata


def _validate_feature_array(
    value: np.ndarray,
    name: str,
    expected_shape: tuple[int, int],
    path: Path,
    schema: _ArchiveSchema,
) -> None:
    if value.shape != expected_shape:
        raise PairedFeatureValidationError(
            f"invalid {schema.label} {name} shape in {path}: "
            f"expected={expected_shape}, actual={value.shape}"
        )
    if value.dtype != np.float32:
        raise PairedFeatureValidationError(
            f"invalid {schema.label} {name} dtype in {path}: "
            f"expected=float32, actual={value.dtype}"
        )
    if not np.all(np.isfinite(value)):
        raise PairedFeatureValidationError(
            f"non-finite {schema.label} {name} feature values in {path}"
        )


def _validate_sample_sets(
    openvla_records: dict[str, _ArchiveRecord],
    pi05_records: dict[str, _ArchiveRecord],
) -> tuple[str, ...]:
    openvla_ids = set(openvla_records)
    pi05_ids = set(pi05_records)
    if openvla_ids != pi05_ids:
        missing_in_openvla = sorted(pi05_ids - openvla_ids)
        missing_in_pi05 = sorted(openvla_ids - pi05_ids)
        raise PairedFeatureValidationError(
            "feature sample sets differ: "
            f"missing_in_openvla={missing_in_openvla}, "
            f"missing_in_pi05={missing_in_pi05}"
        )
    return tuple(sorted(openvla_ids))


def _build_manifest(
    sample_ids: tuple[str, ...],
    openvla_records: dict[str, _ArchiveRecord],
    pi05_records: dict[str, _ArchiveRecord],
    resolved_manifest_parent: Path,
) -> dict[str, Any]:
    pairs: list[dict[str, str]] = []
    for sample_id in sample_ids:
        openvla_record = openvla_records[sample_id]
        pi05_record = pi05_records[sample_id]
        if openvla_record.source_image_hash != pi05_record.source_image_hash:
            raise PairedFeatureValidationError(
                f"source_image_hash mismatch for sample_id={sample_id!r}: "
                f"OpenVLA={openvla_record.source_image_hash!r}, "
                f"pi0.5={pi05_record.source_image_hash!r}"
            )
        pairs.append(
            {
                "sample_id": sample_id,
                "source_image_hash": openvla_record.source_image_hash,
                "openvla_feature_path": _relative_posix_path(
                    openvla_record.path,
                    resolved_manifest_parent,
                ),
                "pi05_feature_path": _relative_posix_path(
                    pi05_record.path,
                    resolved_manifest_parent,
                ),
            }
        )
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "num_samples": len(pairs),
        "pairs": pairs,
    }


def _relative_posix_path(archive: Path, manifest_parent: Path) -> str:
    try:
        relative = Path(os.path.relpath(archive, start=manifest_parent))
    except (OSError, ValueError) as error:
        raise PairedFeatureValidationError(
            f"cannot construct relative manifest path for archive: {archive}"
        ) from error
    if relative.is_absolute():
        raise PairedFeatureValidationError(
            f"relative manifest path unexpectedly became absolute: {archive}"
        )
    return relative.as_posix()


def _write_manifest(path: Path, manifest_json: str) -> None:
    created_identity: tuple[int, int] | None = None
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            stat = os.fstat(output.fileno())
            created_identity = (stat.st_dev, stat.st_ino)
            output.write(manifest_json)
            output.write("\n")
    except FileExistsError as error:
        raise PairedFeatureValidationError(
            f"refusing to overwrite existing manifest: {path}"
        ) from error
    except Exception as error:
        _remove_incomplete_manifest(path, created_identity)
        raise PairedFeatureValidationError(
            f"failed to write paired-feature manifest: {path}"
        ) from error


def _remove_incomplete_manifest(
    path: Path,
    created_identity: tuple[int, int] | None,
) -> None:
    if created_identity is None:
        return
    try:
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) == created_identity:
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
