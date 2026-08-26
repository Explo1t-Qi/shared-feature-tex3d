from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _full_feature_extraction_common as common  # noqa: E402
from scripts import c2_full_feature_extraction as c2  # noqa: E402
from scripts import c3_full_feature_extraction as c3  # noqa: E402
from shared_feature import paired_features as paired  # noqa: E402


EXPECTED_RECORD_COUNT = 200
_MANIFEST_FIELDS = {
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
_SOURCE_COLLECTION_FIELDS = {
    "manifest_path",
    "schema_version",
    "checkpoint_identity",
    "libero_revision",
    "task_count",
    "group_count",
    "observation_count",
}
_RECORD_FIELDS = {"sample_id", "source_observation_path", "feature_path"}


@dataclass(frozen=True)
class _FeatureRecord:
    sample_id: str
    source_observation_path: str
    feature_path: Path


@dataclass(frozen=True)
class _FeatureManifest:
    path: Path
    source_collection: dict[str, Any]
    records: tuple[_FeatureRecord, ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the formal Pilot v0.2 paired-feature manifest."
    )
    parser.add_argument(
        "--openvla-feature-manifest",
        type=Path,
        required=True,
        help="Completed formal C2 OpenVLA feature manifest.",
    )
    parser.add_argument(
        "--pi05-feature-manifest",
        type=Path,
        required=True,
        help="Completed formal C3 pi0.5 feature manifest.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="New paired_features_v1 JSON manifest path.",
    )
    return parser.parse_args(argv)


def _resolve_manifest(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise paired.PairedFeatureValidationError(
            f"{label} feature manifest does not exist: {path}"
        ) from error
    if not resolved.is_file():
        raise paired.PairedFeatureValidationError(
            f"{label} feature manifest is not a file: {path}"
        )
    return resolved


def _load_feature_manifest(
    path: Path,
    *,
    label: str,
    spec: common.FeatureSpec,
) -> _FeatureManifest:
    resolved = _resolve_manifest(path, label)
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as error:
        raise paired.PairedFeatureValidationError(
            f"failed to load {label} feature manifest: {resolved}"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise paired.PairedFeatureValidationError(
            f"invalid {label} feature manifest fields: {resolved}"
        )

    expected_scalars = {
        "schema_version": common.FEATURE_MANIFEST_SCHEMA_VERSION,
        "pilot_version": common.PILOT_VERSION,
        "run_status": "COMPLETED",
        "completeness_status": "COMPLETE",
        "model_family": spec.model_family,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} feature manifest {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )

    source_collection = _validate_source_collection(
        manifest.get("source_collection"),
        label=label,
    )
    _validate_extraction(manifest.get("extraction"), label=label, spec=spec)
    _validate_feature_nodes(manifest.get("feature_nodes"), label=label, spec=spec)
    records = _load_records(
        manifest.get("records"),
        manifest_parent=resolved.parent,
        label=label,
    )
    return _FeatureManifest(
        path=resolved,
        source_collection=source_collection,
        records=records,
    )


def _validate_source_collection(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_COLLECTION_FIELDS:
        raise paired.PairedFeatureValidationError(
            f"invalid {label} source_collection fields"
        )
    expected = {
        "schema_version": common.COLLECTION_SCHEMA_VERSION,
        "checkpoint_identity": common.COLLECTION_CHECKPOINT_IDENTITY,
        "task_count": len(common.EXPECTED_TASK_IDS),
        "group_count": common.EXPECTED_GROUP_COUNT,
        "observation_count": common.EXPECTED_OBSERVATION_COUNT,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} source_collection {key}: "
                f"expected={expected_value!r}, actual={value.get(key)!r}"
            )
    if (
        not isinstance(value.get("manifest_path"), str)
        or not value["manifest_path"].strip()
    ):
        raise paired.PairedFeatureValidationError(
            f"invalid {label} source_collection manifest_path"
        )
    if (
        not isinstance(value.get("libero_revision"), str)
        or not value["libero_revision"].strip()
    ):
        raise paired.PairedFeatureValidationError(
            f"invalid {label} source_collection libero_revision"
        )
    return value


def _validate_extraction(
    value: Any,
    *,
    label: str,
    spec: common.FeatureSpec,
) -> None:
    expected: dict[str, Any] = {
        "feature_checkpoint_identity": spec.checkpoint_identity,
        "runtime_precision": common.RUNTIME_PRECISION,
        "saved_dtype": common.SAVED_DTYPE,
        "batch_size": common.BATCH_SIZE,
        "num_tokens": common.NUM_TOKENS,
        "grid_shape": list(common.GRID_SHAPE),
        "token_order": common.TOKEN_ORDER,
    }
    if spec.feature_config is not None:
        expected["feature_config"] = spec.feature_config
    if value != expected:
        raise paired.PairedFeatureValidationError(
            f"invalid {label} extraction metadata: "
            f"expected={expected!r}, actual={value!r}"
        )


def _validate_feature_nodes(
    value: Any,
    *,
    label: str,
    spec: common.FeatureSpec,
) -> None:
    expected = {
        node.scientific_name: {
            "archive_key": node.archive_key,
            "shape": list(node.shape),
        }
        for node in spec.nodes
    }
    if value != expected:
        raise paired.PairedFeatureValidationError(
            f"invalid {label} feature_nodes: expected={expected!r}, actual={value!r}"
        )


def _load_records(
    value: Any,
    *,
    manifest_parent: Path,
    label: str,
) -> tuple[_FeatureRecord, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_RECORD_COUNT:
        actual_count = len(value) if isinstance(value, list) else None
        raise paired.PairedFeatureValidationError(
            f"{label} feature manifest must contain exactly "
            f"{EXPECTED_RECORD_COUNT} records; actual={actual_count}"
        )

    records: list[_FeatureRecord] = []
    sample_ids: set[str] = set()
    feature_paths: set[Path] = set()
    features_dir = (manifest_parent / "features").resolve(strict=False)
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} record fields at index {index}"
            )
        sample_id = record.get("sample_id")
        source_path = record.get("source_observation_path")
        feature_path = record.get("feature_path")
        if not isinstance(sample_id, str) or not sample_id:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} sample_id at index {index}"
            )
        if sample_id in sample_ids:
            raise paired.PairedFeatureValidationError(
                f"duplicate {label} sample_id: {sample_id!r}"
            )
        expected_source = f"observations/{sample_id}.npz"
        if source_path != expected_source:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} source_observation_path for {sample_id!r}: "
                f"{source_path!r}"
            )
        expected_feature = f"features/{sample_id}.npz"
        if feature_path != expected_feature or "\\" in feature_path:
            raise paired.PairedFeatureValidationError(
                f"invalid {label} feature_path for {sample_id!r}: {feature_path!r}"
            )
        relative = PurePosixPath(feature_path)
        if relative.is_absolute():
            raise paired.PairedFeatureValidationError(
                f"absolute {label} feature_path for {sample_id!r}: {feature_path!r}"
            )
        try:
            resolved_feature = (manifest_parent / Path(*relative.parts)).resolve(
                strict=True
            )
        except (OSError, RuntimeError) as error:
            raise paired.PairedFeatureValidationError(
                f"unresolved {label} feature_path for {sample_id!r}: {feature_path!r}"
            ) from error
        if not resolved_feature.is_file() or resolved_feature.parent != features_dir:
            raise paired.PairedFeatureValidationError(
                f"{label} feature_path escapes features directory: {feature_path!r}"
            )
        if resolved_feature in feature_paths:
            raise paired.PairedFeatureValidationError(
                f"duplicate {label} feature archive path: {feature_path!r}"
            )
        sample_ids.add(sample_id)
        feature_paths.add(resolved_feature)
        records.append(
            _FeatureRecord(
                sample_id=sample_id,
                source_observation_path=source_path,
                feature_path=resolved_feature,
            )
        )
    return tuple(records)


def _validate_shared_collection(
    openvla: _FeatureManifest,
    pi05: _FeatureManifest,
) -> None:
    keys = _SOURCE_COLLECTION_FIELDS - {"manifest_path"}
    openvla_identity = {key: openvla.source_collection[key] for key in keys}
    pi05_identity = {key: pi05.source_collection[key] for key in keys}
    if openvla_identity != pi05_identity:
        raise paired.PairedFeatureValidationError(
            "feature manifests refer to different source collections: "
            f"OpenVLA={openvla_identity!r}, pi0.5={pi05_identity!r}"
        )


def _validate_record_alignment(
    openvla: _FeatureManifest,
    pi05: _FeatureManifest,
) -> None:
    openvla_ids = tuple(record.sample_id for record in openvla.records)
    pi05_ids = tuple(record.sample_id for record in pi05.records)
    if set(openvla_ids) != set(pi05_ids):
        raise paired.PairedFeatureValidationError(
            "feature manifest sample sets differ: "
            f"missing_in_openvla={sorted(set(pi05_ids) - set(openvla_ids))}, "
            f"missing_in_pi05={sorted(set(openvla_ids) - set(pi05_ids))}"
        )
    if openvla_ids != pi05_ids:
        raise paired.PairedFeatureValidationError(
            "feature manifest canonical record order differs"
        )
    for openvla_record, pi05_record in zip(
        openvla.records,
        pi05.records,
        strict=True,
    ):
        if openvla_record.source_observation_path != (
            pi05_record.source_observation_path
        ):
            raise paired.PairedFeatureValidationError(
                "source_observation_path mismatch for "
                f"sample_id={openvla_record.sample_id!r}: "
                f"OpenVLA={openvla_record.source_observation_path!r}, "
                f"pi0.5={pi05_record.source_observation_path!r}"
            )


def _validate_archive(
    record: _FeatureRecord,
    *,
    spec: common.FeatureSpec,
    archive_schema: paired._ArchiveSchema,
) -> paired._ArchiveRecord:
    archive_record = paired._validate_archive(record.feature_path, archive_schema)
    if archive_record.sample_id != record.sample_id:
        raise paired.PairedFeatureValidationError(
            f"feature archive sample_id differs from manifest record: "
            f"expected={record.sample_id!r}, actual={archive_record.sample_id!r}"
        )
    source_record = common.SourceRecord(
        sample_id=record.sample_id,
        source_observation_path=record.source_observation_path,
        resolved_observation_path=record.feature_path,
        source_image_hash=archive_record.source_image_hash,
    )
    try:
        common.validate_feature_archive(record.feature_path, source_record, spec)
    except common.FullFeatureExtractionError as error:
        raise paired.PairedFeatureValidationError(str(error)) from error
    return archive_record


def _build_formal_manifest(
    *,
    openvla: _FeatureManifest,
    pi05: _FeatureManifest,
    output_parent: Path,
) -> dict[str, Any]:
    pairs: list[dict[str, str]] = []
    for openvla_record, pi05_record in zip(
        openvla.records,
        pi05.records,
        strict=True,
    ):
        openvla_archive = _validate_archive(
            openvla_record,
            spec=c2.SPEC,
            archive_schema=paired._OPENVLA_SCHEMA,
        )
        pi05_archive = _validate_archive(
            pi05_record,
            spec=c3.SPEC,
            archive_schema=paired._PI05_SCHEMA,
        )
        if openvla_archive.source_image_hash != pi05_archive.source_image_hash:
            raise paired.PairedFeatureValidationError(
                "source_image_hash mismatch for "
                f"sample_id={openvla_record.sample_id!r}: "
                f"OpenVLA={openvla_archive.source_image_hash!r}, "
                f"pi0.5={pi05_archive.source_image_hash!r}"
            )
        pairs.append(
            {
                "sample_id": openvla_record.sample_id,
                "source_image_hash": openvla_archive.source_image_hash,
                "openvla_feature_path": paired._relative_posix_path(
                    openvla_record.feature_path,
                    output_parent,
                ),
                "pi05_feature_path": paired._relative_posix_path(
                    pi05_record.feature_path,
                    output_parent,
                ),
            }
        )
    return {
        "schema_version": paired._MANIFEST_SCHEMA_VERSION,
        "num_samples": len(pairs),
        "pairs": pairs,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output_path).expanduser().resolve(strict=False)
    if output_path.exists():
        raise paired.PairedFeatureValidationError(
            f"refusing to overwrite existing manifest: {output_path}"
        )

    openvla = _load_feature_manifest(
        Path(args.openvla_feature_manifest),
        label="OpenVLA",
        spec=c2.SPEC,
    )
    pi05 = _load_feature_manifest(
        Path(args.pi05_feature_manifest),
        label="pi0.5",
        spec=c3.SPEC,
    )
    _validate_shared_collection(openvla, pi05)
    _validate_record_alignment(openvla, pi05)
    manifest = _build_formal_manifest(
        openvla=openvla,
        pi05=pi05,
        output_parent=output_path.parent,
    )

    try:
        manifest_json = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as error:
        raise paired.PairedFeatureValidationError(
            "failed to serialize formal paired-feature manifest"
        ) from error
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise paired.PairedFeatureValidationError(
            f"failed to create manifest parent directory: {output_path.parent}"
        ) from error
    paired._write_manifest(output_path, manifest_json)
    return {
        "status": "C4 Formal Paired Feature Materialization — COMPLETE",
        "manifest_path": str(output_path),
        "num_pairs": len(manifest["pairs"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
