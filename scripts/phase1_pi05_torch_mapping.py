from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c3_full_feature_extraction as c3  # noqa: E402
from scripts import c4_full_paired_features as c4  # noqa: E402
from scripts import c5b_explicit_shared_space as c5b  # noqa: E402
from scripts import c5bm_authoritative_mapping as c5bm  # noqa: E402


MATERIALIZATION_ID = "phase1_o2_p2_pi05_torch_v1"
METADATA_SCHEMA_VERSION = "phase1_mapping_metadata_v1"
VALIDATION_SCHEMA_VERSION = "phase1_mapping_validation_v1"
OUTPUT_FILES = {"mapping.npz", "metadata.json", "validation.json", "summary.md"}
HISTORICAL_FILES = {"mapping.npz", "metadata.json", "validation.json", "summary.md"}
PI05_EXTRACTION_IDENTITY = "pi05_libero:PI0Pytorch"


class Phase1MappingError(RuntimeError):
    """Raised when the current-backend authoritative mapping cannot be frozen."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the Phase 1 O2/P2 mapping after current PI0Pytorch C5-B."
        )
    )
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--pi05-feature-manifest", type=Path, required=True)
    parser.add_argument("--c5b-output-dir", type=Path, required=True)
    parser.add_argument("--historical-c5bm-dir", type=Path, required=True)
    parser.add_argument("--pi05-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve_file(path: str | Path, label: str) -> Path:
    try:
        value = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise Phase1MappingError(f"{label} does not exist: {path}") from error
    if not value.is_file():
        raise Phase1MappingError(f"{label} is not a regular file: {value}")
    return value


def _resolve_directory(path: str | Path, label: str) -> Path:
    try:
        value = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise Phase1MappingError(f"{label} does not exist: {path}") from error
    if not value.is_dir():
        raise Phase1MappingError(f"{label} is not a directory: {value}")
    return value


def _validate_output_admission(path: str | Path, historical: Path) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    if output == historical:
        raise Phase1MappingError("output directory cannot be the historical artifact")
    if output.exists():
        if not output.is_dir():
            raise Phase1MappingError(f"output path is not a directory: {output}")
        if tuple(output.iterdir()):
            raise Phase1MappingError(f"output directory must be empty: {output}")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Phase1MappingError(f"failed to hash file: {path}") from error
    return f"sha256:{digest.hexdigest()}"


def _hash_required_directory(
    directory: Path,
    filenames: set[str],
    *,
    label: str,
) -> dict[str, str]:
    try:
        entries = {entry.name: entry for entry in directory.iterdir()}
    except OSError as error:
        raise Phase1MappingError(f"failed to inspect {label}: {directory}") from error
    if set(entries) != filenames or any(not entry.is_file() for entry in entries.values()):
        raise Phase1MappingError(
            f"{label} must contain exactly {sorted(filenames)}"
        )
    return {name: _sha256_file(entries[name]) for name in sorted(filenames)}


def _git_head(repository: Path, label: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase1MappingError(f"failed to identify {label} commit") from error
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise Phase1MappingError(f"invalid {label} commit: {commit!r}")
    return commit


def _validate_pi05_manifest(
    path: Path,
    records: Sequence[Any],
) -> dict[str, Any]:
    if c3.SPEC.feature_config != PI05_EXTRACTION_IDENTITY:
        raise Phase1MappingError("C3 no longer identifies the current PI0Pytorch path")
    try:
        loaded = c4._load_feature_manifest(path, label="pi0.5", spec=c3.SPEC)
        document = json.loads(loaded.path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Phase1MappingError("invalid current PI0Pytorch feature manifest") from error
    if document.get("extraction", {}).get("feature_config") != PI05_EXTRACTION_IDENTITY:
        raise Phase1MappingError("pi0.5 feature manifest is not from PI0Pytorch")
    expected = tuple(
        (record.sample_id, record.pi05_feature_path.resolve()) for record in records
    )
    actual = tuple(
        (record.sample_id, record.feature_path.resolve()) for record in loaded.records
    )
    if actual != expected:
        raise Phase1MappingError(
            "pi0.5 feature manifest order or paths differ from paired manifest"
        )
    return document


def _validate_checkpoint(path: Path) -> tuple[Path, Path]:
    checkpoint = _resolve_directory(path, "PI0Pytorch checkpoint")
    weights = _resolve_file(checkpoint / "model.safetensors", "PI0Pytorch weights")
    if not (checkpoint / "assets").is_dir():
        raise Phase1MappingError("PI0Pytorch checkpoint assets directory is missing")
    return checkpoint, weights


def _source_feature_provenance(records: Sequence[Any]) -> list[dict[str, str]]:
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
    provenance: Sequence[Mapping[str, str]],
) -> None:
    changed = [
        record["sample_id"]
        for record in provenance
        if _sha256_file(Path(record["openvla_feature_path"]))
        != record["openvla_feature_hash"]
        or _sha256_file(Path(record["pi05_feature_path"]))
        != record["pi05_feature_hash"]
    ]
    if changed:
        raise Phase1MappingError(f"source feature artifacts changed: {changed}")


def _current_primary_result(analysis: Any) -> dict[str, Any]:
    primary = analysis.alignment_summary.get("results", {}).get("o2_p2__99")
    if not isinstance(primary, dict):
        raise Phase1MappingError("current C5-B primary result is missing")
    null = primary.get("null_summary", {}).get("top5mean")
    if not isinstance(null, dict):
        raise Phase1MappingError("current C5-B primary null summary is missing")
    if (
        analysis.alignment_summary.get("c5b_result") != "PASS"
        or float(primary.get("heldout_top5mean", 0.0)) <= 0.0
        or float(null.get("empirical_p", 1.0)) > 0.05
    ):
        raise Phase1MappingError(
            "current PI0Pytorch held-out C5-B result is not scientifically meaningful"
        )
    return primary


def _validate_fit_against_analysis(
    fit: Any,
    primary: Mapping[str, Any],
) -> dict[str, dict[str, float | bool]]:
    checks: dict[str, dict[str, float | bool]] = {}
    for name in (
        "train_top5mean",
        "heldout_top1",
        "heldout_top5mean",
        "heldout_top10mean",
    ):
        fitted = float(fit.metrics[name])
        recorded = float(primary[name])
        difference = abs(fitted - recorded)
        passed = difference <= c5bm.SCALAR_TOLERANCE
        checks[name] = {
            "fitted": fitted,
            "recorded_c5b": recorded,
            "absolute_difference": difference,
            "passed": passed,
        }
        if not passed:
            raise Phase1MappingError(
                f"mapping refit differs from current C5-B for {name}: {difference}"
            )
    return checks


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_hash": c5bm._array_hash(value),
        }
        for name, value in sorted(arrays.items())
    }


def _build_metadata(
    *,
    manifest_path: Path,
    pi05_manifest_path: Path,
    analysis: Any,
    historical_dir: Path,
    historical_hashes: Mapping[str, str],
    checkpoint: Path,
    weights: Path,
    openpi_root: Path,
    source_features: list[dict[str, str]],
    fit: Any,
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "mapping_schema_version": c5bm.MAPPING_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "materialization_id": MATERIALIZATION_ID,
        "fit_configuration": {
            **c5bm.FIT_CONFIGURATION,
            "pi05_backend": "PI0Pytorch",
            "pi05_config": c3.CONFIG_NAME,
            "pi05_extraction_identity": PI05_EXTRACTION_IDENTITY,
        },
        "mapping_identity": {
            "openvla_retained_pca_dimensions": fit.pca_a.d_99,
            "pi05_retained_pca_dimensions": fit.pca_b.d_99,
            "canonical_component_count": int(fit.arrays["sigma"].size),
            "canonical_order": "descending_numpy_svd_order",
            "paired_component_arrays": ["w_a", "w_b", "sigma"],
            "sign_rule_version": "c5bm_concat_native_readout_max_abs_smallest_index_v1",
            "sign_anchors": list(fit.sign_anchors),
        },
        "arrays": _array_metadata(fit.arrays),
        "metrics": {
            "train_top5mean": float(primary["train_top5mean"]),
            "heldout_top1": float(primary["heldout_top1"]),
            "heldout_top5mean": float(primary["heldout_top5mean"]),
            "heldout_top10mean": float(primary["heldout_top10mean"]),
            "heldout_null_summary": primary["null_summary"],
        },
        "p2_extraction": {
            "backend": "PI0Pytorch",
            "config": "pi05_libero",
            "input_observation": "frozen Pilot v0.2 PilotObservation",
            "camera_mapping": "base_rgb_raw -> base_0_rgb",
            "client_orientation": "rotate_180_degrees",
            "client_resize": "resize_with_pad_224x224_then_uint8",
            "image_slot_order": [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
            ],
            "image_masks": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "model_preprocessing": "PI0Pytorch._preprocess_observation(train=False)",
            "node": "paligemma_with_expert.embed_image(base_0_rgb)",
            "official_prefix_check": "bitwise_equal_to_embed_prefix_base_slice",
            "shape": [256, 2048],
            "native_dtype": "torch.bfloat16",
            "serialized_dtype": "float32",
            "batch_size": 1,
            "token_order": "model_native_spatial_flatten_order",
        },
        "provenance": {
            "source_paired_manifest": {
                "runtime_path": str(manifest_path),
                "content_hash": _sha256_file(manifest_path),
            },
            "pi05_feature_manifest": {
                "runtime_path": str(pi05_manifest_path),
                "content_hash": _sha256_file(pi05_manifest_path),
            },
            "current_c5b_files": {
                name: {
                    "runtime_path": str(analysis.paths[name]),
                    "content_hash": analysis.hashes[name],
                }
                for name in sorted(c5bm.HISTORICAL_FILES)
            },
            "historical_c5bm_artifact": {
                "runtime_path": str(historical_dir),
                "files": dict(historical_hashes),
            },
            "pi05_checkpoint": {
                "logical_identity": c3.CHECKPOINT_IDENTITY,
                "runtime_path": str(checkpoint),
                "model_safetensors_hash": _sha256_file(weights),
            },
            "openpi": {
                "runtime_path": str(openpi_root),
                "repository_commit": _git_head(openpi_root, "OpenPI"),
            },
            "source_feature_artifacts": source_features,
            "source_feature_validation": {
                "validated_pair_count": len(source_features),
                "result": "PASS",
            },
            "repository_commit": _git_head(PROJECT_ROOT, "project"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }


def _build_validation(
    *,
    fit: Any,
    scalar_checks: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "phase1_result": "PASS",
        "materialization_id": MATERIALIZATION_ID,
        "checks": {
            "pi05_torch_extraction_identity": True,
            "official_embed_prefix_slice_checked_during_extraction": True,
            "paired_sample_and_source_image_identity": True,
            "group_aware_split": True,
            "observation_major_token_rows_0_to_255": True,
            "train_only_pca_and_cca_fit": True,
            "heldout_without_refit": True,
            "current_c5b_gate": True,
            "mapping_array_round_trip": True,
            "source_features_unchanged": True,
            "historical_c5bm_unchanged": True,
            "transactional_publication": True,
        },
        "mapping_refit_checks": dict(scalar_checks),
        "retained_dimensions": {
            "openvla": fit.pca_a.d_99,
            "pi05": fit.pca_b.d_99,
            "canonical": int(fit.arrays["sigma"].size),
        },
    }


def _build_summary(metadata: Mapping[str, Any]) -> str:
    identity = metadata["mapping_identity"]
    metrics = metadata["metrics"]
    null = metrics["heldout_null_summary"]["top5mean"]
    return "\n".join(
        [
            "# Phase 1 PI0Pytorch O2/P2 Authoritative Mapping",
            "",
            "Phase 1 result: **PASS**",
            "",
            f"Materialization ID: `{MATERIALIZATION_ID}`",
            "",
            "- Representation pair: O2 ↔ P2",
            "- P2 backend: PI0Pytorch (`pi05_libero`)",
            "- PCA cutoff: 99% cumulative explained variance",
            "- PCA and ordinary linear CCA fit: TRAIN only",
            "- Evaluation: HELD-OUT without refit",
            f"- O2 retained dimensions: {identity['openvla_retained_pca_dimensions']}",
            f"- P2 retained dimensions: {identity['pi05_retained_pca_dimensions']}",
            f"- Canonical components: {identity['canonical_component_count']}",
            f"- TRAIN Top5Mean: {metrics['train_top5mean']:.12g}",
            f"- HELD-OUT Top5Mean: {metrics['heldout_top5mean']:.12g}",
            f"- HELD-OUT null median: {float(null['null_median']):.12g}",
            f"- HELD-OUT empirical p: {float(null['empirical_p']):.12g}",
            "",
            "This versioned artifact was fitted from the current PI0Pytorch P2 path.",
            "Historical C5-BM files were verified unchanged.",
            "",
        ]
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_published_artifact(path: Path) -> None:
    entries = {entry.name: entry for entry in path.iterdir()}
    if set(entries) != OUTPUT_FILES:
        raise Phase1MappingError("published artifact has an invalid file set")
    try:
        with np.load(entries["mapping.npz"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        c5bm._validate_mapping_arrays(arrays)
        metadata = json.loads(entries["metadata.json"].read_text(encoding="utf-8"))
        validation = json.loads(
            entries["validation.json"].read_text(encoding="utf-8")
        )
    except Exception as error:
        raise Phase1MappingError("failed to load published mapping artifact") from error
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise Phase1MappingError("published metadata schema is invalid")
    if metadata.get("materialization_id") != MATERIALIZATION_ID:
        raise Phase1MappingError("published materialization identity is invalid")
    if metadata.get("arrays") != _array_metadata(arrays):
        raise Phase1MappingError("published mapping array hashes differ")
    if validation.get("schema_version") != VALIDATION_SCHEMA_VERSION or (
        validation.get("phase1_result") != "PASS"
    ):
        raise Phase1MappingError("published validation result is invalid")
    if not entries["summary.md"].read_text(encoding="utf-8").startswith(
        "# Phase 1 PI0Pytorch O2/P2 Authoritative Mapping"
    ):
        raise Phase1MappingError("published summary has an invalid role")


def _publish(
    output: Path,
    *,
    fit: Any,
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-")
    )
    published = False
    try:
        np.savez(staging / "mapping.npz", **fit.arrays)
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "validation.json", validation)
        (staging / "summary.md").write_text(
            summary, encoding="utf-8", newline="\n"
        )
        _validate_published_artifact(staging)
        if output.exists():
            if tuple(output.iterdir()):
                raise Phase1MappingError(f"output directory became non-empty: {output}")
            output.rmdir()
        staging.rename(output)
        published = True
        _validate_published_artifact(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if published and output.exists():
            shutil.rmtree(output)
        raise


def _run(args: argparse.Namespace) -> dict[str, Any]:
    historical_dir = _resolve_directory(
        args.historical_c5bm_dir, "historical C5-BM artifact"
    )
    try:
        c5bm._validate_published_artifact(historical_dir)
    except Exception as error:
        raise Phase1MappingError(
            "historical C5-BM artifact failed its authoritative validator"
        ) from error
    historical_hashes = _hash_required_directory(
        historical_dir, HISTORICAL_FILES, label="historical C5-BM artifact"
    )
    output = _validate_output_admission(args.output_dir, historical_dir)
    manifest_path = _resolve_file(args.paired_manifest, "paired manifest")
    pi05_manifest_path = _resolve_file(
        args.pi05_feature_manifest, "pi0.5 feature manifest"
    )
    checkpoint, weights = _validate_checkpoint(args.pi05_checkpoint_dir)
    openpi_root = _resolve_directory(args.openpi_root, "OpenPI repository")

    try:
        _, records, groups, split = c5b._load_dataset_structure(manifest_path)
        c5b._validate_feature_archives(records)
        _validate_pi05_manifest(pi05_manifest_path, records)
        analysis = c5bm._load_historical_c5b(
            args.c5b_output_dir,
            manifest_path=manifest_path,
            groups=groups,
            split=split,
        )
        primary = _current_primary_result(analysis)
        source_features = _source_feature_provenance(records)
        fit = c5bm._fit_mapping(records, split)
        c5bm._validate_mapping_arrays(fit.arrays)
    except Phase1MappingError:
        raise
    except Exception as error:
        raise Phase1MappingError("failed to validate or fit current O2/P2 data") from error

    scalar_checks = _validate_fit_against_analysis(fit, primary)
    historical_after_fit = _hash_required_directory(
        historical_dir, HISTORICAL_FILES, label="historical C5-BM artifact"
    )
    if historical_after_fit != historical_hashes:
        raise Phase1MappingError("historical C5-BM artifact changed during fitting")
    if {
        name: _sha256_file(path) for name, path in analysis.paths.items()
    } != analysis.hashes:
        raise Phase1MappingError("current C5-B output changed during fitting")
    _assert_source_features_unchanged(source_features)

    metadata = _build_metadata(
        manifest_path=manifest_path,
        pi05_manifest_path=pi05_manifest_path,
        analysis=analysis,
        historical_dir=historical_dir,
        historical_hashes=historical_hashes,
        checkpoint=checkpoint,
        weights=weights,
        openpi_root=openpi_root,
        source_features=source_features,
        fit=fit,
        primary=primary,
    )
    validation = _build_validation(fit=fit, scalar_checks=scalar_checks)
    _publish(
        output,
        fit=fit,
        metadata=metadata,
        validation=validation,
        summary=_build_summary(metadata),
    )
    if _hash_required_directory(
        historical_dir, HISTORICAL_FILES, label="historical C5-BM artifact"
    ) != historical_hashes:
        raise Phase1MappingError("historical C5-BM artifact changed during publication")
    return {
        "status": "Phase 1 PI0Pytorch Mapping — COMPLETED",
        "phase1_result": "PASS",
        "materialization_id": MATERIALIZATION_ID,
        "num_pairs": len(records),
        "output_dir": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
