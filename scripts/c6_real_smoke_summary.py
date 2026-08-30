from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_RESULT_FILENAME = "openvla_results.json"
PI05_RESULT_FILENAME = "pi05_results.json"
RESULT_FILENAME = "results.json"
SUMMARY_FILENAME = "summary.md"
ALPHA = 1e-3
RNG_IMPLEMENTATION = "numpy.random.Generator(PCG64)"
OPENVLA_CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
PI05_CONFIG_NAME = "pi05_libero"
PI05_CHECKPOINT_IDENTITY = "gs://openpi-assets/checkpoints/pi05_libero"
PI05_BACKEND = "jax_nnx"
NOISE_SEED = 2026
NOISE_SHAPE = (1, 10, 32)
NOISE_DTYPE = "float32"


@dataclass(frozen=True)
class FrozenObservation:
    sample_id: str
    task_id: str
    source_image_hash: str
    direction_seeds: dict[str, int]


@dataclass(frozen=True)
class LoadedObservation:
    identity: FrozenObservation
    path: Path
    record: Any


FROZEN_OBSERVATIONS = (
    FrozenObservation(
        sample_id="libero_spatial__task00__state00__step0008",
        task_id="0",
        source_image_hash=(
            "sha256:279786daa449ca71c3e6aa2c8d6941c37814a9286d2dd2fefd53c3123390b879"
        ),
        direction_seeds={"openvla": 202700, "pi05": 202710},
    ),
    FrozenObservation(
        sample_id="libero_spatial__task01__state00__step0011",
        task_id="1",
        source_image_hash=(
            "sha256:b8c55f535389ba9e7ff7a1e106ed2dea696282b62b4202694857372c634f4f4a"
        ),
        direction_seeds={"openvla": 202701, "pi05": 202711},
    ),
)


class C6RealSmokeError(RuntimeError):
    """Raised when the frozen C6 real-smoke boundary is invalid."""


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def load_frozen_observations(
    collection_manifest: str | Path,
) -> tuple[LoadedObservation, ...]:
    from scripts import _full_feature_extraction_common as feature_common
    from shared_feature import PilotObservation

    source = feature_common.load_source_collection(collection_manifest)
    records = {record.sample_id: record for record in source.records}
    loaded: list[LoadedObservation] = []
    for identity in FROZEN_OBSERVATIONS:
        source_record = records.get(identity.sample_id)
        if source_record is None:
            raise C6RealSmokeError(
                f"frozen observation is absent from collection: {identity.sample_id}"
            )
        if source_record.source_image_hash != identity.source_image_hash:
            raise C6RealSmokeError(
                f"collection source hash mismatch for {identity.sample_id}"
            )
        record = PilotObservation.load(source_record.resolved_observation_path)
        digest = (
            "sha256:"
            + hashlib.sha256(
                np.ascontiguousarray(record.base_rgb_raw).tobytes()
            ).hexdigest()
        )
        if (
            record.sample_id != identity.sample_id
            or record.task_id != identity.task_id
            or digest != identity.source_image_hash
        ):
            raise C6RealSmokeError(
                f"frozen observation identity mismatch for {identity.sample_id}"
            )
        loaded.append(
            LoadedObservation(
                identity=identity,
                path=source_record.resolved_observation_path,
                record=record,
            )
        )
    return tuple(loaded)


def make_fixed_noise() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(NOISE_SEED))
    return rng.standard_normal(NOISE_SHAPE, dtype=np.float32)


def make_direction(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    direction = rng.standard_normal(shape, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    if not np.all(np.isfinite(direction)) or not np.any(direction):
        raise C6RealSmokeError("generated intervention direction is invalid")
    return direction


def intended_perturbation(
    clean_native_host: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean_native_host, dtype=np.float32)
    if not np.all(np.isfinite(clean)) or not np.any(clean):
        raise C6RealSmokeError("clean native feature must be finite and nonzero")
    direction = make_direction(clean.shape, seed)
    intended_delta = np.float32(ALPHA * np.linalg.norm(clean)) * direction
    intended_modified = clean + intended_delta
    if not np.all(np.isfinite(intended_modified)):
        raise C6RealSmokeError("intended modified feature is non-finite")
    return intended_delta, intended_modified


def perturbation_metrics(
    clean_native_host: np.ndarray,
    intended_delta: np.ndarray,
    actual_modified_host: np.ndarray,
) -> dict[str, Any]:
    clean = np.asarray(clean_native_host, dtype=np.float32)
    actual_modified = np.asarray(actual_modified_host, dtype=np.float32)
    if clean.shape != actual_modified.shape:
        raise C6RealSmokeError("actual modified feature shape changed")
    actual_delta = actual_modified - clean
    values = (clean, intended_delta, actual_modified, actual_delta)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise C6RealSmokeError("feature perturbation contains non-finite values")
    clean_norm = float(np.linalg.norm(clean.astype(np.float64)))
    intended_norm = float(np.linalg.norm(np.asarray(intended_delta, dtype=np.float64)))
    actual_norm = float(np.linalg.norm(actual_delta.astype(np.float64)))
    if clean_norm == 0.0:
        raise C6RealSmokeError("clean feature norm is zero")
    return {
        "clean_feature_norm": clean_norm,
        "intended_delta_norm": intended_norm,
        "actual_delta_norm": actual_norm,
        "intended_relative_perturbation": intended_norm / clean_norm,
        "actual_relative_perturbation": actual_norm / clean_norm,
        "actual_modified_differs": bool(np.any(actual_delta != 0)),
        "all_features_finite": True,
    }


def array_comparison(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    exact: bool = False,
) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    shape_equal = left.shape == right.shape
    if not shape_equal:
        return {
            "pass": False,
            "shape_equal": False,
            "max_abs_error": None,
        }
    finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    passed = (
        bool(np.array_equal(left, right))
        if exact
        else bool(np.allclose(left, right, rtol=0, atol=atol))
    )
    return {
        "pass": passed and finite,
        "shape_equal": True,
        "max_abs_error": float(difference.max(initial=0.0)),
    }


def git_commit(repository: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(repository).resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise C6RealSmokeError(f"invalid git commit returned for {repository}")
    return commit


def validate_phase_output_slot(output_dir: str | Path, filename: str) -> Path:
    target = Path(output_dir).expanduser().resolve()
    if filename not in {OPENVLA_RESULT_FILENAME, PI05_RESULT_FILENAME}:
        raise ValueError(f"unsupported phase output filename: {filename}")
    if target.exists():
        if not target.is_dir():
            raise C6RealSmokeError(f"output path is not a directory: {target}")
        entries = {path.name for path in target.iterdir()}
        counterpart = {
            OPENVLA_RESULT_FILENAME: PI05_RESULT_FILENAME,
            PI05_RESULT_FILENAME: OPENVLA_RESULT_FILENAME,
        }[filename]
        if entries - {counterpart}:
            raise C6RealSmokeError(
                f"output directory is not fresh for {filename}: {sorted(entries)}"
            )
    return target


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(value)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the two frozen C6 real-smoke phase results."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_phase(path: Path, schema_version: str, model: str) -> dict[str, Any]:
    if not path.is_file():
        raise C6RealSmokeError(f"missing phase result: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise C6RealSmokeError(f"failed to load phase result: {path}") from error
    if not isinstance(payload, dict):
        raise C6RealSmokeError(f"phase result is not an object: {path}")
    expected = {
        "schema_version": schema_version,
        "run_status": "COMPLETED",
        "model": model,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise C6RealSmokeError(
                f"invalid {model} phase field {key}: {payload.get(key)!r}"
            )
    if payload.get("model_result") not in {"PASS", "BLOCKED"}:
        raise C6RealSmokeError(f"invalid {model} model_result")
    _validate_phase_semantics(payload, model)
    return payload


def _validate_phase_semantics(payload: dict[str, Any], model: str) -> None:
    commit = payload.get("repository_commit")
    reference_commit = payload.get("reference_repository_commit")
    if (
        not isinstance(commit, str)
        or _COMMIT_PATTERN.fullmatch(commit) is None
        or not isinstance(reference_commit, str)
        or _COMMIT_PATTERN.fullmatch(reference_commit) is None
    ):
        raise C6RealSmokeError(f"invalid {model} git provenance")
    versions = payload.get("runtime_versions")
    if (
        not isinstance(versions, dict)
        or not versions
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise C6RealSmokeError(f"invalid {model} runtime versions")

    identity = payload.get("model_identity")
    protocol = payload.get("protocol")
    if not isinstance(identity, dict) or not isinstance(protocol, dict):
        raise C6RealSmokeError(f"missing {model} frozen identity/protocol")
    if model == "openvla":
        expected_identity = {
            "scientific_checkpoint": OPENVLA_CHECKPOINT_IDENTITY,
            "unnorm_key": "libero_spatial_no_noops",
            "center_crop": True,
            "do_sample": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
        }
        expected_protocol = {
            "batch_size": 1,
            "global_seed": 7,
            "alpha": ALPHA,
            "direction_rng": RNG_IMPLEMENTATION,
            "direction_dtype": "float32",
            "clean_atol": 1e-8,
            "translation_floor": 1e-8,
        }
        seed_key = "openvla"
        floor = 1e-8
    elif model == "pi05":
        expected_identity = {
            "config": PI05_CONFIG_NAME,
            "scientific_checkpoint": PI05_CHECKPOINT_IDENTITY,
            "backend": PI05_BACKEND,
        }
        expected_protocol = {
            "batch_size": 1,
            "alpha": ALPHA,
            "noise_rng": RNG_IMPLEMENTATION,
            "noise_seed": NOISE_SEED,
            "noise_dtype": NOISE_DTYPE,
            "noise_shape": list(NOISE_SHAPE),
            "direction_rng": RNG_IMPLEMENTATION,
            "direction_dtype": "float32",
            "clean_atol": 1e-6,
            "translation_floor": 1e-6,
        }
        seed_key = "pi05"
        floor = 1e-6
    else:
        raise C6RealSmokeError(f"unsupported phase model: {model}")
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise C6RealSmokeError(f"invalid {model} model identity")
    resolved_checkpoint = identity.get("resolved_checkpoint_path")
    if (
        not isinstance(resolved_checkpoint, str)
        or not Path(resolved_checkpoint).is_absolute()
    ):
        raise C6RealSmokeError(f"invalid {model} resolved checkpoint path")
    if protocol != expected_protocol:
        raise C6RealSmokeError(f"invalid {model} frozen protocol")

    expected_ids = [item.sample_id for item in FROZEN_OBSERVATIONS]
    expected_hashes = [item.source_image_hash for item in FROZEN_OBSERVATIONS]
    observations = payload.get("observations")
    if (
        not isinstance(observations, list)
        or [value.get("sample_id") for value in observations if isinstance(value, dict)]
        != expected_ids
    ):
        raise C6RealSmokeError(f"invalid {model} observation order")
    if [value.get("source_image_hash") for value in observations] != expected_hashes:
        raise C6RealSmokeError(f"invalid {model} observation hashes")
    clean_passes: list[bool] = []
    response_norms: list[float] = []
    interventions_valid = True
    for frozen, observation in zip(FROZEN_OBSERVATIONS, observations, strict=True):
        if (
            observation.get("task_id") != frozen.task_id
            or observation.get("direction_seed") != frozen.direction_seeds[seed_key]
            or not isinstance(observation.get("observation_path"), str)
            or not Path(observation["observation_path"]).is_absolute()
        ):
            raise C6RealSmokeError(f"invalid {model} observation provenance")
        clean = observation.get("clean_equivalence")
        if not isinstance(clean, dict) or type(clean.get("pass")) is not bool:
            raise C6RealSmokeError(f"invalid {model} clean-equivalence result")
        checks = clean.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or any(
                not isinstance(value, dict) or type(value.get("pass")) is not bool
                for value in checks.values()
            )
            or clean["pass"] != all(value["pass"] for value in checks.values())
        ):
            raise C6RealSmokeError(f"inconsistent {model} clean checks")
        clean_passes.append(clean["pass"])
        intervention = observation.get("intervention")
        if intervention is None:
            interventions_valid = False
            continue
        required_booleans = (
            "actual_modified_differs",
            "all_features_finite",
            "all_actions_finite",
            "continuation_consumed_supplied_native_feature",
        )
        if any(intervention.get(key) is not True for key in required_booleans):
            interventions_valid = False
        numeric_keys = (
            "clean_feature_norm",
            "intended_delta_norm",
            "actual_delta_norm",
            "intended_relative_perturbation",
            "actual_relative_perturbation",
            "translation_delta_norm",
        )
        numeric = [intervention.get(key) for key in numeric_keys]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in numeric
        ):
            raise C6RealSmokeError(f"invalid {model} intervention metrics")
        if (
            numeric[0] <= 0
            or numeric[1] < 0
            or numeric[2] < 0
            or not np.isclose(
                intervention["intended_relative_perturbation"],
                ALPHA,
                rtol=1e-5,
                atol=1e-8,
            )
        ):
            raise C6RealSmokeError(f"invalid {model} perturbation scale")
        if intervention["actual_modified_differs"] != (
            intervention["actual_delta_norm"] > 0
        ):
            raise C6RealSmokeError(f"inconsistent {model} applied perturbation")
        for key in ("clean_translation", "modified_translation", "translation_delta"):
            vector = intervention.get(key)
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    for value in vector
                )
            ):
                raise C6RealSmokeError(f"invalid {model} translation metrics")
        response_norms.append(float(intervention["translation_delta_norm"]))

    clean_gate = all(clean_passes)
    if payload.get("clean_gate") != ("PASS" if clean_gate else "BLOCKED"):
        raise C6RealSmokeError(f"inconsistent {model} clean gate")
    if not clean_gate and any(
        observation.get("intervention") is not None for observation in observations
    ):
        raise C6RealSmokeError(f"{model} intervened after a failed clean gate")
    if clean_gate and any(
        observation.get("intervention") is None for observation in observations
    ):
        raise C6RealSmokeError(f"{model} omitted an authorized intervention")
    expected_result = (
        "PASS"
        if clean_gate
        and interventions_valid
        and len(response_norms) == len(FROZEN_OBSERVATIONS)
        and any(value > floor for value in response_norms)
        else "BLOCKED"
    )
    if payload.get("model_result") != expected_result:
        raise C6RealSmokeError(f"inconsistent {model} model_result")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise C6RealSmokeError(f"phase output directory does not exist: {output_dir}")
    entries = {path.name for path in output_dir.iterdir()}
    expected = {OPENVLA_RESULT_FILENAME, PI05_RESULT_FILENAME}
    if entries != expected:
        raise C6RealSmokeError(
            f"aggregator requires exactly the two phase files: {sorted(entries)}"
        )
    openvla = _load_phase(
        output_dir / OPENVLA_RESULT_FILENAME,
        "c6_openvla_real_smoke_v1",
        "openvla",
    )
    pi05 = _load_phase(
        output_dir / PI05_RESULT_FILENAME,
        "c6_pi05_real_smoke_v1",
        "pi05",
    )
    if openvla.get("repository_commit") != pi05.get("repository_commit"):
        raise C6RealSmokeError("phase results use different repository commits")
    overall = (
        "PASS"
        if openvla["model_result"] == pi05["model_result"] == "PASS"
        else "BLOCKED"
    )
    result = {
        "schema_version": "c6_real_smoke_v1",
        "run_status": "COMPLETED",
        "overall_result": overall,
        "repository_commit": openvla["repository_commit"],
        "observation_ids": [item.sample_id for item in FROZEN_OBSERVATIONS],
        "observation_image_hashes": {
            item.sample_id: item.source_image_hash for item in FROZEN_OBSERVATIONS
        },
        "openvla": openvla,
        "pi05": pi05,
    }
    write_json_atomic(output_dir / RESULT_FILENAME, result)
    summary = "\n".join(
        [
            "# C6 Real Clean-Equivalence / Intervention Smoke",
            "",
            f"- OpenVLA: {openvla['model_result']}",
            f"- pi0.5: {pi05['model_result']}",
            f"- Overall: {overall}",
            "",
            "This smoke validates integration closure only; it does not establish policy/action relevance.",
            "",
        ]
    )
    summary_path = output_dir / SUMMARY_FILENAME
    write_text_atomic(summary_path, summary)
    return {
        "status": "C6 real-smoke aggregation — COMPLETED",
        "overall_result": overall,
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
