from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c6_real_smoke_summary as common  # noqa: E402


CONFIG_NAME = common.PI05_CONFIG_NAME
CHECKPOINT_IDENTITY = common.PI05_CHECKPOINT_IDENTITY
BACKEND = common.PI05_BACKEND
DEFAULT_OPENPI_ROOT = PROJECT_ROOT.parent / "openpi"
DEFAULT_CHECKPOINT_DIR = Path(
    "/data/xiaomengqi/checkpoints/pi05_libero/openpi-assets/checkpoints/pi05_libero"
)
RESULT_FILENAME = common.PI05_RESULT_FILENAME
CLEAN_ATOL = 1e-6
TRANSLATION_FLOOR = 1e-6


@dataclass(frozen=True)
class _Runtime:
    jax: Any
    jnp: Any
    image_tools: Any
    policy: Any
    checkpoint_path: Path
    prepare_context: Callable[..., Any]
    run_reference: Callable[..., Any]
    continue_from_p2: Callable[..., Any]
    versions: dict[str, str]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen real pi0.5 C6 clean/intervention smoke."
    )
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    openpi_root = args.openpi_root.expanduser().resolve()
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"OpenPI source package not found: {openpi_root}")
    if not (
        openpi_root / "packages" / "openpi-client" / "src" / "openpi_client"
    ).is_dir():
        raise FileNotFoundError(f"OpenPI client package not found: {openpi_root}")
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"pi0.5 JAX params not found: {checkpoint / 'params'}")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(
            f"pi0.5 checkpoint assets not found: {checkpoint / 'assets'}"
        )
    if (checkpoint / "model.safetensors").exists():
        raise RuntimeError("frozen pi0.5 smoke requires the JAX/NNX checkpoint backend")
    return openpi_root, checkpoint


def _load_runtime(openpi_root: Path, checkpoint: Path) -> _Runtime:
    source_roots = (
        PROJECT_ROOT,
        openpi_root / "packages" / "openpi-client" / "src",
        openpi_root / "src",
    )
    for source_root in source_roots:
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

    import jax
    import jax.numpy as jnp
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools
    from shared_feature import (
        continue_pi05_from_p2,
        prepare_pi05_context,
        run_pi05_reference,
    )

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"pi0.5 JAX backend is not GPU: {jax.default_backend()}")
    train_config = config.get_config(CONFIG_NAME)
    model_config = train_config.model
    if (
        train_config.name != CONFIG_NAME
        or model_config.pi05 is not True
        or model_config.action_horizon != 10
        or model_config.action_dim != 32
        or model_config.discrete_state_input is not False
    ):
        raise RuntimeError("local pi05_libero TrainConfig violates frozen semantics")
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    if getattr(policy, "_is_pytorch_model", None) is not False:
        raise RuntimeError("loaded pi0.5 policy is not the frozen JAX/NNX backend")

    return _Runtime(
        jax=jax,
        jnp=jnp,
        image_tools=image_tools,
        policy=policy,
        checkpoint_path=checkpoint,
        prepare_context=prepare_pi05_context,
        run_reference=run_pi05_reference,
        continue_from_p2=continue_pi05_from_p2,
        versions={
            "python": platform.python_version(),
            "jax": str(jax.__version__),
            "jaxlib": _package_version("jaxlib"),
            "flax": _package_version("flax"),
            "jax_backend": str(jax.default_backend()),
            "jax_devices": ",".join(str(device) for device in jax.devices()),
        },
    )


def _policy_input(record: Any, image_tools: Any) -> dict[str, Any]:
    def prepare_image(value: np.ndarray) -> np.ndarray:
        rotated = np.ascontiguousarray(value[::-1, ::-1])
        resized = image_tools.resize_with_pad(rotated, 224, 224)
        return image_tools.convert_to_uint8(resized)

    return {
        "observation/image": prepare_image(record.base_rgb_raw),
        "observation/wrist_image": prepare_image(record.wrist_rgb_raw),
        "observation/state": record.state.copy(),
        "prompt": record.prompt,
    }


def _clean_report(reference: Any, continued: Any) -> dict[str, Any]:
    checks = {
        "normalized_action_chunk_32": common.array_comparison(
            reference.normalized_action_chunk_32,
            continued.normalized_action_chunk_32,
            atol=CLEAN_ATOL,
        ),
        "normalized_action_chunk": common.array_comparison(
            reference.normalized_action_chunk,
            continued.normalized_action_chunk,
            atol=CLEAN_ATOL,
        ),
        "unnormalized_action_chunk": common.array_comparison(
            reference.unnormalized_action_chunk,
            continued.unnormalized_action_chunk,
            atol=CLEAN_ATOL,
        ),
    }
    return {
        "pass": all(value["pass"] for value in checks.values()),
        "checks": checks,
    }


def _intervention_report(
    *,
    clean_result: Any,
    modified_result: Any,
    perturbation: dict[str, Any],
) -> dict[str, Any]:
    clean_translation = np.asarray(
        clean_result.unnormalized_action_chunk[0, 0, :3], dtype=np.float64
    )
    modified_translation = np.asarray(
        modified_result.unnormalized_action_chunk[0, 0, :3], dtype=np.float64
    )
    delta = modified_translation - clean_translation
    arrays = (
        clean_result.normalized_action_chunk_32,
        clean_result.normalized_action_chunk,
        clean_result.unnormalized_action_chunk,
        modified_result.normalized_action_chunk_32,
        modified_result.normalized_action_chunk,
        modified_result.unnormalized_action_chunk,
    )
    all_actions_finite = all(np.all(np.isfinite(value)) for value in arrays)
    return {
        **perturbation,
        "clean_translation": clean_translation.tolist(),
        "modified_translation": modified_translation.tolist(),
        "translation_delta": delta.tolist(),
        "translation_delta_norm": float(np.linalg.norm(delta)),
        "all_actions_finite": bool(all_actions_finite),
        "continuation_consumed_supplied_native_feature": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = common.validate_phase_output_slot(args.output_dir, RESULT_FILENAME)
    observations = common.load_frozen_observations(args.collection_manifest)
    openpi_root, checkpoint = _validate_paths(args)
    runtime = _load_runtime(openpi_root, checkpoint)
    fixed_noise = common.make_fixed_noise()

    prepared_records: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    for loaded in observations:
        prepared = runtime.prepare_context(
            policy=runtime.policy,
            observation=_policy_input(loaded.record, runtime.image_tools),
            noise=fixed_noise,
        )
        prepared_noise = np.asarray(runtime.jax.device_get(prepared.noise))
        if (
            prepared.config_name != CONFIG_NAME
            or prepared.checkpoint != CHECKPOINT_IDENTITY
            or prepared.backend != BACKEND
            or prepared.right_image_mask is not False
            or tuple(prepared.base_p2.shape) != (1, 256, 2048)
            or tuple(prepared.left_p2.shape) != (1, 256, 2048)
            or tuple(prepared.right_p2.shape) != (1, 256, 2048)
            or not np.array_equal(prepared_noise, fixed_noise)
        ):
            raise RuntimeError("prepared pi0.5 context violates frozen identity")
        reference = runtime.run_reference(prepared=prepared)
        continued = runtime.continue_from_p2(
            prepared=prepared,
            base_p2=prepared.base_p2,
        )
        clean = _clean_report(reference, continued)
        report = {
            "sample_id": loaded.identity.sample_id,
            "task_id": loaded.identity.task_id,
            "source_image_hash": loaded.identity.source_image_hash,
            "observation_path": str(loaded.path),
            "direction_seed": loaded.identity.direction_seeds["pi05"],
            "clean_equivalence": clean,
            "intervention": None,
        }
        prepared_records.append((loaded, prepared, continued, report))
        reports.append(report)

    clean_gate = all(report["clean_equivalence"]["pass"] for report in reports)
    if clean_gate:
        for loaded, prepared, clean_result, report in prepared_records:
            clean_host = np.asarray(
                runtime.jax.device_get(prepared.base_p2), dtype=np.float32
            ).copy()
            intended_delta, intended_modified = common.intended_perturbation(
                clean_host,
                seed=loaded.identity.direction_seeds["pi05"],
            )
            actual_modified = runtime.jnp.asarray(
                intended_modified,
                dtype=prepared.base_p2.dtype,
            )
            actual_host = np.asarray(
                runtime.jax.device_get(actual_modified), dtype=np.float32
            ).copy()
            perturbation = common.perturbation_metrics(
                clean_host,
                intended_delta,
                actual_host,
            )
            modified_result = runtime.continue_from_p2(
                prepared=prepared,
                base_p2=actual_modified,
            )
            report["intervention"] = _intervention_report(
                clean_result=clean_result,
                modified_result=modified_result,
                perturbation=perturbation,
            )

    intervention_pass = bool(
        clean_gate
        and all(
            report["intervention"] is not None
            and report["intervention"]["actual_modified_differs"]
            and report["intervention"]["all_features_finite"]
            and report["intervention"]["all_actions_finite"]
            and report["intervention"]["continuation_consumed_supplied_native_feature"]
            for report in reports
        )
        and any(
            report["intervention"]["translation_delta_norm"] > TRANSLATION_FLOOR
            for report in reports
        )
    )
    result = {
        "schema_version": "c6_pi05_real_smoke_v1",
        "run_status": "COMPLETED",
        "model": "pi05",
        "model_result": "PASS" if intervention_pass else "BLOCKED",
        "repository_commit": common.git_commit(PROJECT_ROOT),
        "reference_repository_commit": common.git_commit(openpi_root),
        "model_identity": {
            "config": CONFIG_NAME,
            "scientific_checkpoint": CHECKPOINT_IDENTITY,
            "resolved_checkpoint_path": str(runtime.checkpoint_path),
            "backend": BACKEND,
        },
        "runtime_versions": runtime.versions,
        "protocol": {
            "batch_size": 1,
            "alpha": common.ALPHA,
            "noise_rng": common.RNG_IMPLEMENTATION,
            "noise_seed": common.NOISE_SEED,
            "noise_dtype": common.NOISE_DTYPE,
            "noise_shape": list(common.NOISE_SHAPE),
            "direction_rng": common.RNG_IMPLEMENTATION,
            "direction_dtype": "float32",
            "clean_atol": CLEAN_ATOL,
            "translation_floor": TRANSLATION_FLOOR,
        },
        "clean_gate": "PASS" if clean_gate else "BLOCKED",
        "observations": reports,
    }
    common.write_json_atomic(output_dir / RESULT_FILENAME, result)
    return {
        "status": "C6 pi0.5 real smoke — COMPLETED",
        "model_result": result["model_result"],
        "output_path": str(output_dir / RESULT_FILENAME),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
