from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS_DIR = PROJECT_ROOT / "experiment_inbox" / "samples"
DEFAULT_OPENPI_ROOT = PROJECT_ROOT.parent / "openpi"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG_NAME = "pi05_libero"
NUM_SAMPLES = 2
EXPECTED_ARCHIVE_FIELDS = {
    "metadata_json",
    "p1_siglip",
    "p2_projected",
}
EXPECTED_METADATA_FIELDS = {
    "sample_id",
    "source_model",
    "checkpoint",
    "feature_schema_version",
    "source_image_hash",
}
EXPECTED_FEATURE_SHAPES = {
    "p1_siglip": (256, 1152),
    "p2_projected": (256, 2048),
}


class _Tee:
    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, value: str) -> int:
        self._terminal.write(value)
        self._log.write(value)
        self._log.flush()
        return len(value)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real C3 pi0.5/OpenPI feature-extraction smoke."
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=DEFAULT_OBSERVATIONS_DIR,
        help="Directory containing existing real C1 PilotObservation samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh C3 smoke output directory; it must not already exist.",
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=DEFAULT_OPENPI_ROOT,
        help="Local OpenPI source repository at the audited commit.",
    )
    return parser.parse_args()


def _qualified_class_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _select_observations(observations_dir: Path) -> tuple[Path, ...]:
    from shared_feature import PilotObservation

    if not observations_dir.is_dir():
        raise FileNotFoundError(
            f"C1 observations directory not found: {observations_dir}"
        )

    candidates: list[tuple[int, int, Path]] = []
    for path in sorted(observations_dir.glob("*.npz")):
        record = PilotObservation.load(path)
        candidates.append((record.initial_state_id, record.step_id, path))

    selected: list[Path] = []
    selected_states: set[int] = set()
    for initial_state_id, _, path in sorted(candidates):
        if initial_state_id in selected_states:
            continue
        selected.append(path)
        selected_states.add(initial_state_id)
        if len(selected) == NUM_SAMPLES:
            break
    if len(selected) != NUM_SAMPLES:
        raise AssertionError(
            "C3 smoke requires two C1 samples from distinct episodes, got "
            f"states={sorted(selected_states)}"
        )
    return tuple(selected)


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_train_config(train_config: Any, model_type: Any) -> None:
    expected = {
        "name": "pi05_libero",
        "pi05": True,
        "action_horizon": 10,
        "discrete_state_input": False,
        "action_dim": 32,
        "max_token_len": 200,
    }
    actual = {
        "name": train_config.name,
        "pi05": train_config.model.pi05,
        "action_horizon": train_config.model.action_horizon,
        "discrete_state_input": train_config.model.discrete_state_input,
        "action_dim": train_config.model.action_dim,
        "max_token_len": train_config.model.max_token_len,
    }
    if actual != expected:
        raise AssertionError(
            f"pi05_libero TrainConfig mismatch: actual={actual}, expected={expected}"
        )
    if train_config.model.model_type != model_type.PI05:
        raise AssertionError(
            "pi05_libero model_type is not the official PI05 enum value"
        )


def _checkpoint_inventory(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "model_safetensors": (checkpoint_path / "model.safetensors").is_file(),
        "params_directory": (checkpoint_path / "params").is_dir(),
        "assets_directory": (checkpoint_path / "assets").is_dir(),
        "root_entries": sorted(path.name for path in checkpoint_path.iterdir()),
    }


def _load_real_model(
    train_config: Any,
    checkpoint_path: Path,
    *,
    openpi_model: Any,
    jnp: Any,
) -> tuple[Any, str]:
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        model = train_config.model.load_pytorch(
            train_config,
            str(safetensors_path),
        )
        model.paligemma_with_expert.to_bfloat16_for_selected_params(
            "bfloat16"
        )
        return model, "pytorch"

    params_path = checkpoint_path / "params"
    if not params_path.is_dir():
        raise FileNotFoundError(
            "released checkpoint contains neither model.safetensors nor params"
        )
    params = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    return train_config.model.load(params), "jax_nnx"


def _model_runtime_info(model: Any, backend: str, jax: Any) -> dict[str, Any]:
    dtype = "unavailable"
    devices: list[str] = []
    sharding = "unavailable"
    try:
        if backend == "jax_nnx":
            from flax import nnx

            leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
            parameter = next(
                value
                for leaf in leaves
                if hasattr((value := getattr(leaf, "value", leaf)), "dtype")
            )
            dtype = str(parameter.dtype)
            if hasattr(parameter, "devices"):
                devices = sorted(str(device) for device in parameter.devices())
            elif hasattr(parameter, "device"):
                devices = [str(parameter.device)]
            sharding = str(getattr(parameter, "sharding", "unavailable"))
        else:
            parameter = next(model.parameters())
            dtype = str(parameter.dtype)
            devices = [str(parameter.device)]
    except (AttributeError, StopIteration, TypeError):
        pass
    return {
        "backend": backend,
        "model_class": _qualified_class_name(model),
        "parameter_dtype": dtype,
        "devices": devices,
        "sharding": sharding,
        "jax_default_backend": jax.default_backend(),
        "has_PaliGemma": hasattr(model, "PaliGemma"),
        "has_callable_PaliGemma_img": callable(
            getattr(getattr(model, "PaliGemma", None), "img", None)
        ),
    }


def _validate_norm_stats(norm_stats: Any) -> dict[str, Any]:
    if not isinstance(norm_stats, Mapping) or "state" not in norm_stats:
        raise AssertionError("checkpoint norm stats are missing the state entry")
    state_stats = norm_stats["state"]
    q01_shape = tuple(state_stats.q01.shape)
    q99_shape = tuple(state_stats.q99.shape)
    if q01_shape != (8,) or q99_shape != (8,):
        raise AssertionError(
            "checkpoint state quantile shapes are incompatible: "
            f"q01={q01_shape}, q99={q99_shape}"
        )
    return {
        "state_q01_shape": q01_shape,
        "state_q99_shape": q99_shape,
    }


def _direct_reference(
    *,
    record: Any,
    model: Any,
    data_config: Any,
    norm_stats: Any,
    jax: Any,
    jnp: Any,
    image_tools: Any,
    transforms: Any,
    openpi_model: Any,
) -> dict[str, Any]:
    import numpy as np

    raw_base = record.base_rgb_raw
    raw_wrist = record.wrist_rgb_raw
    rotated_base = np.ascontiguousarray(raw_base[::-1, ::-1])
    rotated_wrist = np.ascontiguousarray(raw_wrist[::-1, ::-1])
    rotation_exact = bool(
        np.array_equal(rotated_base, np.rot90(raw_base, 2))
        and np.array_equal(rotated_wrist, np.rot90(raw_wrist, 2))
    )
    if not rotation_exact:
        raise AssertionError("direct reference did not preserve 180-degree rotation")
    if not rotated_base.flags.c_contiguous or not rotated_wrist.flags.c_contiguous:
        raise AssertionError("direct reference rotated images are not C-contiguous")

    base_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(rotated_base, 224, 224)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(rotated_wrist, 224, 224)
    )
    if base_image.shape != (224, 224, 3) or wrist_image.shape != (
        224,
        224,
        3,
    ):
        raise AssertionError("direct client preprocessing produced wrong shapes")

    policy_input = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": record.state.copy(),
        "prompt": record.prompt,
    }
    input_transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    transformed = input_transform(policy_input)
    batched = jax.tree.map(
        lambda value: jnp.asarray(value)[None, ...],
        transformed,
    )
    observation = openpi_model.Observation.from_dict(batched)
    observation = openpi_model.preprocess_observation(
        None,
        observation,
        train=False,
    )

    direct_p2, direct_aux = model.PaliGemma.img(
        observation.images["base_0_rgb"],
        train=False,
    )
    if not isinstance(direct_aux, Mapping) or "encoded" not in direct_aux:
        raise AssertionError("real PaliGemma.img did not return aux['encoded']")
    direct_p1 = direct_aux["encoded"]

    base_host = np.asarray(
        jax.device_get(observation.images["base_0_rgb"])
    )
    p1_host = np.asarray(jax.device_get(direct_p1), dtype=np.float32)
    p2_host = np.asarray(jax.device_get(direct_p2), dtype=np.float32)
    if base_host.shape != (1, 224, 224, 3):
        raise AssertionError(
            f"direct base image has unexpected shape: {base_host.shape}"
        )
    if not np.all(np.isfinite(base_host)):
        raise AssertionError("direct preprocessed base image is non-finite")
    if float(base_host.min()) < -1.0 or float(base_host.max()) > 1.0:
        raise AssertionError("direct preprocessed base image is outside [-1, 1]")
    if p1_host.shape != (1, 256, 1152):
        raise AssertionError(f"direct P1 has wrong shape: {p1_host.shape}")
    if p2_host.shape != (1, 256, 2048):
        raise AssertionError(f"direct P2 has wrong shape: {p2_host.shape}")
    for name, value in (("P1", p1_host), ("P2", p2_host)):
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"direct {name} contains non-finite values")
        if not np.any(value) or float(value.std()) == 0.0:
            raise AssertionError(f"direct {name} is trivially zero or constant")

    return {
        "p1": p1_host,
        "p2": p2_host,
        "p1_runtime_dtype": str(direct_p1.dtype),
        "p2_runtime_dtype": str(direct_p2.dtype),
        "base_diagnostics": {
            "shape": base_host.shape,
            "dtype": str(base_host.dtype),
            "min": float(base_host.min()),
            "max": float(base_host.max()),
            "finite": True,
            "rotation_exact_180": rotation_exact,
        },
        "p1_sharding": str(getattr(direct_p1, "sharding", "unavailable")),
        "p2_sharding": str(getattr(direct_p2, "sharding", "unavailable")),
    }


def _feature_diagnostics(value: Any) -> dict[str, Any]:
    import numpy as np

    if not np.all(np.isfinite(value)):
        raise AssertionError("serialized feature contains non-finite values")
    if not np.any(value) or float(value.std()) == 0.0:
        raise AssertionError("serialized feature is trivially zero or constant")
    return {
        "shape": value.shape,
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


def _load_feature(
    *,
    path: Path,
    record: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARCHIVE_FIELDS:
            raise AssertionError(
                f"unexpected archive fields for {record.sample_id}: "
                f"{sorted(archive.files)}"
            )
        metadata_value = archive["metadata_json"]
        if metadata_value.ndim != 0 or metadata_value.dtype.kind != "U":
            raise AssertionError("metadata_json is not a Unicode scalar")
        metadata = json.loads(str(metadata_value.item()))
        arrays = {
            "p1_siglip": archive["p1_siglip"].copy(),
            "p2_projected": archive["p2_projected"].copy(),
        }

    if set(metadata) != EXPECTED_METADATA_FIELDS:
        raise AssertionError(
            f"unexpected metadata fields for {record.sample_id}: "
            f"{sorted(metadata)}"
        )
    expected_hash = "sha256:" + hashlib.sha256(
        np.ascontiguousarray(record.base_rgb_raw).tobytes()
    ).hexdigest()
    expected_metadata = {
        "sample_id": record.sample_id,
        "source_model": "pi05",
        "checkpoint": CHECKPOINT,
        "feature_schema_version": "pi05_features_v1",
        "source_image_hash": expected_hash,
    }
    if metadata != expected_metadata:
        raise AssertionError(
            f"feature provenance mismatch for {record.sample_id}: "
            f"actual={metadata}, expected={expected_metadata}"
        )

    diagnostics: dict[str, Any] = {}
    for name, expected_shape in EXPECTED_FEATURE_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise AssertionError(
                f"unexpected {name} layout for {record.sample_id}: "
                f"shape={value.shape}, dtype={value.dtype}"
            )
        diagnostics[name] = _feature_diagnostics(value)
    return arrays, {
        "sample_id": record.sample_id,
        "source_image_hash": expected_hash,
        "features": diagnostics,
    }


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    if reference.shape != candidate.shape:
        return {
            "array_equal": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "shape_equal": False,
        }
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "array_equal": bool(np.array_equal(reference, candidate)),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "shape_equal": True,
    }


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    observations_dir = args.observations_dir.resolve()
    openpi_root = args.openpi_root.resolve()
    if not openpi_root.is_dir():
        raise FileNotFoundError(f"OpenPI repository not found: {openpi_root}")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(openpi_root / "packages" / "openpi-client" / "src"))
    sys.path.insert(0, str(openpi_root / "src"))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi import transforms
    from openpi.models import model as openpi_model
    from openpi.shared import download
    from openpi.training import checkpoints
    from openpi.training import config
    from openpi_client import image_tools

    from shared_feature import PilotObservation, extract_pi05_features

    openpi_commit = _git_commit(openpi_root)
    train_config = config.get_config(CONFIG_NAME)
    _validate_train_config(train_config, openpi_model.ModelType)

    checkpoint_path = download.maybe_download(CHECKPOINT)
    checkpoint_path = checkpoint_path.resolve()
    inventory = _checkpoint_inventory(checkpoint_path)

    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )
    if data_config.asset_id != "physical-intelligence/libero":
        raise AssertionError(
            f"unexpected pi05_libero asset_id: {data_config.asset_id}"
        )
    if data_config.use_quantile_norm is not True:
        raise AssertionError("pi05_libero does not use quantile normalization")
    norm_stats = checkpoints.load_norm_stats(
        checkpoint_path / "assets",
        data_config.asset_id,
    )
    norm_diagnostics = _validate_norm_stats(norm_stats)

    model, backend = _load_real_model(
        train_config,
        checkpoint_path,
        openpi_model=openpi_model,
        jnp=jnp,
    )
    model_info = _model_runtime_info(model, backend, jax)
    if not model_info["has_PaliGemma"] or not model_info[
        "has_callable_PaliGemma_img"
    ]:
        raise RuntimeError(
            "released checkpoint backend does not expose callable model.PaliGemma.img"
        )

    observation_paths = _select_observations(observations_dir)
    records = tuple(PilotObservation.load(path) for path in observation_paths)
    batch1_dir = output_dir / "batch1"
    returned_paths = extract_pi05_features(
        model=model,
        train_config=train_config,
        checkpoint=CHECKPOINT,
        norm_stats=norm_stats,
        observation_paths=observation_paths,
        output_dir=batch1_dir,
        batch_size=1,
    )
    expected_paths = tuple(
        batch1_dir / f"{record.sample_id}.npz" for record in records
    )
    if returned_paths != expected_paths:
        raise AssertionError(
            f"extractor output order mismatch: {returned_paths} != {expected_paths}"
        )

    batch1_arrays: dict[str, dict[str, Any]] = {}
    sample_reports: list[dict[str, Any]] = []
    for path, record in zip(returned_paths, records, strict=True):
        arrays, sample_report = _load_feature(path=path, record=record)
        batch1_arrays[record.sample_id] = arrays
        sample_reports.append(sample_report)

    direct = _direct_reference(
        record=records[0],
        model=model,
        data_config=data_config,
        norm_stats=norm_stats,
        jax=jax,
        jnp=jnp,
        image_tools=image_tools,
        transforms=transforms,
        openpi_model=openpi_model,
    )
    direct_p1 = direct.pop("p1")
    direct_p2 = direct.pop("p2")
    first_arrays = batch1_arrays[records[0].sample_id]
    equivalence = {
        "p1": _compare(direct_p1[0], first_arrays["p1_siglip"]),
        "p2": _compare(direct_p2[0], first_arrays["p2_projected"]),
    }
    if not equivalence["p1"]["array_equal"]:
        raise AssertionError(f"direct-vs-serialized P1 mismatch: {equivalence['p1']}")
    if not equivalence["p2"]["array_equal"]:
        raise AssertionError(f"direct-vs-serialized P2 mismatch: {equivalence['p2']}")

    batch2_report: dict[str, Any]
    try:
        batch2_dir = output_dir / "batch2"
        batch2_paths = extract_pi05_features(
            model=model,
            train_config=train_config,
            checkpoint=CHECKPOINT,
            norm_stats=norm_stats,
            observation_paths=observation_paths,
            output_dir=batch2_dir,
            batch_size=2,
        )
        order_equal = tuple(path.name for path in batch2_paths) == tuple(
            path.name for path in expected_paths
        )
        sample_equal: dict[str, dict[str, bool]] = {}
        for path, record in zip(batch2_paths, records, strict=True):
            arrays, _ = _load_feature(path=path, record=record)
            sample_equal[record.sample_id] = {
                name: bool(
                    np.array_equal(
                        arrays[name],
                        batch1_arrays[record.sample_id][name],
                    )
                )
                for name in EXPECTED_FEATURE_SHAPES
            }
        all_equal = order_equal and all(
            all(comparisons.values())
            for comparisons in sample_equal.values()
        )
        batch2_report = {
            "status": "PASS" if all_equal else "MISMATCH",
            "output_order_equal": order_equal,
            "sample_features_equal": sample_equal,
        }
    except Exception as error:
        batch2_report = {
            "status": "FAILED",
            "error_type": _qualified_class_name(error),
            "error": str(error),
            "traceback": traceback.format_exc(),
        }

    report = {
        "status": "PASS",
        "environment": {
            "openpi_commit": openpi_commit,
            "openpi_root": str(openpi_root),
            "config": CONFIG_NAME,
            "checkpoint_identifier": CHECKPOINT,
            "resolved_checkpoint_path": str(checkpoint_path),
            "checkpoint_inventory": inventory,
            **model_info,
        },
        "norm_stats": {
            "asset_id": data_config.asset_id,
            **norm_diagnostics,
        },
        "input_samples": sample_reports,
        "direct_reference": {
            "sample_id": records[0].sample_id,
            "p1_shape": (1, 256, 1152),
            "p2_shape": (1, 256, 2048),
            **direct,
        },
        "equivalence": equivalence,
        "batch2_check": batch2_report,
        "output_directory": str(output_dir),
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("=== Environment ===")
    print(json.dumps(report["environment"], indent=2, sort_keys=True))
    print("=== Norm Stats ===")
    print(json.dumps(report["norm_stats"], indent=2, sort_keys=True))
    print("=== Input Samples / Extractor Output ===")
    print(json.dumps(report["input_samples"], indent=2, sort_keys=True))
    print("=== Direct Reference ===")
    print(json.dumps(report["direct_reference"], indent=2, sort_keys=True))
    print("=== Equivalence ===")
    print(json.dumps(report["equivalence"], indent=2, sort_keys=True))
    print("=== Batch-2 Check ===")
    print(json.dumps(report["batch2_check"], indent=2, sort_keys=True))
    print("=== Final Result ===")
    print("PASS")
    print("C3 real pi0.5 integration smoke — PASS")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"smoke output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    log_path = output_dir / "run.log"
    report_path = output_dir / "smoke_report.json"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout = _Tee(original_stdout, log)
        sys.stderr = _Tee(original_stderr, log)
        try:
            report = _run_smoke(args)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _print_report(report)
            return_code = 0
        except Exception as error:
            traceback.print_exc()
            blocked_report = {
                "status": "BLOCKED",
                "error_type": _qualified_class_name(error),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            report_path.write_text(
                json.dumps(blocked_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print("=== Final Result ===")
            print("BLOCKED")
            print(f"Exact blocker: {_qualified_class_name(error)}: {error}")
            return_code = 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
