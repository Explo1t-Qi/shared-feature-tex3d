from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c6_real_smoke_summary as common  # noqa: E402


CHECKPOINT_IDENTITY = common.OPENVLA_CHECKPOINT_IDENTITY
UNNORM_KEY = "libero_spatial_no_noops"
GLOBAL_SEED = 7
DEFAULT_TEX3D_OPENVLA_ROOT = PROJECT_ROOT.parent / "tex3d" / "openvla"
RESULT_FILENAME = common.OPENVLA_RESULT_FILENAME
CLEAN_ATOL = 1e-8
TRANSLATION_FLOOR = 1e-8


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    get_model: Callable[[Any], Any]
    get_processor: Callable[[Any], Any]
    set_seed_everywhere: Callable[[int], None]
    build_policy_observation: Callable[[Any], dict[str, np.ndarray]]
    prepare_context: Callable[..., Any]
    run_reference: Callable[..., Any]
    continue_from_o2: Callable[..., Any]
    validate_model: Callable[[Any], None]
    versions: dict[str, str]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen real OpenVLA C6 clean/intervention smoke."
    )
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=DEFAULT_TEX3D_OPENVLA_ROOT,
    )
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _validate_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    tex3d_root = args.tex3d_openvla_root.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenVLA checkpoint directory not found: {checkpoint}")
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(f"dataset statistics not found: {statistics_path}")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if not isinstance(statistics, dict) or UNNORM_KEY not in statistics:
        raise KeyError(f"{UNNORM_KEY!r} is absent from {statistics_path}")
    if not tex3d_root.is_dir():
        raise FileNotFoundError(f"Tex3D OpenVLA source root not found: {tex3d_root}")
    return checkpoint, tex3d_root


def _load_runtime(tex3d_root: Path) -> _Runtime:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for source_root in (PROJECT_ROOT, tex3d_root):
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

    import torch
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from shared_feature import (
        continue_openvla_from_o2,
        prepare_openvla_context,
        run_openvla_reference,
    )
    from shared_feature import openvla_features

    preprocessing = openvla_features._load_preprocessing_runtime()

    def build_policy_observation(record: Any) -> dict[str, np.ndarray]:
        image = openvla_features._build_policy_image(
            record.base_rgb_raw,
            preprocessing,
            center_crop=False,
        )
        return {
            "full_image": np.asarray(image, dtype=np.uint8).copy(),
            "state": record.state.copy(),
        }

    def validate_model(model: Any) -> None:
        try:
            parameter = next(
                value
                for value in model.vision_backbone.parameters()
                if torch.is_floating_point(value)
            )
        except (AttributeError, StopIteration) as error:
            raise RuntimeError(
                "OpenVLA vision backbone has no floating parameter"
            ) from error
        if parameter.device.type != "cuda":
            raise RuntimeError(f"OpenVLA model is not on CUDA: {parameter.device}")
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"OpenVLA model is not BF16: {parameter.dtype}")

    return _Runtime(
        torch=torch,
        get_model=get_model,
        get_processor=get_processor,
        set_seed_everywhere=set_seed_everywhere,
        build_policy_observation=build_policy_observation,
        prepare_context=prepare_openvla_context,
        run_reference=run_openvla_reference,
        continue_from_o2=continue_openvla_from_o2,
        validate_model=validate_model,
        versions={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "unavailable"
            ),
            "transformers": _package_version("transformers"),
            "tensorflow": _package_version("tensorflow"),
        },
    )


def _clean_report(reference: Any, continued: Any) -> dict[str, Any]:
    checks = {
        "action_token_ids": common.array_comparison(
            reference.action_token_ids,
            continued.action_token_ids,
            atol=0,
            exact=True,
        ),
        "normalized_action": common.array_comparison(
            reference.normalized_action,
            continued.normalized_action,
            atol=CLEAN_ATOL,
        ),
        "unnormalized_action": common.array_comparison(
            reference.unnormalized_action,
            continued.unnormalized_action,
            atol=CLEAN_ATOL,
        ),
        "deployed_action": common.array_comparison(
            reference.deployed_action,
            continued.deployed_action,
            atol=CLEAN_ATOL,
        ),
        "discrete_gripper": common.array_comparison(
            reference.deployed_action[:, -1],
            continued.deployed_action[:, -1],
            atol=0,
            exact=True,
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
        clean_result.unnormalized_action[0, :3], dtype=np.float64
    )
    modified_translation = np.asarray(
        modified_result.unnormalized_action[0, :3], dtype=np.float64
    )
    delta = modified_translation - clean_translation
    arrays = (
        clean_result.normalized_action,
        clean_result.unnormalized_action,
        clean_result.deployed_action,
        modified_result.normalized_action,
        modified_result.unnormalized_action,
        modified_result.deployed_action,
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
    checkpoint, tex3d_root = _validate_runtime_paths(args)
    runtime = _load_runtime(tex3d_root)
    if not runtime.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the OpenVLA smoke runtime")

    runtime.set_seed_everywhere(GLOBAL_SEED)
    model_config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    model = runtime.get_model(model_config)
    processor = runtime.get_processor(model_config)
    runtime.validate_model(model)

    prepared_records: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    for loaded in observations:
        policy_observation = runtime.build_policy_observation(loaded.record)
        prepared = runtime.prepare_context(
            model=model,
            processor=processor,
            observation=policy_observation,
            task_description=loaded.record.prompt,
            pretrained_checkpoint=CHECKPOINT_IDENTITY,
            unnorm_key=UNNORM_KEY,
            center_crop=True,
        )
        if (
            prepared.checkpoint_identity != CHECKPOINT_IDENTITY
            or prepared.unnorm_key != UNNORM_KEY
            or prepared.center_crop is not True
            or tuple(prepared.o2.shape) != (1, 256, 4096)
        ):
            raise RuntimeError("prepared OpenVLA context violates frozen identity")
        reference = runtime.run_reference(prepared=prepared)
        continued = runtime.continue_from_o2(prepared=prepared, o2=prepared.o2)
        clean = _clean_report(reference, continued)
        report = {
            "sample_id": loaded.identity.sample_id,
            "task_id": loaded.identity.task_id,
            "source_image_hash": loaded.identity.source_image_hash,
            "observation_path": str(loaded.path),
            "direction_seed": loaded.identity.direction_seeds["openvla"],
            "clean_equivalence": clean,
            "intervention": None,
        }
        prepared_records.append((loaded, prepared, continued, report))
        reports.append(report)

    clean_gate = all(report["clean_equivalence"]["pass"] for report in reports)
    if clean_gate:
        for loaded, prepared, clean_result, report in prepared_records:
            clean_native = prepared.o2.detach().to(
                device="cpu", dtype=runtime.torch.float32
            )
            clean_host = clean_native.numpy().copy()
            intended_delta, intended_modified = common.intended_perturbation(
                clean_host,
                seed=loaded.identity.direction_seeds["openvla"],
            )
            actual_modified = runtime.torch.as_tensor(
                intended_modified,
                dtype=prepared.o2.dtype,
                device=prepared.o2.device,
            )
            actual_host = (
                actual_modified.detach()
                .to(device="cpu", dtype=runtime.torch.float32)
                .numpy()
                .copy()
            )
            perturbation = common.perturbation_metrics(
                clean_host,
                intended_delta,
                actual_host,
            )
            modified_result = runtime.continue_from_o2(
                prepared=prepared,
                o2=actual_modified,
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
        "schema_version": "c6_openvla_real_smoke_v1",
        "run_status": "COMPLETED",
        "model": "openvla",
        "model_result": "PASS" if intervention_pass else "BLOCKED",
        "repository_commit": common.git_commit(PROJECT_ROOT),
        "reference_repository_commit": common.git_commit(tex3d_root.parent),
        "model_identity": {
            "scientific_checkpoint": CHECKPOINT_IDENTITY,
            "resolved_checkpoint_path": str(checkpoint),
            "unnorm_key": UNNORM_KEY,
            "center_crop": True,
            "do_sample": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
        },
        "runtime_versions": runtime.versions,
        "protocol": {
            "batch_size": 1,
            "global_seed": GLOBAL_SEED,
            "alpha": common.ALPHA,
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
        "status": "C6 OpenVLA real smoke — COMPLETED",
        "model_result": result["model_result"],
        "output_path": str(output_dir / RESULT_FILENAME),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
