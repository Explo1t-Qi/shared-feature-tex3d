from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX3D_OPENVLA_ROOT = PROJECT_ROOT.parent / "tex3d" / "openvla"
TASK_IDS = (0, 1)
TARGET_GROUPS_PER_TASK = 1
TARGET_PROGRESS = (0.10, 0.40, 0.70, 0.90)
UNNORM_KEY = "libero_spatial_no_noops"
GLOBAL_SEED = 7


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reduced real C5-D0 Pilot v0.2 collector smoke."
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Local OpenVLA LIBERO-Spatial checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-identity",
        required=True,
        help="Approved portable identity for the OpenVLA checkpoint.",
    )
    parser.add_argument(
        "--libero-revision",
        required=True,
        help="Exact LIBERO source revision used by the runtime.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh or pre-existing empty smoke output directory.",
    )
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=DEFAULT_TEX3D_OPENVLA_ROOT,
        help="Official Tex3D OpenVLA source root used by the validated C1 path.",
    )
    return parser.parse_args(argv)


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tex3d_openvla_root = args.tex3d_openvla_root.expanduser().resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(f"dataset statistics file not found: {statistics_path}")
    with statistics_path.open(encoding="utf-8") as source:
        statistics = json.load(source)
    if UNNORM_KEY not in statistics:
        raise KeyError(
            f"{UNNORM_KEY!r} missing from {statistics_path}; "
            f"available={sorted(statistics)}"
        )
    if not isinstance(args.checkpoint_identity, str) or not (
        args.checkpoint_identity.strip()
    ):
        raise ValueError("checkpoint identity must be a non-empty string")
    if not isinstance(args.libero_revision, str) or not args.libero_revision.strip():
        raise ValueError("LIBERO revision must be a non-empty string")
    if not tex3d_openvla_root.is_dir():
        raise FileNotFoundError(
            f"official Tex3D OpenVLA root not found: {tex3d_openvla_root}"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"smoke output directory is not empty: {output_dir}")

    return checkpoint, output_dir, tex3d_openvla_root


def _load_runtime_components(tex3d_openvla_root: Path) -> tuple[Any, ...]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for source_root in (PROJECT_ROOT, tex3d_openvla_root):
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

    import torch
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from shared_feature.pilot_v02_collector import (
        _collect_pilot_v02_with_plan,
    )

    return (
        torch,
        get_model,
        get_processor,
        set_seed_everywhere,
        _collect_pilot_v02_with_plan,
    )


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, output_dir, tex3d_openvla_root = _validate_inputs(args)
    (
        torch,
        get_model,
        get_processor,
        set_seed_everywhere,
        collect_with_plan,
    ) = _load_runtime_components(tex3d_openvla_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the smoke runtime")

    set_seed_everywhere(GLOBAL_SEED)
    model_config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    model = get_model(model_config)
    processor = get_processor(model_config)

    result = collect_with_plan(
        model=model,
        processor=processor,
        pretrained_checkpoint=checkpoint,
        checkpoint_identity=args.checkpoint_identity,
        libero_revision=args.libero_revision,
        output_dir=output_dir,
        task_ids=TASK_IDS,
        target_groups_per_task=TARGET_GROUPS_PER_TASK,
    )
    return _validate_result(result, output_dir)


def _validate_result(result: Any, output_dir: Path) -> dict[str, Any]:
    from shared_feature import PilotObservation

    if result.run_status != "SMOKE_COMPLETED":
        raise AssertionError(f"unexpected smoke run_status: {result.run_status!r}")
    if result.completeness_status is not None:
        raise AssertionError(
            "reduced smoke must not emit a formal completeness status: "
            f"{result.completeness_status!r}"
        )
    expected_manifest_path = (output_dir / "collection_manifest.json").resolve()
    if result.manifest_path.resolve() != expected_manifest_path:
        raise AssertionError(
            f"unexpected manifest path: {result.manifest_path.resolve()}"
        )
    with expected_manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)

    if manifest.get("schema_version") != "pilot_v0_2_collection_v1":
        raise AssertionError("unexpected collection manifest schema_version")
    if manifest.get("run_status") != "SMOKE_COMPLETED":
        raise AssertionError("manifest does not record SMOKE_COMPLETED")
    if manifest.get("completeness_status") is not None:
        raise AssertionError("manifest assigns a formal completeness label to smoke")
    coverage = manifest.get("coverage", {})
    if coverage.get("actual_total_groups") != 2:
        raise AssertionError(
            f"expected 2 accepted groups, got {coverage.get('actual_total_groups')!r}"
        )
    if coverage.get("actual_total_observations") != 8:
        raise AssertionError(
            "expected 8 accepted observations, got "
            f"{coverage.get('actual_total_observations')!r}"
        )

    task_results = manifest.get("task_results")
    if not isinstance(task_results, list) or [
        task.get("task_id") for task in task_results
    ] != list(TASK_IDS):
        raise AssertionError("smoke manifest does not contain tasks 0 and 1 in order")

    manifest_sample_ids: list[str] = []
    manifest_paths: list[Path] = []
    for task in task_results:
        if task.get("actual_accepted_groups") != 1:
            raise AssertionError(
                f"task {task.get('task_id')} did not produce one accepted group"
            )
        groups = task.get("accepted_groups")
        if not isinstance(groups, list) or len(groups) != 1:
            raise AssertionError("smoke task has malformed accepted_groups")
        samples = groups[0].get("samples")
        if not isinstance(samples, list) or len(samples) != 4:
            raise AssertionError("smoke accepted group does not contain four samples")
        if [sample.get("target_relative_progress") for sample in samples] != list(
            TARGET_PROGRESS
        ):
            raise AssertionError("smoke samples do not use the frozen target progress")
        for sample in samples:
            sample_id = sample.get("sample_id")
            relative_path = sample.get("observation_path")
            if not isinstance(sample_id, str) or not isinstance(relative_path, str):
                raise AssertionError("smoke sample metadata is malformed")
            if Path(relative_path).is_absolute() or "\\" in relative_path:
                raise AssertionError(
                    f"observation path is not relative POSIX: {relative_path}"
                )
            path = (output_dir / relative_path).resolve()
            if path.parent != (output_dir / "observations").resolve():
                raise AssertionError(
                    f"observation path escaped canonical layout: {path}"
                )
            manifest_sample_ids.append(sample_id)
            manifest_paths.append(path)

    if len(set(manifest_sample_ids)) != 8:
        raise AssertionError("smoke sample IDs are not globally unique")
    returned_paths = [path.resolve() for path in result.sample_paths]
    if returned_paths != manifest_paths:
        raise AssertionError("returned sample path order differs from manifest order")
    archive_paths = sorted((output_dir / "observations").glob("*.npz"))
    if len(archive_paths) != 8 or {path.resolve() for path in archive_paths} != set(
        manifest_paths
    ):
        raise AssertionError("canonical observations directory does not contain 8 NPZs")

    loaded_ids = []
    for sample_id, path in zip(manifest_sample_ids, manifest_paths, strict=True):
        record = PilotObservation.load(path)
        if record.sample_id != sample_id:
            raise AssertionError(f"sample ID mismatch in {path}")
        if not record.episode_success:
            raise AssertionError(
                f"accepted smoke sample is not successful: {sample_id}"
            )
        loaded_ids.append(record.sample_id)

    return {
        "run_status": result.run_status,
        "completeness_status": result.completeness_status,
        "manifest_path": str(expected_manifest_path),
        "accepted_groups": 2,
        "observation_file_count": len(archive_paths),
        "sample_ids": loaded_ids,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = _run_smoke(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
