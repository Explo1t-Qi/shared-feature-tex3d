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
CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial_no_noops"
GLOBAL_SEED = 7
EXPECTED_TASK_IDS = tuple(range(10))
TARGET_PROGRESS = (0.10, 0.40, 0.70, 0.90)
COMPLETENESS_STATUSES = {"COMPLETE", "USABLE_WITH_SHORTFALL", "BLOCKED"}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Pilot v0.2 OpenVLA/LIBERO collection."
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Local OpenVLA LIBERO-Spatial checkpoint directory.",
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
        help="Fresh or pre-existing empty canonical collection directory.",
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
    if not isinstance(statistics, dict) or UNNORM_KEY not in statistics:
        available = sorted(statistics) if isinstance(statistics, dict) else []
        raise KeyError(
            f"{UNNORM_KEY!r} missing from {statistics_path}; available={available}"
        )
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
            raise FileExistsError(
                f"full-collection output directory is not empty: {output_dir}"
            )

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
    from shared_feature import collect_pilot_v02_observations

    return (
        torch,
        get_model,
        get_processor,
        set_seed_everywhere,
        collect_pilot_v02_observations,
    )


def _run_collection(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, output_dir, tex3d_openvla_root = _validate_inputs(args)
    (
        torch,
        get_model,
        get_processor,
        set_seed_everywhere,
        collect_formal,
    ) = _load_runtime_components(tex3d_openvla_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the full-collection runtime")

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

    result = collect_formal(
        model=model,
        processor=processor,
        pretrained_checkpoint=checkpoint,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        libero_revision=args.libero_revision,
        output_dir=output_dir,
    )
    return _validate_result(result, output_dir, args.libero_revision)


def _validate_result(
    result: Any,
    output_dir: Path,
    supplied_libero_revision: str,
) -> dict[str, Any]:
    from shared_feature import PilotObservation

    if result.run_status != "COMPLETED":
        raise AssertionError(
            f"formal collection returned unexpected run_status: {result.run_status!r}"
        )
    if result.completeness_status not in COMPLETENESS_STATUSES:
        raise AssertionError(
            "formal collection returned invalid completeness_status: "
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
    if manifest.get("run_status") != "COMPLETED":
        raise AssertionError("manifest does not record COMPLETED")
    if manifest.get("completeness_status") != result.completeness_status:
        raise AssertionError("result and manifest completeness statuses differ")

    rollout = manifest.get("rollout")
    if (
        not isinstance(rollout, dict)
        or rollout.get("checkpoint_identity") != CHECKPOINT_IDENTITY
    ):
        raise AssertionError("manifest checkpoint identity is not the frozen identity")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise AssertionError("manifest runtime record is malformed")
    if runtime.get("libero_revision") != supplied_libero_revision:
        raise AssertionError("manifest LIBERO revision differs from the supplied value")
    if runtime.get("discovered_task_count") != 10 or runtime.get("task_ids") != list(
        EXPECTED_TASK_IDS
    ):
        raise AssertionError("manifest runtime does not expose tasks 0..9")

    task_results = manifest.get("task_results")
    if not isinstance(task_results, list) or [
        task.get("task_id") for task in task_results if isinstance(task, dict)
    ] != list(EXPECTED_TASK_IDS):
        raise AssertionError("manifest task_results are not ordered tasks 0..9")

    manifest_sample_ids: list[str] = []
    manifest_paths: list[Path] = []
    accepted_counts: dict[str, int] = {}
    attempted_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    group_count = 0

    observations_dir = (output_dir / "observations").resolve()
    for task in task_results:
        task_id = task["task_id"]
        task_key = str(task_id)
        task_language = task.get("task_language")
        if not isinstance(task_language, str) or not task_language:
            raise AssertionError(f"task {task_id} language is malformed")
        attempted_state_ids = task.get("attempted_state_ids")
        accepted_state_ids = task.get("accepted_state_ids")
        rejected_states = task.get("rejected_states")
        accepted_groups = task.get("accepted_groups")
        if not all(
            isinstance(value, list)
            for value in (
                attempted_state_ids,
                accepted_state_ids,
                rejected_states,
                accepted_groups,
            )
        ):
            raise AssertionError(f"task {task_id} collection records are malformed")
        if task.get("actual_accepted_groups") != len(accepted_groups):
            raise AssertionError(f"task {task_id} accepted-group count differs")
        if accepted_state_ids != [
            group.get("initial_state_id") for group in accepted_groups
        ]:
            raise AssertionError(f"task {task_id} accepted state order differs")

        accepted_counts[task_key] = len(accepted_groups)
        attempted_counts[task_key] = len(attempted_state_ids)
        rejected_counts[task_key] = len(rejected_states)
        group_count += len(accepted_groups)

        for group in accepted_groups:
            initial_state_id = group.get("initial_state_id")
            if group.get("task_id") != task_id or not group.get("episode_success"):
                raise AssertionError(f"task {task_id} accepted group metadata differs")
            samples = group.get("samples")
            if not isinstance(samples, list) or len(samples) != 4:
                raise AssertionError(
                    f"task {task_id} state {initial_state_id} lacks four samples"
                )
            if [sample.get("target_relative_progress") for sample in samples] != list(
                TARGET_PROGRESS
            ):
                raise AssertionError(
                    f"task {task_id} state {initial_state_id} target progress differs"
                )

            for sample in samples:
                sample_id = sample.get("sample_id")
                relative_path = sample.get("observation_path")
                if not isinstance(sample_id, str) or not isinstance(relative_path, str):
                    raise AssertionError("manifest sample metadata is malformed")
                if Path(relative_path).is_absolute() or "\\" in relative_path:
                    raise AssertionError(
                        f"observation path is not relative POSIX: {relative_path}"
                    )
                if relative_path != f"observations/{sample_id}.npz":
                    raise AssertionError(
                        f"observation path is not canonical: {relative_path}"
                    )
                path = (output_dir / relative_path).resolve()
                if path.parent != observations_dir:
                    raise AssertionError(
                        f"observation path escaped canonical layout: {path}"
                    )
                if path.stem != sample_id:
                    raise AssertionError(
                        f"sample filename differs from ID: {sample_id}"
                    )
                if sample_id in manifest_sample_ids:
                    raise AssertionError(f"duplicate manifest sample ID: {sample_id}")

                record = PilotObservation.load(path)
                if (
                    record.sample_id != sample_id
                    or record.task_id != str(task_id)
                    or record.initial_state_id != initial_state_id
                    or record.episode_id != initial_state_id
                    or record.step_id != sample.get("step_id")
                    or record.normalized_episode_progress
                    != sample.get("actual_normalized_episode_progress")
                    or record.prompt != task_language
                    or not record.episode_success
                ):
                    raise AssertionError(f"PilotObservation metadata differs: {path}")
                manifest_sample_ids.append(sample_id)
                manifest_paths.append(path)

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise AssertionError("manifest coverage record is malformed")
    observation_count = len(manifest_paths)
    if (
        coverage.get("target_total_groups") != 50
        or coverage.get("target_total_observations") != 200
    ):
        raise AssertionError("manifest formal coverage targets differ")
    if (
        coverage.get("actual_total_groups") != group_count
        or coverage.get("actual_total_observations") != observation_count
    ):
        raise AssertionError("manifest actual coverage counts differ")
    if coverage.get("accepted_groups_per_task") != accepted_counts:
        raise AssertionError("manifest per-task accepted-group counts differ")

    returned_paths = [path.resolve() for path in result.sample_paths]
    if returned_paths != manifest_paths:
        raise AssertionError("returned sample path order differs from manifest order")
    archive_paths = sorted(observations_dir.glob("*.npz"))
    if len(archive_paths) != observation_count or {
        path.resolve() for path in archive_paths
    } != set(manifest_paths):
        raise AssertionError(
            "canonical observations directory differs from manifest archives"
        )
    if set(output_dir.iterdir()) != {
        output_dir / "observations",
        output_dir / "collection_manifest.json",
    }:
        raise AssertionError("canonical output directory contains unexpected entries")

    return {
        "run_status": result.run_status,
        "completeness_status": result.completeness_status,
        "checkpoint_identity": CHECKPOINT_IDENTITY,
        "libero_revision": supplied_libero_revision,
        "manifest_path": str(expected_manifest_path),
        "target_total_groups": coverage["target_total_groups"],
        "actual_total_groups": coverage["actual_total_groups"],
        "target_total_observations": coverage["target_total_observations"],
        "actual_total_observations": coverage["actual_total_observations"],
        "accepted_groups_per_task": accepted_counts,
        "attempted_states_per_task": attempted_counts,
        "rejected_states_per_task": rejected_counts,
        "observation_file_count": len(archive_paths),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = _run_collection(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
