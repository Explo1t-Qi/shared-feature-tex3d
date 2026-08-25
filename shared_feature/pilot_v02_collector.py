from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import libero_collector as _c1
from .pilot_observation import PilotObservation


_SCHEMA_VERSION = "pilot_v0_2_collection_v1"
_PILOT_VERSION = "0.2"
_FORMAL_TASK_IDS = tuple(range(10))
_FORMAL_GROUPS_PER_TASK = 5
_TARGET_PROGRESS = (0.10, 0.40, 0.70, 0.90)
_UNNORM_KEY = "libero_spatial_no_noops"
_CENTER_CROP = True
_GLOBAL_SEED = 7
_ENVIRONMENT_SEED = 0

_TOP_LEVEL_KEYS = {
    "schema_version",
    "pilot_version",
    "suite",
    "run_status",
    "completeness_status",
    "rollout",
    "protocol",
    "runtime",
    "coverage",
    "task_results",
}
_ROLLOUT_KEYS = {
    "policy_family",
    "checkpoint_identity",
    "resolved_checkpoint_path",
    "unnorm_key",
    "center_crop",
    "camera_resolution",
    "dummy_steps",
    "max_valid_policy_actions",
    "load_in_4bit",
    "load_in_8bit",
    "global_seed",
    "environment_seed",
    "do_sample",
}
_PROTOCOL_KEYS = {
    "target_task_count",
    "target_groups_per_task",
    "observations_per_group",
    "target_relative_progress",
    "sampling_rounding",
    "group_identity_fields",
    "candidate_state_order",
    "resume_enabled",
}
_RUNTIME_KEYS = {
    "libero_revision",
    "discovered_task_count",
    "task_ids",
    "official_initial_state_counts",
}
_COVERAGE_KEYS = {
    "target_total_groups",
    "target_total_observations",
    "actual_total_groups",
    "actual_total_observations",
    "accepted_groups_per_task",
}
_TASK_RESULT_KEYS = {
    "task_id",
    "task_language",
    "available_initial_state_count",
    "target_accepted_groups",
    "actual_accepted_groups",
    "attempted_state_ids",
    "accepted_state_ids",
    "rejected_states",
    "accepted_groups",
}
_REJECTED_STATE_KEYS = {"initial_state_id", "reason", "trajectory_length"}
_ACCEPTED_GROUP_KEYS = {
    "task_id",
    "initial_state_id",
    "trajectory_length",
    "episode_success",
    "samples",
}
_SAMPLE_KEYS = {
    "sample_id",
    "step_id",
    "target_relative_progress",
    "actual_normalized_episode_progress",
    "observation_path",
}
_FATAL_ERROR_KEYS = {"category", "message", "task_id", "initial_state_id"}
_FATAL_CATEGORIES = {
    "RUNTIME_ERROR",
    "DUMMY_PHASE_ERROR",
    "OBSERVATION_ERROR",
    "ACTION_ERROR",
    "INTEGRITY_ERROR",
    "SERIALIZATION_ERROR",
    "FILESYSTEM_ERROR",
    "MANIFEST_WRITE_ERROR",
}


class PilotV02CollectionError(RuntimeError):
    """Raised when Pilot v0.2 collection cannot complete safely."""


@dataclass(frozen=True)
class PilotV02CollectionResult:
    sample_paths: tuple[Path, ...]
    manifest_path: Path
    run_status: str
    completeness_status: str | None


@dataclass(frozen=True)
class _CollectionPlan:
    task_ids: tuple[int, ...]
    target_groups_per_task: int
    formal: bool


@dataclass(frozen=True)
class _Preflight:
    destination: Path
    checkpoint_path: Path
    runtime: _c1._OfficialRuntime
    tasks: dict[int, Any]
    initial_states: dict[int, Sequence[Any]]
    task_languages: dict[int, str]
    discovered_task_ids: tuple[int, ...]
    initial_state_counts: dict[int, int]


class _ActiveFailure(RuntimeError):
    def __init__(
        self,
        category: str,
        message: str,
        *,
        task_id: int | None = None,
        initial_state_id: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.task_id = task_id
        self.initial_state_id = initial_state_id


def collect_pilot_v02_observations(
    *,
    model: Any,
    processor: Any,
    pretrained_checkpoint: str | Path,
    checkpoint_identity: str,
    libero_revision: str,
    output_dir: str | Path,
) -> PilotV02CollectionResult:
    """Collect the fixed formal Pilot v0.2 LIBERO-Spatial observation plan."""
    return _collect_pilot_v02_with_plan(
        model=model,
        processor=processor,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoint_identity=checkpoint_identity,
        libero_revision=libero_revision,
        output_dir=output_dir,
        task_ids=_FORMAL_TASK_IDS,
        target_groups_per_task=_FORMAL_GROUPS_PER_TASK,
        _formal=True,
    )


def _collect_pilot_v02_with_plan(
    *,
    model: Any,
    processor: Any,
    pretrained_checkpoint: str | Path,
    checkpoint_identity: str,
    libero_revision: str,
    output_dir: str | Path,
    task_ids: Sequence[int],
    target_groups_per_task: int,
    _formal: bool = False,
) -> PilotV02CollectionResult:
    """Run a reduced deterministic plan for later real-integration smoke tests."""
    plan = _validate_plan(task_ids, target_groups_per_task, formal=_formal)
    preflight = _run_preflight(
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoint_identity=checkpoint_identity,
        libero_revision=libero_revision,
        output_dir=output_dir,
        plan=plan,
    )
    action_config = _c1._OpenVLAActionConfig(
        pretrained_checkpoint=str(preflight.checkpoint_path),
        unnorm_key=_UNNORM_KEY,
        center_crop=_CENTER_CROP,
    )
    task_results: list[dict[str, Any]] = []
    sample_paths: list[Path] = []
    sample_ids: set[str] = set()
    group_ids: set[tuple[int, int]] = set()
    manifest_path = (preflight.destination / "collection_manifest.json").resolve()

    try:
        _create_output_layout(preflight.destination)
        observations_dir = preflight.destination / "observations"
        for task_id in plan.task_ids:
            task_result = _new_task_result(preflight, plan, task_id)
            task_results.append(task_result)
            for initial_state_id, initial_state in enumerate(
                preflight.initial_states[task_id]
            ):
                if task_result["actual_accepted_groups"] == plan.target_groups_per_task:
                    break
                task_result["attempted_state_ids"].append(initial_state_id)
                try:
                    trajectory, episode_success = _c1._collect_episode(
                        runtime=preflight.runtime,
                        action_config=action_config,
                        model=model,
                        processor=processor,
                        task=preflight.tasks[task_id],
                        initial_state=initial_state,
                        initial_state_id=initial_state_id,
                        task_id=task_id,
                    )
                except _c1._EpisodeCollectionError as error:
                    raise _ActiveFailure(
                        error.category,
                        str(error),
                        task_id=task_id,
                        initial_state_id=initial_state_id,
                    ) from error
                except Exception as error:
                    raise _ActiveFailure(
                        "RUNTIME_ERROR",
                        "unexpected LIBERO rollout failure",
                        task_id=task_id,
                        initial_state_id=initial_state_id,
                    ) from error

                trajectory_length = len(trajectory)
                if not episode_success:
                    _record_rejection(
                        task_result,
                        initial_state_id,
                        "policy_failure",
                        trajectory_length,
                    )
                    continue
                if trajectory_length < len(_TARGET_PROGRESS):
                    _record_rejection(
                        task_result,
                        initial_state_id,
                        "trajectory_too_short",
                        trajectory_length,
                    )
                    continue
                indices = _sample_indices(trajectory_length)
                if len(set(indices)) != len(_TARGET_PROGRESS):
                    _record_rejection(
                        task_result,
                        initial_state_id,
                        "sampling_index_collision",
                        trajectory_length,
                    )
                    continue

                group_id = (task_id, initial_state_id)
                if group_id in group_ids:
                    raise _ActiveFailure(
                        "INTEGRITY_ERROR",
                        f"duplicate accepted group identity: {group_id}",
                        task_id=task_id,
                        initial_state_id=initial_state_id,
                    )
                records, sample_records = _build_group_records(
                    trajectory=trajectory,
                    indices=indices,
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                    task_language=preflight.task_languages[task_id],
                    observations_dir=observations_dir,
                    known_sample_ids=sample_ids,
                )
                group_paths = _commit_group(
                    records,
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                )
                group_ids.add(group_id)
                sample_ids.update(record.sample_id for record, _ in records)
                sample_paths.extend(group_paths)
                task_result["accepted_state_ids"].append(initial_state_id)
                task_result["accepted_groups"].append(
                    {
                        "task_id": task_id,
                        "initial_state_id": initial_state_id,
                        "trajectory_length": trajectory_length,
                        "episode_success": True,
                        "samples": sample_records,
                    }
                )
                task_result["actual_accepted_groups"] += 1

        run_status = "COMPLETED" if plan.formal else "SMOKE_COMPLETED"
        completeness = _classify_completeness(task_results) if plan.formal else None
        manifest = _build_manifest(
            preflight=preflight,
            plan=plan,
            checkpoint_identity=checkpoint_identity,
            libero_revision=libero_revision,
            task_results=task_results,
            run_status=run_status,
            completeness_status=completeness,
        )
        _validate_manifest_structure(manifest, fatal=False)
        _write_manifest(manifest_path, manifest)
    except _ActiveFailure as failure:
        _handle_fatal_failure(
            failure=failure,
            preflight=preflight,
            plan=plan,
            checkpoint_identity=checkpoint_identity,
            libero_revision=libero_revision,
            task_results=task_results,
            manifest_path=manifest_path,
        )
    except Exception as error:
        failure = _ActiveFailure("RUNTIME_ERROR", "unexpected collection failure")
        _handle_fatal_failure(
            failure=failure,
            preflight=preflight,
            plan=plan,
            checkpoint_identity=checkpoint_identity,
            libero_revision=libero_revision,
            task_results=task_results,
            manifest_path=manifest_path,
            original_error=error,
        )

    return PilotV02CollectionResult(
        sample_paths=tuple(path.resolve() for path in sample_paths),
        manifest_path=manifest_path,
        run_status=run_status,
        completeness_status=completeness,
    )


def _validate_plan(
    task_ids: Sequence[int], target_groups_per_task: int, *, formal: bool
) -> _CollectionPlan:
    selected = tuple(task_ids)
    if not selected or any(type(task_id) is not int for task_id in selected):
        raise PilotV02CollectionError("task_ids must be a non-empty integer sequence")
    if tuple(sorted(selected)) != selected or len(set(selected)) != len(selected):
        raise PilotV02CollectionError("task_ids must be unique and sorted ascending")
    if type(target_groups_per_task) is not int or not 1 <= target_groups_per_task <= 5:
        raise PilotV02CollectionError("target_groups_per_task must be within [1, 5]")
    if formal and (
        selected != _FORMAL_TASK_IDS
        or target_groups_per_task != _FORMAL_GROUPS_PER_TASK
    ):
        raise PilotV02CollectionError("formal plan does not match the frozen protocol")
    return _CollectionPlan(selected, target_groups_per_task, formal)


def _run_preflight(
    *,
    pretrained_checkpoint: str | Path,
    checkpoint_identity: str,
    libero_revision: str,
    output_dir: str | Path,
    plan: _CollectionPlan,
) -> _Preflight:
    if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
        raise PilotV02CollectionError("checkpoint_identity must be a non-empty string")
    if not isinstance(libero_revision, str) or not libero_revision.strip():
        raise PilotV02CollectionError("libero_revision must be a non-empty string")
    checkpoint_path = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise PilotV02CollectionError(
            f"pretrained_checkpoint must be an existing local directory: {checkpoint_path}"
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise PilotV02CollectionError(
                f"output_dir is not a directory: {destination}"
            )
        try:
            if any(destination.iterdir()):
                raise PilotV02CollectionError(
                    f"output_dir must be empty: {destination}"
                )
        except OSError as error:
            raise PilotV02CollectionError(
                f"failed to inspect output_dir: {destination}"
            ) from error

    try:
        runtime = _c1._load_official_runtime()
        factory = runtime.benchmark.get_benchmark_dict()[_c1.PILOT_SUITE]
        suite = factory()
        discovered_task_count = suite.get_num_tasks()
        if type(discovered_task_count) is not int or discovered_task_count != 10:
            raise ValueError(
                f"expected 10 LIBERO-Spatial tasks, got {discovered_task_count!r}"
            )
        discovered_task_ids = tuple(range(discovered_task_count))
        if any(task_id not in discovered_task_ids for task_id in plan.task_ids):
            raise ValueError(f"requested task IDs are unavailable: {plan.task_ids}")

        tasks: dict[int, Any] = {}
        initial_states: dict[int, Sequence[Any]] = {}
        languages: dict[int, str] = {}
        counts: dict[int, int] = {}
        for task_id in discovered_task_ids:
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            language = task.language
            if not isinstance(language, str) or not language:
                raise ValueError(f"task {task_id} language must be non-empty")
            count = len(states)
            if task_id in plan.task_ids and count < plan.target_groups_per_task:
                raise ValueError(
                    f"task {task_id} has only {count} official initial states"
                )
            tasks[task_id] = task
            initial_states[task_id] = states
            languages[task_id] = language
            counts[task_id] = count
    except Exception as error:
        raise PilotV02CollectionError(
            "failed to initialize and validate the frozen LIBERO-Spatial runtime"
        ) from error

    return _Preflight(
        destination=destination,
        checkpoint_path=checkpoint_path,
        runtime=runtime,
        tasks=tasks,
        initial_states=initial_states,
        task_languages=languages,
        discovered_task_ids=discovered_task_ids,
        initial_state_counts=counts,
    )


def _create_output_layout(destination: Path) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(f"output_dir became non-empty: {destination}")
        (destination / "observations").mkdir()
    except OSError as error:
        raise _ActiveFailure(
            "FILESYSTEM_ERROR", "failed to create output layout"
        ) from error


def _new_task_result(
    preflight: _Preflight, plan: _CollectionPlan, task_id: int
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_language": preflight.task_languages[task_id],
        "available_initial_state_count": preflight.initial_state_counts[task_id],
        "target_accepted_groups": plan.target_groups_per_task,
        "actual_accepted_groups": 0,
        "attempted_state_ids": [],
        "accepted_state_ids": [],
        "rejected_states": [],
        "accepted_groups": [],
    }


def _record_rejection(
    task_result: dict[str, Any],
    initial_state_id: int,
    reason: str,
    trajectory_length: int | None,
) -> None:
    task_result["rejected_states"].append(
        {
            "initial_state_id": initial_state_id,
            "reason": reason,
            "trajectory_length": trajectory_length,
        }
    )


def _sample_indices(
    trajectory_length: int,
    target_progress: Sequence[float] = _TARGET_PROGRESS,
) -> tuple[int, ...]:
    return tuple(
        math.floor(float(progress) * (trajectory_length - 1) + 0.5)
        for progress in target_progress
    )


def _sample_id(task_id: int, initial_state_id: int, step_id: int) -> str:
    return (
        f"libero_spatial__task{task_id:02d}__state{initial_state_id:02d}"
        f"__step{step_id:04d}"
    )


def _build_group_records(
    *,
    trajectory: Sequence[_c1._BufferedObservation],
    indices: Sequence[int],
    task_id: int,
    initial_state_id: int,
    task_language: str,
    observations_dir: Path,
    known_sample_ids: set[str],
) -> tuple[
    list[tuple[PilotObservation, Path]],
    list[dict[str, Any]],
]:
    trajectory_length = len(trajectory)
    records: list[tuple[PilotObservation, Path]] = []
    sample_records: list[dict[str, Any]] = []
    local_ids: set[str] = set()
    for target_progress, index in zip(_TARGET_PROGRESS, indices, strict=True):
        buffered = trajectory[index]
        sample_id = _sample_id(task_id, initial_state_id, buffered.step_id)
        if sample_id in known_sample_ids or sample_id in local_ids:
            raise _ActiveFailure(
                "INTEGRITY_ERROR",
                f"duplicate sample_id: {sample_id}",
                task_id=task_id,
                initial_state_id=initial_state_id,
            )
        local_ids.add(sample_id)
        final_path = observations_dir / f"{sample_id}.npz"
        if final_path.exists():
            raise _ActiveFailure(
                "INTEGRITY_ERROR",
                f"observation destination already exists: {final_path}",
                task_id=task_id,
                initial_state_id=initial_state_id,
            )
        actual_progress = buffered.step_id / (trajectory_length - 1)
        try:
            record = PilotObservation(
                sample_id=sample_id,
                task_id=str(task_id),
                initial_state_id=initial_state_id,
                episode_id=initial_state_id,
                step_id=buffered.step_id,
                normalized_episode_progress=actual_progress,
                base_rgb_raw=buffered.base_rgb_raw,
                wrist_rgb_raw=buffered.wrist_rgb_raw,
                state=buffered.state,
                prompt=task_language,
                episode_success=True,
            )
        except Exception as error:
            raise _ActiveFailure(
                "OBSERVATION_ERROR",
                f"invalid PilotObservation for sample {sample_id}",
                task_id=task_id,
                initial_state_id=initial_state_id,
            ) from error
        records.append((record, final_path))
        sample_records.append(
            {
                "sample_id": sample_id,
                "step_id": buffered.step_id,
                "target_relative_progress": target_progress,
                "actual_normalized_episode_progress": actual_progress,
                "observation_path": f"observations/{sample_id}.npz",
            }
        )
    return records, sample_records


def _commit_group(
    records: Sequence[tuple[PilotObservation, Path]],
    *,
    task_id: int | None = None,
    initial_state_id: int | None = None,
) -> tuple[Path, ...]:
    temporary_paths: list[Path] = []
    committed_paths: list[Path] = []
    try:
        for record, final_path in records:
            temporary_path = final_path.with_name(f".{final_path.name}.tmp")
            temporary_paths.append(temporary_path)
            try:
                record.save(temporary_path)
            except OSError as error:
                raise _ActiveFailure(
                    "FILESYSTEM_ERROR",
                    f"failed to write sample {record.sample_id}",
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                ) from error
            except Exception as error:
                raise _ActiveFailure(
                    "SERIALIZATION_ERROR",
                    f"failed to serialize sample {record.sample_id}",
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                ) from error
        for temporary_path, (_, final_path) in zip(
            temporary_paths, records, strict=True
        ):
            try:
                os.replace(temporary_path, final_path)
            except OSError as error:
                raise _ActiveFailure(
                    "FILESYSTEM_ERROR",
                    f"failed to commit sample {final_path.name}",
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                ) from error
            committed_paths.append(final_path)
    except _ActiveFailure as failure:
        cleanup_errors = _cleanup_paths((*temporary_paths, *committed_paths))
        if cleanup_errors:
            rollback_failure = _ActiveFailure(
                "FILESYSTEM_ERROR",
                "failed to rollback incomplete observation group",
                task_id=task_id,
                initial_state_id=initial_state_id,
            )
            for cleanup_error in cleanup_errors:
                rollback_failure.add_note(f"rollback failure: {cleanup_error}")
            raise rollback_failure from failure
        raise
    return tuple(path.resolve() for path in committed_paths)


def _cleanup_paths(paths: Sequence[Path]) -> list[OSError]:
    errors: list[OSError] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(error)
    return errors


def _build_manifest(
    *,
    preflight: _Preflight,
    plan: _CollectionPlan,
    checkpoint_identity: str,
    libero_revision: str,
    task_results: list[dict[str, Any]],
    run_status: str,
    completeness_status: str | None,
) -> dict[str, Any]:
    accepted_counts = {
        str(task_id): next(
            (
                result["actual_accepted_groups"]
                for result in task_results
                if result["task_id"] == task_id
            ),
            0,
        )
        for task_id in plan.task_ids
    }
    actual_groups = sum(accepted_counts.values())
    return {
        "schema_version": _SCHEMA_VERSION,
        "pilot_version": _PILOT_VERSION,
        "suite": _c1.PILOT_SUITE,
        "run_status": run_status,
        "completeness_status": completeness_status,
        "rollout": {
            "policy_family": "openvla",
            "checkpoint_identity": checkpoint_identity,
            "resolved_checkpoint_path": str(preflight.checkpoint_path),
            "unnorm_key": _UNNORM_KEY,
            "center_crop": _CENTER_CROP,
            "camera_resolution": _c1._CAMERA_RESOLUTION,
            "dummy_steps": _c1._NUM_DUMMY_STEPS,
            "max_valid_policy_actions": _c1._MAX_POLICY_ACTIONS,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "global_seed": _GLOBAL_SEED,
            "environment_seed": _ENVIRONMENT_SEED,
            "do_sample": False,
        },
        "protocol": {
            "target_task_count": len(plan.task_ids),
            "target_groups_per_task": plan.target_groups_per_task,
            "observations_per_group": len(_TARGET_PROGRESS),
            "target_relative_progress": list(_TARGET_PROGRESS),
            "sampling_rounding": "floor(q*(T-1)+0.5)",
            "group_identity_fields": ["task_id", "initial_state_id"],
            "candidate_state_order": "ascending_official_index",
            "resume_enabled": False,
        },
        "runtime": {
            "libero_revision": libero_revision,
            "discovered_task_count": len(preflight.discovered_task_ids),
            "task_ids": list(preflight.discovered_task_ids),
            "official_initial_state_counts": {
                str(task_id): preflight.initial_state_counts[task_id]
                for task_id in preflight.discovered_task_ids
            },
        },
        "coverage": {
            "target_total_groups": len(plan.task_ids) * plan.target_groups_per_task,
            "target_total_observations": len(plan.task_ids)
            * plan.target_groups_per_task
            * len(_TARGET_PROGRESS),
            "actual_total_groups": actual_groups,
            "actual_total_observations": actual_groups * len(_TARGET_PROGRESS),
            "accepted_groups_per_task": accepted_counts,
        },
        "task_results": task_results,
    }


def _classify_completeness(task_results: Sequence[dict[str, Any]]) -> str:
    _validate_accepted_structure(task_results)
    counts = [result["actual_accepted_groups"] for result in task_results]
    total = sum(counts)
    if len(counts) == 10 and all(count == 5 for count in counts) and total == 50:
        return "COMPLETE"
    if (
        len(counts) == 10
        and all(count in (4, 5) for count in counts)
        and 4 in counts
        and 40 <= total <= 49
    ):
        return "USABLE_WITH_SHORTFALL"
    return "BLOCKED"


def _validate_accepted_structure(task_results: Sequence[dict[str, Any]]) -> None:
    sample_ids: set[str] = set()
    group_ids: set[tuple[int, int]] = set()
    for task_result in task_results:
        groups = task_result["accepted_groups"]
        if task_result["actual_accepted_groups"] != len(groups):
            raise _ActiveFailure("INTEGRITY_ERROR", "accepted-group count mismatch")
        for group in groups:
            group_id = (group["task_id"], group["initial_state_id"])
            if group_id in group_ids or len(group["samples"]) != 4:
                raise _ActiveFailure("INTEGRITY_ERROR", "malformed accepted group")
            group_ids.add(group_id)
            for sample in group["samples"]:
                if sample["sample_id"] in sample_ids:
                    raise _ActiveFailure(
                        "INTEGRITY_ERROR", "duplicate manifest sample_id"
                    )
                sample_ids.add(sample["sample_id"])


def _validate_manifest_structure(manifest: dict[str, Any], *, fatal: bool) -> None:
    expected_top = _TOP_LEVEL_KEYS | ({"fatal_error"} if fatal else set())
    _require_exact_keys(manifest, expected_top, "manifest")
    _require_exact_keys(manifest["rollout"], _ROLLOUT_KEYS, "rollout")
    _require_exact_keys(manifest["protocol"], _PROTOCOL_KEYS, "protocol")
    _require_exact_keys(manifest["runtime"], _RUNTIME_KEYS, "runtime")
    _require_exact_keys(manifest["coverage"], _COVERAGE_KEYS, "coverage")
    for task_result in manifest["task_results"]:
        _require_exact_keys(task_result, _TASK_RESULT_KEYS, "task_result")
        for rejected in task_result["rejected_states"]:
            _require_exact_keys(rejected, _REJECTED_STATE_KEYS, "rejected_state")
        for group in task_result["accepted_groups"]:
            _require_exact_keys(group, _ACCEPTED_GROUP_KEYS, "accepted_group")
            for sample in group["samples"]:
                _require_exact_keys(sample, _SAMPLE_KEYS, "sample")
    if fatal:
        _require_exact_keys(manifest["fatal_error"], _FATAL_ERROR_KEYS, "fatal_error")
        if manifest["fatal_error"]["category"] not in _FATAL_CATEGORIES:
            raise _ActiveFailure("INTEGRITY_ERROR", "invalid fatal category")
    _validate_accepted_structure(manifest["task_results"])


def _require_exact_keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise _ActiveFailure(
            "INTEGRITY_ERROR",
            f"invalid {context} fields: {sorted(value) if isinstance(value, dict) else type(value).__name__}",
        )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        try:
            payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        except Exception as error:
            raise _ActiveFailure(
                "SERIALIZATION_ERROR", "failed to serialize collection manifest"
            ) from error
        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
                output.write(payload)
            os.replace(temporary_path, path)
        except OSError as error:
            raise _ActiveFailure(
                "MANIFEST_WRITE_ERROR", "failed to write collection manifest"
            ) from error
    finally:
        _cleanup_paths((temporary_path,))


def _handle_fatal_failure(
    *,
    failure: _ActiveFailure,
    preflight: _Preflight,
    plan: _CollectionPlan,
    checkpoint_identity: str,
    libero_revision: str,
    task_results: list[dict[str, Any]],
    manifest_path: Path,
    original_error: Exception | None = None,
) -> None:
    fatal_manifest = _build_manifest(
        preflight=preflight,
        plan=plan,
        checkpoint_identity=checkpoint_identity,
        libero_revision=libero_revision,
        task_results=task_results,
        run_status="FATAL",
        completeness_status=None,
    )
    fatal_manifest["fatal_error"] = {
        "category": failure.category,
        "message": str(failure),
        "task_id": failure.task_id,
        "initial_state_id": failure.initial_state_id,
    }
    try:
        _validate_manifest_structure(fatal_manifest, fatal=True)
        _write_manifest(manifest_path, fatal_manifest)
    except Exception as manifest_error:
        failure.add_note(f"fatal manifest could not be written: {manifest_error}")
    if original_error is not None and failure.__cause__ is None:
        failure.__cause__ = original_error
    raise PilotV02CollectionError(str(failure)) from failure
