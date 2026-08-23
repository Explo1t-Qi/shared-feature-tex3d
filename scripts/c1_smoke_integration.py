from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX3D_OPENVLA_ROOT = PROJECT_ROOT.parent / "tex3d" / "openvla"
UNNORM_KEY = "libero_spatial_no_noops"
INITIAL_STATE_IDS = (0, 1)
NUM_SAMPLES_PER_EPISODE = 5
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


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
        description="Run and verify the real C1 LIBERO/OpenVLA smoke integration."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local OpenVLA LIBERO-spatial checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New dedicated smoke output directory; it must not already exist.",
    )
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=DEFAULT_TEX3D_OPENVLA_ROOT,
        help="Official Tex3D OpenVLA source root.",
    )
    return parser.parse_args()


def _image_digest(value: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    tex3d_openvla_root = args.tex3d_openvla_root.resolve()
    output_dir = args.output_dir.resolve()
    samples_dir = output_dir / "samples"

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"dataset statistics file not found: {statistics_path}"
        )
    if not tex3d_openvla_root.is_dir():
        raise FileNotFoundError(
            f"official Tex3D OpenVLA root not found: {tex3d_openvla_root}"
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(tex3d_openvla_root))

    import numpy as np
    import torch

    import shared_feature.libero_collector as collector
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from shared_feature import PilotObservation

    with statistics_path.open(encoding="utf-8") as source:
        dataset_statistics = json.load(source)
    if UNNORM_KEY not in dataset_statistics:
        raise KeyError(
            f"{UNNORM_KEY!r} missing from {statistics_path}; "
            f"available={sorted(dataset_statistics)}"
        )

    class TrackedEnv:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.actions: list[list[float]] = []
            self.query_env_step_counts: list[int] = []
            self.policy_input_raw_hashes: list[dict[str, str]] = []
            self.last_done = False
            self.last_success = False
            self.previous_returned_hashes: dict[str, str] | None = None
            self.last_returned_hashes: dict[str, str] | None = None
            self.close_called = False
            self.closed_cleanly = False

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def step(self, action: list[float]) -> Any:
            self.actions.append(list(action))
            result = self.inner.step(action)
            observation, _, done, _ = result
            self.previous_returned_hashes = self.last_returned_hashes
            self.last_returned_hashes = {
                "base": _image_digest(observation["agentview_image"]),
                "wrist": _image_digest(
                    observation["robot0_eye_in_hand_image"]
                ),
            }
            self.last_done = bool(done)
            return result

        def check_success(self) -> bool:
            self.last_success = bool(self.inner.check_success())
            return self.last_success

        def close(self) -> None:
            self.close_called = True
            self.inner.close()
            self.closed_cleanly = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the smoke runtime")

    set_seed_everywhere(7)
    model_config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    print("CUDA_DEVICE:", torch.cuda.get_device_name(0))
    print("MODEL_CONFIG:", vars(model_config))

    model = get_model(model_config)
    processor = get_processor(model_config)

    official_runtime = collector._load_official_runtime()
    tracked_envs: list[TrackedEnv] = []
    task_descriptions: list[str] = []

    def tracked_get_libero_env(
        task: Any,
        model_family: str,
        resolution: int,
    ) -> tuple[TrackedEnv, str]:
        inner, description = official_runtime.get_libero_env(
            task,
            model_family,
            resolution=resolution,
        )
        tracked = TrackedEnv(inner)
        tracked_envs.append(tracked)
        task_descriptions.append(description)
        return tracked, description

    def tracked_get_action(
        cfg: Any,
        action_model: Any,
        observation: Any,
        task_description: str,
        processor: Any = None,
    ) -> Any:
        tracked = tracked_envs[-1]
        tracked.query_env_step_counts.append(len(tracked.actions))
        if tracked.last_returned_hashes is None:
            raise AssertionError("policy queried before an environment observation")
        tracked.policy_input_raw_hashes.append(
            dict(tracked.last_returned_hashes)
        )
        return official_runtime.get_action(
            cfg,
            action_model,
            observation,
            task_description,
            processor=processor,
        )

    instrumented_runtime = replace(
        official_runtime,
        get_libero_env=tracked_get_libero_env,
        get_action=tracked_get_action,
    )
    collector._load_official_runtime = lambda: instrumented_runtime

    returned_paths = collector.collect_pilot_observations(
        model=model,
        processor=processor,
        pretrained_checkpoint=checkpoint,
        output_dir=samples_dir,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
        initial_state_ids=INITIAL_STATE_IDS,
        num_samples_per_episode=NUM_SAMPLES_PER_EPISODE,
    )

    files = sorted(samples_dir.glob("*.npz"))
    if len(returned_paths) != 10 or len(files) != 10:
        raise AssertionError(
            "smoke collector must produce exactly 10 NPZ files: "
            f"returned={len(returned_paths)}, files={len(files)}"
        )
    if {path.resolve() for path in returned_paths} != {
        path.resolve() for path in files
    }:
        raise AssertionError("returned paths do not match serialized NPZ files")
    if len(tracked_envs) != 2:
        raise AssertionError(
            f"expected two real environments, got {len(tracked_envs)}"
        )

    report: dict[str, Any] = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "model_family": "openvla",
        "unnorm_key": UNNORM_KEY,
        "center_crop": True,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "camera_resolution": 512,
        "dummy_steps": 10,
        "max_policy_actions": 300,
        "initial_state_ids": list(INITIAL_STATE_IDS),
        "num_samples_per_episode": NUM_SAMPLES_PER_EPISODE,
        "output_directory": str(samples_dir),
        "file_count": len(files),
        "episodes": [],
    }

    for episode_index, initial_state_id in enumerate(INITIAL_STATE_IDS):
        episode_files = [
            path
            for path in files
            if f"__state{initial_state_id:02d}__" in path.stem
        ]
        if len(episode_files) != NUM_SAMPLES_PER_EPISODE:
            raise AssertionError(
                f"state {initial_state_id} produced {len(episode_files)} files"
            )

        records = [PilotObservation.load(path) for path in episode_files]
        records.sort(key=lambda record: record.step_id)
        tracked = tracked_envs[episode_index]
        policy_query_count = len(tracked.query_env_step_counts)
        steps = [record.step_id for record in records]
        progress = [
            float(record.normalized_episode_progress) for record in records
        ]
        successes = {record.episode_success for record in records}

        if len(successes) != 1:
            raise AssertionError(
                f"state {initial_state_id} has inconsistent success metadata"
            )
        if steps[0] != 0 or any(
            left >= right for left, right in zip(steps, steps[1:])
        ):
            raise AssertionError(
                f"state {initial_state_id} has invalid sampled steps: {steps}"
            )
        if steps[-1] != policy_query_count - 1:
            raise AssertionError(
                "final sample is not the final valid-policy observation: "
                f"state={initial_state_id}, final={steps[-1]}, "
                f"queries={policy_query_count}"
            )
        if progress[0] != 0.0 or progress[-1] != 1.0:
            raise AssertionError(
                f"state {initial_state_id} does not span progress [0, 1]"
            )

        for record in records:
            expected_sample_id = (
                f"libero_spatial__task02__state{initial_state_id:02d}"
                f"__step{record.step_id:04d}"
            )
            if record.sample_id != expected_sample_id:
                raise AssertionError(
                    f"unexpected sample_id: {record.sample_id}"
                )
            if record.task_id != "2":
                raise AssertionError(f"unexpected task_id: {record.task_id}")
            if record.initial_state_id != initial_state_id:
                raise AssertionError("initial_state_id mismatch")
            if record.episode_id != initial_state_id:
                raise AssertionError("episode_id mismatch")
            if record.state.shape != (8,):
                raise AssertionError(
                    f"unexpected state shape: {record.state.shape}"
                )
            if record.prompt != task_descriptions[episode_index]:
                raise AssertionError("prompt does not match LIBERO task language")

            input_hashes = tracked.policy_input_raw_hashes[record.step_id]
            if _image_digest(record.base_rgb_raw) != input_hashes["base"]:
                raise AssertionError(
                    "saved base image differs from raw policy-query observation"
                )
            if _image_digest(record.wrist_rgb_raw) != input_hashes["wrist"]:
                raise AssertionError(
                    "saved wrist image differs from raw policy-query observation"
                )

        base_shapes = {tuple(record.base_rgb_raw.shape) for record in records}
        wrist_shapes = {
            tuple(record.wrist_rgb_raw.shape) for record in records
        }
        base_dtypes = {str(record.base_rgb_raw.dtype) for record in records}
        wrist_dtypes = {
            str(record.wrist_rgb_raw.dtype) for record in records
        }
        state_shapes = {tuple(record.state.shape) for record in records}
        state_dtypes = {str(record.state.dtype) for record in records}

        if base_shapes != {(512, 512, 3)}:
            raise AssertionError(f"unexpected base image shapes: {base_shapes}")
        if wrist_shapes != {(512, 512, 3)}:
            raise AssertionError(
                f"unexpected wrist image shapes: {wrist_shapes}"
            )
        if (224, 224, 3) in base_shapes or (224, 224, 3) in wrist_shapes:
            raise AssertionError("stored raw images are 224x224 model inputs")

        if len(tracked.actions) != 10 + policy_query_count:
            raise AssertionError("environment action count mismatch")
        if tracked.actions[:10] != [DUMMY_ACTION] * 10:
            raise AssertionError("the first 10 actions were not dummy actions")
        expected_query_steps = list(range(10, 10 + policy_query_count))
        if tracked.query_env_step_counts != expected_query_steps:
            raise AssertionError(
                "policy queries did not begin immediately after 10 dummy steps"
            )
        if not tracked.close_called or not tracked.closed_cleanly:
            raise AssertionError("LIBERO environment did not close cleanly")
        if tracked.previous_returned_hashes is None:
            raise AssertionError("missing final policy-input provenance")

        final_record = records[-1]
        if (
            _image_digest(final_record.base_rgb_raw)
            != tracked.previous_returned_hashes["base"]
        ):
            raise AssertionError("terminal base observation was serialized")
        if (
            _image_digest(final_record.wrist_rgb_raw)
            != tracked.previous_returned_hashes["wrist"]
        ):
            raise AssertionError("terminal wrist observation was serialized")

        episode_success = successes.pop()
        if episode_success != tracked.last_success:
            raise AssertionError("saved success does not match env.check_success()")

        if tracked.last_success:
            termination_reason = "task_success"
        elif tracked.last_done:
            termination_reason = "done_without_success"
        elif policy_query_count == 300:
            termination_reason = "max_policy_actions"
        else:
            raise AssertionError(
                "episode stopped without success, done, or policy-action limit"
            )

        report["episodes"].append(
            {
                "initial_state_id": initial_state_id,
                "episode_success": episode_success,
                "termination_reason": termination_reason,
                "sampled_step_ids": steps,
                "progress_values": progress,
                "policy_query_count": policy_query_count,
                "dummy_steps_verified": 10,
                "environment_closed_cleanly": tracked.closed_cleanly,
                "base_rgb_raw_shapes": sorted(base_shapes),
                "base_rgb_raw_dtypes": sorted(base_dtypes),
                "wrist_rgb_raw_shapes": sorted(wrist_shapes),
                "wrist_rgb_raw_dtypes": sorted(wrist_dtypes),
                "state_shapes": sorted(state_shapes),
                "state_dtypes": sorted(state_dtypes),
                "prompt": records[0].prompt,
                "raw_policy_input_provenance_verified": True,
                "terminal_observation_excluded": True,
                "terminal_image_differs_from_last_policy_input": (
                    tracked.last_returned_hashes
                    != tracked.previous_returned_hashes
                ),
            }
        )

    return report


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing smoke output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)

    log_path = args.output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log)
        sys.stderr = _Tee(original_stderr, log)
        try:
            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("TEX3D_OPENVLA_ROOT:", args.tex3d_openvla_root.resolve())
            print("CHECKPOINT:", args.checkpoint.resolve())
            print("OUTPUT_DIR:", args.output_dir)
            report = _run_smoke(args)
            report_path = args.output_dir / "smoke_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("SMOKE_REPORT:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print("REPORT_PATH:", report_path)
            return 0
        except Exception as error:
            traceback.print_exc()
            failure_path = args.output_dir / "smoke_failure.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print("FAILURE_PATH:", failure_path, file=sys.stderr)
            return 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
