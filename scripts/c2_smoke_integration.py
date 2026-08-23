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
DEFAULT_CHECKPOINT = Path(
    "/data/huangsimin/openvla-7b-finetuned-libero-spatial"
)
DEFAULT_OBSERVATIONS_DIR = PROJECT_ROOT / "experiment_inbox" / "samples"
DEFAULT_TEX3D_OPENVLA_ROOT = PROJECT_ROOT.parent / "tex3d" / "openvla"
UNNORM_KEY = "libero_spatial_no_noops"
NUM_SAMPLES = 2
EXPECTED_ARCHIVE_FIELDS = {
    "metadata_json",
    "o1_siglip",
    "o1_fused",
    "o2_projected",
}
EXPECTED_FEATURE_SHAPES = {
    "o1_siglip": (256, 1152),
    "o1_fused": (256, 2176),
    "o2_projected": (256, 4096),
}
EXPECTED_METADATA_FIELDS = {
    "sample_id",
    "source_model",
    "checkpoint",
    "feature_schema_version",
    "source_image_hash",
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


class _RecordingProcessor:
    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.image_arrays: list[Any] = []
        self.pixel_values: list[Any] = []

    def __call__(self, text: Any, images: Any, **kwargs: Any) -> Any:
        import numpy as np

        image_list = images if isinstance(images, list) else [images]
        self.image_arrays.extend(
            np.array(image, copy=True) for image in image_list
        )
        output = self._processor(text, images, **kwargs)
        self.pixel_values.extend(
            tensor.detach().cpu().clone()
            for tensor in output["pixel_values"]
        )
        return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real C2 OpenVLA/GPU feature-extraction smoke."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Local OpenVLA LIBERO-spatial checkpoint directory.",
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=DEFAULT_OBSERVATIONS_DIR,
        help="Directory containing the existing C1 PilotObservation samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh C2 smoke output directory; it must not already exist.",
    )
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=DEFAULT_TEX3D_OPENVLA_ROOT,
        help="Official Tex3D OpenVLA source root used by the C1 runtime.",
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
            "C2 smoke requires two C1 samples from distinct episodes, got "
            f"states={sorted(selected_states)}"
        )
    return tuple(selected)


def _build_prompt(
    task_label: str,
    checkpoint: str,
    openvla_v01_system_prompt: str,
) -> str:
    if "openvla-v01" in checkpoint:
        return (
            f"{openvla_v01_system_prompt} USER: What action should the robot "
            f"take to {task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def _reference_preprocessing(
    *,
    record: Any,
    checkpoint: str,
    processor: Any,
    get_libero_image: Any,
    crop_and_resize: Any,
    openvla_v01_system_prompt: str,
    tf: Any,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    libero_512 = get_libero_image(
        {"agentview_image": record.base_rgb_raw},
        512,
    )
    pre_center_crop = np.array(
        Image.fromarray(libero_512).resize((224, 224))
    )
    image = Image.fromarray(pre_center_crop).convert("RGB")

    tensor = tf.convert_to_tensor(np.array(image))
    original_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = crop_and_resize(tensor, 0.9, 1)
    tensor = tf.clip_by_value(tensor, 0, 1)
    tensor = tf.image.convert_image_dtype(
        tensor,
        original_dtype,
        saturate=True,
    )
    image = Image.fromarray(tensor.numpy()).convert("RGB")
    post_center_crop = np.array(image, copy=True)

    prompt = _build_prompt(
        record.prompt,
        checkpoint,
        openvla_v01_system_prompt,
    )
    processor_output = processor(prompt, image)
    pixel_values = processor_output["pixel_values"].detach().cpu().clone()
    return {
        "libero_512": np.array(libero_512, copy=True),
        "pre_center_crop": pre_center_crop,
        "post_center_crop": post_center_crop,
        "pixel_values": pixel_values,
    }


def _tensor_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_equal": False,
            "torch_equal": False,
            "torch_allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    difference = (reference.float() - candidate.float()).abs()
    return {
        "shape_equal": True,
        "torch_equal": bool(torch.equal(reference, candidate)),
        "torch_allclose": bool(torch.allclose(reference, candidate)),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
    }


def _numpy_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "array_equal": False,
            "allclose": False,
            "max_abs_diff": None,
        }
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "shape_equal": True,
        "array_equal": bool(np.array_equal(reference, candidate)),
        "allclose": bool(np.allclose(reference, candidate)),
        "max_abs_diff": float(difference.max()),
    }


def _load_and_validate_feature(
    *,
    path: Path,
    record: Any,
    checkpoint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARCHIVE_FIELDS:
            raise AssertionError(
                f"unexpected feature archive fields for {record.sample_id}: "
                f"{sorted(archive.files)}"
            )
        metadata_value = archive["metadata_json"]
        if metadata_value.ndim != 0 or metadata_value.dtype.kind != "U":
            raise AssertionError("metadata_json is not a Unicode scalar")
        metadata = json.loads(str(metadata_value.item()))
        arrays = {
            name: archive[name].copy() for name in EXPECTED_FEATURE_SHAPES
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
        "source_model": "openvla",
        "checkpoint": checkpoint,
        "feature_schema_version": "openvla_features_v1",
        "source_image_hash": expected_hash,
    }
    if metadata != expected_metadata:
        raise AssertionError(
            f"feature provenance mismatch for {record.sample_id}: "
            f"actual={metadata}, expected={expected_metadata}"
        )

    for name, expected_shape in EXPECTED_FEATURE_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise AssertionError(
                f"unexpected {name} layout for {record.sample_id}: "
                f"shape={value.shape}, dtype={value.dtype}"
            )
        if value.ndim != 2 or value.shape[0] != 256:
            raise AssertionError(f"{name} does not preserve all patch tokens")
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"{name} contains non-finite values")
        if not np.any(value != 0):
            raise AssertionError(f"{name} contains only zeros")
    return metadata, arrays


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    observations_dir = args.observations_dir.resolve()
    output_dir = args.output_dir.resolve()
    features_dir = output_dir / "features"
    tex3d_openvla_root = args.tex3d_openvla_root.resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"dataset statistics file not found: {statistics_path}"
        )
    with statistics_path.open(encoding="utf-8") as source:
        statistics = json.load(source)
    if UNNORM_KEY not in statistics:
        raise KeyError(
            f"{UNNORM_KEY!r} missing from {statistics_path}; "
            f"available={sorted(statistics)}"
        )
    if not tex3d_openvla_root.is_dir():
        raise FileNotFoundError(
            f"official Tex3D OpenVLA root not found: {tex3d_openvla_root}"
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(tex3d_openvla_root))

    import numpy as np
    import tensorflow as tf
    import torch
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.openvla_utils import (
        OPENVLA_V01_SYSTEM_PROMPT,
        crop_and_resize,
        get_processor,
    )
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from shared_feature import PilotObservation, extract_openvla_features
    import shared_feature.openvla_features as openvla_features

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the smoke runtime")

    observation_paths = _select_observations(observations_dir)
    records = [PilotObservation.load(path) for path in observation_paths]
    checkpoint_string = str(checkpoint)

    set_seed_everywhere(7)
    model_config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint_string,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    model = get_model(model_config)
    processor = get_processor(model_config)

    vision_backbone = getattr(model, "vision_backbone", None)
    dino_featurizer = getattr(vision_backbone, "featurizer", None)
    siglip_featurizer = getattr(vision_backbone, "fused_featurizer", None)
    projector = getattr(model, "projector", None)
    if not all(
        callable(value)
        for value in (dino_featurizer, siglip_featurizer, projector)
    ):
        raise AssertionError(
            "real model does not expose the frozen fused OpenVLA structure"
        )
    visual_parameter = next(
        parameter
        for parameter in vision_backbone.parameters()
        if torch.is_floating_point(parameter)
    )
    if visual_parameter.device.type != "cuda":
        raise AssertionError(
            f"visual backbone is not on GPU: {visual_parameter.device}"
        )
    if visual_parameter.dtype != torch.bfloat16:
        raise AssertionError(
            f"visual backbone is not BF16: {visual_parameter.dtype}"
        )

    references = [
        _reference_preprocessing(
            record=record,
            checkpoint=checkpoint_string,
            processor=processor,
            get_libero_image=get_libero_image,
            crop_and_resize=crop_and_resize,
            openvla_v01_system_prompt=OPENVLA_V01_SYSTEM_PROMPT,
            tf=tf,
        )
        for record in records
    ]

    c2_libero_images: list[Any] = []
    c2_pre_crop_images: list[Any] = []
    c2_post_crop_images: list[Any] = []
    base_runtime = openvla_features._load_preprocessing_runtime()

    def recording_get_libero_image(observation: Any, resize_size: int) -> Any:
        value = base_runtime.get_libero_image(observation, resize_size)
        c2_libero_images.append(np.array(value, copy=True))
        return value

    def recording_center_crop_image(image: Any) -> Any:
        c2_pre_crop_images.append(np.array(image, copy=True))
        value = base_runtime.center_crop_image(image)
        c2_post_crop_images.append(np.array(value, copy=True))
        return value

    instrumented_runtime = replace(
        base_runtime,
        get_libero_image=recording_get_libero_image,
        center_crop_image=recording_center_crop_image,
    )
    recording_processor = _RecordingProcessor(processor)
    original_runtime_loader = openvla_features._load_preprocessing_runtime
    openvla_features._load_preprocessing_runtime = lambda: instrumented_runtime
    try:
        feature_paths = extract_openvla_features(
            model=model,
            processor=recording_processor,
            pretrained_checkpoint=checkpoint_string,
            observation_paths=observation_paths,
            output_dir=features_dir,
            center_crop=True,
            batch_size=1,
        )
    finally:
        openvla_features._load_preprocessing_runtime = original_runtime_loader

    captured_lengths = {
        len(c2_libero_images),
        len(c2_pre_crop_images),
        len(c2_post_crop_images),
        len(recording_processor.image_arrays),
        len(recording_processor.pixel_values),
        len(feature_paths),
    }
    if captured_lengths != {NUM_SAMPLES}:
        raise AssertionError(
            f"C2 smoke capture counts are inconsistent: {captured_lengths}"
        )

    report: dict[str, Any] = {
        "status": "PASS",
        "checkpoint": checkpoint_string,
        "unnorm_key": UNNORM_KEY,
        "center_crop": True,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "batch_size": 1,
        "samples": [record.sample_id for record in records],
        "observation_paths": [str(path) for path in observation_paths],
        "output_directory": str(features_dir),
        "runtime_audit": {
            "cuda_device": torch.cuda.get_device_name(visual_parameter.device),
            "model_class": _qualified_class_name(model),
            "processor_class": _qualified_class_name(processor),
            "vision_backbone_class": _qualified_class_name(vision_backbone),
            "dino_featurizer_class": _qualified_class_name(dino_featurizer),
            "siglip_featurizer_class": _qualified_class_name(siglip_featurizer),
            "projector_class": _qualified_class_name(projector),
            "visual_parameter_device": str(visual_parameter.device),
            "visual_parameter_dtype": str(visual_parameter.dtype),
            "processor_pixel_values_dtype": str(
                references[0]["pixel_values"].dtype
            ),
        },
        "sample_results": [],
    }

    for index, (record, feature_path, reference) in enumerate(
        zip(records, feature_paths, references, strict=True)
    ):
        c2_pixel_values = recording_processor.pixel_values[index].unsqueeze(0)
        if tuple(reference["pixel_values"].shape) != (1, 6, 224, 224):
            raise AssertionError(
                "reference processor pixel_values has unexpected shape: "
                f"{tuple(reference['pixel_values'].shape)}"
            )
        if tuple(c2_pixel_values.shape) != (1, 6, 224, 224):
            raise AssertionError(
                "C2 processor pixel_values has unexpected shape: "
                f"{tuple(c2_pixel_values.shape)}"
            )

        stage_equality = {
            "libero_512": bool(
                np.array_equal(reference["libero_512"], c2_libero_images[index])
            ),
            "pre_center_crop": bool(
                np.array_equal(
                    reference["pre_center_crop"],
                    c2_pre_crop_images[index],
                )
            ),
            "post_center_crop": bool(
                np.array_equal(
                    reference["post_center_crop"],
                    c2_post_crop_images[index],
                )
            ),
            "processor_input": bool(
                np.array_equal(
                    reference["post_center_crop"],
                    recording_processor.image_arrays[index],
                )
            ),
        }
        pixel_comparison = _tensor_comparison(
            reference["pixel_values"],
            c2_pixel_values,
        )
        if not all(stage_equality.values()) or not pixel_comparison["torch_equal"]:
            first_difference = next(
                (name for name, equal in stage_equality.items() if not equal),
                "processor",
            )
            raise AssertionError(
                "C1-reference/C2 preprocessing mismatch: "
                f"sample={record.sample_id}, first_difference={first_difference}, "
                f"stages={stage_equality}, pixels={pixel_comparison}"
            )

        metadata, serialized = _load_and_validate_feature(
            path=feature_path,
            record=record,
            checkpoint=checkpoint_string,
        )
        model_pixels = c2_pixel_values.to(
            device=visual_parameter.device,
            dtype=visual_parameter.dtype,
        )
        model.eval()
        with torch.inference_mode():
            dino_pixels, siglip_pixels = torch.split(
                model_pixels,
                [3, 3],
                dim=1,
            )
            dino = dino_featurizer(dino_pixels)
            siglip = siglip_featurizer(siglip_pixels)
            fused = torch.cat([dino, siglip], dim=-1)
            projected = projector(fused)

        direct_tensors = {
            "o1_siglip": siglip,
            "o1_fused": fused,
            "o2_projected": projected,
        }
        direct_arrays = {
            name: tensor[0].detach().to(dtype=torch.float32, device="cpu").numpy()
            for name, tensor in direct_tensors.items()
        }
        comparisons = {
            name: _numpy_comparison(direct_arrays[name], serialized[name])
            for name in EXPECTED_FEATURE_SHAPES
        }
        if not all(result["array_equal"] for result in comparisons.values()):
            raise AssertionError(
                "serialized/direct feature mismatch: "
                f"sample={record.sample_id}, comparisons={comparisons}"
            )

        report["sample_results"].append(
            {
                "sample_id": record.sample_id,
                "preprocessing": {
                    "pixel_values_shape": list(c2_pixel_values.shape),
                    "pixel_values_dtype": str(c2_pixel_values.dtype),
                    "stage_equality": stage_equality,
                    **pixel_comparison,
                },
                "features": {
                    name: {
                        "shape": list(serialized[name].shape),
                        "serialized_dtype": str(serialized[name].dtype),
                        "model_tensor_dtype": str(direct_tensors[name].dtype),
                        "finite": bool(np.all(np.isfinite(serialized[name]))),
                        "nonzero": bool(np.any(serialized[name] != 0)),
                    }
                    for name in EXPECTED_FEATURE_SHAPES
                },
                "source_image_hash": metadata["source_image_hash"],
                "provenance_verified": True,
                "serialized_vs_direct": comparisons,
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
            print("CHECKPOINT:", args.checkpoint.resolve())
            print("OBSERVATIONS_DIR:", args.observations_dir.resolve())
            print("TEX3D_OPENVLA_ROOT:", args.tex3d_openvla_root.resolve())
            print("OUTPUT_DIR:", args.output_dir)
            report = _run_smoke(args)
            report_path = args.output_dir / "smoke_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("C2 REAL SMOKE: PASS")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print("REPORT_PATH:", report_path)
            return 0
        except Exception as error:
            print("C2 REAL SMOKE: FAIL", file=sys.stderr)
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
