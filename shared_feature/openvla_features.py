from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .pilot_observation import PilotObservation


_CAMERA_RESOLUTION = 512
_MODEL_INPUT_SIZE = 224
_CROP_SCALE = 0.9
_FEATURE_SCHEMA_VERSION = "openvla_features_v1"
_EXPECTED_FEATURE_SHAPES = {
    "o1_siglip": (256, 1152),
    "o1_fused": (256, 2176),
    "o2_projected": (256, 4096),
}


class OpenVLAFeatureExtractionError(RuntimeError):
    """Raised when C2 cannot produce a valid OpenVLA feature record."""


@dataclass(frozen=True)
class _PreprocessingRuntime:
    get_libero_image: Callable[..., Any]
    center_crop_image: Callable[[Any], Any]
    openvla_v01_system_prompt: str


@dataclass(frozen=True)
class _InputRecord:
    observation: PilotObservation
    output_path: Path
    source_image_hash: str


def extract_openvla_features(
    *,
    model: Any,
    processor: Any,
    pretrained_checkpoint: str | Path,
    observation_paths: Sequence[str | Path],
    output_dir: str | Path,
    center_crop: bool = True,
    batch_size: int = 1,
) -> tuple[Path, ...]:
    """Extract and serialize the frozen C2 OpenVLA representation nodes."""
    if type(batch_size) is not int or batch_size <= 0:
        raise OpenVLAFeatureExtractionError(
            "batch_size must be a positive integer"
        )
    if type(center_crop) is not bool:
        raise OpenVLAFeatureExtractionError("center_crop must be a boolean")

    paths = tuple(Path(path) for path in observation_paths)
    if not paths:
        raise OpenVLAFeatureExtractionError("observation_paths must be non-empty")
    _validate_unique_input_paths(paths)

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise OpenVLAFeatureExtractionError(
            f"output path is not a directory: {destination}"
        )

    checkpoint = str(pretrained_checkpoint)
    records = _load_input_records(paths, destination)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OpenVLAFeatureExtractionError(
            f"failed to create output directory: {destination}"
        ) from error

    runtime = _load_preprocessing_runtime()
    torch = _load_torch()
    vision_backbone, projector, device, dtype = _validate_model(model, torch)

    try:
        model.eval()
    except Exception as error:
        raise OpenVLAFeatureExtractionError(
            "failed to put the OpenVLA model in inference mode"
        ) from error

    written_paths: list[Path] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        _validate_batch_output_paths(batch)
        try:
            images = [
                _build_policy_image(
                    item.observation.base_rgb_raw,
                    runtime,
                    center_crop=center_crop,
                )
                for item in batch
            ]
            prompts = [
                _build_prompt(
                    item.observation.prompt,
                    checkpoint,
                    runtime.openvla_v01_system_prompt,
                )
                for item in batch
            ]
        except Exception as error:
            raise OpenVLAFeatureExtractionError(
                f"OpenVLA preprocessing failed for batch starting at index {start}"
            ) from error

        try:
            processor_output = processor(
                prompts,
                images,
                padding=True,
                return_tensors="pt",
            )
            pixel_values = processor_output["pixel_values"]
        except Exception as error:
            raise OpenVLAFeatureExtractionError(
                f"OpenVLA processor failed for batch starting at index {start}"
            ) from error

        try:
            pixel_values = _validate_pixel_values(
                pixel_values,
                expected_batch_size=len(batch),
                torch=torch,
            )
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            dino_pixels, siglip_pixels = torch.split(
                pixel_values,
                [3, 3],
                dim=1,
            )
            with torch.inference_mode():
                dino_feature = vision_backbone.featurizer(dino_pixels)
                siglip_feature = vision_backbone.fused_featurizer(siglip_pixels)
                _validate_batched_feature(
                    "DINOv2 branch",
                    dino_feature,
                    (len(batch), 256, 1024),
                    torch,
                )
                _validate_batched_feature(
                    "O1-S SigLIP branch",
                    siglip_feature,
                    (len(batch), 256, 1152),
                    torch,
                )
                fused_feature = torch.cat(
                    [dino_feature, siglip_feature],
                    dim=-1,
                )
                _validate_batched_feature(
                    "O1-F fused feature",
                    fused_feature,
                    (len(batch), 256, 2176),
                    torch,
                )
                projected_feature = projector(fused_feature)
                _validate_batched_feature(
                    "O2 projected feature",
                    projected_feature,
                    (len(batch), 256, 4096),
                    torch,
                )
        except OpenVLAFeatureExtractionError:
            raise
        except Exception as error:
            raise OpenVLAFeatureExtractionError(
                f"OpenVLA model forward failed for batch starting at index {start}"
            ) from error

        for index, item in enumerate(batch):
            metadata = {
                "sample_id": item.observation.sample_id,
                "source_model": "openvla",
                "checkpoint": checkpoint,
                "feature_schema_version": _FEATURE_SCHEMA_VERSION,
                "source_image_hash": item.source_image_hash,
            }
            try:
                arrays = {
                    "o1_siglip": _to_serialized_array(
                        siglip_feature[index],
                        torch,
                    ),
                    "o1_fused": _to_serialized_array(
                        fused_feature[index],
                        torch,
                    ),
                    "o2_projected": _to_serialized_array(
                        projected_feature[index],
                        torch,
                    ),
                }
                _validate_serialized_arrays(arrays)
                _save_feature_record(item.output_path, arrays, metadata)
            except Exception as error:
                raise OpenVLAFeatureExtractionError(
                    f"failed to serialize OpenVLA features for "
                    f"{item.observation.sample_id}"
                ) from error
            written_paths.append(item.output_path)

    return tuple(written_paths)


def _validate_unique_input_paths(paths: tuple[Path, ...]) -> None:
    resolved_paths: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise OpenVLAFeatureExtractionError(
                f"observation file does not exist: {path}"
            ) from error
        if not resolved.is_file():
            raise OpenVLAFeatureExtractionError(
                f"observation path is not a file: {path}"
            )
        if resolved in resolved_paths:
            raise OpenVLAFeatureExtractionError(
                f"duplicate observation path: {path}"
            )
        resolved_paths.add(resolved)


def _load_input_records(
    paths: tuple[Path, ...],
    destination: Path,
) -> tuple[_InputRecord, ...]:
    sample_ids: set[str] = set()
    records: list[_InputRecord] = []
    for path in paths:
        try:
            observation = PilotObservation.load(path)
        except Exception as error:
            raise OpenVLAFeatureExtractionError(
                f"failed to load PilotObservation: {path}"
            ) from error
        _validate_base_image(observation.base_rgb_raw, observation.sample_id)
        if observation.sample_id in sample_ids:
            raise OpenVLAFeatureExtractionError(
                f"duplicate sample_id: {observation.sample_id}"
            )
        sample_ids.add(observation.sample_id)

        output_path = destination / f"{observation.sample_id}.npz"
        if output_path.exists():
            raise OpenVLAFeatureExtractionError(
                f"refusing to overwrite existing feature file: {output_path}"
            )
        image_bytes = np.ascontiguousarray(observation.base_rgb_raw).tobytes()
        source_image_hash = f"sha256:{hashlib.sha256(image_bytes).hexdigest()}"
        records.append(
            _InputRecord(
                observation=observation,
                output_path=output_path,
                source_image_hash=source_image_hash,
            )
        )
    return tuple(records)


def _validate_base_image(image: Any, sample_id: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.shape != (_CAMERA_RESOLUTION, _CAMERA_RESOLUTION, 3)
        or image.dtype != np.uint8
    ):
        raise OpenVLAFeatureExtractionError(
            "base_rgb_raw must be a uint8 array with shape (512, 512, 3): "
            f"sample_id={sample_id}"
        )


def _validate_model(model: Any, torch: Any) -> tuple[Any, Any, Any, Any]:
    vision_backbone = getattr(model, "vision_backbone", None)
    projector = getattr(model, "projector", None)
    if vision_backbone is None or not callable(projector):
        raise OpenVLAFeatureExtractionError(
            "model must expose vision_backbone and callable projector"
        )
    if not callable(getattr(vision_backbone, "featurizer", None)):
        raise OpenVLAFeatureExtractionError(
            "model vision_backbone must expose callable featurizer"
        )
    if not callable(getattr(vision_backbone, "fused_featurizer", None)):
        raise OpenVLAFeatureExtractionError(
            "model vision_backbone must expose callable fused_featurizer"
        )

    try:
        parameter = next(
            value
            for value in vision_backbone.parameters()
            if torch.is_floating_point(value)
        )
    except (AttributeError, StopIteration) as error:
        raise OpenVLAFeatureExtractionError(
            "vision_backbone must expose a floating-point parameter"
        ) from error
    return vision_backbone, projector, parameter.device, parameter.dtype


def _build_policy_image(
    base_rgb_raw: np.ndarray,
    runtime: _PreprocessingRuntime,
    *,
    center_crop: bool,
) -> Any:
    from PIL import Image

    policy_source = runtime.get_libero_image(
        {"agentview_image": base_rgb_raw},
        _CAMERA_RESOLUTION,
    )
    policy_array = np.array(
        Image.fromarray(policy_source).resize(
            (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE)
        )
    )
    image = Image.fromarray(policy_array).convert("RGB")
    if center_crop:
        image = runtime.center_crop_image(image)
    return image


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


def _validate_pixel_values(
    pixel_values: Any,
    *,
    expected_batch_size: int,
    torch: Any,
) -> Any:
    if not torch.is_tensor(pixel_values):
        raise OpenVLAFeatureExtractionError(
            "processor pixel_values must be a PyTorch tensor"
        )
    expected_shape = (expected_batch_size, 6, 224, 224)
    if tuple(pixel_values.shape) != expected_shape:
        raise OpenVLAFeatureExtractionError(
            f"processor pixel_values must have shape {expected_shape}, got "
            f"{tuple(pixel_values.shape)}"
        )
    return pixel_values


def _validate_batched_feature(
    name: str,
    value: Any,
    expected_shape: tuple[int, int, int],
    torch: Any,
) -> None:
    if not torch.is_tensor(value):
        raise OpenVLAFeatureExtractionError(f"{name} must be a PyTorch tensor")
    if value.ndim != 3:
        raise OpenVLAFeatureExtractionError(
            f"{name} must be rank 3, got rank {value.ndim}"
        )
    if tuple(value.shape) != expected_shape:
        raise OpenVLAFeatureExtractionError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
        )


def _to_serialized_array(value: Any, torch: Any) -> np.ndarray:
    return value.detach().to(dtype=torch.float32, device="cpu").numpy()


def _validate_serialized_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, expected_shape in _EXPECTED_FEATURE_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise OpenVLAFeatureExtractionError(
                f"serialized {name} must have shape {expected_shape} and dtype "
                f"float32, got shape={value.shape}, dtype={value.dtype}"
            )


def _validate_batch_output_paths(batch: tuple[_InputRecord, ...]) -> None:
    collisions = [str(item.output_path) for item in batch if item.output_path.exists()]
    if collisions:
        raise OpenVLAFeatureExtractionError(
            f"refusing to overwrite existing feature files: {collisions}"
        )


def _save_feature_record(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> None:
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("xb") as output:
        np.savez_compressed(
            output,
            metadata_json=np.asarray(metadata_json),
            **arrays,
        )


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise OpenVLAFeatureExtractionError(
            "PyTorch is required for OpenVLA feature extraction"
        ) from error
    return torch


def _load_preprocessing_runtime() -> _PreprocessingRuntime:
    try:
        import tensorflow as tf
        from experiments.robot.libero.libero_utils import get_libero_image
        from experiments.robot.openvla_utils import (
            OPENVLA_V01_SYSTEM_PROMPT,
            crop_and_resize,
        )
        from PIL import Image
    except ImportError as error:
        raise OpenVLAFeatureExtractionError(
            "official OpenVLA and LIBERO preprocessing modules must be importable"
        ) from error

    def center_crop_image(image: Any) -> Any:
        tensor = tf.convert_to_tensor(np.array(image))
        original_dtype = tensor.dtype
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        tensor = crop_and_resize(tensor, _CROP_SCALE, 1)
        tensor = tf.clip_by_value(tensor, 0, 1)
        tensor = tf.image.convert_image_dtype(
            tensor,
            original_dtype,
            saturate=True,
        )
        return Image.fromarray(tensor.numpy()).convert("RGB")

    return _PreprocessingRuntime(
        get_libero_image=get_libero_image,
        center_crop_image=center_crop_image,
        openvla_v01_system_prompt=OPENVLA_V01_SYSTEM_PROMPT,
    )
