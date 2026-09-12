from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pilot_observation import PilotObservation


_MODEL_INPUT_SIZE = 224
_FEATURE_SCHEMA_VERSION = "pi05_torch_features_v1"
_EXPECTED_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
_EXPECTED_FEATURE_SHAPES = {
    "p1_siglip": (256, 1152),
    "p2_projected": (256, 2048),
}


class Pi05FeatureExtractionError(RuntimeError):
    """Raised when C3 cannot produce a valid current-backend pi0.5 record."""


@dataclass(frozen=True)
class _OpenPIRuntime:
    torch: Any
    image_tools: Any
    observation_type: Any
    native_feature_dtype: Any


@dataclass(frozen=True)
class _InputRecord:
    observation: PilotObservation
    output_path: Path
    source_image_hash: str


def extract_pi05_features(
    *,
    model: Any,
    policy: Any,
    checkpoint: str | Path,
    observation_paths: Sequence[str | Path],
    output_dir: str | Path,
    batch_size: int = 1,
) -> tuple[Path, ...]:
    """Extract current PI0Pytorch P1/P2 records for frozen observations.

    P2 is the direct result of ``paligemma_with_expert.embed_image``. Every
    batch is also passed through the official ``embed_prefix`` path, and the
    direct result must be bitwise equal to the base-camera prefix slice. P1 is
    retained only for the historical control-pair schema and is captured at
    the input of the official multimodal projector during that same direct
    image-embedding call.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise Pi05FeatureExtractionError("batch_size must be a positive integer")

    paths = tuple(Path(path) for path in observation_paths)
    if not paths:
        raise Pi05FeatureExtractionError("observation_paths must be non-empty")
    _validate_unique_input_paths(paths)

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise Pi05FeatureExtractionError(
            f"output path is not a directory: {destination}"
        )

    records = _load_input_records(paths, destination)
    runtime = _load_openpi_runtime()
    input_transform, device = _validate_policy(policy)
    embed_image, projector = _validate_model(model, runtime)

    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Pi05FeatureExtractionError(
            f"failed to create output directory: {destination}"
        ) from error

    checkpoint_id = str(checkpoint)
    written_paths: list[Path] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        _validate_batch_output_paths(batch)
        transformed_records = tuple(
            _transform_record(item.observation, input_transform, runtime)
            for item in batch
        )
        observation = _build_batched_observation(
            transformed_records,
            expected_batch_size=len(batch),
            device=device,
            runtime=runtime,
        )

        try:
            p1, p2 = _extract_projected_base_tokens(
                model=model,
                observation=observation,
                embed_image=embed_image,
                projector=projector,
                expected_batch_size=len(batch),
                runtime=runtime,
            )
        except Pi05FeatureExtractionError:
            raise
        except Exception as error:
            raise Pi05FeatureExtractionError(
                f"pi0.5 visual extraction failed for batch starting at index {start}"
            ) from error

        for index, item in enumerate(batch):
            try:
                arrays = {
                    "p1_siglip": _to_serialized_array(p1[index]),
                    "p2_projected": _to_serialized_array(p2[index]),
                }
            except Exception as error:
                raise Pi05FeatureExtractionError(
                    "failed to transfer pi0.5 features to CPU for "
                    f"{item.observation.sample_id}"
                ) from error
            _validate_serialized_arrays(arrays)
            metadata = {
                "sample_id": item.observation.sample_id,
                "source_model": "pi05",
                "checkpoint": checkpoint_id,
                "feature_schema_version": _FEATURE_SCHEMA_VERSION,
                "source_image_hash": item.source_image_hash,
            }
            try:
                _save_feature_record(item.output_path, arrays, metadata)
            except Pi05FeatureExtractionError:
                raise
            except Exception as error:
                raise Pi05FeatureExtractionError(
                    "failed to serialize pi0.5 features for "
                    f"{item.observation.sample_id}"
                ) from error
            written_paths.append(item.output_path)

    return tuple(written_paths)


def _load_openpi_runtime() -> _OpenPIRuntime:
    try:
        import torch
        from openpi.models import model as openpi_model
        from openpi_client import image_tools
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to load required PI0Pytorch preprocessing dependencies"
        ) from error
    return _OpenPIRuntime(
        torch=torch,
        image_tools=image_tools,
        observation_type=openpi_model.Observation,
        native_feature_dtype=torch.bfloat16,
    )


def _validate_unique_input_paths(paths: tuple[Path, ...]) -> None:
    resolved_paths: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise Pi05FeatureExtractionError(
                f"observation file does not exist: {path}"
            ) from error
        if not resolved.is_file():
            raise Pi05FeatureExtractionError(f"observation path is not a file: {path}")
        if resolved in resolved_paths:
            raise Pi05FeatureExtractionError(f"duplicate observation path: {path}")
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
            raise Pi05FeatureExtractionError(
                f"failed to load PilotObservation: {path}"
            ) from error
        _validate_sample_id(observation.sample_id)
        _validate_raw_image(
            "base_rgb_raw", observation.base_rgb_raw, observation.sample_id
        )
        _validate_raw_image(
            "wrist_rgb_raw", observation.wrist_rgb_raw, observation.sample_id
        )
        _validate_state(observation.state, observation.sample_id)

        if observation.sample_id in sample_ids:
            raise Pi05FeatureExtractionError(
                f"duplicate sample_id: {observation.sample_id}"
            )
        sample_ids.add(observation.sample_id)

        output_path = destination / f"{observation.sample_id}.npz"
        if output_path.exists():
            raise Pi05FeatureExtractionError(
                f"refusing to overwrite existing feature file: {output_path}"
            )
        image_bytes = np.ascontiguousarray(observation.base_rgb_raw).tobytes()
        records.append(
            _InputRecord(
                observation=observation,
                output_path=output_path,
                source_image_hash=f"sha256:{hashlib.sha256(image_bytes).hexdigest()}",
            )
        )
    return tuple(records)


def _validate_sample_id(sample_id: str) -> None:
    if (
        sample_id in {".", ".."}
        or Path(sample_id).is_absolute()
        or os.sep in sample_id
        or (os.altsep is not None and os.altsep in sample_id)
        or "/" in sample_id
        or "\\" in sample_id
        or "\x00" in sample_id
    ):
        raise Pi05FeatureExtractionError(
            f"sample_id must be a safe path component: {sample_id!r}"
        )


def _validate_raw_image(name: str, image: Any, sample_id: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[-1] != 3
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise Pi05FeatureExtractionError(
            f"{name} must be a non-empty rank-3 uint8 RGB array: "
            f"sample_id={sample_id}"
        )


def _validate_state(state: Any, sample_id: str) -> None:
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (8,)
        or state.dtype.hasobject
        or not np.issubdtype(state.dtype, np.number)
        or not np.isrealobj(state)
        or not np.all(np.isfinite(state))
    ):
        raise Pi05FeatureExtractionError(
            "state must be a finite real numeric array with shape (8,): "
            f"sample_id={sample_id}"
        )


def _validate_policy(policy: Any) -> tuple[Any, Any]:
    transform = getattr(policy, "_input_transform", None)
    device = getattr(policy, "_pytorch_device", None)
    if (
        getattr(policy, "_is_pytorch_model", None) is not True
        or not callable(transform)
        or device is None
    ):
        raise Pi05FeatureExtractionError(
            "policy must expose the current PI0Pytorch input-transform path"
        )
    return transform, device


def _validate_model(model: Any, runtime: _OpenPIRuntime) -> tuple[Any, Any]:
    paligemma_with_expert = getattr(model, "paligemma_with_expert", None)
    embed_image = getattr(paligemma_with_expert, "embed_image", None)
    paligemma = getattr(paligemma_with_expert, "paligemma", None)
    projector = getattr(
        getattr(paligemma, "model", None), "multi_modal_projector", None
    )
    if not callable(embed_image) or not hasattr(projector, "register_forward_pre_hook"):
        raise Pi05FeatureExtractionError(
            "model must expose PI0Pytorch paligemma_with_expert.embed_image "
            "and its multimodal projector"
        )
    if not isinstance(model, runtime.torch.nn.Module):
        raise Pi05FeatureExtractionError("pi0.5 model must be a torch.nn.Module")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise Pi05FeatureExtractionError(
            "PI0Pytorch feature extraction requires frozen model parameters"
        )
    return embed_image, projector


def _transform_record(
    observation: PilotObservation,
    input_transform: Any,
    runtime: _OpenPIRuntime,
) -> dict[str, Any]:
    try:
        base_image = _preprocess_client_image(observation.base_rgb_raw, runtime)
        wrist_image = _preprocess_client_image(observation.wrist_rgb_raw, runtime)
        transformed = input_transform(
            {
                "observation/image": base_image,
                "observation/wrist_image": wrist_image,
                "observation/state": observation.state.copy(),
                "prompt": observation.prompt,
            }
        )
    except Pi05FeatureExtractionError:
        raise
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "pi05_libero policy preprocessing failed for "
            f"sample_id={observation.sample_id}"
        ) from error
    if not isinstance(transformed, dict):
        raise Pi05FeatureExtractionError(
            "pi05_libero input transform must return a dictionary"
        )
    _validate_transformed_slots(
        transformed,
        expected_base=base_image,
        expected_wrist=wrist_image,
    )
    return transformed


def _preprocess_client_image(
    image: np.ndarray,
    runtime: _OpenPIRuntime,
) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    resized = runtime.image_tools.resize_with_pad(
        rotated, _MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE
    )
    converted = runtime.image_tools.convert_to_uint8(resized)
    if (
        not isinstance(converted, np.ndarray)
        or converted.shape != (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE, 3)
        or converted.dtype != np.uint8
    ):
        raise Pi05FeatureExtractionError(
            "official OpenPI client preprocessing produced an invalid RGB image"
        )
    return converted


def _validate_transformed_slots(
    transformed: dict[str, Any],
    *,
    expected_base: np.ndarray,
    expected_wrist: np.ndarray,
) -> None:
    images = transformed.get("image")
    masks = transformed.get("image_mask")
    if not isinstance(images, Mapping) or tuple(images) != _EXPECTED_IMAGE_KEYS:
        raise Pi05FeatureExtractionError(
            "pi05_libero transforms changed the frozen image-slot ordering"
        )
    if not isinstance(masks, Mapping) or tuple(masks) != _EXPECTED_IMAGE_KEYS:
        raise Pi05FeatureExtractionError(
            "pi05_libero transforms changed the frozen image-mask ordering"
        )
    if not np.array_equal(np.asarray(images["base_0_rgb"]), expected_base):
        raise Pi05FeatureExtractionError(
            "base_0_rgb does not preserve the client-preprocessed base image"
        )
    if not np.array_equal(np.asarray(images["left_wrist_0_rgb"]), expected_wrist):
        raise Pi05FeatureExtractionError(
            "left_wrist_0_rgb does not preserve the client-preprocessed wrist image"
        )
    if not np.array_equal(
        np.asarray(images["right_wrist_0_rgb"]), np.zeros_like(expected_base)
    ):
        raise Pi05FeatureExtractionError("right_wrist_0_rgb must be zero padding")

    expected_masks = {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    for name, expected in expected_masks.items():
        value = np.asarray(masks[name])
        if value.ndim != 0 or bool(value.item()) is not expected:
            raise Pi05FeatureExtractionError(
                f"unexpected PI05 image mask for {name}"
            )


def _tree_stack_to_torch(
    values: tuple[Any, ...],
    *,
    device: Any,
    runtime: _OpenPIRuntime,
) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values):
            raise Pi05FeatureExtractionError(
                "transformed records have inconsistent mapping order"
            )
        return {
            key: _tree_stack_to_torch(
                tuple(value[key] for value in values),
                device=device,
                runtime=runtime,
            )
            for key in keys
        }
    try:
        array = np.stack([np.asarray(value) for value in values], axis=0)
        return runtime.torch.from_numpy(array.copy()).to(device)
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to stack transformed PI0Pytorch values"
        ) from error


def _build_batched_observation(
    transformed_records: tuple[dict[str, Any], ...],
    *,
    expected_batch_size: int,
    device: Any,
    runtime: _OpenPIRuntime,
) -> Any:
    inputs = _tree_stack_to_torch(
        transformed_records,
        device=device,
        runtime=runtime,
    )
    try:
        observation = runtime.observation_type.from_dict(inputs)
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to construct batched PI0Pytorch Observation"
        ) from error
    _validate_model_observation(observation, expected_batch_size)
    return observation


def _validate_model_observation(observation: Any, expected_batch_size: int) -> None:
    images = getattr(observation, "images", None)
    masks = getattr(observation, "image_masks", None)
    if not isinstance(images, Mapping) or tuple(images) != _EXPECTED_IMAGE_KEYS:
        raise Pi05FeatureExtractionError(
            "PI0Pytorch Observation has unexpected image-slot ordering"
        )
    if not isinstance(masks, Mapping) or tuple(masks) != _EXPECTED_IMAGE_KEYS:
        raise Pi05FeatureExtractionError(
            "PI0Pytorch Observation has unexpected image-mask ordering"
        )
    for name in _EXPECTED_IMAGE_KEYS:
        image_shape = tuple(getattr(images[name], "shape", ()))
        mask_shape = tuple(getattr(masks[name], "shape", ()))
        if image_shape != (
            expected_batch_size,
            3,
            _MODEL_INPUT_SIZE,
            _MODEL_INPUT_SIZE,
        ):
            raise Pi05FeatureExtractionError(
                f"unexpected PI0Pytorch image shape for {name}: {image_shape}"
            )
        if mask_shape != (expected_batch_size,):
            raise Pi05FeatureExtractionError(
                f"unexpected PI0Pytorch image-mask shape for {name}: {mask_shape}"
            )
        if images[name].dtype != images[name].new_zeros(()).float().dtype:
            raise Pi05FeatureExtractionError(
                f"PI0Pytorch input image dtype must be float32 for {name}"
            )
        if masks[name].dtype != masks[name].new_zeros(()).bool().dtype:
            raise Pi05FeatureExtractionError(
                f"PI0Pytorch input mask dtype must be bool for {name}"
            )
    state_shape = tuple(getattr(observation.state, "shape", ()))
    prompt_shape = tuple(getattr(observation.tokenized_prompt, "shape", ()))
    prompt_mask_shape = tuple(
        getattr(observation.tokenized_prompt_mask, "shape", ())
    )
    if state_shape != (expected_batch_size, 32):
        raise Pi05FeatureExtractionError(
            f"unexpected PI0Pytorch state shape: {state_shape}"
        )
    if prompt_shape != (expected_batch_size, 200) or prompt_mask_shape != (
        expected_batch_size,
        200,
    ):
        raise Pi05FeatureExtractionError(
            "unexpected PI0Pytorch tokenized-prompt shape"
        )


def _extract_projected_base_tokens(
    *,
    model: Any,
    observation: Any,
    embed_image: Any,
    projector: Any,
    expected_batch_size: int,
    runtime: _OpenPIRuntime,
) -> tuple[Any, Any]:
    torch = runtime.torch
    captures: list[Any] = []

    def capture_projector_input(_module: Any, args: tuple[Any, ...]) -> None:
        if len(args) != 1:
            raise Pi05FeatureExtractionError(
                "PI0Pytorch multimodal projector received unexpected arguments"
            )
        captures.append(args[0])

    model.eval()
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise Pi05FeatureExtractionError("torch.no_grad guard is not active")
        prepared = model._preprocess_observation(observation, train=False)
        if not isinstance(prepared, tuple) or len(prepared) != 5:
            raise Pi05FeatureExtractionError(
                "PI0Pytorch._preprocess_observation must return five values"
            )
        images, image_masks, lang_tokens, lang_masks, _ = prepared
        if len(images) != len(_EXPECTED_IMAGE_KEYS) or len(image_masks) != len(
            _EXPECTED_IMAGE_KEYS
        ):
            raise Pi05FeatureExtractionError(
                "PI0Pytorch preprocessing must preserve three image slots"
            )
        expected_masks = (True, True, False)
        for key, image, mask, expected_mask in zip(
            _EXPECTED_IMAGE_KEYS,
            images,
            image_masks,
            expected_masks,
            strict=True,
        ):
            if tuple(image.shape) != (
                expected_batch_size,
                3,
                _MODEL_INPUT_SIZE,
                _MODEL_INPUT_SIZE,
            ):
                raise Pi05FeatureExtractionError(
                    f"PI0Pytorch preprocessing changed image layout for {key}"
                )
            expected = torch.full_like(mask, expected_mask, dtype=torch.bool)
            if mask.dtype != torch.bool or not torch.equal(mask, expected):
                raise Pi05FeatureExtractionError(
                    f"PI0Pytorch preprocessing changed mask semantics for {key}"
                )
        image_by_key = dict(zip(_EXPECTED_IMAGE_KEYS, images, strict=True))
        prefix, _, _ = model.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )

        handle = projector.register_forward_pre_hook(capture_projector_input)
        try:
            direct = {
                key: embed_image(image_by_key[key]) for key in _EXPECTED_IMAGE_KEYS
            }
        finally:
            handle.remove()

        if len(captures) != len(_EXPECTED_IMAGE_KEYS):
            raise Pi05FeatureExtractionError(
                "PI0Pytorch projector hook did not capture one P1 per image slot"
            )
        for index, key in enumerate(_EXPECTED_IMAGE_KEYS):
            p2 = direct[key]
            p1 = captures[index]
            _validate_batched_feature(
                f"{key} P1",
                p1,
                (expected_batch_size, *_EXPECTED_FEATURE_SHAPES["p1_siglip"]),
            )
            _validate_batched_feature(
                f"{key} P2",
                p2,
                (expected_batch_size, *_EXPECTED_FEATURE_SHAPES["p2_projected"]),
            )
            start = index * _EXPECTED_FEATURE_SHAPES["p2_projected"][0]
            stop = start + _EXPECTED_FEATURE_SHAPES["p2_projected"][0]
            if p1.requires_grad or p2.requires_grad:
                raise Pi05FeatureExtractionError(
                    f"{key} PI0Pytorch features must be detached"
                )
            if (
                p1.dtype != runtime.native_feature_dtype
                or p2.dtype != runtime.native_feature_dtype
            ):
                raise Pi05FeatureExtractionError(
                    f"{key} PI0Pytorch native feature dtype must be "
                    f"{runtime.native_feature_dtype}, got P1={p1.dtype}, P2={p2.dtype}"
                )
            if not bool(torch.isfinite(p1).all()) or not bool(torch.isfinite(p2).all()):
                raise Pi05FeatureExtractionError(
                    f"{key} PI0Pytorch features contain non-finite values"
                )
            if not torch.equal(p2, prefix[:, start:stop]):
                raise Pi05FeatureExtractionError(
                    f"{key} P2 differs from its official embed_prefix token slice"
                )
    return captures[0], direct["base_0_rgb"]


def _validate_batched_feature(
    name: str,
    value: Any,
    expected_shape: tuple[int, int, int],
) -> None:
    shape = tuple(getattr(value, "shape", ()))
    if shape != expected_shape:
        raise Pi05FeatureExtractionError(
            f"{name} must have shape {expected_shape}, got {shape}"
        )


def _to_serialized_array(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _validate_serialized_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, expected_shape in _EXPECTED_FEATURE_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise Pi05FeatureExtractionError(
                f"serialized {name} must have shape {expected_shape} and dtype "
                f"float32, got shape={value.shape}, dtype={value.dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise Pi05FeatureExtractionError(
                f"serialized {name} contains non-finite values"
            )
        if not np.any(value):
            raise Pi05FeatureExtractionError(
                f"serialized {name} must not be all-zero"
            )


def _validate_batch_output_paths(batch: tuple[_InputRecord, ...]) -> None:
    collisions = [str(item.output_path) for item in batch if item.output_path.exists()]
    if collisions:
        raise Pi05FeatureExtractionError(
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
        np.savez_compressed(output, metadata_json=np.asarray(metadata_json), **arrays)
