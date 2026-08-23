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
_FEATURE_SCHEMA_VERSION = "pi05_features_v1"
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
    """Raised when C3 cannot produce a valid pi0.5 feature record."""


@dataclass(frozen=True)
class _OpenPIRuntime:
    jax: Any
    jnp: Any
    image_tools: Any
    transforms: Any
    observation_type: Any
    preprocess_observation: Any
    model_type: Any


@dataclass(frozen=True)
class _InputRecord:
    observation: PilotObservation
    output_path: Path
    source_image_hash: str


def extract_pi05_features(
    *,
    model: Any,
    train_config: Any,
    checkpoint: str | Path,
    norm_stats: Any,
    observation_paths: Sequence[str | Path],
    output_dir: str | Path,
    batch_size: int = 1,
) -> tuple[Path, ...]:
    """Extract and serialize the frozen C3 pi0.5 representation nodes."""
    if type(batch_size) is not int or batch_size <= 0:
        raise Pi05FeatureExtractionError(
            "batch_size must be a positive integer"
        )

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
    stats = _validate_norm_stats(norm_stats)
    transform = _build_input_transform(train_config, stats, runtime)
    image_encoder = _validate_model(model)

    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Pi05FeatureExtractionError(
            f"failed to create output directory: {destination}"
        ) from error

    checkpoint_id = str(checkpoint)
    written_paths: list[Path] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        _validate_batch_output_paths(batch)
        transformed_records = tuple(
            _transform_record(item.observation, transform, runtime)
            for item in batch
        )
        observation = _build_batched_observation(
            transformed_records,
            expected_batch_size=len(batch),
            runtime=runtime,
        )

        try:
            p2, aux = image_encoder(
                observation.images["base_0_rgb"],
                train=False,
            )
        except Exception as error:
            raise Pi05FeatureExtractionError(
                f"pi0.5 visual extraction failed for batch starting at index {start}"
            ) from error
        if not isinstance(aux, Mapping) or "encoded" not in aux:
            raise Pi05FeatureExtractionError(
                "model.PaliGemma.img output is missing aux['encoded']"
            )
        p1 = aux["encoded"]
        _validate_batched_feature(
            "P1",
            p1,
            (len(batch), *_EXPECTED_FEATURE_SHAPES["p1_siglip"]),
        )
        _validate_batched_feature(
            "P2",
            p2,
            (len(batch), *_EXPECTED_FEATURE_SHAPES["p2_projected"]),
        )

        for index, item in enumerate(batch):
            try:
                arrays = {
                    "p1_siglip": _to_serialized_array(p1[index], runtime),
                    "p2_projected": _to_serialized_array(p2[index], runtime),
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
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as openpi_model
        from openpi_client import image_tools
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to load required OpenPI preprocessing dependencies"
        ) from error

    return _OpenPIRuntime(
        jax=jax,
        jnp=jnp,
        image_tools=image_tools,
        transforms=transforms,
        observation_type=openpi_model.Observation,
        preprocess_observation=openpi_model.preprocess_observation,
        model_type=openpi_model.ModelType,
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
            raise Pi05FeatureExtractionError(
                f"observation path is not a file: {path}"
            )
        if resolved in resolved_paths:
            raise Pi05FeatureExtractionError(
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
            raise Pi05FeatureExtractionError(
                f"failed to load PilotObservation: {path}"
            ) from error
        _validate_sample_id(observation.sample_id)
        _validate_raw_image(
            "base_rgb_raw",
            observation.base_rgb_raw,
            observation.sample_id,
        )
        _validate_raw_image(
            "wrist_rgb_raw",
            observation.wrist_rgb_raw,
            observation.sample_id,
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
        image_bytes = np.ascontiguousarray(
            observation.base_rgb_raw
        ).tobytes()
        records.append(
            _InputRecord(
                observation=observation,
                output_path=output_path,
                source_image_hash=(
                    f"sha256:{hashlib.sha256(image_bytes).hexdigest()}"
                ),
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


def _validate_norm_stats(norm_stats: Any) -> dict[str, Any]:
    if not isinstance(norm_stats, Mapping) or not norm_stats:
        raise Pi05FeatureExtractionError(
            "norm_stats must be a non-empty mapping"
        )
    if "state" not in norm_stats:
        raise Pi05FeatureExtractionError(
            "norm_stats must contain checkpoint state statistics"
        )

    stats = dict(norm_stats)
    for name, value in stats.items():
        q01 = getattr(value, "q01", None)
        q99 = getattr(value, "q99", None)
        if q01 is None or q99 is None:
            raise Pi05FeatureExtractionError(
                f"norm_stats entry {name!r} must contain q01 and q99"
            )
        try:
            lower = np.asarray(q01)
            upper = np.asarray(q99)
        except Exception as error:
            raise Pi05FeatureExtractionError(
                f"norm_stats entry {name!r} has malformed quantiles"
            ) from error
        if (
            lower.shape != upper.shape
            or lower.size == 0
            or not np.issubdtype(lower.dtype, np.number)
            or not np.issubdtype(upper.dtype, np.number)
            or not np.isrealobj(lower)
            or not np.isrealobj(upper)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise Pi05FeatureExtractionError(
                f"norm_stats entry {name!r} has invalid quantiles"
            )

    state_q01 = np.asarray(stats["state"].q01)
    if state_q01.shape != (8,):
        raise Pi05FeatureExtractionError(
            "state norm_stats quantiles must have shape (8,)"
        )
    return stats


def _build_input_transform(
    train_config: Any,
    norm_stats: dict[str, Any],
    runtime: _OpenPIRuntime,
) -> Any:
    _validate_train_config(train_config, runtime)
    try:
        data_config = train_config.data.create(
            train_config.assets_dirs,
            train_config.model,
        )
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to construct pi05_libero DataConfig"
        ) from error
    _validate_data_config(data_config)

    try:
        return runtime.transforms.compose(
            [
                runtime.transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                runtime.transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to construct pi05_libero input transforms"
        ) from error


def _validate_train_config(
    train_config: Any,
    runtime: _OpenPIRuntime,
) -> None:
    model_config = getattr(train_config, "model", None)
    if (
        getattr(train_config, "name", None) != "pi05_libero"
        or model_config is None
        or getattr(model_config, "model_type", None)
        != runtime.model_type.PI05
        or getattr(model_config, "pi05", None) is not True
        or getattr(model_config, "action_horizon", None) != 10
        or getattr(model_config, "discrete_state_input", None) is not False
        or getattr(model_config, "action_dim", None) != 32
        or getattr(model_config, "max_token_len", None) != 200
        or not hasattr(train_config, "data")
        or not hasattr(train_config, "assets_dirs")
    ):
        raise Pi05FeatureExtractionError(
            "train_config is incompatible with frozen pi05_libero semantics"
        )


def _validate_data_config(data_config: Any) -> None:
    if (
        getattr(data_config, "repo_id", None) != "physical-intelligence/libero"
        or getattr(data_config, "asset_id", None)
        != "physical-intelligence/libero"
        or getattr(data_config, "use_quantile_norm", None) is not True
        or not hasattr(getattr(data_config, "data_transforms", None), "inputs")
        or not hasattr(getattr(data_config, "model_transforms", None), "inputs")
    ):
        raise Pi05FeatureExtractionError(
            "DataConfig is incompatible with frozen pi05_libero semantics"
        )


def _transform_record(
    observation: PilotObservation,
    transform: Any,
    runtime: _OpenPIRuntime,
) -> dict[str, Any]:
    try:
        base_image = _preprocess_client_image(
            observation.base_rgb_raw,
            runtime,
        )
        wrist_image = _preprocess_client_image(
            observation.wrist_rgb_raw,
            runtime,
        )
        policy_input = {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": observation.state.copy(),
            "prompt": observation.prompt,
        }
        transformed = transform(policy_input)
    except Pi05FeatureExtractionError:
        raise
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "pi05_libero preprocessing failed for "
            f"sample_id={observation.sample_id}"
        ) from error
    if not isinstance(transformed, dict):
        raise Pi05FeatureExtractionError(
            "pi05_libero input transforms must return a dictionary"
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
        rotated,
        _MODEL_INPUT_SIZE,
        _MODEL_INPUT_SIZE,
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
    expected_keys = set(_EXPECTED_IMAGE_KEYS)
    if not isinstance(images, Mapping) or set(images) != expected_keys:
        raise Pi05FeatureExtractionError(
            "pi05_libero transforms produced malformed image slots"
        )
    if not isinstance(masks, Mapping) or set(masks) != expected_keys:
        raise Pi05FeatureExtractionError(
            "pi05_libero transforms produced malformed image masks"
        )
    if not np.array_equal(np.asarray(images["base_0_rgb"]), expected_base):
        raise Pi05FeatureExtractionError(
            "base_0_rgb does not preserve the client-preprocessed base image"
        )
    if not np.array_equal(
        np.asarray(images["left_wrist_0_rgb"]),
        expected_wrist,
    ):
        raise Pi05FeatureExtractionError(
            "left_wrist_0_rgb does not preserve the client-preprocessed wrist image"
        )
    if not np.array_equal(
        np.asarray(images["right_wrist_0_rgb"]),
        np.zeros_like(expected_base),
    ):
        raise Pi05FeatureExtractionError(
            "right_wrist_0_rgb must be zero padding"
        )

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


def _build_batched_observation(
    transformed_records: tuple[dict[str, Any], ...],
    *,
    expected_batch_size: int,
    runtime: _OpenPIRuntime,
) -> Any:
    try:
        batched = runtime.jax.tree.map(
            lambda *values: runtime.jnp.asarray(
                np.stack([np.asarray(value) for value in values], axis=0)
            ),
            *transformed_records,
        )
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to stack transformed pi05_libero records"
        ) from error

    try:
        observation = runtime.observation_type.from_dict(batched)
        observation = runtime.preprocess_observation(
            None,
            observation,
            train=False,
        )
    except Exception as error:
        raise Pi05FeatureExtractionError(
            "failed to construct batched pi0.5 Observation"
        ) from error
    _validate_batched_observation(
        observation,
        expected_batch_size,
        runtime,
    )
    return observation


def _validate_batched_observation(
    observation: Any,
    expected_batch_size: int,
    runtime: _OpenPIRuntime,
) -> None:
    images = getattr(observation, "images", None)
    masks = getattr(observation, "image_masks", None)
    if not isinstance(images, Mapping) or set(images) != set(
        _EXPECTED_IMAGE_KEYS
    ):
        raise Pi05FeatureExtractionError(
            "batched Observation has malformed image slots"
        )
    if not isinstance(masks, Mapping) or set(masks) != set(
        _EXPECTED_IMAGE_KEYS
    ):
        raise Pi05FeatureExtractionError(
            "batched Observation has malformed image masks"
        )
    for name in _EXPECTED_IMAGE_KEYS:
        image_shape = tuple(getattr(images[name], "shape", ()))
        mask_shape = tuple(getattr(masks[name], "shape", ()))
        if image_shape != (
            expected_batch_size,
            _MODEL_INPUT_SIZE,
            _MODEL_INPUT_SIZE,
            3,
        ):
            raise Pi05FeatureExtractionError(
                f"unexpected batched image shape for {name}: {image_shape}"
            )
        if mask_shape != (expected_batch_size,):
            raise Pi05FeatureExtractionError(
                f"unexpected batched image mask shape for {name}: {mask_shape}"
            )

    expected_masks = {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    for name, expected in expected_masks.items():
        values = np.asarray(runtime.jax.device_get(masks[name]))
        if not np.all(values == expected):
            raise Pi05FeatureExtractionError(
                f"unexpected batched PI05 image mask for {name}"
            )

    state_shape = tuple(getattr(observation.state, "shape", ()))
    prompt_shape = tuple(
        getattr(observation.tokenized_prompt, "shape", ())
    )
    prompt_mask_shape = tuple(
        getattr(observation.tokenized_prompt_mask, "shape", ())
    )
    if state_shape != (expected_batch_size, 32):
        raise Pi05FeatureExtractionError(
            f"unexpected batched state shape: {state_shape}"
        )
    if prompt_shape != (expected_batch_size, 200) or prompt_mask_shape != (
        expected_batch_size,
        200,
    ):
        raise Pi05FeatureExtractionError(
            "unexpected batched tokenized prompt shape"
        )


def _validate_model(model: Any) -> Any:
    paligemma = getattr(model, "PaliGemma", None)
    image_encoder = getattr(paligemma, "img", None)
    if not callable(image_encoder):
        raise Pi05FeatureExtractionError(
            "model must expose callable PaliGemma.img"
        )
    return image_encoder


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


def _to_serialized_array(value: Any, runtime: _OpenPIRuntime) -> np.ndarray:
    return np.asarray(runtime.jax.device_get(value), dtype=np.float32)


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
        np.savez_compressed(
            output,
            metadata_json=np.asarray(metadata_json),
            **arrays,
        )
