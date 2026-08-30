from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


OPENVLA_CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
OPENVLA_UNNORM_KEY = "libero_spatial_no_noops"
_O2_SHAPE = (1, 256, 4096)
_ACTION_SHAPE = (1, 7)
_EMPTY_ACTION_TOKEN_ID = 29871


class OpenVLAInterventionError(RuntimeError):
    """Raised when the frozen OpenVLA O2 interface cannot run safely."""


@dataclass(frozen=True)
class OpenVLAContinuationResult:
    action_token_ids: np.ndarray
    normalized_action: np.ndarray
    unnormalized_action: np.ndarray
    deployed_action: np.ndarray


@dataclass(frozen=True)
class PreparedOpenVLAContext:
    o2: Any
    checkpoint_identity: str
    unnorm_key: str
    center_crop: bool
    task_description: str
    prompt: str
    processor_identity: str
    _model: Any = field(repr=False)
    _pixel_values: Any = field(repr=False)
    _input_ids: Any = field(repr=False)
    _text_embeddings: Any = field(repr=False)
    _text_attention_mask: Any = field(repr=False)
    _multimodal_attention_mask: Any = field(repr=False)
    _runtime: Any = field(repr=False)


@dataclass(frozen=True)
class _OpenVLARuntime:
    torch: Any
    image_type: Any
    center_crop_image: Any
    build_prompt: Any
    openvla_v01_system_prompt: str
    normalize_gripper_action: Any
    invert_gripper_action: Any


def prepare_openvla_context(
    *,
    model: Any,
    processor: Any,
    observation: Mapping[str, Any],
    task_description: str,
    pretrained_checkpoint: str = OPENVLA_CHECKPOINT_IDENTITY,
    unnorm_key: str = OPENVLA_UNNORM_KEY,
    center_crop: bool = True,
) -> PreparedOpenVLAContext:
    """Prepare the frozen OpenVLA context and expose its native O2 tensor."""
    _validate_frozen_configuration(
        pretrained_checkpoint=pretrained_checkpoint,
        unnorm_key=unnorm_key,
        center_crop=center_crop,
    )
    if not isinstance(task_description, str) or not task_description.strip():
        raise OpenVLAInterventionError("task_description must be a non-empty string")
    image_array = _validate_policy_observation(observation)
    runtime = _load_openvla_runtime()
    torch = runtime.torch
    device, dtype = _model_device_dtype(model, torch)
    _validate_model(model)

    try:
        model.eval()
        image = runtime.image_type.fromarray(image_array).convert("RGB")
        image = runtime.center_crop_image(image)
        prompt = runtime.build_prompt(
            task_description,
            pretrained_checkpoint,
            runtime.openvla_v01_system_prompt,
        )
        processor_output = processor(prompt, image)
        processor_output = _move_processor_output(
            processor_output,
            device=device,
            dtype=dtype,
            torch=torch,
        )
        input_ids = processor_output["input_ids"]
        pixel_values = processor_output["pixel_values"]
        attention_mask = processor_output.get("attention_mask")
        _validate_processor_tensors(
            input_ids,
            pixel_values,
            attention_mask,
            torch,
        )
        input_ids = _append_empty_action_token(input_ids, torch)
        if attention_mask is not None and attention_mask.shape[1] != input_ids.shape[1]:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

        with torch.inference_mode():
            patch_features = model.vision_backbone(pixel_values)
            o2 = model.projector(patch_features)
            _validate_o2(o2, torch)
            text_embeddings = model.get_input_embeddings()(input_ids)
        multimodal_attention_mask = _build_multimodal_attention_mask(
            attention_mask,
            o2,
            torch,
        )
    except OpenVLAInterventionError:
        raise
    except Exception as error:
        raise OpenVLAInterventionError(
            "failed to prepare the OpenVLA O2 continuation context"
        ) from error

    return PreparedOpenVLAContext(
        o2=o2,
        checkpoint_identity=pretrained_checkpoint,
        unnorm_key=unnorm_key,
        center_crop=center_crop,
        task_description=task_description,
        prompt=prompt,
        processor_identity=(
            f"{type(processor).__module__}.{type(processor).__qualname__}"
        ),
        _model=model,
        _pixel_values=pixel_values,
        _input_ids=input_ids,
        _text_embeddings=text_embeddings,
        _text_attention_mask=attention_mask,
        _multimodal_attention_mask=multimodal_attention_mask,
        _runtime=runtime,
    )


def continue_openvla_from_o2(
    *,
    prepared: PreparedOpenVLAContext,
    o2: Any,
) -> OpenVLAContinuationResult:
    """Continue deterministic OpenVLA decoding from a supplied native O2."""
    if not isinstance(prepared, PreparedOpenVLAContext):
        raise OpenVLAInterventionError("prepared must be a PreparedOpenVLAContext")
    runtime = prepared._runtime
    torch = runtime.torch
    converted = _validate_and_convert_override(o2, prepared.o2, torch)
    embeddings = _multimodal_embeddings(
        prepared._text_embeddings,
        converted,
        torch,
    )
    action_dim = _validate_action_dim(prepared._model, prepared.unnorm_key)
    try:
        with torch.inference_mode():
            generated = prepared._model.language_model.generate(
                inputs_embeds=embeddings,
                attention_mask=prepared._multimodal_attention_mask,
                max_new_tokens=action_dim,
                do_sample=False,
            )
    except Exception as error:
        raise OpenVLAInterventionError(
            "OpenVLA continuation from supplied O2 failed"
        ) from error
    return _decode_result(prepared, generated)


def run_openvla_reference(
    *,
    prepared: PreparedOpenVLAContext,
) -> OpenVLAContinuationResult:
    """Run one authoritative original OpenVLA generation and decode all fields."""
    if not isinstance(prepared, PreparedOpenVLAContext):
        raise OpenVLAInterventionError("prepared must be a PreparedOpenVLAContext")
    action_dim = _validate_action_dim(prepared._model, prepared.unnorm_key)
    kwargs = {
        "input_ids": prepared._input_ids,
        "pixel_values": prepared._pixel_values,
        "max_new_tokens": action_dim,
        "do_sample": False,
    }
    if prepared._text_attention_mask is not None:
        kwargs["attention_mask"] = prepared._text_attention_mask
    try:
        with prepared._runtime.torch.inference_mode():
            generated = prepared._model.generate(**kwargs)
    except Exception as error:
        raise OpenVLAInterventionError(
            "authoritative OpenVLA reference generation failed"
        ) from error
    return _decode_result(prepared, generated)


def _validate_frozen_configuration(
    *,
    pretrained_checkpoint: str,
    unnorm_key: str,
    center_crop: bool,
) -> None:
    if pretrained_checkpoint != OPENVLA_CHECKPOINT_IDENTITY:
        raise OpenVLAInterventionError(
            "pretrained_checkpoint must equal the frozen OpenVLA identity"
        )
    if unnorm_key != OPENVLA_UNNORM_KEY:
        raise OpenVLAInterventionError("unnorm_key must equal libero_spatial_no_noops")
    if center_crop is not True:
        raise OpenVLAInterventionError("center_crop must be True")


def _validate_policy_observation(
    observation: Mapping[str, Any],
) -> np.ndarray:
    if not isinstance(observation, Mapping) or "full_image" not in observation:
        raise OpenVLAInterventionError(
            "observation must be the C1 policy mapping with full_image"
        )
    image = observation["full_image"]
    if (
        not isinstance(image, np.ndarray)
        or image.shape != (224, 224, 3)
        or image.dtype != np.uint8
    ):
        raise OpenVLAInterventionError(
            "observation full_image must be uint8 with shape (224, 224, 3)"
        )
    return image


def _validate_model(model: Any) -> None:
    if (
        not callable(getattr(model, "vision_backbone", None))
        or not callable(getattr(model, "projector", None))
        or not callable(getattr(model, "get_input_embeddings", None))
        or not callable(getattr(model, "generate", None))
        or not callable(
            getattr(getattr(model, "language_model", None), "generate", None)
        )
        or not callable(getattr(model, "get_action_dim", None))
        or not callable(getattr(model, "get_action_stats", None))
    ):
        raise OpenVLAInterventionError(
            "model is incompatible with the frozen OpenVLA continuation path"
        )


def _model_device_dtype(model: Any, torch: Any) -> tuple[Any, Any]:
    try:
        parameter = next(
            value for value in model.parameters() if torch.is_floating_point(value)
        )
    except (AttributeError, StopIteration) as error:
        raise OpenVLAInterventionError(
            "OpenVLA model must expose a floating-point parameter"
        ) from error
    return parameter.device, parameter.dtype


def _move_processor_output(
    output: Any,
    *,
    device: Any,
    dtype: Any,
    torch: Any,
) -> Mapping[str, Any]:
    if hasattr(output, "to"):
        moved = output.to(device=device, dtype=dtype)
        if moved is not None:
            output = moved
    elif isinstance(output, Mapping):
        output = {
            key: value.to(
                device=device,
                dtype=dtype if torch.is_floating_point(value) else value.dtype,
            )
            if torch.is_tensor(value)
            else value
            for key, value in output.items()
        }
    if not isinstance(output, Mapping):
        raise OpenVLAInterventionError("processor output must be a mapping")
    return output


def _validate_processor_tensors(
    input_ids: Any,
    pixel_values: Any,
    attention_mask: Any,
    torch: Any,
) -> None:
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise OpenVLAInterventionError(
            "processor input_ids must have shape [1, sequence]"
        )
    if not torch.is_tensor(pixel_values) or tuple(pixel_values.shape) != (
        1,
        6,
        224,
        224,
    ):
        raise OpenVLAInterventionError(
            "processor pixel_values must have shape [1, 6, 224, 224]"
        )
    if attention_mask is not None and (
        not torch.is_tensor(attention_mask)
        or tuple(attention_mask.shape) != tuple(input_ids.shape)
    ):
        raise OpenVLAInterventionError("processor attention_mask must match input_ids")


def _append_empty_action_token(input_ids: Any, torch: Any) -> Any:
    if bool(torch.all(input_ids[:, -1] == _EMPTY_ACTION_TOKEN_ID)):
        return input_ids
    suffix = torch.full(
        (1, 1),
        _EMPTY_ACTION_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, suffix], dim=1)


def _validate_o2(o2: Any, torch: Any) -> None:
    if not torch.is_tensor(o2) or tuple(o2.shape) != _O2_SHAPE:
        raise OpenVLAInterventionError(f"OpenVLA O2 must have shape {_O2_SHAPE}")
    if not torch.is_floating_point(o2) or not bool(torch.all(torch.isfinite(o2))):
        raise OpenVLAInterventionError("OpenVLA O2 must be a finite floating tensor")


def _validate_and_convert_override(o2: Any, clean_o2: Any, torch: Any) -> Any:
    _validate_o2(o2, torch)
    return o2.to(device=clean_o2.device, dtype=clean_o2.dtype)


def _build_multimodal_attention_mask(
    attention_mask: Any,
    o2: Any,
    torch: Any,
) -> Any:
    if attention_mask is None:
        return None
    patch_mask = torch.ones(
        (1, o2.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat(
        [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]],
        dim=1,
    )


def _multimodal_embeddings(text_embeddings: Any, o2: Any, torch: Any) -> Any:
    return torch.cat(
        [text_embeddings[:, :1], o2, text_embeddings[:, 1:]],
        dim=1,
    )


def _validate_action_dim(model: Any, unnorm_key: str) -> int:
    try:
        action_dim = model.get_action_dim(unnorm_key)
    except Exception as error:
        raise OpenVLAInterventionError(
            "failed to resolve OpenVLA action dimension"
        ) from error
    if action_dim != 7:
        raise OpenVLAInterventionError(
            "frozen OpenVLA LIBERO action dimension must be 7"
        )
    return action_dim


def _generated_sequences(generated: Any, torch: Any) -> Any:
    sequences = getattr(generated, "sequences", generated)
    if not torch.is_tensor(sequences) or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise OpenVLAInterventionError(
            "OpenVLA generation must return token sequences with batch size 1"
        )
    if sequences.shape[1] < 7:
        raise OpenVLAInterventionError(
            "OpenVLA generation returned fewer than seven action tokens"
        )
    return sequences


def _decode_result(
    prepared: PreparedOpenVLAContext,
    generated: Any,
) -> OpenVLAContinuationResult:
    torch = prepared._runtime.torch
    sequences = _generated_sequences(generated, torch)
    token_ids = sequences[:, -7:].detach().to(device="cpu").numpy().astype(np.int64)
    model = prepared._model
    try:
        bin_centers = np.asarray(model.bin_centers)
        discretized = np.clip(
            model.vocab_size - token_ids - 1,
            a_min=0,
            a_max=bin_centers.shape[0] - 1,
        )
        normalized = np.asarray(bin_centers[discretized])
        stats = model.get_action_stats(prepared.unnorm_key)
        low = np.asarray(stats["q01"])
        high = np.asarray(stats["q99"])
        mask = np.asarray(
            stats.get("mask", np.ones_like(low, dtype=bool)),
            dtype=bool,
        )
        if low.shape != (7,) or high.shape != (7,) or mask.shape != (7,):
            raise ValueError("malformed action statistics")
        unnormalized = np.where(
            mask[None, :],
            0.5 * (normalized + 1.0) * (high - low)[None, :] + low[None, :],
            normalized,
        )
        deployed = np.asarray(unnormalized).copy()
        deployed = prepared._runtime.normalize_gripper_action(
            deployed,
            binarize=True,
        )
        deployed = prepared._runtime.invert_gripper_action(deployed)
    except Exception as error:
        raise OpenVLAInterventionError(
            "failed to decode OpenVLA action tokens"
        ) from error

    arrays = (normalized, unnormalized, deployed)
    if any(
        np.asarray(value).shape != _ACTION_SHAPE
        or not np.issubdtype(np.asarray(value).dtype, np.floating)
        or not np.all(np.isfinite(value))
        for value in arrays
    ):
        raise OpenVLAInterventionError(
            "decoded OpenVLA actions are malformed or non-finite"
        )
    return OpenVLAContinuationResult(
        action_token_ids=token_ids,
        normalized_action=np.asarray(normalized).copy(),
        unnormalized_action=np.asarray(unnormalized).copy(),
        deployed_action=np.asarray(deployed).copy(),
    )


def _load_openvla_runtime() -> _OpenVLARuntime:
    try:
        import torch
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )
        from PIL import Image

        from .openvla_features import (
            _build_prompt,
            _load_preprocessing_runtime,
        )
    except ImportError as error:
        raise OpenVLAInterventionError(
            "official OpenVLA runtime dependencies must be importable"
        ) from error
    preprocessing = _load_preprocessing_runtime()
    return _OpenVLARuntime(
        torch=torch,
        image_type=Image,
        center_crop_image=preprocessing.center_crop_image,
        build_prompt=_build_prompt,
        openvla_v01_system_prompt=preprocessing.openvla_v01_system_prompt,
        normalize_gripper_action=normalize_gripper_action,
        invert_gripper_action=invert_gripper_action,
    )
