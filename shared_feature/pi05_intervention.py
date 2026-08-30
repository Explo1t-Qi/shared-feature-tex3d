from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


PI05_CONFIG_NAME = "pi05_libero"
PI05_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
_P2_SHAPE = (1, 256, 2048)
_NOISE_SHAPE = (1, 10, 32)


class Pi05InterventionError(RuntimeError):
    """Raised when the frozen pi0.5 P2 interface cannot run safely."""


@dataclass(frozen=True)
class Pi05ContinuationResult:
    normalized_action_chunk_32: np.ndarray
    normalized_action_chunk: np.ndarray
    unnormalized_action_chunk: np.ndarray


@dataclass(frozen=True)
class PreparedPi05Context:
    base_p2: Any
    left_p2: Any
    right_p2: Any
    right_image_mask: bool
    noise: Any
    config_name: str
    checkpoint: str
    backend: str
    _model: Any = field(repr=False)
    _raw_model_observation: Any = field(repr=False)
    _model_observation: Any = field(repr=False)
    _prompt_embeddings: Any = field(repr=False)
    _prefix_mask: Any = field(repr=False)
    _prefix_ar_mask: Any = field(repr=False)
    _model_input_state: Any = field(repr=False)
    _output_transform: Any = field(repr=False)
    _num_steps: int = field(repr=False)
    _runtime: Any = field(repr=False)


@dataclass(frozen=True)
class _Pi05Runtime:
    jax: Any
    jnp: Any
    einops: Any
    observation_type: Any
    preprocess_observation: Any
    make_attn_mask: Any


def prepare_pi05_context(
    *,
    policy: Any,
    observation: Mapping[str, Any],
    noise: Any,
) -> PreparedPi05Context:
    """Apply the deployed pi05_libero transforms and expose all three P2 slots."""
    runtime = _load_pi05_runtime()
    model, input_transform, output_transform, num_steps = _validate_policy(policy)
    if not isinstance(observation, Mapping):
        raise Pi05InterventionError(
            "observation must be one unbatched raw LIBERO inference mapping"
        )
    noise_array = _validate_noise(noise)

    try:
        copied = runtime.jax.tree.map(lambda value: value, observation)
        transformed = input_transform(copied)
        if not isinstance(transformed, Mapping):
            raise Pi05InterventionError(
                "pi05_libero input transforms must return a mapping"
            )
        batched = runtime.jax.tree.map(
            lambda value: runtime.jnp.asarray(value)[None, ...],
            transformed,
        )
        raw_model_observation = runtime.observation_type.from_dict(batched)
        model_observation = runtime.preprocess_observation(
            None,
            raw_model_observation,
            train=False,
        )
    except Pi05InterventionError:
        raise
    except Exception as error:
        raise Pi05InterventionError(
            "failed to apply the authoritative pi05_libero input path"
        ) from error

    _validate_model_observation(model_observation)
    image_tokens: dict[str, Any] = {}
    try:
        for name in _IMAGE_KEYS:
            p2, _ = model.PaliGemma.img(
                model_observation.images[name],
                train=False,
            )
            _validate_p2(name, p2)
            image_tokens[name] = p2
        prompt_embeddings = model.PaliGemma.llm(
            model_observation.tokenized_prompt,
            method="embed",
        )
        prefix_mask, prefix_ar_mask = _build_prefix_masks(
            model_observation,
            image_tokens,
            runtime,
        )
        native_noise = runtime.jnp.asarray(noise_array)
    except Pi05InterventionError:
        raise
    except Exception as error:
        raise Pi05InterventionError(
            "failed to prepare pi0.5 P2 continuation context"
        ) from error

    right_mask = np.asarray(
        runtime.jax.device_get(model_observation.image_masks["right_wrist_0_rgb"])
    )
    if right_mask.shape != (1,) or bool(right_mask[0]) is not False:
        raise Pi05InterventionError(
            "deployed pi05_libero right image mask must be False"
        )

    return PreparedPi05Context(
        base_p2=image_tokens["base_0_rgb"],
        left_p2=image_tokens["left_wrist_0_rgb"],
        right_p2=image_tokens["right_wrist_0_rgb"],
        right_image_mask=False,
        noise=native_noise,
        config_name=PI05_CONFIG_NAME,
        checkpoint=PI05_CHECKPOINT,
        backend="jax_nnx",
        _model=model,
        _raw_model_observation=raw_model_observation,
        _model_observation=model_observation,
        _prompt_embeddings=prompt_embeddings,
        _prefix_mask=prefix_mask,
        _prefix_ar_mask=prefix_ar_mask,
        _model_input_state=batched["state"],
        _output_transform=output_transform,
        _num_steps=num_steps,
        _runtime=runtime,
    )


def continue_pi05_from_p2(
    *,
    prepared: PreparedPi05Context,
    base_p2: Any,
) -> Pi05ContinuationResult:
    """Continue pi0.5 flow matching from a supplied native base-camera P2."""
    if not isinstance(prepared, PreparedPi05Context):
        raise Pi05InterventionError("prepared must be a PreparedPi05Context")
    converted = _validate_and_convert_p2_override(base_p2, prepared)
    actions = _sample_from_prefix(prepared, converted)
    return _build_result(prepared, actions)


def run_pi05_reference(
    *,
    prepared: PreparedPi05Context,
) -> Pi05ContinuationResult:
    """Run one authoritative fixed-noise pi0.5 sample_actions invocation."""
    if not isinstance(prepared, PreparedPi05Context):
        raise Pi05InterventionError("prepared must be a PreparedPi05Context")
    runtime = prepared._runtime
    try:
        actions = prepared._model.sample_actions(
            runtime.jax.random.key(0),
            prepared._raw_model_observation,
            num_steps=prepared._num_steps,
            noise=prepared.noise,
        )
    except Exception as error:
        raise Pi05InterventionError(
            "authoritative fixed-noise pi0.5 reference inference failed"
        ) from error
    return _build_result(prepared, actions)


def _validate_policy(policy: Any) -> tuple[Any, Any, Any, int]:
    if getattr(policy, "_is_pytorch_model", None) is not False:
        raise Pi05InterventionError(
            "policy must use the frozen JAX/NNX pi05_libero backend"
        )
    model = getattr(policy, "_model", None)
    input_transform = getattr(policy, "_input_transform", None)
    output_transform = getattr(policy, "_output_transform", None)
    sample_kwargs = getattr(policy, "_sample_kwargs", None)
    if (
        model is None
        or not callable(input_transform)
        or not callable(output_transform)
        or not isinstance(sample_kwargs, Mapping)
    ):
        raise Pi05InterventionError(
            "policy must expose the audited OpenPI model and transforms"
        )
    if set(sample_kwargs) - {"num_steps"}:
        raise Pi05InterventionError("pi05_libero policy has unsupported sample kwargs")
    num_steps = sample_kwargs.get("num_steps", getattr(model, "num_steps", 10))
    if type(num_steps) is not int or num_steps <= 0:
        raise Pi05InterventionError("pi05_libero num_steps must be a positive integer")
    if (
        getattr(model, "pi05", None) is not True
        or getattr(model, "action_horizon", None) != 10
        or getattr(model, "action_dim", None) != 32
        or not callable(getattr(model, "embed_suffix", None))
        or not callable(getattr(model, "action_out_proj", None))
        or not callable(getattr(model, "sample_actions", None))
        or not callable(getattr(getattr(model, "PaliGemma", None), "img", None))
        or not callable(getattr(getattr(model, "PaliGemma", None), "llm", None))
    ):
        raise Pi05InterventionError(
            "policy model is incompatible with frozen pi05_libero semantics"
        )
    return model, input_transform, output_transform, num_steps


def _validate_noise(noise: Any) -> np.ndarray:
    try:
        value = np.asarray(noise)
    except Exception as error:
        raise Pi05InterventionError(
            "noise must be a floating array with shape [1, 10, 32]"
        ) from error
    if (
        value.shape != _NOISE_SHAPE
        or not np.issubdtype(value.dtype, np.floating)
        or not np.all(np.isfinite(value))
    ):
        raise Pi05InterventionError(
            "noise must be a finite floating array with shape [1, 10, 32]"
        )
    return np.array(value, copy=True)


def _validate_model_observation(observation: Any) -> None:
    images = getattr(observation, "images", None)
    masks = getattr(observation, "image_masks", None)
    if not isinstance(images, Mapping) or tuple(images) != _IMAGE_KEYS:
        raise Pi05InterventionError(
            "pi05_libero observation must preserve canonical image-slot order"
        )
    if not isinstance(masks, Mapping) or tuple(masks) != _IMAGE_KEYS:
        raise Pi05InterventionError(
            "pi05_libero observation must preserve canonical mask order"
        )
    if getattr(observation, "state", np.empty(0)).shape != (1, 32):
        raise Pi05InterventionError("pi05_libero model state must have shape [1, 32]")
    prompt = getattr(observation, "tokenized_prompt", None)
    prompt_mask = getattr(observation, "tokenized_prompt_mask", None)
    if prompt is None or prompt_mask is None or prompt.shape[0] != 1:
        raise Pi05InterventionError(
            "pi05_libero tokenized prompt context is missing or batched incorrectly"
        )
    expected_masks = (True, True, False)
    for name, expected in zip(_IMAGE_KEYS, expected_masks, strict=True):
        if getattr(images[name], "shape", None) != (1, 224, 224, 3):
            raise Pi05InterventionError(
                f"pi05_libero image slot {name} must have shape [1, 224, 224, 3]"
            )
        value = np.asarray(masks[name])
        if value.shape != (1,) or bool(value[0]) is not expected:
            raise Pi05InterventionError(f"unexpected pi05_libero image mask for {name}")


def _validate_p2(name: str, value: Any) -> None:
    if getattr(value, "shape", None) != _P2_SHAPE:
        raise Pi05InterventionError(f"{name} P2 must have shape {_P2_SHAPE}")
    try:
        array = np.asarray(value)
    except Exception as error:
        raise Pi05InterventionError(f"{name} P2 is not array-like") from error
    if not _is_floating_dtype(array.dtype) or not np.all(np.isfinite(array)):
        raise Pi05InterventionError(f"{name} P2 must be finite and floating")


def _build_prefix_masks(
    observation: Any,
    image_tokens: Mapping[str, Any],
    runtime: _Pi05Runtime,
) -> tuple[Any, Any]:
    masks = []
    ar_mask: list[bool] = []
    for name in _IMAGE_KEYS:
        token_count = image_tokens[name].shape[1]
        masks.append(
            runtime.einops.repeat(
                observation.image_masks[name],
                "b -> b s",
                s=token_count,
            )
        )
        ar_mask.extend([False] * token_count)
    masks.append(observation.tokenized_prompt_mask)
    ar_mask.extend([False] * observation.tokenized_prompt.shape[1])
    return runtime.jnp.concatenate(masks, axis=1), runtime.jnp.array(ar_mask)


def _validate_and_convert_p2_override(
    base_p2: Any,
    prepared: PreparedPi05Context,
) -> Any:
    _validate_p2("base_0_rgb override", base_p2)
    runtime = prepared._runtime
    clean_dtype = getattr(prepared.base_p2, "dtype", None)
    try:
        converted = runtime.jnp.asarray(base_p2, dtype=clean_dtype)
    except TypeError:
        converted = runtime.jnp.asarray(base_p2)
    if getattr(converted, "shape", None) != _P2_SHAPE:
        raise Pi05InterventionError(
            "converted base P2 does not preserve the frozen shape"
        )
    return converted


def _sample_from_prefix(
    prepared: PreparedPi05Context,
    base_p2: Any,
) -> Any:
    runtime = prepared._runtime
    jnp = runtime.jnp
    model = prepared._model
    prefix_tokens = jnp.concatenate(
        [
            base_p2,
            prepared.left_p2,
            prepared.right_p2,
            prepared._prompt_embeddings,
        ],
        axis=1,
    )
    try:
        prefix_attn_mask = runtime.make_attn_mask(
            prepared._prefix_mask,
            prepared._prefix_ar_mask,
        )
        positions = jnp.cumsum(prepared._prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        dt = -1.0 / prepared._num_steps

        def step(carry: tuple[Any, Any]) -> tuple[Any, Any]:
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
                model.embed_suffix(
                    prepared._model_observation,
                    x_t,
                    jnp.broadcast_to(time, 1),
                )
            )
            suffix_attn_mask = runtime.make_attn_mask(
                suffix_mask,
                suffix_ar_mask,
            )
            prefix_for_suffix = runtime.einops.repeat(
                prepared._prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attn_mask = jnp.concatenate(
                [prefix_for_suffix, suffix_attn_mask],
                axis=-1,
            )
            suffix_positions = (
                jnp.sum(prepared._prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise Pi05InterventionError(
                    "pi0.5 suffix continuation unexpectedly returned prefix output"
                )
            velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry: tuple[Any, Any]) -> Any:
            _, time = carry
            return time >= -dt / 2

        actions, _ = runtime.jax.lax.while_loop(
            cond,
            step,
            (prepared.noise, 1.0),
        )
        return actions
    except Pi05InterventionError:
        raise
    except Exception as error:
        raise Pi05InterventionError(
            "pi0.5 continuation from supplied base P2 failed"
        ) from error


def _build_result(
    prepared: PreparedPi05Context,
    actions: Any,
) -> Pi05ContinuationResult:
    runtime = prepared._runtime
    normalized_32 = np.asarray(runtime.jax.device_get(actions)).copy()
    if (
        normalized_32.shape != _NOISE_SHAPE
        or not _is_floating_dtype(normalized_32.dtype)
        or not np.all(np.isfinite(normalized_32))
    ):
        raise Pi05InterventionError(
            "pi0.5 native normalized action chunk is malformed or non-finite"
        )
    normalized = normalized_32[..., :7].copy()
    state = np.asarray(runtime.jax.device_get(prepared._model_input_state[0])).copy()
    outputs = {
        "state": state,
        "actions": normalized_32[0].copy(),
    }
    try:
        transformed = prepared._output_transform(outputs)
        unnormalized = np.asarray(transformed["actions"])[None, ...].copy()
    except Exception as error:
        raise Pi05InterventionError("pi05_libero output transforms failed") from error
    if (
        unnormalized.shape != (1, 10, 7)
        or not _is_floating_dtype(unnormalized.dtype)
        or not np.all(np.isfinite(unnormalized))
    ):
        raise Pi05InterventionError(
            "pi05_libero unnormalized action chunk must be finite [1, 10, 7]"
        )
    return Pi05ContinuationResult(
        normalized_action_chunk_32=normalized_32,
        normalized_action_chunk=normalized,
        unnormalized_action_chunk=unnormalized,
    )


def _is_floating_dtype(dtype: Any) -> bool:
    try:
        if np.issubdtype(dtype, np.floating):
            return True
    except TypeError:
        pass
    return str(dtype) == "bfloat16"


def _load_pi05_runtime() -> _Pi05Runtime:
    try:
        import einops
        import jax
        import jax.numpy as jnp
        from openpi.models import model as openpi_model
        from openpi.models.pi0 import make_attn_mask
    except ImportError as error:
        raise Pi05InterventionError(
            "OpenPI JAX/NNX runtime dependencies must be importable"
        ) from error
    return _Pi05Runtime(
        jax=jax,
        jnp=jnp,
        einops=einops,
        observation_type=openpi_model.Observation,
        preprocess_observation=openpi_model.preprocess_observation,
        make_attn_mask=make_attn_mask,
    )
