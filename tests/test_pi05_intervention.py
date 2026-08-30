from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import shared_feature.pi05_intervention as target


class _Tree:
    @classmethod
    def map(cls, function, value):
        if isinstance(value, dict):
            return {key: cls.map(function, item) for key, item in value.items()}
        return function(value)


class _Lax:
    @staticmethod
    def while_loop(condition, step, carry):
        iterations = 0
        while bool(condition(carry)):
            carry = step(carry)
            iterations += 1
            assert iterations <= 20
        return carry


class _Random:
    @staticmethod
    def key(seed):
        return seed


class _Jax:
    tree = _Tree
    lax = _Lax
    random = _Random

    @staticmethod
    def device_get(value):
        return value


class _Jnp:
    asarray = staticmethod(np.asarray)
    concatenate = staticmethod(np.concatenate)
    array = staticmethod(np.array)
    cumsum = staticmethod(np.cumsum)
    broadcast_to = staticmethod(np.broadcast_to)
    sum = staticmethod(np.sum)


class _Einops:
    @staticmethod
    def repeat(value, pattern, **axes):
        if pattern == "b -> b s":
            return np.repeat(np.asarray(value)[:, None], axes["s"], axis=1)
        if pattern == "b p -> b s p":
            return np.repeat(np.asarray(value)[:, None, :], axes["s"], axis=1)
        raise AssertionError(pattern)


@dataclass
class _Observation:
    images: dict
    image_masks: dict
    state: np.ndarray
    tokenized_prompt: np.ndarray
    tokenized_prompt_mask: np.ndarray

    @classmethod
    def from_dict(cls, value):
        return cls(
            images=value["images"],
            image_masks=value["image_masks"],
            state=value["state"],
            tokenized_prompt=value["tokenized_prompt"],
            tokenized_prompt_mask=value["tokenized_prompt_mask"],
        )


class _PaliGemma:
    def __init__(self):
        self.last_prefix = None

    def img(self, image, *, train):
        assert train is False
        marker = float(np.asarray(image).mean())
        return np.full((1, 256, 2048), marker, dtype=np.float32), {}

    def llm(self, tokens, **kwargs):
        if kwargs.get("method") == "embed":
            return np.zeros((1, np.asarray(tokens).shape[1], 2048), dtype=np.float32)
        if tokens[0] is not None:
            self.last_prefix = np.asarray(tokens[0]).copy()
            marker = float(self.last_prefix[:, :256].mean())
            return None, marker
        marker = kwargs["kv_cache"]
        suffix = np.full((1, 10, 32), marker, dtype=np.float32)
        return (None, suffix), None


class _Model:
    pi05 = True
    action_horizon = 10
    action_dim = 32
    num_steps = 10

    def __init__(self, preprocess):
        self.PaliGemma = _PaliGemma()
        self._preprocess = preprocess
        self.reference_noise = None

    def embed_suffix(self, observation, x_t, time):
        assert observation.state.shape == (1, 32)
        assert time.shape == (1,)
        return (
            np.zeros((1, 10, 32), dtype=np.float32),
            np.ones((1, 10), dtype=bool),
            np.zeros(10, dtype=bool),
            None,
        )

    def action_out_proj(self, suffix):
        return suffix

    def sample_actions(self, rng, observation, *, num_steps, noise):
        assert rng == 0
        assert num_steps == 10
        self.reference_noise = np.asarray(noise).copy()
        observation = self._preprocess(None, observation, train=False)
        base, _ = self.PaliGemma.img(observation.images["base_0_rgb"], train=False)
        return np.asarray(noise) - float(base.mean())


class _Policy:
    _is_pytorch_model = False
    _sample_kwargs = {}

    def __init__(self, model, calls, *, batched_images=False):
        self._model = model

        def transform(observation):
            calls.append("input_transform")
            image_shape = (1, 224, 224, 3) if batched_images else (224, 224, 3)
            return {
                "images": {
                    "base_0_rgb": np.full(image_shape, 1.0, dtype=np.float32),
                    "left_wrist_0_rgb": np.full(image_shape, 2.0, dtype=np.float32),
                    "right_wrist_0_rgb": np.zeros(image_shape, dtype=np.float32),
                },
                "image_masks": {
                    "base_0_rgb": np.array(True),
                    "left_wrist_0_rgb": np.array(True),
                    "right_wrist_0_rgb": np.array(False),
                },
                "state": np.zeros(32, dtype=np.float32),
                "tokenized_prompt": np.array([1, 2], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True]),
            }

        def output_transform(output):
            calls.append("output_transform")
            return {"actions": np.asarray(output["actions"])[..., :7] + 5.0}

        self._input_transform = transform
        self._output_transform = output_transform


@pytest.fixture
def setup_runtime(monkeypatch):
    calls = []

    def preprocess(rng, observation, *, train):
        assert rng is None
        assert train is False
        calls.append("preprocess")
        return observation

    runtime = target._Pi05Runtime(
        jax=_Jax,
        jnp=_Jnp,
        einops=_Einops,
        observation_type=_Observation,
        preprocess_observation=preprocess,
        make_attn_mask=lambda mask, ar_mask: np.ones(
            (mask.shape[0], mask.shape[1], mask.shape[1]),
            dtype=bool,
        ),
    )
    monkeypatch.setattr(target, "_load_pi05_runtime", lambda: runtime)
    return calls, preprocess


def _prepare(setup_runtime, *, noise=None, batched_images=False):
    calls, preprocess = setup_runtime
    model = _Model(preprocess)
    policy = _Policy(model, calls, batched_images=batched_images)
    if noise is None:
        noise = np.zeros((1, 10, 32), dtype=np.float32)
    prepared = target.prepare_pi05_context(
        policy=policy,
        observation={
            "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/state": np.zeros(8),
            "prompt": "task",
        },
        noise=noise,
    )
    return model, prepared


def test_prepare_pi05_context_preserves_slots_context_and_explicit_noise(
    setup_runtime,
):
    supplied_noise = np.arange(320, dtype=np.float32).reshape(1, 10, 32)
    model, prepared = _prepare(setup_runtime, noise=supplied_noise)
    calls, _ = setup_runtime

    assert calls == ["input_transform", "preprocess"]
    assert prepared.base_p2.shape == (1, 256, 2048)
    assert prepared.left_p2.shape == (1, 256, 2048)
    assert prepared.right_p2.shape == (1, 256, 2048)
    assert prepared.right_image_mask is False
    assert prepared.config_name == target.PI05_CONFIG_NAME
    assert prepared.checkpoint == target.PI05_CHECKPOINT
    assert prepared.backend == "jax_nnx"
    np.testing.assert_array_equal(prepared.noise, supplied_noise)
    supplied_noise.fill(-99)
    assert not np.any(prepared.noise == -99)
    assert model.reference_noise is None
    with pytest.raises(FrozenInstanceError):
        prepared.right_image_mask = True


def test_pi05_clean_reference_equivalence_and_override_consumption(setup_runtime):
    model, prepared = _prepare(setup_runtime)
    left_before = prepared.left_p2.copy()
    right_before = prepared.right_p2.copy()
    noise_before = prepared.noise.copy()

    reference = target.run_pi05_reference(prepared=prepared)
    clean = target.continue_pi05_from_p2(
        prepared=prepared,
        base_p2=prepared.base_p2,
    )
    np.testing.assert_allclose(
        clean.normalized_action_chunk_32,
        reference.normalized_action_chunk_32,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        clean.normalized_action_chunk,
        reference.normalized_action_chunk,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        clean.unnormalized_action_chunk,
        reference.unnormalized_action_chunk,
        rtol=0,
        atol=1e-6,
    )
    assert clean.normalized_action_chunk_32.shape == (1, 10, 32)
    assert clean.normalized_action_chunk.shape == (1, 10, 7)
    assert clean.unnormalized_action_chunk.shape == (1, 10, 7)
    assert not np.array_equal(
        clean.normalized_action_chunk,
        clean.unnormalized_action_chunk,
    )
    np.testing.assert_array_equal(model.reference_noise, noise_before)

    override = prepared.base_p2 + 2.0
    intervened = target.continue_pi05_from_p2(
        prepared=prepared,
        base_p2=override,
    )
    np.testing.assert_array_equal(model.PaliGemma.last_prefix[:, :256], override)
    np.testing.assert_array_equal(
        model.PaliGemma.last_prefix[:, 256:512],
        prepared.left_p2,
    )
    np.testing.assert_array_equal(
        model.PaliGemma.last_prefix[:, 512:768],
        prepared.right_p2,
    )
    assert not np.array_equal(
        intervened.normalized_action_chunk_32,
        clean.normalized_action_chunk_32,
    )
    np.testing.assert_array_equal(prepared.left_p2, left_before)
    np.testing.assert_array_equal(prepared.right_p2, right_before)
    np.testing.assert_array_equal(prepared.noise, noise_before)


@pytest.mark.parametrize(
    "noise",
    [
        np.zeros((10, 32), dtype=np.float32),
        np.zeros((2, 10, 32), dtype=np.float32),
        np.zeros((1, 10, 32), dtype=np.int32),
        np.full((1, 10, 32), np.nan, dtype=np.float32),
    ],
)
def test_prepare_pi05_context_rejects_invalid_noise(setup_runtime, noise):
    with pytest.raises(target.Pi05InterventionError):
        _prepare(setup_runtime, noise=noise)


def test_prepare_pi05_context_enforces_unbatched_input(setup_runtime):
    with pytest.raises(target.Pi05InterventionError, match=r"\[1, 224, 224, 3\]"):
        _prepare(setup_runtime, batched_images=True)


def test_continue_pi05_validates_shape_and_finiteness(setup_runtime):
    _, prepared = _prepare(setup_runtime)
    wrong_shape = np.zeros((2, 256, 2048), dtype=np.float32)
    nonfinite = np.zeros((1, 256, 2048), dtype=np.float32)
    nonfinite[0, 0, 0] = np.inf

    with pytest.raises(target.Pi05InterventionError):
        target.continue_pi05_from_p2(prepared=prepared, base_p2=wrong_shape)
    with pytest.raises(target.Pi05InterventionError):
        target.continue_pi05_from_p2(prepared=prepared, base_p2=nonfinite)


def test_prepare_pi05_context_rejects_non_jax_policy(setup_runtime):
    calls, preprocess = setup_runtime
    policy = _Policy(_Model(preprocess), calls)
    policy._is_pytorch_model = True
    with pytest.raises(target.Pi05InterventionError, match="JAX/NNX"):
        target.prepare_pi05_context(
            policy=policy,
            observation={},
            noise=np.zeros((1, 10, 32), dtype=np.float32),
        )
