from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

import shared_feature.openvla_intervention as target


class _Batch(dict):
    def to(self, *, device, dtype):
        for key, value in self.items():
            if torch.is_floating_point(value):
                self[key] = value.to(device=device, dtype=dtype)
            else:
                self[key] = value.to(device=device)
        return self


class _Processor:
    def __init__(self):
        self.prompt = None
        self.image = None

    def __call__(self, prompt, image):
        self.prompt = prompt
        self.image = image
        return _Batch(
            input_ids=torch.tensor([[11, 12]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            pixel_values=torch.zeros((1, 6, 224, 224)),
        )


class _LanguageModel:
    def __init__(self, owner):
        self.owner = owner
        self.last_o2 = None

    def generate(self, *, inputs_embeds, attention_mask, max_new_tokens, do_sample):
        assert attention_mask.shape == (1, 259)
        assert max_new_tokens == 7
        assert do_sample is False
        self.last_o2 = inputs_embeds[:, 1:257].detach().clone()
        return self.owner.tokens_for_o2(self.last_o2)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.language_model = _LanguageModel(self)
        self.vocab_size = 100
        self.bin_centers = np.linspace(-1.0, 1.0, 100, dtype=np.float64)

    def vision_backbone(self, pixel_values):
        assert pixel_values.shape == (1, 6, 224, 224)
        return torch.ones((1, 256, 8), dtype=pixel_values.dtype)

    def projector(self, patch_features):
        return torch.full(
            (1, 256, 4096),
            0.25,
            dtype=patch_features.dtype,
            device=patch_features.device,
        )

    def get_input_embeddings(self):
        return lambda input_ids: torch.zeros(
            (1, input_ids.shape[1], 4096),
            dtype=self.weight.dtype,
            device=input_ids.device,
        )

    def get_action_dim(self, unnorm_key):
        assert unnorm_key == target.OPENVLA_UNNORM_KEY
        return 7

    def get_action_stats(self, unnorm_key):
        assert unnorm_key == target.OPENVLA_UNNORM_KEY
        return {
            "q01": np.zeros(7, dtype=np.float64),
            "q99": np.full(7, 2.0, dtype=np.float64),
            "mask": np.ones(7, dtype=bool),
        }

    def tokens_for_o2(self, o2):
        token = 90 + int(round(float(o2.mean()) * 4))
        return torch.full((1, 7), token, dtype=torch.long)

    def generate(
        self,
        *,
        input_ids,
        pixel_values,
        attention_mask,
        max_new_tokens,
        do_sample,
    ):
        assert input_ids.shape == attention_mask.shape == (1, 3)
        assert max_new_tokens == 7
        assert do_sample is False
        return self.tokens_for_o2(self.projector(self.vision_backbone(pixel_values)))


@pytest.fixture
def runtime(monkeypatch):
    def normalize_gripper_action(actions, *, binarize):
        assert binarize is True
        result = np.asarray(actions).copy()
        result[..., -1] = np.where(result[..., -1] > 0, 1.0, -1.0)
        return result

    def invert_gripper_action(actions):
        result = np.asarray(actions).copy()
        result[..., -1] *= -1
        return result

    value = target._OpenVLARuntime(
        torch=torch,
        image_type=__import__("PIL.Image", fromlist=["Image"]),
        center_crop_image=lambda image: image,
        build_prompt=lambda task, checkpoint, system: (
            f"{system}|{checkpoint}|{task.lower()}"
        ),
        openvla_v01_system_prompt="system",
        normalize_gripper_action=normalize_gripper_action,
        invert_gripper_action=invert_gripper_action,
    )
    monkeypatch.setattr(target, "_load_openvla_runtime", lambda: value)
    return value


def _prepare(runtime):
    model = _Model()
    processor = _Processor()
    prepared = target.prepare_openvla_context(
        model=model,
        processor=processor,
        observation={
            "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros(8),
        },
        task_description="Pick Up The Bowl",
    )
    return model, processor, prepared


def test_prepare_openvla_context_freezes_authoritative_semantics(runtime):
    _, processor, prepared = _prepare(runtime)

    assert prepared.o2.shape == (1, 256, 4096)
    assert prepared.checkpoint_identity == target.OPENVLA_CHECKPOINT_IDENTITY
    assert prepared.unnorm_key == target.OPENVLA_UNNORM_KEY
    assert prepared.center_crop is True
    assert prepared.task_description == "Pick Up The Bowl"
    assert processor.prompt == (
        "system|openvla/openvla-7b-finetuned-libero-spatial|pick up the bowl"
    )
    assert processor.image.size == (224, 224)
    with pytest.raises(FrozenInstanceError):
        prepared.task_description = "changed"


def test_openvla_clean_reference_equivalence_and_override_consumption(runtime):
    model, _, prepared = _prepare(runtime)
    clean_o2 = prepared.o2.detach().clone()

    reference = target.run_openvla_reference(prepared=prepared)
    clean = target.continue_openvla_from_o2(prepared=prepared, o2=prepared.o2)
    np.testing.assert_array_equal(clean.action_token_ids, reference.action_token_ids)
    np.testing.assert_allclose(clean.normalized_action, reference.normalized_action)
    np.testing.assert_allclose(clean.unnormalized_action, reference.unnormalized_action)
    np.testing.assert_allclose(clean.deployed_action, reference.deployed_action)
    assert clean.action_token_ids.shape == (1, 7)
    assert np.issubdtype(clean.action_token_ids.dtype, np.integer)
    assert clean.normalized_action.shape == (1, 7)
    assert clean.unnormalized_action.shape == (1, 7)
    assert clean.deployed_action.shape == (1, 7)
    assert not np.array_equal(clean.normalized_action, clean.unnormalized_action)
    assert clean.deployed_action[0, -1] == -1.0

    override = torch.full_like(prepared.o2, -2.25)
    intervened = target.continue_openvla_from_o2(
        prepared=prepared,
        o2=override,
    )
    torch.testing.assert_close(model.language_model.last_o2, override)
    assert not np.array_equal(
        intervened.action_token_ids,
        clean.action_token_ids,
    )
    torch.testing.assert_close(prepared.o2, clean_o2)


@pytest.mark.parametrize(
    ("observation", "kwargs"),
    [
        ({"full_image": np.zeros((1, 224, 224, 3), dtype=np.uint8)}, {}),
        ({"full_image": np.zeros((224, 224, 3), dtype=np.float32)}, {}),
        (
            {"full_image": np.zeros((224, 224, 3), dtype=np.uint8)},
            {"pretrained_checkpoint": "wrong"},
        ),
        (
            {"full_image": np.zeros((224, 224, 3), dtype=np.uint8)},
            {"unnorm_key": "wrong"},
        ),
        (
            {"full_image": np.zeros((224, 224, 3), dtype=np.uint8)},
            {"center_crop": False},
        ),
    ],
)
def test_prepare_openvla_context_rejects_non_frozen_inputs(
    runtime,
    observation,
    kwargs,
):
    with pytest.raises(target.OpenVLAInterventionError):
        target.prepare_openvla_context(
            model=_Model(),
            processor=_Processor(),
            observation=observation,
            task_description="task",
            **kwargs,
        )


def test_continue_openvla_validates_shape_finiteness_and_batch(runtime):
    _, _, prepared = _prepare(runtime)
    wrong_shape = torch.zeros((2, 256, 4096))
    nonfinite = torch.zeros((1, 256, 4096))
    nonfinite[0, 0, 0] = torch.nan

    with pytest.raises(target.OpenVLAInterventionError):
        target.continue_openvla_from_o2(prepared=prepared, o2=wrong_shape)
    with pytest.raises(target.OpenVLAInterventionError):
        target.continue_openvla_from_o2(prepared=prepared, o2=nonfinite)
