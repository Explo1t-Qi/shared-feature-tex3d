import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import shared_feature.pi05_features as features
from shared_feature import (
    Pi05FeatureExtractionError,
    PilotObservation,
    extract_pi05_features,
)


def make_observation(
    path: Path,
    sample_id: str,
    value: int,
    *,
    state: np.ndarray | None = None,
    base_image: np.ndarray | None = None,
) -> Path:
    if base_image is None:
        base_image = np.zeros((3, 4, 3), dtype=np.uint8)
        base_image[-1, -1] = (value, value + 1, value + 2)
    wrist_image = np.zeros((2, 5, 3), dtype=np.uint8)
    wrist_image[-1, -1] = (value + 10, value + 11, value + 12)
    if state is None:
        state = np.arange(8, dtype=np.float32) + value
    PilotObservation(
        sample_id=sample_id,
        task_id="2",
        initial_state_id=0,
        episode_id=0,
        step_id=value,
        normalized_episode_progress=0.0,
        base_rgb_raw=base_image,
        wrist_rgb_raw=wrist_image,
        state=state,
        prompt=f"Pick object {value}",
        episode_success=True,
    ).save(path)
    return path


class FakeImageTools:
    def __init__(self) -> None:
        self.resize_inputs: list[np.ndarray] = []

    def resize_with_pad(self, image, height, width):
        self.resize_inputs.append(image.copy())
        assert image.flags.c_contiguous
        assert (height, width) == (224, 224)
        return np.broadcast_to(image[0, 0], (height, width, 3)).copy()

    @staticmethod
    def convert_to_uint8(image):
        return image.astype(np.uint8, copy=True)


class FakeObservation:
    @classmethod
    def from_dict(cls, data):
        images = {
            key: value.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            for key, value in data["image"].items()
        }
        return SimpleNamespace(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data["tokenized_prompt"],
            tokenized_prompt_mask=data["tokenized_prompt_mask"],
        )


class FakeProjector(nn.Module):
    def forward(self, value):
        return value[..., :1].expand(-1, -1, 2048) + 1.0


class FakePaliGemmaWithExpert(nn.Module):
    def __init__(self, mode: str = "valid") -> None:
        super().__init__()
        self.mode = mode
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(multi_modal_projector=FakeProjector())
        )

    def embed_image(self, image):
        batch = image.shape[0]
        marker = image[:, 0, 0, 0].reshape(batch, 1, 1)
        token = torch.arange(256, device=image.device).reshape(1, 256, 1) / 256
        p1 = (marker + token + 2.0).expand(batch, 256, 1152)
        if self.mode == "wrong_p1":
            p1 = p1[:, :, :-1]
        if self.mode == "nonfinite":
            p1 = p1.clone()
            p1[0, 0, 0] = torch.nan
        if self.mode == "wrong_dtype":
            p1 = p1.to(torch.float64)
        p2 = self.paligemma.model.multi_modal_projector(p1)
        if self.mode == "wrong_p2":
            p2 = p2[:, :, :-1]
        if self.mode == "zero":
            p2 = torch.zeros_like(p2)
        return p2


class FakeModel(nn.Module):
    def __init__(self, mode: str = "valid") -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.paligemma_with_expert = FakePaliGemmaWithExpert(mode)
        self.preprocess_calls: list[bool] = []
        self.prefix_corrupt = mode == "prefix_mismatch"

    def _preprocess_observation(self, observation, *, train):
        self.preprocess_calls.append(train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, image_masks, lang_tokens, lang_masks):
        del image_masks, lang_tokens, lang_masks
        prefix = torch.cat(
            [self.paligemma_with_expert.embed_image(image) for image in images],
            dim=1,
        )
        if self.prefix_corrupt:
            prefix = prefix.clone()
            prefix[0, 0, 0] += 1.0
        batch = prefix.shape[0]
        return (
            prefix,
            torch.ones((batch, prefix.shape[1]), dtype=torch.bool),
            torch.zeros((batch, prefix.shape[1]), dtype=torch.bool),
        )


class FakePolicy:
    _is_pytorch_model = True
    _pytorch_device = "cpu"

    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def _input_transform(self, data):
        self.inputs.append(data)
        base = data["observation/image"]
        wrist = data["observation/wrist_image"]
        return {
            "state": np.pad(data["observation/state"], (0, 24)),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "tokenized_prompt": np.arange(200, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(200, dtype=bool),
        }


def install_runtime(monkeypatch):
    image_tools = FakeImageTools()
    runtime = features._OpenPIRuntime(
        torch=torch,
        image_tools=image_tools,
        observation_type=FakeObservation,
        native_feature_dtype=torch.float32,
    )
    monkeypatch.setattr(features, "_load_openpi_runtime", lambda: runtime)
    return image_tools


def extract(monkeypatch, paths, output, *, model=None, batch_size=1):
    image_tools = install_runtime(monkeypatch)
    policy = FakePolicy()
    written = extract_pi05_features(
        model=model or FakeModel(),
        policy=policy,
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        observation_paths=paths,
        output_dir=output,
        batch_size=batch_size,
    )
    return written, policy, image_tools


def load_feature(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return (
            json.loads(str(archive["metadata_json"].item())),
            archive["p1_siglip"].copy(),
            archive["p2_projected"].copy(),
            set(archive.files),
        )


def test_extracts_current_pytorch_p2_with_official_order_and_identity(
    monkeypatch, tmp_path
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    paths = [
        make_observation(input_dir / "z.npz", "sample-z", 3),
        make_observation(input_dir / "a.npz", "sample-a", 7),
        make_observation(input_dir / "m.npz", "sample-m", 11),
    ]
    originals = [PilotObservation.load(path) for path in paths]
    model = FakeModel()

    written, policy, image_tools = extract(
        monkeypatch,
        paths,
        tmp_path / "features",
        model=model,
        batch_size=2,
    )

    assert [path.stem for path in written] == [
        "sample-z",
        "sample-a",
        "sample-m",
    ]
    assert model.preprocess_calls == [False, False]
    assert len(policy.inputs) == 3
    assert len(image_tools.resize_inputs) == 6
    for index, original in enumerate(originals):
        np.testing.assert_array_equal(
            image_tools.resize_inputs[index * 2],
            original.base_rgb_raw[::-1, ::-1],
        )
        np.testing.assert_array_equal(
            image_tools.resize_inputs[index * 2 + 1],
            original.wrist_rgb_raw[::-1, ::-1],
        )

    for path, original in zip(written, originals, strict=True):
        metadata, p1, p2, keys = load_feature(path)
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(original.base_rgb_raw).tobytes()
        ).hexdigest()
        assert metadata == {
            "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
            "feature_schema_version": "pi05_torch_features_v1",
            "sample_id": original.sample_id,
            "source_image_hash": f"sha256:{expected_hash}",
            "source_model": "pi05",
        }
        assert keys == {"metadata_json", "p1_siglip", "p2_projected"}
        assert p1.shape == (256, 1152)
        assert p2.shape == (256, 2048)
        assert p1.dtype == p2.dtype == np.float32
        assert np.all(np.diff(p1[:, 0]) > 0)
        np.testing.assert_allclose(p2[:, 0], p1[:, 0] + 1.0)


@pytest.mark.parametrize(
    ("sample_id", "state", "base_image", "match"),
    [
        ("../escape", np.zeros(8), None, "safe path"),
        ("nested/name", np.zeros(8), None, "safe path"),
        ("nested\\name", np.zeros(8), None, "safe path"),
        ("bad-shape", np.zeros(7), None, r"shape \(8,\)"),
        ("bad-dtype", np.asarray(list("abcdefgh")), None, "finite real"),
        (
            "bad-finite",
            np.asarray([0, 1, 2, 3, 4, 5, 6, np.nan]),
            None,
            "finite real",
        ),
        (
            "bad-image",
            np.zeros(8),
            np.zeros((3, 4, 1), dtype=np.uint8),
            "uint8 RGB",
        ),
    ],
)
def test_rejects_unsafe_or_malformed_observations(
    monkeypatch, tmp_path, sample_id, state, base_image, match
) -> None:
    path = make_observation(
        tmp_path / "input.npz",
        sample_id,
        1,
        state=state,
        base_image=base_image,
    )
    with pytest.raises(Pi05FeatureExtractionError, match=match):
        extract(monkeypatch, [path], tmp_path / "output")


def test_rejects_identity_output_and_historical_overwrite_collisions(
    monkeypatch, tmp_path
) -> None:
    first = make_observation(tmp_path / "first.npz", "same", 1)
    second = make_observation(tmp_path / "second.npz", "same", 2)
    with pytest.raises(Pi05FeatureExtractionError, match="duplicate observation"):
        extract(monkeypatch, [first, first], tmp_path / "duplicate-path")
    with pytest.raises(Pi05FeatureExtractionError, match="duplicate sample_id"):
        extract(monkeypatch, [first, second], tmp_path / "duplicate-id")

    output = tmp_path / "historical-features"
    output.mkdir()
    existing = output / "same.npz"
    existing.write_bytes(b"keep")
    with pytest.raises(Pi05FeatureExtractionError, match="overwrite"):
        extract(monkeypatch, [first], output)
    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("wrong_p1", "P1"),
        ("wrong_p2", "P2"),
        ("prefix_mismatch", "embed_prefix"),
        ("nonfinite", "non-finite"),
        ("wrong_dtype", "native feature dtype"),
        ("zero", "all-zero"),
    ],
)
def test_rejects_wrong_node_or_prefix_semantics(
    monkeypatch, tmp_path, mode, match
) -> None:
    path = make_observation(tmp_path / "input.npz", "sample", 1)
    with pytest.raises(Pi05FeatureExtractionError, match=match):
        extract(
            monkeypatch,
            [path],
            tmp_path / "output",
            model=FakeModel(mode),
        )


def test_rejects_unfrozen_or_non_pytorch_model(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    path = make_observation(tmp_path / "input.npz", "sample", 1)
    unfrozen = FakeModel()
    unfrozen.anchor.requires_grad_(True)
    with pytest.raises(Pi05FeatureExtractionError, match="frozen"):
        extract_pi05_features(
            model=unfrozen,
            policy=FakePolicy(),
            checkpoint="checkpoint",
            observation_paths=[path],
            output_dir=tmp_path / "unfrozen",
        )
    policy = FakePolicy()
    policy._is_pytorch_model = False
    with pytest.raises(Pi05FeatureExtractionError, match="PI0Pytorch"):
        extract_pi05_features(
            model=FakeModel(),
            policy=policy,
            checkpoint="checkpoint",
            observation_paths=[path],
            output_dir=tmp_path / "jax",
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_batch_size(tmp_path, batch_size) -> None:
    with pytest.raises(Pi05FeatureExtractionError, match="batch_size"):
        extract_pi05_features(
            model=FakeModel(),
            policy=FakePolicy(),
            checkpoint="checkpoint",
            observation_paths=[tmp_path / "unused.npz"],
            output_dir=tmp_path / "output",
            batch_size=batch_size,
        )
