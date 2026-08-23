import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import shared_feature.openvla_features as features
from shared_feature import (
    OpenVLAFeatureExtractionError,
    PilotObservation,
    extract_openvla_features,
)


def make_observation(path: Path, sample_id: str, value: int) -> Path:
    PilotObservation(
        sample_id=sample_id,
        task_id="2",
        initial_state_id=0,
        episode_id=0,
        step_id=value,
        normalized_episode_progress=0.0,
        base_rgb_raw=np.full((512, 512, 3), value, dtype=np.uint8),
        wrist_rgb_raw=np.zeros((512, 512, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float64),
        prompt=f"Pick object {value}",
        episode_success=True,
    ).save(path)
    return path


class FakeBranch(nn.Module):
    def __init__(self, output_dim: int, first_channel_offset: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.output_dim = output_dim
        self.first_channel_offset = first_channel_offset
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.inputs.append(value.detach().clone())
        assert value.dtype == self.anchor.dtype
        expected_offsets = torch.arange(
            self.first_channel_offset,
            self.first_channel_offset + 3,
            dtype=value.dtype,
        )
        torch.testing.assert_close(
            value[:, :, 0, 0] - value[:, :1, 0, 0],
            (expected_offsets - self.first_channel_offset).expand_as(
                value[:, :, 0, 0]
            ),
        )
        markers = value[:, 0, 0, 0].reshape(-1, 1, 1)
        return markers.expand(-1, 256, self.output_dim)


class FakeProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.inputs.append(value.detach().clone())
        return value[:, :, :1].expand(-1, -1, 4096)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.featurizer = FakeBranch(1024, 1)
        self.vision_backbone.fused_featurizer = FakeBranch(
            1152, 4
        )
        self.projector = FakeProjector()


class FakeProcessor:
    def __init__(self, *, malformed_channels: bool = False):
        self.malformed_channels = malformed_channels
        self.calls: list[tuple[list[str], list[Image.Image], bool, str]] = []

    def __call__(self, prompts, images, *, padding, return_tensors):
        self.calls.append((prompts, images, padding, return_tensors))
        channel_count = 5 if self.malformed_channels else 6
        pixel_values = torch.empty(
            (len(images), channel_count, 224, 224), dtype=torch.float32
        )
        for sample_index, image in enumerate(images):
            image_value = int(np.asarray(image)[0, 0, 0])
            for channel in range(channel_count):
                pixel_values[sample_index, channel].fill_(
                    image_value + channel + 1
                )
        return {"pixel_values": pixel_values}


class FakePreprocessingRuntime:
    def __init__(self):
        self.libero_calls: list[tuple[np.ndarray, int]] = []
        self.crop_calls: list[Image.Image] = []
        self.openvla_v01_system_prompt = "SYSTEM"

    def get_libero_image(self, observation, resize_size):
        self.libero_calls.append(
            (observation["agentview_image"].copy(), resize_size)
        )
        return observation["agentview_image"][::-1, ::-1].copy()

    def center_crop_image(self, image):
        self.crop_calls.append(image.copy())
        return image


def install_runtime(monkeypatch) -> FakePreprocessingRuntime:
    runtime = FakePreprocessingRuntime()
    monkeypatch.setattr(features, "_load_preprocessing_runtime", lambda: runtime)
    return runtime


def load_feature(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return (
            json.loads(str(archive["metadata_json"].item())),
            {name: archive[name].copy() for name in features._EXPECTED_FEATURE_SHAPES},
            set(archive.files),
        )


def test_extracts_batched_features_with_exact_identity_and_provenance(
    monkeypatch, tmp_path
) -> None:
    runtime = install_runtime(monkeypatch)
    observations = tmp_path / "observations"
    observations.mkdir()
    paths = [
        make_observation(observations / "z.npz", "sample-z", 3),
        make_observation(observations / "a.npz", "sample-a", 7),
        make_observation(observations / "m.npz", "sample-m", 11),
    ]
    output_dir = tmp_path / "features"
    model = FakeModel()
    processor = FakeProcessor()

    written = extract_openvla_features(
        model=model,
        processor=processor,
        pretrained_checkpoint="checkpoint/openvla-v01-test",
        observation_paths=paths,
        output_dir=output_dir,
        center_crop=True,
        batch_size=2,
    )

    assert written == tuple(
        output_dir / f"{sample_id}.npz"
        for sample_id in ("sample-z", "sample-a", "sample-m")
    )
    assert [len(call[0]) for call in processor.calls] == [2, 1]
    assert model.vision_backbone.featurizer.calls == 2
    assert model.vision_backbone.fused_featurizer.calls == 2
    assert model.projector.calls == 2
    assert all(call[2:] == (True, "pt") for call in processor.calls)
    assert [
        prompt for call in processor.calls for prompt in call[0]
    ] == [
        "SYSTEM USER: What action should the robot take to pick object 3? ASSISTANT:",
        "SYSTEM USER: What action should the robot take to pick object 7? ASSISTANT:",
        "SYSTEM USER: What action should the robot take to pick object 11? ASSISTANT:",
    ]
    assert [image.size for call in processor.calls for image in call[1]] == [
        (224, 224),
        (224, 224),
        (224, 224),
    ]
    assert [resize_size for _, resize_size in runtime.libero_calls] == [512, 512, 512]
    assert len(runtime.crop_calls) == 3

    dino_input = model.vision_backbone.featurizer.inputs[0]
    siglip_input = model.vision_backbone.fused_featurizer.inputs[0]
    assert tuple(dino_input[0, :, 0, 0].tolist()) == (4.0, 5.0, 6.0)
    assert tuple(siglip_input[0, :, 0, 0].tolist()) == (7.0, 8.0, 9.0)
    expected_fused = torch.cat(
        [
            torch.tensor([4.0, 8.0], dtype=torch.float64)
            .reshape(2, 1, 1)
            .expand(-1, 256, 1024),
            torch.tensor([7.0, 11.0], dtype=torch.float64)
            .reshape(2, 1, 1)
            .expand(-1, 256, 1152),
        ],
        dim=-1,
    )
    torch.testing.assert_close(model.projector.inputs[0], expected_fused)

    for path, sample_id, raw_value in zip(
        written,
        ("sample-z", "sample-a", "sample-m"),
        (3, 7, 11),
        strict=True,
    ):
        metadata, arrays, archive_fields = load_feature(path)
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(
                np.full((512, 512, 3), raw_value, dtype=np.uint8)
            ).tobytes()
        ).hexdigest()
        assert metadata == {
            "checkpoint": "checkpoint/openvla-v01-test",
            "feature_schema_version": "openvla_features_v1",
            "sample_id": sample_id,
            "source_image_hash": f"sha256:{expected_hash}",
            "source_model": "openvla",
        }
        assert archive_fields == {
            "metadata_json",
            "o1_siglip",
            "o1_fused",
            "o2_projected",
        }
        for name, expected_shape in features._EXPECTED_FEATURE_SHAPES.items():
            assert arrays[name].shape == expected_shape
            assert arrays[name].dtype == np.float32
        assert arrays["o1_siglip"].ndim == 2
        assert arrays["o1_fused"].ndim == 2
        assert arrays["o2_projected"].ndim == 2
        assert arrays["o1_siglip"][0, 0] == raw_value + 4
        assert arrays["o1_fused"][0, 0] == raw_value + 1
        assert arrays["o1_fused"][0, 1024] == raw_value + 4
        assert arrays["o2_projected"][0, 0] == raw_value + 1


def test_standard_prompt_and_disabled_crop(monkeypatch, tmp_path) -> None:
    runtime = install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    processor = FakeProcessor()

    extract_openvla_features(
        model=FakeModel(),
        processor=processor,
        pretrained_checkpoint="openvla-checkpoint",
        observation_paths=[observation],
        output_dir=tmp_path / "output",
        center_crop=False,
    )

    assert processor.calls[0][0] == [
        "In: What action should the robot take to pick object 1?\nOut:"
    ]
    assert runtime.crop_calls == []


def test_duplicate_sample_id_is_rejected(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    first = make_observation(tmp_path / "first.npz", "same", 1)
    second = make_observation(tmp_path / "second.npz", "same", 2)

    with pytest.raises(OpenVLAFeatureExtractionError, match="duplicate sample_id"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[first, second],
            output_dir=tmp_path / "output",
        )


def test_duplicate_input_path_is_rejected(tmp_path) -> None:
    observation = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(OpenVLAFeatureExtractionError, match="duplicate observation"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation, observation],
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize("paths", [[], [Path("missing.npz")]])
def test_empty_or_missing_input_is_rejected(tmp_path, paths) -> None:
    with pytest.raises(OpenVLAFeatureExtractionError, match="non-empty|does not exist"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=paths,
            output_dir=tmp_path / "output",
        )


def test_invalid_observation_archive_is_chained(tmp_path) -> None:
    path = tmp_path / "invalid.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(OpenVLAFeatureExtractionError, match="PilotObservation") as info:
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[path],
            output_dir=tmp_path / "output",
        )

    assert info.value.__cause__ is not None


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "sample.npz"
    existing.write_bytes(b"keep")

    with pytest.raises(OpenVLAFeatureExtractionError, match="overwrite"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=output_dir,
        )

    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(tmp_path, batch_size) -> None:
    with pytest.raises(OpenVLAFeatureExtractionError, match="batch_size"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[tmp_path / "unused"],
            output_dir=tmp_path / "output",
            batch_size=batch_size,
        )


def test_invalid_raw_base_image_is_rejected(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    path = make_observation(tmp_path / "input.npz", "sample", 1)
    observation = PilotObservation.load(path)
    observation.base_rgb_raw = np.zeros((4, 5, 3), dtype=np.uint8)
    observation.save(path)

    with pytest.raises(OpenVLAFeatureExtractionError, match="512, 512, 3"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[path],
            output_dir=tmp_path / "output",
        )


def test_malformed_processor_channels_are_rejected(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(OpenVLAFeatureExtractionError, match="pixel_values"):
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(malformed_channels=True),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )


def test_incompatible_model_is_rejected(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(OpenVLAFeatureExtractionError, match="vision_backbone"):
        extract_openvla_features(
            model=nn.Module(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )


def test_malformed_feature_shape_is_rejected(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    model = FakeModel()
    model.vision_backbone.fused_featurizer.output_dim = 1151

    with pytest.raises(OpenVLAFeatureExtractionError, match="O1-S"):
        extract_openvla_features(
            model=model,
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )


def test_model_error_is_chained(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    model = FakeModel()
    failure = RuntimeError("branch failed")

    def fail(_value):
        raise failure

    model.vision_backbone.featurizer.forward = fail

    with pytest.raises(OpenVLAFeatureExtractionError, match="model forward") as info:
        extract_openvla_features(
            model=model,
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )

    assert info.value.__cause__ is failure


def test_processor_error_is_chained(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    failure = RuntimeError("processor failed")

    def fail(*_args, **_kwargs):
        raise failure

    with pytest.raises(OpenVLAFeatureExtractionError, match="processor failed") as info:
        extract_openvla_features(
            model=FakeModel(),
            processor=fail,
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )

    assert info.value.__cause__ is failure


def test_serialization_error_is_chained(monkeypatch, tmp_path) -> None:
    install_runtime(monkeypatch)
    observation = make_observation(tmp_path / "input.npz", "sample", 1)
    failure = OSError("write failed")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(features, "_save_feature_record", fail)

    with pytest.raises(OpenVLAFeatureExtractionError, match="serialize") as info:
        extract_openvla_features(
            model=FakeModel(),
            processor=FakeProcessor(),
            pretrained_checkpoint="checkpoint",
            observation_paths=[observation],
            output_dir=tmp_path / "output",
        )

    assert info.value.__cause__ is failure
