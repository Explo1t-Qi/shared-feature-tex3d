import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import shared_feature.pi05_features as features
from shared_feature import (
    Pi05FeatureExtractionError,
    PilotObservation,
    extract_pi05_features,
)


_UNSET = object()


def make_observation(
    path: Path,
    sample_id: str,
    value: int,
    *,
    state: np.ndarray | None = None,
    base_image: np.ndarray | None = None,
    wrist_image: np.ndarray | None = None,
) -> Path:
    if base_image is None:
        base_image = np.zeros((3, 4, 3), dtype=np.uint8)
        base_image[-1, -1] = (value, value + 1, value + 2)
    if wrist_image is None:
        wrist_image = np.zeros((2, 5, 3), dtype=np.uint8)
        wrist_image[-1, -1] = (value + 10, value + 11, value + 12)
    if state is None:
        state = np.arange(8, dtype=np.float64) + value
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


def make_norm_stats() -> dict[str, SimpleNamespace]:
    return {
        "state": SimpleNamespace(
            mean=np.zeros(8),
            std=np.ones(8),
            q01=np.zeros(8),
            q99=np.ones(8) * 10,
        )
    }


class FakeImageTools:
    def __init__(self):
        self.resize_inputs: list[np.ndarray] = []
        self.convert_inputs: list[np.ndarray] = []

    def resize_with_pad(self, image, height, width):
        self.resize_inputs.append(image.copy())
        assert image.flags.c_contiguous
        assert (height, width) == (224, 224)
        return np.broadcast_to(image[0, 0], (height, width, 3)).copy()

    def convert_to_uint8(self, image):
        self.convert_inputs.append(image.copy())
        return image.astype(np.uint8, copy=True)


class FakeTree:
    @classmethod
    def map(cls, function, *trees):
        first = trees[0]
        if isinstance(first, dict):
            return {
                key: cls.map(function, *(tree[key] for tree in trees))
                for key in first
            }
        return function(*trees)


class FakeJax:
    tree = FakeTree

    @staticmethod
    def device_get(value):
        return np.asarray(value)


class FakeJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value)


class FakeModelType:
    PI05 = object()


class FakeLiberoInputs:
    def __init__(self, events):
        self.events = events
        self.inputs: list[dict] = []

    def __call__(self, data):
        self.events.append("libero")
        self.inputs.append(data)
        base = data["observation/image"]
        wrist = data["observation/wrist_image"]
        return {
            "state": data["observation/state"],
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
            "prompt": data["prompt"],
        }


class FakeSentinelTransform:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def __call__(self, data):
        self.events.append(self.name)
        return data


class FakeTokenizeTransform(FakeSentinelTransform):
    def __call__(self, data):
        super().__call__(data)
        data.pop("prompt")
        data["tokenized_prompt"] = np.arange(200, dtype=np.int32)
        data["tokenized_prompt_mask"] = np.ones(200, dtype=bool)
        return data


class FakePadTransform(FakeSentinelTransform):
    def __call__(self, data):
        super().__call__(data)
        data["state"] = np.pad(data["state"], (0, 24))
        return data


class FakeInjectDefaultPrompt:
    def __init__(self, events, prompt):
        self.events = events
        self.prompt = prompt

    def __call__(self, data):
        self.events.append("inject")
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = self.prompt
        return data


class FakeNormalize:
    def __init__(self, owner, norm_stats, *, use_quantiles):
        self.owner = owner
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        owner.normalize_instances.append(self)

    def __call__(self, data):
        self.owner.events.append("normalize")
        return data


class FakeTransforms:
    def __init__(self, events):
        self.events = events
        self.normalize_instances: list[FakeNormalize] = []
        self.composed: list[object] = []

    def InjectDefaultPrompt(self, prompt):
        return FakeInjectDefaultPrompt(self.events, prompt)

    def Normalize(self, norm_stats, *, use_quantiles):
        return FakeNormalize(
            self,
            norm_stats,
            use_quantiles=use_quantiles,
        )

    def compose(self, transforms):
        self.composed = list(transforms)

        def apply(data):
            for transform in transforms:
                data = transform(data)
            return data

        return apply


class FakeDataFactory:
    def __init__(self, data_config):
        self.data_config = data_config
        self.calls: list[tuple[object, object]] = []

    def create(self, assets_dirs, model):
        self.calls.append((assets_dirs, model))
        return self.data_config


class FakeObservationType:
    calls: list[dict] = []

    @classmethod
    def from_dict(cls, data):
        cls.calls.append(data)
        images = {
            name: value.astype(np.float32) / 255.0 * 2.0 - 1.0
            if value.dtype == np.uint8
            else value
            for name, value in data["image"].items()
        }
        return SimpleNamespace(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data["tokenized_prompt"],
            tokenized_prompt_mask=data["tokenized_prompt_mask"],
        )


class FakeImageEncoder:
    def __init__(self, mode="valid"):
        self.mode = mode
        self.inputs: list[np.ndarray] = []

    def __call__(self, image, *, train):
        assert train is False
        self.inputs.append(np.asarray(image).copy())
        if self.mode == "raise":
            raise RuntimeError("encoder failed")
        batch_size = image.shape[0]
        markers = np.asarray(image)[:, 0, 0, 0].reshape(-1, 1, 1)
        p1 = np.broadcast_to(
            markers + 2.0,
            (batch_size, 256, 1152),
        ).copy()
        p2 = np.broadcast_to(
            markers + 3.0,
            (batch_size, 256, 2048),
        ).copy()
        if self.mode == "missing_encoded":
            return p2, {}
        if self.mode == "wrong_p1":
            p1 = p1[:, :, :-1]
        if self.mode == "wrong_p2":
            p2 = p2[:, :, :-1]
        if self.mode == "nonfinite":
            p1[0, 0, 0] = np.nan
        if self.mode == "zero":
            p2.fill(0)
        return p2, {"encoded": p1}


class FakeModel:
    def __init__(self, mode="valid"):
        self.encoder = FakeImageEncoder(mode)
        self.PaliGemma = SimpleNamespace(img=self.encoder)


def make_runtime():
    events: list[str] = []
    image_tools = FakeImageTools()
    transforms = FakeTransforms(events)
    libero_inputs = FakeLiberoInputs(events)
    data_config = SimpleNamespace(
        repo_id="physical-intelligence/libero",
        asset_id="physical-intelligence/libero",
        use_quantile_norm=True,
        norm_stats="must-not-be-used",
        data_transforms=SimpleNamespace(inputs=(libero_inputs,)),
        model_transforms=SimpleNamespace(
            inputs=(
                FakeSentinelTransform(events, "model_prompt"),
                FakeSentinelTransform(events, "resize"),
                FakeTokenizeTransform(events, "tokenize"),
                FakePadTransform(events, "pad"),
            )
        ),
    )
    data_factory = FakeDataFactory(data_config)
    model_config = SimpleNamespace(
        model_type=FakeModelType.PI05,
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        action_dim=32,
        max_token_len=200,
    )
    train_config = SimpleNamespace(
        name="pi05_libero",
        model=model_config,
        data=data_factory,
        assets_dirs=Path("configured-assets"),
    )
    preprocess_calls: list[tuple[object, bool, object]] = []

    def preprocess_observation(rng, observation, *, train):
        preprocess_calls.append((rng, train, observation))
        return observation

    FakeObservationType.calls = []
    runtime = features._OpenPIRuntime(
        jax=FakeJax,
        jnp=FakeJnp,
        image_tools=image_tools,
        transforms=transforms,
        observation_type=FakeObservationType,
        preprocess_observation=preprocess_observation,
        model_type=FakeModelType,
    )
    return SimpleNamespace(
        runtime=runtime,
        events=events,
        image_tools=image_tools,
        transforms=transforms,
        libero_inputs=libero_inputs,
        data_factory=data_factory,
        train_config=train_config,
        preprocess_calls=preprocess_calls,
    )


def install_runtime(monkeypatch):
    context = make_runtime()
    monkeypatch.setattr(
        features,
        "_load_openpi_runtime",
        lambda: context.runtime,
    )
    return context


def load_feature(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return (
            json.loads(str(archive["metadata_json"].item())),
            archive["p1_siglip"].copy(),
            archive["p2_projected"].copy(),
            set(archive.files),
        )


def extract(
    context,
    observation_paths,
    output_dir,
    *,
    model=None,
    norm_stats=_UNSET,
    batch_size=1,
):
    return extract_pi05_features(
        model=model or FakeModel(),
        train_config=context.train_config,
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        norm_stats=make_norm_stats() if norm_stats is _UNSET else norm_stats,
        observation_paths=observation_paths,
        output_dir=output_dir,
        batch_size=batch_size,
    )


def test_extracts_ordered_batched_features_with_official_boundaries(
    monkeypatch,
    tmp_path,
) -> None:
    context = install_runtime(monkeypatch)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = [
        make_observation(inputs / "z.npz", "sample-z", 3),
        make_observation(inputs / "a.npz", "sample.a", 7),
        make_observation(inputs / "m.npz", "sample-m", 11),
    ]
    originals = [PilotObservation.load(path) for path in paths]
    model = FakeModel()
    output_dir = tmp_path / "features"
    supplied_stats = make_norm_stats()

    written = extract(
        context,
        paths,
        output_dir,
        model=model,
        norm_stats=supplied_stats,
        batch_size=2,
    )

    assert written == tuple(
        output_dir / f"{sample_id}.npz"
        for sample_id in ("sample-z", "sample.a", "sample-m")
    )
    assert context.data_factory.calls == [
        (context.train_config.assets_dirs, context.train_config.model)
    ]
    normalize = context.transforms.normalize_instances[0]
    assert normalize.use_quantiles is True
    assert normalize.norm_stats["state"] is supplied_stats["state"]
    assert context.events == [
        event
        for _ in paths
        for event in (
            "inject",
            "libero",
            "normalize",
            "model_prompt",
            "resize",
            "tokenize",
            "pad",
        )
    ]
    assert len({id(value) for value in context.libero_inputs.inputs}) == len(paths)

    assert len(context.image_tools.resize_inputs) == 6
    assert len(context.image_tools.convert_inputs) == 6
    for index, original in enumerate(originals):
        base_input = context.image_tools.resize_inputs[index * 2]
        wrist_input = context.image_tools.resize_inputs[index * 2 + 1]
        np.testing.assert_array_equal(
            base_input,
            original.base_rgb_raw[::-1, ::-1],
        )
        np.testing.assert_array_equal(
            wrist_input,
            original.wrist_rgb_raw[::-1, ::-1],
        )
        assert base_input.flags.c_contiguous
        assert wrist_input.flags.c_contiguous
        reloaded = PilotObservation.load(paths[index])
        np.testing.assert_array_equal(
            reloaded.base_rgb_raw,
            original.base_rgb_raw,
        )
        np.testing.assert_array_equal(
            reloaded.wrist_rgb_raw,
            original.wrist_rgb_raw,
        )

    assert [call["state"].shape[0] for call in FakeObservationType.calls] == [
        2,
        1,
    ]
    first_batched = FakeObservationType.calls[0]
    assert np.all(first_batched["image"]["right_wrist_0_rgb"] == 0)
    np.testing.assert_array_equal(
        first_batched["image_mask"]["base_0_rgb"],
        (True, True),
    )
    np.testing.assert_array_equal(
        first_batched["image_mask"]["left_wrist_0_rgb"],
        (True, True),
    )
    np.testing.assert_array_equal(
        first_batched["image_mask"]["right_wrist_0_rgb"],
        (False, False),
    )
    assert [(rng, train) for rng, train, _ in context.preprocess_calls] == [
        (None, False),
        (None, False),
    ]
    assert [value.shape for value in model.encoder.inputs] == [
        (2, 224, 224, 3),
        (1, 224, 224, 3),
    ]

    for path, original in zip(written, originals, strict=True):
        metadata, p1, p2, keys = load_feature(path)
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(original.base_rgb_raw).tobytes()
        ).hexdigest()
        assert metadata == {
            "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
            "feature_schema_version": "pi05_features_v1",
            "sample_id": original.sample_id,
            "source_image_hash": f"sha256:{expected_hash}",
            "source_model": "pi05",
        }
        assert keys == {"metadata_json", "p1_siglip", "p2_projected"}
        assert p1.shape == (256, 1152)
        assert p2.shape == (256, 2048)
        assert p1.dtype == np.float32
        assert p2.dtype == np.float32
        assert np.all(p1 == p1[0, 0])
        assert np.all(p2 == p2[0, 0])


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
        (
            "bad-image-dtype",
            np.zeros(8),
            np.zeros((3, 4, 3), dtype=np.float32),
            "uint8 RGB",
        ),
    ],
)
def test_rejects_unsafe_or_malformed_observations(
    monkeypatch,
    tmp_path,
    sample_id,
    state,
    base_image,
    match,
) -> None:
    context = install_runtime(monkeypatch)
    path = make_observation(
        tmp_path / "input.npz",
        sample_id,
        1,
        state=state,
        base_image=base_image,
    )

    with pytest.raises(Pi05FeatureExtractionError, match=match):
        extract(context, [path], tmp_path / "output")


@pytest.mark.parametrize(
    "norm_stats",
    [
        None,
        {},
        {"actions": SimpleNamespace(q01=np.zeros(7), q99=np.ones(7))},
        {"state": SimpleNamespace(q01=None, q99=np.ones(8))},
        {
            "state": SimpleNamespace(
                q01=np.zeros(7),
                q99=np.ones(7),
            )
        },
        {
            "state": SimpleNamespace(
                q01=np.zeros(9),
                q99=np.ones(9),
            )
        },
        {
            "state": SimpleNamespace(
                q01=np.ones(8),
                q99=np.zeros(8),
            )
        },
    ],
)
def test_rejects_missing_or_malformed_norm_stats(
    monkeypatch,
    tmp_path,
    norm_stats,
) -> None:
    context = install_runtime(monkeypatch)
    path = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(Pi05FeatureExtractionError, match="norm_stats"):
        extract(
            context,
            [path],
            tmp_path / "output",
            norm_stats=norm_stats,
        )


def test_rejects_identity_and_output_collisions(monkeypatch, tmp_path) -> None:
    context = install_runtime(monkeypatch)
    first = make_observation(tmp_path / "first.npz", "same", 1)
    second = make_observation(tmp_path / "second.npz", "same", 2)

    with pytest.raises(Pi05FeatureExtractionError, match="duplicate observation"):
        extract(context, [first, first], tmp_path / "duplicate-path")
    with pytest.raises(Pi05FeatureExtractionError, match="duplicate sample_id"):
        extract(context, [first, second], tmp_path / "duplicate-id")

    output_dir = tmp_path / "collision"
    output_dir.mkdir()
    existing = output_dir / "same.npz"
    existing.write_bytes(b"keep")
    with pytest.raises(Pi05FeatureExtractionError, match="overwrite"):
        extract(context, [first], output_dir)
    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing_encoded", "encoded"),
        ("wrong_p1", "P1"),
        ("wrong_p2", "P2"),
        ("nonfinite", "non-finite"),
        ("zero", "all-zero"),
    ],
)
def test_rejects_malformed_model_outputs(
    monkeypatch,
    tmp_path,
    mode,
    match,
) -> None:
    context = install_runtime(monkeypatch)
    path = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(Pi05FeatureExtractionError, match=match):
        extract(
            context,
            [path],
            tmp_path / "output",
            model=FakeModel(mode),
        )


def test_external_model_failure_is_chained(monkeypatch, tmp_path) -> None:
    context = install_runtime(monkeypatch)
    path = make_observation(tmp_path / "input.npz", "sample", 1)

    with pytest.raises(
        Pi05FeatureExtractionError,
        match="visual extraction",
    ) as info:
        extract(
            context,
            [path],
            tmp_path / "output",
            model=FakeModel("raise"),
        )

    assert isinstance(info.value.__cause__, RuntimeError)


def test_rejects_missing_model_boundary_and_invalid_input_archive(
    monkeypatch,
    tmp_path,
) -> None:
    context = install_runtime(monkeypatch)
    valid = make_observation(tmp_path / "valid.npz", "sample", 1)
    with pytest.raises(Pi05FeatureExtractionError, match="PaliGemma.img"):
        extract(
            context,
            [valid],
            tmp_path / "missing-model",
            model=SimpleNamespace(),
        )

    invalid = tmp_path / "invalid.npz"
    invalid.write_bytes(b"not an archive")
    with pytest.raises(
        Pi05FeatureExtractionError,
        match="PilotObservation",
    ) as info:
        extract(context, [invalid], tmp_path / "invalid-output")
    assert info.value.__cause__ is not None

    with pytest.raises(Pi05FeatureExtractionError, match="does not exist"):
        extract(
            context,
            [tmp_path / "missing.npz"],
            tmp_path / "missing-output",
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_batch_size(tmp_path, batch_size) -> None:
    with pytest.raises(Pi05FeatureExtractionError, match="batch_size"):
        extract_pi05_features(
            model=FakeModel(),
            train_config=object(),
            checkpoint="checkpoint",
            norm_stats=make_norm_stats(),
            observation_paths=[tmp_path / "unused.npz"],
            output_dir=tmp_path / "output",
            batch_size=batch_size,
        )
