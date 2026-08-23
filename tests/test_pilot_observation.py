import numpy as np
import pytest

from shared_feature import PilotObservation


def make_observation(**overrides: object) -> PilotObservation:
    values = {
        "sample_id": "task-0/episode-2/step-17",
        "task_id": "libero_task_0",
        "initial_state_id": 2,
        "episode_id": 2,
        "step_id": 17,
        "normalized_episode_progress": 0.5,
        "base_rgb_raw": np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3),
        "wrist_rgb_raw": np.arange(3 * 4 * 2, dtype=np.uint16).reshape(3, 4, 2),
        "state": np.arange(12, dtype=np.float32).reshape(3, 4),
        "prompt": "把红色方块放入左侧托盘。\nPreserve spacing exactly.",
        "episode_success": True,
    }
    values.update(overrides)
    return PilotObservation(**values)


def test_save_load_round_trip_preserves_record(tmp_path) -> None:
    original = make_observation()
    path = tmp_path / "observation.npz"

    original.save(path)
    loaded = PilotObservation.load(path)

    assert loaded.sample_id == original.sample_id
    assert loaded.task_id == original.task_id
    assert loaded.initial_state_id == original.initial_state_id
    assert loaded.episode_id == original.episode_id
    assert loaded.step_id == original.step_id
    assert loaded.normalized_episode_progress == original.normalized_episode_progress
    assert loaded.prompt == original.prompt
    assert loaded.episode_success is original.episode_success

    for field in ("base_rgb_raw", "wrist_rgb_raw", "state"):
        original_array = getattr(original, field)
        loaded_array = getattr(loaded, field)
        np.testing.assert_array_equal(loaded_array, original_array)
        assert loaded_array.shape == original_array.shape
        assert loaded_array.dtype == original_array.dtype


@pytest.mark.parametrize("progress", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_rejects_invalid_normalized_episode_progress(progress) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        make_observation(normalized_episode_progress=progress)


@pytest.mark.parametrize("field", ["base_rgb_raw", "wrist_rgb_raw"])
@pytest.mark.parametrize(
    "image",
    [np.zeros((4, 5), dtype=np.uint8), np.zeros((1, 4, 5, 3), dtype=np.uint8)],
)
def test_rejects_malformed_image_rank(field, image) -> None:
    with pytest.raises(ValueError, match="rank-3"):
        make_observation(**{field: image})


def test_different_sample_ids_remain_distinguishable(tmp_path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    make_observation(sample_id="sample-a").save(first_path)
    make_observation(sample_id="sample-b").save(second_path)

    first = PilotObservation.load(first_path)
    second = PilotObservation.load(second_path)

    assert first.sample_id == "sample-a"
    assert second.sample_id == "sample-b"
    assert first.sample_id != second.sample_id


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("sample_id", "  ", ValueError),
        ("task_id", "", ValueError),
        ("initial_state_id", -1, ValueError),
        ("episode_id", True, TypeError),
        ("step_id", 1.5, TypeError),
    ],
)
def test_rejects_invalid_identifiers(field, value, error) -> None:
    with pytest.raises(error):
        make_observation(**{field: value})


@pytest.mark.parametrize(
    "state",
    [np.asarray(1.0), np.asarray([], dtype=np.float32)],
)
def test_rejects_state_without_non_empty_shape(state) -> None:
    with pytest.raises(ValueError, match="non-empty shape"):
        make_observation(state=state)


def test_revalidates_mutable_record_before_save(tmp_path) -> None:
    observation = make_observation()
    observation.base_rgb_raw = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="rank-3"):
        observation.save(tmp_path / "invalid.npz")
