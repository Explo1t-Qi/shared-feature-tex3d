from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.c6_openvla_real_smoke as openvla
import scripts.c6_pi05_real_smoke as pi05
import scripts.c6_real_smoke_summary as common
from shared_feature import PilotObservation


COMMIT = "a" * 40


def _record(identity: common.FrozenObservation) -> PilotObservation:
    return PilotObservation(
        sample_id=identity.sample_id,
        task_id=identity.task_id,
        initial_state_id=0,
        episode_id=0,
        step_id=8,
        normalized_episode_progress=0.1,
        base_rgb_raw=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist_rgb_raw=np.zeros((2, 2, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt=f"task {identity.task_id}",
        episode_success=True,
    )


def _loaded() -> tuple[common.LoadedObservation, ...]:
    return tuple(
        common.LoadedObservation(
            identity=identity,
            path=Path(f"/{identity.sample_id}.npz"),
            record=_record(identity),
        )
        for identity in common.FROZEN_OBSERVATIONS
    )


def _openvla_result(feature: torch.Tensor) -> SimpleNamespace:
    marker = float(feature.to(dtype=torch.float32).sum())
    normalized = np.full((1, 7), marker, dtype=np.float64)
    return SimpleNamespace(
        action_token_ids=np.full((1, 7), int(marker) % 100, dtype=np.int64),
        normalized_action=normalized,
        unnormalized_action=normalized + 1.0,
        deployed_action=normalized + 2.0,
    )


def _pi05_result(feature: np.ndarray) -> SimpleNamespace:
    marker = float(np.asarray(feature, dtype=np.float32).sum())
    native = np.full((1, 10, 32), marker, dtype=np.float32)
    return SimpleNamespace(
        normalized_action_chunk_32=native,
        normalized_action_chunk=native[..., :7].copy(),
        unnormalized_action_chunk=native[..., :7] + 1.0,
    )


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr(common, "load_frozen_observations", lambda path: _loaded())
    monkeypatch.setattr(common, "git_commit", lambda path: COMMIT)


def _fake_openvla_runtime(events: list[object]) -> openvla._Runtime:
    torch_runtime = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32=torch.float32,
        as_tensor=torch.as_tensor,
    )
    model = object()
    processor = object()

    def get_model(config):
        events.append(("model", vars(config)))
        return model

    def get_processor(config):
        events.append(("processor", config.center_crop))
        return processor

    def prepare_context(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["processor"] is processor
        assert kwargs["pretrained_checkpoint"] == openvla.CHECKPOINT_IDENTITY
        assert kwargs["unnorm_key"] == openvla.UNNORM_KEY
        assert kwargs["center_crop"] is True
        events.append(("prepare", kwargs["task_description"]))
        return SimpleNamespace(
            o2=torch.ones((1, 256, 4096), dtype=torch.float32),
            checkpoint_identity=openvla.CHECKPOINT_IDENTITY,
            unnorm_key=openvla.UNNORM_KEY,
            center_crop=True,
        )

    def run_reference(*, prepared):
        events.append("reference")
        return _openvla_result(prepared.o2)

    def continue_from_o2(*, prepared, o2):
        events.append(("continue", o2 is prepared.o2))
        return _openvla_result(o2)

    return openvla._Runtime(
        torch=torch_runtime,
        get_model=get_model,
        get_processor=get_processor,
        set_seed_everywhere=lambda seed: events.append(("seed", seed)),
        build_policy_observation=lambda record: {
            "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": record.state.copy(),
        },
        prepare_context=prepare_context,
        run_reference=run_reference,
        continue_from_o2=continue_from_o2,
        validate_model=lambda value: events.append(("validate_model", value is model)),
        versions={"torch": "test"},
    )


class _ImageTools:
    @staticmethod
    def resize_with_pad(image, width, height):
        assert image.flags.c_contiguous
        assert (width, height) == (224, 224)
        return np.resize(image, (height, width, 3))

    @staticmethod
    def convert_to_uint8(image):
        return np.asarray(image, dtype=np.uint8)


def _fake_pi05_runtime(
    events: list[object],
) -> tuple[pi05._Runtime, list[np.ndarray]]:
    fixed_noise: list[np.ndarray] = []

    def prepare_context(*, policy, observation, noise):
        assert policy == "policy"
        assert observation["observation/image"].shape == (224, 224, 3)
        assert observation["observation/wrist_image"].shape == (224, 224, 3)
        fixed_noise.append(np.asarray(noise).copy())
        events.append(("prepare", observation["prompt"]))
        return SimpleNamespace(
            base_p2=np.ones((1, 256, 2048), dtype=np.float32),
            left_p2=np.ones((1, 256, 2048), dtype=np.float32),
            right_p2=np.zeros((1, 256, 2048), dtype=np.float32),
            right_image_mask=False,
            noise=np.asarray(noise).copy(),
            config_name=pi05.CONFIG_NAME,
            checkpoint=pi05.CHECKPOINT_IDENTITY,
            backend=pi05.BACKEND,
        )

    def run_reference(*, prepared):
        events.append("reference")
        return _pi05_result(prepared.base_p2)

    def continue_from_p2(*, prepared, base_p2):
        events.append(("continue", base_p2 is prepared.base_p2))
        return _pi05_result(base_p2)

    runtime = pi05._Runtime(
        jax=SimpleNamespace(device_get=lambda value: value),
        jnp=SimpleNamespace(asarray=np.asarray),
        image_tools=_ImageTools,
        policy="policy",
        checkpoint_path=Path("/checkpoint").resolve(),
        prepare_context=prepare_context,
        run_reference=run_reference,
        continue_from_p2=continue_from_p2,
        versions={"jax": "test"},
    )
    return runtime, fixed_noise


def test_frozen_rng_replay_and_independent_directions() -> None:
    first_noise = common.make_fixed_noise()
    second_noise = common.make_fixed_noise()
    assert first_noise.shape == common.NOISE_SHAPE
    assert first_noise.dtype == np.float32
    np.testing.assert_array_equal(first_noise, second_noise)

    shape = (1, 4, 8)
    first = common.make_direction(shape, 202700)
    replay = common.make_direction(shape, 202700)
    other = common.make_direction(shape, 202701)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert first.dtype == np.float32
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)


def test_native_perturbation_metrics_use_actual_cast_delta() -> None:
    clean = np.ones((1, 4, 8), dtype=np.float32)
    intended_delta, intended_modified = common.intended_perturbation(clean, seed=202700)
    actual = intended_modified.astype(np.float16).astype(np.float32)
    metrics = common.perturbation_metrics(clean, intended_delta, actual)
    expected = actual - clean

    assert metrics["actual_delta_norm"] == pytest.approx(
        np.linalg.norm(expected.astype(np.float64))
    )
    assert metrics["actual_delta_norm"] != pytest.approx(
        metrics["intended_delta_norm"], abs=1e-12
    )


def test_pi05_policy_input_rotates_images_before_resize() -> None:
    base = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = base + 20
    record = SimpleNamespace(
        base_rgb_raw=base,
        wrist_rgb_raw=wrist,
        state=np.arange(8, dtype=np.float32),
        prompt="task",
    )
    captured: list[np.ndarray] = []

    class RecordingImageTools:
        @staticmethod
        def resize_with_pad(image, width, height):
            assert (width, height) == (224, 224)
            captured.append(np.asarray(image).copy())
            return image

        @staticmethod
        def convert_to_uint8(image):
            return np.asarray(image, dtype=np.uint8)

    result = pi05._policy_input(record, RecordingImageTools)

    np.testing.assert_array_equal(captured[0], base[::-1, ::-1])
    np.testing.assert_array_equal(captured[1], wrist[::-1, ::-1])
    np.testing.assert_array_equal(result["observation/state"], record.state)
    assert result["prompt"] == "task"


def test_load_frozen_observations_validates_manifest_and_raw_hash(
    monkeypatch, tmp_path
) -> None:
    identities = []
    source_records = []
    for index in range(2):
        array = np.full((2, 2, 3), index, dtype=np.uint8)
        digest = (
            "sha256:"
            + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        )
        identity = common.FrozenObservation(
            sample_id=f"sample-{index}",
            task_id=str(index),
            source_image_hash=digest,
            direction_seeds={"openvla": index, "pi05": index + 10},
        )
        identities.append(identity)
        record = _record(identity)
        record.base_rgb_raw = array
        path = tmp_path / f"sample-{index}.npz"
        record.save(path)
        source_records.append(
            SimpleNamespace(
                sample_id=identity.sample_id,
                source_image_hash=digest,
                resolved_observation_path=path.resolve(),
            )
        )

    from scripts import _full_feature_extraction_common as feature_common

    monkeypatch.setattr(common, "FROZEN_OBSERVATIONS", tuple(identities))
    monkeypatch.setattr(
        feature_common,
        "load_source_collection",
        lambda path: SimpleNamespace(records=tuple(source_records)),
    )
    loaded = common.load_frozen_observations(tmp_path / "manifest.json")
    assert [value.identity.sample_id for value in loaded] == ["sample-0", "sample-1"]

    source_records[0].source_image_hash = "sha256:" + "0" * 64
    with pytest.raises(common.C6RealSmokeError, match="source hash mismatch"):
        common.load_frozen_observations(tmp_path / "manifest.json")


def test_openvla_and_pi05_runners_then_aggregator(monkeypatch, tmp_path) -> None:
    _patch_common(monkeypatch)
    events: list[object] = []
    open_runtime = _fake_openvla_runtime(events)
    pi_runtime, fixed_noise = _fake_pi05_runtime(events)
    checkpoint = tmp_path / "openvla-checkpoint"
    tex3d_root = tmp_path / "tex3d" / "openvla"
    openpi_root = tmp_path / "openpi"
    pi_checkpoint = tmp_path / "pi-checkpoint"
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        openvla,
        "_validate_runtime_paths",
        lambda args: (checkpoint.resolve(), tex3d_root.resolve()),
    )
    monkeypatch.setattr(openvla, "_load_runtime", lambda root: open_runtime)
    open_summary = openvla._run(
        SimpleNamespace(
            output_dir=output_dir,
            collection_manifest=tmp_path / "collection.json",
            pretrained_checkpoint=checkpoint,
            tex3d_openvla_root=tex3d_root,
        )
    )
    assert open_summary["model_result"] == "PASS"

    monkeypatch.setattr(
        pi05,
        "_validate_paths",
        lambda args: (openpi_root.resolve(), pi_checkpoint.resolve()),
    )
    monkeypatch.setattr(
        pi05,
        "_load_runtime",
        lambda root, checkpoint: pi_runtime,
    )
    pi_summary = pi05._run(
        SimpleNamespace(
            output_dir=output_dir,
            collection_manifest=tmp_path / "collection.json",
            openpi_root=openpi_root,
            checkpoint_dir=pi_checkpoint,
        )
    )
    assert pi_summary["model_result"] == "PASS"
    assert len(fixed_noise) == 2
    np.testing.assert_array_equal(fixed_noise[0], fixed_noise[1])

    aggregate = common._run(SimpleNamespace(output_dir=output_dir))
    assert aggregate["overall_result"] == "PASS"
    assert {path.name for path in output_dir.iterdir()} == {
        "openvla_results.json",
        "pi05_results.json",
        "results.json",
        "summary.md",
    }
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert result["overall_result"] == "PASS"
    assert result["observation_ids"] == [
        identity.sample_id for identity in common.FROZEN_OBSERVATIONS
    ]
    assert events.count("reference") == 4
    assert events.count(("continue", True)) == 4
    assert events.count(("continue", False)) == 4
    model_event = next(value for value in events if value[0] == "model")
    assert model_event[1] == {
        "model_family": "openvla",
        "pretrained_checkpoint": str(checkpoint.resolve()),
        "load_in_8bit": False,
        "load_in_4bit": False,
        "unnorm_key": "libero_spatial_no_noops",
        "center_crop": True,
    }

    altered = json.loads(json.dumps(result["pi05"]))
    altered["protocol"]["noise_seed"] = 999
    with pytest.raises(common.C6RealSmokeError, match="frozen protocol"):
        common._validate_phase_semantics(altered, "pi05")


def test_clean_failure_blocks_intervention(monkeypatch, tmp_path) -> None:
    _patch_common(monkeypatch)
    events: list[object] = []
    runtime = _fake_openvla_runtime(events)
    original_reference = runtime.run_reference

    def mismatched_reference(*, prepared):
        result = original_reference(prepared=prepared)
        result.unnormalized_action[0, 0] += 1.0
        return result

    runtime = openvla._Runtime(
        **{
            **runtime.__dict__,
            "run_reference": mismatched_reference,
        }
    )
    monkeypatch.setattr(
        openvla,
        "_validate_runtime_paths",
        lambda args: (tmp_path / "checkpoint", tmp_path / "tex3d" / "openvla"),
    )
    monkeypatch.setattr(openvla, "_load_runtime", lambda root: runtime)
    summary = openvla._run(
        SimpleNamespace(
            output_dir=tmp_path / "output",
            collection_manifest=tmp_path / "collection.json",
            pretrained_checkpoint=tmp_path / "checkpoint",
            tex3d_openvla_root=tmp_path / "tex3d" / "openvla",
        )
    )
    payload = json.loads(Path(summary["output_path"]).read_text(encoding="utf-8"))
    assert summary["model_result"] == "BLOCKED"
    assert payload["clean_gate"] == "BLOCKED"
    assert all(value["intervention"] is None for value in payload["observations"])
    assert events.count(("continue", False)) == 0


def test_phase_and_aggregator_refuse_overwrite_or_unexpected_files(
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(common.C6RealSmokeError, match="not fresh"):
        common.validate_phase_output_slot(
            output_dir,
            common.OPENVLA_RESULT_FILENAME,
        )
    with pytest.raises(common.C6RealSmokeError, match="exactly the two phase files"):
        common._run(SimpleNamespace(output_dir=output_dir))

    target = tmp_path / "value.json"
    common.write_json_atomic(target, {"value": 1})
    with pytest.raises(FileExistsError, match="overwrite"):
        common.write_json_atomic(target, {"value": 2})
