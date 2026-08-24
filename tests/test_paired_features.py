from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shared_feature import (
    PairedFeatureValidationError,
    build_paired_feature_manifest,
)


_PI05_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
_OPENVLA_SHAPES = {
    "o1_siglip": (256, 1152),
    "o1_fused": (256, 2176),
    "o2_projected": (256, 4096),
}
_PI05_SHAPES = {
    "p1_siglip": (256, 1152),
    "p2_projected": (256, 2048),
}
_DEFAULT_METADATA = object()


def _source_hash(sample_id: str) -> str:
    return f"sha256:{hashlib.sha256(sample_id.encode()).hexdigest()}"


def _metadata(model: str, sample_id: str) -> dict[str, Any]:
    if model == "openvla":
        checkpoint = "/some/local/openvla/checkpoint"
        feature_schema_version = "openvla_features_v1"
    else:
        checkpoint = _PI05_CHECKPOINT
        feature_schema_version = "pi05_features_v1"
    return {
        "sample_id": sample_id,
        "source_model": model,
        "checkpoint": checkpoint,
        "feature_schema_version": feature_schema_version,
        "source_image_hash": _source_hash(sample_id),
    }


def _write_archive(
    directory: Path,
    filename: str,
    *,
    model: str,
    sample_id: str,
    metadata_updates: dict[str, Any] | None = None,
    metadata_value: Any = _DEFAULT_METADATA,
    array_updates: dict[str, np.ndarray] | None = None,
    remove_key: str | None = None,
    extra_key: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shapes = _OPENVLA_SHAPES if model == "openvla" else _PI05_SHAPES
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    if array_updates:
        arrays.update(array_updates)
    if remove_key is not None:
        arrays.pop(remove_key)
    if extra_key:
        arrays["unexpected"] = np.zeros(1, dtype=np.float32)

    if metadata_value is _DEFAULT_METADATA:
        metadata = _metadata(model, sample_id)
        if metadata_updates:
            metadata.update(metadata_updates)
        metadata_value = np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        )
    arrays["metadata_json"] = metadata_value

    path = directory / filename
    np.savez_compressed(path, **arrays)
    return path


def _write_pair(
    openvla_dir: Path,
    pi05_dir: Path,
    sample_id: str,
    *,
    openvla_filename: str | None = None,
    pi05_filename: str | None = None,
) -> tuple[Path, Path]:
    openvla_path = _write_archive(
        openvla_dir,
        openvla_filename or f"openvla-{sample_id}.npz",
        model="openvla",
        sample_id=sample_id,
    )
    pi05_path = _write_archive(
        pi05_dir,
        pi05_filename or f"pi05-{sample_id}.npz",
        model="pi05",
        sample_id=sample_id,
    )
    return openvla_path, pi05_path


def test_builds_sorted_manifest_from_authoritative_metadata_ids(tmp_path) -> None:
    openvla_dir = tmp_path / "features" / "openvla"
    pi05_dir = tmp_path / "features" / "pi05"
    filenames = {
        "sample-b": ("z_file.npz", "pi_z.npz"),
        "sample-c": ("a_file.npz", "pi_a.npz"),
        "sample-a": ("m_file.npz", "pi_m.npz"),
    }
    expected_paths: dict[str, tuple[Path, Path]] = {}
    for sample_id, (openvla_name, pi05_name) in filenames.items():
        expected_paths[sample_id] = _write_pair(
            openvla_dir,
            pi05_dir,
            sample_id,
            openvla_filename=openvla_name,
            pi05_filename=pi05_name,
        )
    (openvla_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
    (pi05_dir / "ignored.json").write_text("ignored", encoding="utf-8")
    original_bytes = {
        path: path.read_bytes()
        for pair in expected_paths.values()
        for path in pair
    }

    output_path = tmp_path / "results" / "nested" / "manifest.json"
    returned_path = build_paired_feature_manifest(
        openvla_feature_dir=openvla_dir,
        pi05_feature_dir=pi05_dir,
        output_path=output_path,
    )

    assert returned_path == output_path
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"schema_version", "num_samples", "pairs"}
    assert manifest["schema_version"] == "paired_features_v1"
    assert manifest["num_samples"] == 3 == len(manifest["pairs"])
    assert [pair["sample_id"] for pair in manifest["pairs"]] == [
        "sample-a",
        "sample-b",
        "sample-c",
    ]
    for pair in manifest["pairs"]:
        assert set(pair) == {
            "sample_id",
            "source_image_hash",
            "openvla_feature_path",
            "pi05_feature_path",
        }
        sample_id = pair["sample_id"]
        assert pair["source_image_hash"] == _source_hash(sample_id)
        for key, expected_path in zip(
            ("openvla_feature_path", "pi05_feature_path"),
            expected_paths[sample_id],
            strict=True,
        ):
            stored_path = pair[key]
            assert not Path(stored_path).is_absolute()
            assert "\\" not in stored_path
            assert (output_path.resolve().parent / stored_path).resolve() == (
                expected_path.resolve()
            )
    assert output_path.read_bytes().decode("utf-8").endswith("\n")
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


def test_reports_both_sides_of_sample_set_mismatch_before_output_creation(
    tmp_path,
) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    for sample_id in ("sample-a", "sample-b", "sample-c"):
        _write_archive(
            openvla_dir,
            f"{sample_id}.npz",
            model="openvla",
            sample_id=sample_id,
        )
    for sample_id in ("sample-a", "sample-c", "sample-d"):
        _write_archive(
            pi05_dir,
            f"{sample_id}.npz",
            model="pi05",
            sample_id=sample_id,
        )
    output_path = tmp_path / "must-not-exist" / "manifest.json"

    with pytest.raises(PairedFeatureValidationError) as error:
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=output_path,
        )

    message = str(error.value)
    assert "missing_in_openvla=['sample-d']" in message
    assert "missing_in_pi05=['sample-b']" in message
    assert not output_path.parent.exists()


def test_rejects_cross_model_hash_mismatch_with_full_context(tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    hash_a = f"sha256:{'a' * 64}"
    hash_b = f"sha256:{'b' * 64}"
    _write_archive(
        openvla_dir,
        "one.npz",
        model="openvla",
        sample_id="sample-a",
        metadata_updates={"source_image_hash": hash_a},
    )
    _write_archive(
        pi05_dir,
        "two.npz",
        model="pi05",
        sample_id="sample-a",
        metadata_updates={"source_image_hash": hash_b},
    )

    with pytest.raises(PairedFeatureValidationError) as error:
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )

    assert "sample-a" in str(error.value)
    assert hash_a in str(error.value)
    assert hash_b in str(error.value)


@pytest.mark.parametrize("model", ["openvla", "pi05"])
def test_rejects_duplicate_metadata_sample_ids(model, tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    _write_pair(openvla_dir, pi05_dir, "sample-a")
    target_dir = openvla_dir if model == "openvla" else pi05_dir
    _write_archive(
        target_dir,
        "unrelated-duplicate-name.npz",
        model=model,
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError, match="duplicate.*sample-a"):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    ("model", "remove_key", "extra_key"),
    [
        ("openvla", "o1_siglip", False),
        ("openvla", None, True),
        ("pi05", "p2_projected", False),
        ("pi05", None, True),
    ],
)
def test_rejects_non_exact_archive_keys(
    model,
    remove_key,
    extra_key,
    tmp_path,
) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    other_model = "pi05" if model == "openvla" else "openvla"
    other_dir = pi05_dir if model == "openvla" else openvla_dir
    _write_archive(
        other_dir,
        "valid.npz",
        model=other_model,
        sample_id="sample-a",
    )
    target_dir = openvla_dir if model == "openvla" else pi05_dir
    _write_archive(
        target_dir,
        "invalid.npz",
        model=model,
        sample_id="sample-a",
        remove_key=remove_key,
        extra_key=extra_key,
    )

    with pytest.raises(PairedFeatureValidationError, match="archive fields"):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    ("model", "feature_name", "invalid_value", "message"),
    [
        (
            "openvla",
            "o1_fused",
            np.zeros((1, 2176), dtype=np.float32),
            "shape",
        ),
        (
            "pi05",
            "p2_projected",
            np.zeros((256, 2048), dtype=np.float64),
            "dtype",
        ),
        (
            "openvla",
            "o1_siglip",
            np.full((256, 1152), np.nan, dtype=np.float32),
            "non-finite",
        ),
        (
            "pi05",
            "p1_siglip",
            np.full((256, 1152), np.inf, dtype=np.float32),
            "non-finite",
        ),
    ],
)
def test_rejects_invalid_feature_arrays(
    model,
    feature_name,
    invalid_value,
    message,
    tmp_path,
) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    target_dir = openvla_dir if model == "openvla" else pi05_dir
    other_dir = pi05_dir if model == "openvla" else openvla_dir
    other_model = "pi05" if model == "openvla" else "openvla"
    _write_archive(
        target_dir,
        "invalid.npz",
        model=model,
        sample_id="sample-a",
        array_updates={feature_name: invalid_value},
    )
    _write_archive(
        other_dir,
        "valid.npz",
        model=other_model,
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError, match=message):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    ("metadata_value", "message"),
    [
        (np.asarray("not-json"), "malformed"),
        (np.asarray(["{}"]), "Unicode scalar"),
        (np.asarray(b"{}"), "Unicode scalar"),
        (np.asarray("[]"), "encode an object"),
        (np.asarray({}, dtype=object), "load or validate"),
    ],
)
def test_rejects_malformed_metadata_values(metadata_value, message, tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    _write_archive(
        openvla_dir,
        "invalid.npz",
        model="openvla",
        sample_id="sample-a",
        metadata_value=metadata_value,
    )
    _write_archive(
        pi05_dir,
        "valid.npz",
        model="pi05",
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError, match=message):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    ("model", "updates", "message"),
    [
        ("openvla", {"sample_id": "  "}, "sample_id"),
        ("openvla", {"source_model": "pi05"}, "source_model"),
        (
            "pi05",
            {"feature_schema_version": "wrong"},
            "feature_schema_version",
        ),
        ("openvla", {"source_image_hash": "sha256:ABC"}, "source_image_hash"),
        ("openvla", {"checkpoint": ""}, "checkpoint"),
        ("openvla", {"checkpoint": None}, "checkpoint"),
        ("openvla", {"checkpoint": 7}, "checkpoint"),
        ("pi05", {"checkpoint": ""}, "checkpoint"),
        ("pi05", {"checkpoint": None}, "checkpoint"),
        ("pi05", {"checkpoint": 7}, "checkpoint"),
        (
            "pi05",
            {"checkpoint": "gs://openpi-assets/checkpoints/pi05_base"},
            "checkpoint",
        ),
        ("pi05", {"checkpoint": "/local/pi05"}, "checkpoint"),
    ],
)
def test_rejects_invalid_model_metadata(model, updates, message, tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    target_dir = openvla_dir if model == "openvla" else pi05_dir
    other_dir = pi05_dir if model == "openvla" else openvla_dir
    other_model = "pi05" if model == "openvla" else "openvla"
    _write_archive(
        target_dir,
        "invalid.npz",
        model=model,
        sample_id="sample-a",
        metadata_updates=updates,
    )
    _write_archive(
        other_dir,
        "valid.npz",
        model=other_model,
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError, match=message):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize("model", ["openvla", "pi05"])
def test_rejects_missing_and_additional_metadata_fields(model, tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    metadata = _metadata(model, "sample-a")
    metadata.pop("source_image_hash")
    metadata["unexpected"] = "value"
    target_dir = openvla_dir if model == "openvla" else pi05_dir
    other_dir = pi05_dir if model == "openvla" else openvla_dir
    other_model = "pi05" if model == "openvla" else "openvla"
    _write_archive(
        target_dir,
        "invalid.npz",
        model=model,
        sample_id="sample-a",
        metadata_value=np.asarray(json.dumps(metadata)),
    )
    _write_archive(
        other_dir,
        "valid.npz",
        model=other_model,
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError) as error:
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )
    assert "missing=['source_image_hash']" in str(error.value)
    assert "unexpected=['unexpected']" in str(error.value)


@pytest.mark.parametrize("case", ["missing", "file", "empty"])
def test_rejects_invalid_feature_directories(case, tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    _write_archive(
        pi05_dir,
        "valid.npz",
        model="pi05",
        sample_id="sample-a",
    )
    if case == "file":
        openvla_dir.write_text("not a directory", encoding="utf-8")
    elif case == "empty":
        openvla_dir.mkdir()
        (openvla_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

    with pytest.raises(PairedFeatureValidationError):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


def test_rejects_malformed_npz_and_ignores_non_npz_files(tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    openvla_dir.mkdir()
    (openvla_dir / "broken.npz").write_bytes(b"not an npz")
    (openvla_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
    _write_archive(
        pi05_dir,
        "valid.npz",
        model="pi05",
        sample_id="sample-a",
    )

    with pytest.raises(PairedFeatureValidationError, match="load or validate"):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=tmp_path / "manifest.json",
        )


def test_refuses_to_overwrite_existing_manifest(tmp_path) -> None:
    openvla_dir = tmp_path / "openvla"
    pi05_dir = tmp_path / "pi05"
    _write_pair(openvla_dir, pi05_dir, "sample-a")
    output_path = tmp_path / "manifest.json"
    original = b"existing manifest"
    output_path.write_bytes(original)

    with pytest.raises(PairedFeatureValidationError, match="overwrite"):
        build_paired_feature_manifest(
            openvla_feature_dir=openvla_dir,
            pi05_feature_dir=pi05_dir,
            output_path=output_path,
        )

    assert output_path.read_bytes() == original
