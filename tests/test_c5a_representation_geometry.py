from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts._full_feature_extraction_common as common
import scripts.c5a_representation_geometry as c5a


def _small_specs(monkeypatch) -> tuple[common.FeatureSpec, common.FeatureSpec]:
    openvla_spec = common.FeatureSpec(
        model_family="openvla",
        source_model="openvla",
        checkpoint_identity="openvla/openvla-7b-finetuned-libero-spatial",
        feature_schema_version="openvla_features_v1",
        manifest_filename="openvla_feature_manifest.json",
        nodes=(
            common.FeatureNode("O1-S", "o1_siglip", (256, 3)),
            common.FeatureNode("O1-F", "o1_fused", (256, 4)),
            common.FeatureNode("O2", "o2_projected", (256, 5)),
        ),
    )
    pi05_spec = common.FeatureSpec(
        model_family="pi05",
        source_model="pi05",
        checkpoint_identity="gs://openpi-assets/checkpoints/pi05_libero",
        feature_schema_version="pi05_features_v1",
        manifest_filename="pi05_feature_manifest.json",
        nodes=(
            common.FeatureNode("P1", "p1_siglip", (256, 3)),
            common.FeatureNode("P2", "p2_projected", (256, 4)),
        ),
        feature_config="pi05_libero",
    )
    monkeypatch.setattr(c5a.c2, "SPEC", openvla_spec)
    monkeypatch.setattr(c5a.c3, "SPEC", pi05_spec)
    return openvla_spec, pi05_spec


def _source_hash(sample_id: str) -> str:
    return f"sha256:{hashlib.sha256(sample_id.encode()).hexdigest()}"


def _write_archive(
    path: Path,
    *,
    sample_id: str,
    source_hash: str,
    spec: common.FeatureSpec,
    node_values: dict[str, np.ndarray] | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "sample_id": sample_id,
        "source_model": spec.source_model,
        "checkpoint": spec.checkpoint_identity,
        "feature_schema_version": spec.feature_schema_version,
        "source_image_hash": source_hash,
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    arrays = {
        node.archive_key: np.zeros(node.shape, dtype=np.float32) for node in spec.nodes
    }
    if node_values:
        arrays.update(node_values)
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _pair_record(
    *,
    task_id: int,
    state_id: int,
    step_id: int,
    index: int,
) -> c5a.PairRecord:
    sample_id = (
        f"libero_spatial__task{task_id:02d}__state{state_id:02d}__step{step_id:04d}"
    )
    placeholder = Path(f"/archive/{index}.npz")
    return c5a.PairRecord(
        sample_id=sample_id,
        source_image_hash=_source_hash(sample_id),
        openvla_feature_path=placeholder,
        pi05_feature_path=placeholder,
        task_id=task_id,
        initial_state_id=state_id,
        step_id=step_id,
    )


def _canonical_records() -> tuple[c5a.PairRecord, ...]:
    records = []
    index = 0
    for task_id in range(10):
        for state_id in range(5):
            for step_id in (8, 32, 56, 72):
                records.append(
                    _pair_record(
                        task_id=task_id,
                        state_id=state_id,
                        step_id=step_id,
                        index=index,
                    )
                )
                index += 1
    return tuple(records)


def _relative_posix(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path, start=parent)).as_posix()


def _prepare_formal_inputs(
    monkeypatch,
    tmp_path: Path,
) -> tuple[Path, tuple[Path, ...]]:
    openvla_spec, pi05_spec = _small_specs(monkeypatch)
    manifest_dir = tmp_path / "c4"
    manifest_dir.mkdir()
    openvla_dir = tmp_path / "c2" / "features"
    pi05_dir = tmp_path / "c3" / "features"
    rng = np.random.default_rng(2026)
    pairs = []
    archives: list[Path] = []

    for index, record in enumerate(_canonical_records()):
        latent = rng.normal(size=3).astype(np.float32)
        o1 = np.tile(latent, (256, 1)).astype(np.float32)
        p1 = np.tile(
            latent
            @ np.asarray(
                [[1.0, 0.2, -0.1], [0.1, 0.8, 0.3], [-0.2, 0.1, 1.1]],
                dtype=np.float32,
            ),
            (256, 1),
        ).astype(np.float32)
        o2_vector = np.concatenate(
            (latent, [latent[0] + latent[1], latent[2] - latent[0]])
        ).astype(np.float32)
        p2_vector = np.asarray(
            [
                latent[0] + 0.2 * latent[1],
                latent[1] - 0.1 * latent[2],
                latent[2] + 0.3 * latent[0],
                latent.sum(),
            ],
            dtype=np.float32,
        )
        openvla_path = openvla_dir / f"{record.sample_id}.npz"
        pi05_path = pi05_dir / f"{record.sample_id}.npz"
        _write_archive(
            openvla_path,
            sample_id=record.sample_id,
            source_hash=record.source_image_hash,
            spec=openvla_spec,
            node_values={
                "o1_siglip": o1,
                "o1_fused": np.tile(
                    np.concatenate((latent, [latent.sum()])),
                    (256, 1),
                ).astype(np.float32),
                "o2_projected": np.tile(o2_vector, (256, 1)),
            },
        )
        _write_archive(
            pi05_path,
            sample_id=record.sample_id,
            source_hash=record.source_image_hash,
            spec=pi05_spec,
            node_values={
                "p1_siglip": p1,
                "p2_projected": np.tile(p2_vector, (256, 1)),
            },
        )
        archives.extend((openvla_path, pi05_path))
        pairs.append(
            {
                "sample_id": record.sample_id,
                "source_image_hash": record.source_image_hash,
                "openvla_feature_path": _relative_posix(openvla_path, manifest_dir),
                "pi05_feature_path": _relative_posix(pi05_path, manifest_dir),
            }
        )

    manifest_path = manifest_dir / "paired_features_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": c5a.PAIRED_SCHEMA_VERSION,
                "num_samples": 200,
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, tuple(archives)


def _prepare_single_pair(
    monkeypatch,
    tmp_path: Path,
) -> tuple[c5a.PairRecord, common.FeatureSpec, common.FeatureSpec]:
    openvla_spec, pi05_spec = _small_specs(monkeypatch)
    sample_id = "libero_spatial__task00__state00__step0008"
    source_hash = _source_hash(sample_id)
    openvla_path = tmp_path / "openvla.npz"
    pi05_path = tmp_path / "pi05.npz"
    _write_archive(
        openvla_path,
        sample_id=sample_id,
        source_hash=source_hash,
        spec=openvla_spec,
    )
    _write_archive(
        pi05_path,
        sample_id=sample_id,
        source_hash=source_hash,
        spec=pi05_spec,
    )
    return (
        c5a.PairRecord(
            sample_id=sample_id,
            source_image_hash=source_hash,
            openvla_feature_path=openvla_path,
            pi05_feature_path=pi05_path,
            task_id=0,
            initial_state_id=0,
            step_id=8,
        ),
        openvla_spec,
        pi05_spec,
    )


def test_end_to_end_writes_frozen_outputs_without_modifying_inputs(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path, archive_paths = _prepare_formal_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(c5a, "NULL_REPEATS", 2)
    original_bytes = {
        path: path.read_bytes() for path in (manifest_path, *archive_paths)
    }
    output_dir = tmp_path / "c5a"

    summary = c5a._run(
        SimpleNamespace(paired_manifest=manifest_path, output_dir=output_dir)
    )

    assert summary["status"] == "C5-A Representation Geometry — COMPLETED"
    assert summary["num_pairs"] == 200
    assert summary["train_groups"] == 40
    assert summary["heldout_groups"] == 10
    assert {path.name for path in output_dir.iterdir()} == {
        "split_manifest.json",
        "metric_summary.json",
        "null_metrics.npz",
        "summary.md",
    }
    split_manifest = json.loads(
        (output_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert split_manifest["schema_version"] == c5a.SPLIT_SCHEMA_VERSION
    assert split_manifest["counts"] == {
        "groups": 50,
        "observations": 200,
        "train_groups": 40,
        "heldout_groups": 10,
        "train_observations": 160,
        "heldout_observations": 40,
    }
    assert all(
        group["inherited_progress_slots"] == [0.1, 0.4, 0.7, 0.9]
        for group in split_manifest["groups"]
    )
    metric_summary = json.loads(
        (output_dir / "metric_summary.json").read_text(encoding="utf-8")
    )
    assert metric_summary["schema_version"] == c5a.METRIC_SCHEMA_VERSION
    assert metric_summary["run_status"] == "COMPLETED"
    for split_name in c5a.SPLIT_NAMES:
        for pair_name in (c5a.PRIMARY_PAIR, c5a.CONTROL_PAIR):
            assert set(metric_summary["splits"][split_name]["pairs"][pair_name]) == set(
                c5a.METRIC_NAMES
            )
    with np.load(output_dir / "null_metrics.npz", allow_pickle=False) as archive:
        assert archive["train_permutations"].dtype == np.int64
        assert archive["heldout_permutations"].dtype == np.int64
        assert archive["train_permutations"].shape == (2, 40)
        assert archive["heldout_permutations"].shape == (2, 10)
        metric_keys = set(archive.files) - {
            "train_permutations",
            "heldout_permutations",
        }
        assert len(metric_keys) == 12
        assert all(archive[key].dtype == np.float64 for key in metric_keys)
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


@pytest.mark.parametrize(
    ("model", "updates", "message"),
    [
        ("openvla", {"checkpoint": "/local/checkpoint"}, "metadata differs"),
        ("pi05", {"checkpoint": "/local/checkpoint"}, "metadata differs"),
        ("openvla", {"source_model": "pi05"}, "metadata differs"),
        ("pi05", {"feature_schema_version": "wrong"}, "metadata differs"),
        ("openvla", {"sample_id": "wrong"}, "metadata differs"),
        ("pi05", {"source_image_hash": f"sha256:{'f' * 64}"}, "metadata differs"),
    ],
)
def test_rejects_wrong_formal_archive_provenance(
    monkeypatch,
    tmp_path,
    model,
    updates,
    message,
) -> None:
    record, openvla_spec, pi05_spec = _prepare_single_pair(monkeypatch, tmp_path)
    path = (
        record.openvla_feature_path if model == "openvla" else record.pi05_feature_path
    )
    spec = openvla_spec if model == "openvla" else pi05_spec
    _write_archive(
        path,
        sample_id=record.sample_id,
        source_hash=record.source_image_hash,
        spec=spec,
        metadata_updates=updates,
    )

    with pytest.raises(c5a.C5AValidationError, match=message):
        c5a._load_and_pool_record(record)


@pytest.mark.parametrize("case", ["shape", "dtype", "nonfinite", "missing"])
def test_rejects_invalid_required_feature_array(
    monkeypatch,
    tmp_path,
    case,
) -> None:
    record, openvla_spec, _ = _prepare_single_pair(monkeypatch, tmp_path)
    if case == "shape":
        value = np.zeros((255, 5), dtype=np.float32)
    elif case == "dtype":
        value = np.zeros((256, 5), dtype=np.float64)
    elif case == "nonfinite":
        value = np.full((256, 5), np.nan, dtype=np.float32)
    else:
        value = None
    metadata = {
        "sample_id": record.sample_id,
        "source_model": openvla_spec.source_model,
        "checkpoint": openvla_spec.checkpoint_identity,
        "feature_schema_version": openvla_spec.feature_schema_version,
        "source_image_hash": record.source_image_hash,
    }
    arrays = {
        node.archive_key: np.zeros(node.shape, dtype=np.float32)
        for node in openvla_spec.nodes
        if not (case == "missing" and node.archive_key == "o2_projected")
    }
    if value is not None:
        arrays["o2_projected"] = value
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(record.openvla_feature_path, **arrays)

    with pytest.raises(c5a.C5AValidationError):
        c5a._load_and_pool_record(record)


def test_reconstructs_groups_and_exact_sha256_split() -> None:
    records = _canonical_records()

    groups = c5a._reconstruct_groups(records)
    split = c5a._build_split(groups)

    assert len(groups) == 50
    assert all(len(group.record_indices) == 4 for group in groups)
    assert [group.canonical_index for group in groups] == list(range(50))
    assert all(
        list(group.record_indices)
        == list(range(group.record_indices[0], group.record_indices[0] + 4))
        for group in groups
    )
    canonical = "pilot-v0.2-c5-split-v1|task_id=2|initial_state_id=1"
    expected_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert c5a._split_digest(2, 1) == expected_digest
    assert len(split.train_group_indices) == 40
    assert len(split.heldout_group_indices) == 10
    assert split.train_observation_indices.shape == (160,)
    assert split.heldout_observation_indices.shape == (40,)
    assert split.heldout_state_by_task == {
        0: 3,
        1: 2,
        2: 1,
        3: 1,
        4: 0,
        5: 0,
        6: 3,
        7: 1,
        8: 4,
        9: 0,
    }
    for task_id in range(10):
        task_groups = [group for group in groups if group.task_id == task_id]
        expected = min(
            task_groups,
            key=lambda group: (
                c5a._split_digest(group.task_id, group.initial_state_id),
                group.initial_state_id,
            ),
        )
        assert split.heldout_state_by_task[task_id] == expected.initial_state_id


@pytest.mark.parametrize("case", ["noncontiguous", "nonincreasing"])
def test_rejects_invalid_group_order(case) -> None:
    records = list(_canonical_records())
    if case == "noncontiguous":
        records[1], records[4] = records[4], records[1]
        message = "not contiguous|exactly 4"
    else:
        source = records[1]
        records[1] = c5a.PairRecord(
            sample_id=source.sample_id,
            source_image_hash=source.source_image_hash,
            openvla_feature_path=source.openvla_feature_path,
            pi05_feature_path=source.pi05_feature_path,
            task_id=source.task_id,
            initial_state_id=source.initial_state_id,
            step_id=records[0].step_id,
        )
        message = "strictly increasing"

    with pytest.raises(c5a.C5AValidationError, match=message):
        c5a._reconstruct_groups(records)


def test_mean_pooling_uses_all_tokens_and_float64() -> None:
    value = np.arange(256 * 3, dtype=np.float32).reshape(256, 3)

    pooled = c5a._mean_pool(value)

    assert pooled.shape == (3,)
    assert pooled.dtype == np.float64
    np.testing.assert_allclose(pooled, value.astype(np.float64).mean(axis=0))


def test_unbiased_hsic_matches_frozen_formula() -> None:
    gram_x = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0],
        ]
    )
    gram_y = np.asarray(
        [
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 6.0, 5.0, 2.0],
            [4.0, 5.0, 9.0, 8.0],
            [3.0, 2.0, 8.0, 7.0],
        ]
    )
    k = gram_x.copy()
    ell = gram_y.copy()
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(ell, 0.0)
    n = 4
    expected = (
        np.trace(k @ ell)
        + (k.sum() * ell.sum()) / ((n - 1) * (n - 2))
        - (2.0 / (n - 2)) * np.ones(n) @ k @ ell @ np.ones(n)
    ) / (n * (n - 3))

    assert c5a._unbiased_hsic(gram_x, gram_y) == pytest.approx(expected)


def test_debiased_and_biased_cka_identical_data_equal_one() -> None:
    rng = np.random.default_rng(11)
    value = rng.normal(size=(12, 7))
    gram = value @ value.T

    assert c5a._debiased_linear_cka_from_grams(gram, gram) == pytest.approx(1.0)
    assert c5a._biased_linear_cka_from_grams(gram, gram) == pytest.approx(1.0)


def test_debiased_cka_preserves_negative_finite_value() -> None:
    x = np.asarray(
        [[-1.2, 0.4], [0.7, 1.1], [1.5, -0.3], [-0.8, -1.4], [0.2, 0.9], [1.0, 0.2]]
    )
    y = np.asarray(
        [[0.1, 1.2], [-1.1, 0.3], [0.5, 0.8], [1.3, -0.6], [-0.4, -1.0], [0.9, -0.2]]
    )

    value = c5a._debiased_linear_cka_from_grams(x @ x.T, y @ y.T)

    assert np.isfinite(value)
    assert value < 0.0


def test_debiased_cka_rejects_nonpositive_self_hsic() -> None:
    gram = np.ones((6, 6), dtype=np.float64)

    with pytest.raises(c5a.C5AValidationError, match="self-HSIC"):
        c5a._debiased_linear_cka_from_grams(gram, gram)


def test_squared_euclidean_spearman_rsa_and_invalid_inputs() -> None:
    x = np.asarray([[0.0], [1.0], [3.0], [7.0]])
    y = 2.0 * x + 5.0

    value = c5a._spearman_rsa_from_distances(
        c5a._distance_matrix(x),
        c5a._distance_matrix(y),
    )

    assert value == pytest.approx(1.0)
    with pytest.raises(c5a.C5AValidationError, match="constant"):
        c5a._spearman_rsa_from_distances(
            np.zeros((4, 4)),
            c5a._distance_matrix(y),
        )
    invalid = c5a._distance_matrix(y)
    invalid[0, 1] = np.nan
    with pytest.raises(c5a.C5AValidationError, match="non-finite"):
        c5a._spearman_rsa_from_distances(c5a._distance_matrix(x), invalid)


def test_frozen_rng_is_deterministic_independent_and_deranged() -> None:
    train_a, heldout_a = c5a._generate_null_permutations(40, 10)
    train_b, heldout_b = c5a._generate_null_permutations(40, 10)

    assert train_a.shape == (50, 40)
    assert heldout_a.shape == (50, 10)
    assert train_a.dtype == heldout_a.dtype == np.int64
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(heldout_a, heldout_b)
    assert np.all(train_a != np.arange(40))
    assert np.all(heldout_a != np.arange(10))
    assert not np.array_equal(train_a[0, :10], heldout_a[0])


def test_derangement_sampler_accepts_multiple_cycles() -> None:
    class FakeRng:
        @staticmethod
        def permutation(n_groups):
            assert n_groups == 4
            return np.asarray([1, 0, 3, 2])

    permutations = c5a._draw_derangements(FakeRng(), 4, 1)

    np.testing.assert_array_equal(permutations, [[1, 0, 3, 2]])


def test_block_permutation_moves_pi05_groups_and_preserves_slots() -> None:
    permutation = np.asarray([2, 0, 1], dtype=np.int64)

    rows = c5a._block_row_permutation(permutation)

    np.testing.assert_array_equal(rows, [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])


def test_empirical_p_value_summary_and_gate_boundary() -> None:
    null = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)

    summary = c5a._summarize_metric(0.7, null)

    assert summary["empirical_p_value"] == pytest.approx(2 / 5)
    assert summary["null_std"] == pytest.approx(np.std(null, ddof=0))
    assert summary["null_q95"] == pytest.approx(
        np.quantile(null, 0.95, method="linear")
    )
    fake_dataset = SimpleNamespace(manifest_path=Path("/paired.json"))
    passing = {
        "pairs": {
            c5a.PRIMARY_PAIR: {
                "debiased_linear_cka": {"true": 1.0, "empirical_p_value": 1 / 51}
            }
        },
        "pass": True,
    }
    summary_value = c5a._build_metric_summary(
        fake_dataset,
        {"train": passing, "heldout": passing},
    )
    assert summary_value["c5a_gate"] == "GO"
    failing = {**passing, "pass": False}
    summary_value = c5a._build_metric_summary(
        fake_dataset,
        {"train": passing, "heldout": failing},
    )
    assert summary_value["c5a_gate"] == "NO-GO"


def test_rejects_nonempty_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing").write_text("keep", encoding="utf-8")

    with pytest.raises(c5a.C5AValidationError, match="must be empty"):
        c5a._validate_output_admission(output_dir)


def test_failed_publish_leaves_no_final_outputs(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        c5a.np,
        "savez_compressed",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("forced failure")),
    )

    with pytest.raises(c5a.C5AValidationError, match="failed to publish"):
        c5a._publish_outputs(
            output_dir,
            split_manifest={"schema_version": "split"},
            metric_summary={"schema_version": "metric"},
            null_arrays={"values": np.zeros(1, dtype=np.float64)},
            summary_markdown="# summary\n",
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output.*"))
