from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import c5a_representation_geometry as c5a
from scripts import c5b_explicit_shared_space as c5b


def _records_and_split(tmp_path: Path):
    records = []
    for task_id in range(10):
        for state_id in range(5):
            for step_id in range(4):
                sample_id = (
                    f"libero_spatial__task{task_id:02d}__state{state_id:02d}"
                    f"__step{step_id:04d}"
                )
                records.append(
                    c5a.PairRecord(
                        sample_id=sample_id,
                        source_image_hash="sha256:" + "a" * 64,
                        openvla_feature_path=tmp_path / f"openvla-{sample_id}.npz",
                        pi05_feature_path=tmp_path / f"pi05-{sample_id}.npz",
                        task_id=task_id,
                        initial_state_id=state_id,
                        step_id=step_id,
                    )
                )
    groups = c5a._reconstruct_groups(records)
    return tuple(records), groups, c5a._build_split(groups)


def _write_c5a_output(
    tmp_path: Path,
    paired_manifest: Path,
    records,
    groups,
    split,
) -> Path:
    output_dir = tmp_path / "c5a"
    output_dir.mkdir()
    split_manifest = c5b._expected_c5a_split_manifest(
        paired_manifest,
        records,
        groups,
        split,
    )
    metric_summary = {
        "schema_version": c5a.METRIC_SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "source_paired_manifest": str(paired_manifest),
        "primary_pair": {},
        "control_pair": {},
        "metric_conventions": {},
        "null": {},
        "splits": {
            "train": {
                "pass": True,
                "num_groups": 40,
                "num_observations": 160,
            },
            "heldout": {
                "pass": True,
                "num_groups": 10,
                "num_observations": 40,
            },
        },
        "c5a_gate": "GO",
        "heldout_null_limitation": "validated upstream",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest), encoding="utf-8"
    )
    (output_dir / "metric_summary.json").write_text(
        json.dumps(metric_summary), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "null_metrics.npz", placeholder=np.array([1]))
    (output_dir / "summary.md").write_text("C5-A GO\n", encoding="utf-8")
    return output_dir


def _valid_c5a_fixture(tmp_path: Path):
    paired_manifest = tmp_path / "paired_features_manifest.json"
    paired_manifest.write_text("{}\n", encoding="utf-8")
    records, groups, split = _records_and_split(tmp_path)
    output_dir = _write_c5a_output(
        tmp_path,
        paired_manifest.resolve(),
        records,
        groups,
        split,
    )
    return paired_manifest.resolve(), records, groups, split, output_dir


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _provenance_args(fixture):
    paired_manifest, records, groups, split, output_dir = fixture
    return output_dir, {
        "paired_manifest_path": paired_manifest,
        "records": records,
        "groups": groups,
        "split": split,
    }


def _full_rank_rows(seed: int, rows: int = 100, dimensions: int = 12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, dimensions)).astype(np.float64)


def _configuration_inputs(seed: int = 3):
    rng = np.random.default_rng(seed)
    train_a = rng.normal(size=(80, 12))
    train_b = train_a @ rng.normal(size=(12, 12)) + rng.normal(
        scale=0.05, size=(80, 12)
    )
    heldout_a = rng.normal(size=(20, 12))
    heldout_b = heldout_a @ rng.normal(size=(12, 12)) + rng.normal(
        scale=0.05, size=(20, 12)
    )
    return tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            train_a,
            train_b,
            heldout_a,
            heldout_b,
        )
    )


def _fake_results(primary_top5: float = 0.4, empirical_p: float = 0.01):
    results = {}
    for name in ("o2_p2__99", "o2_p2__95", "o1s_p1__99", "o1s_p1__95"):
        summaries = {}
        for metric in ("top1", "top5mean", "top10mean"):
            summaries[metric] = {
                "true": 0.4,
                "null_mean": 0.0,
                "null_std": 0.1,
                "null_median": 0.0,
                "null_q95": 0.2,
                "true_minus_null_median": 0.4,
                "empirical_p": empirical_p if metric == "top5mean" else 0.02,
            }
        results[name] = {
            "train_top5mean": 0.8,
            "heldout_top1": 0.3,
            "heldout_top5mean": (primary_top5 if name == "o2_p2__99" else 0.3),
            "heldout_top10mean": 0.2,
            "train_to_heldout_top5mean_gap": 0.4,
            "null_summary": summaries,
        }
    return results


def _pca_metadata():
    return {
        name: {
            "full_rank": 12,
            "d_95": 11,
            "d_99": 12,
            "explained_variance_95": 0.96,
            "explained_variance_99": 1.0,
        }
        for name in ("o2", "p2", "o1s", "p1")
    }


def _null_arrays(repeats: int):
    arrays = {
        "train_permutations": np.stack(
            [np.roll(np.arange(40, dtype=np.int64), 1) for _ in range(repeats)]
        ),
        "heldout_permutations": np.stack(
            [np.roll(np.arange(10, dtype=np.int64), 1) for _ in range(repeats)]
        ),
    }
    for name in c5b._expected_null_keys() - set(arrays):
        arrays[name] = np.linspace(0.0, 1.0, repeats, dtype=np.float64)
    return arrays


def test_c4_provenance_errors_are_rejected(monkeypatch):
    def fail(_path):
        raise c5a.C5AValidationError("bad paired manifest")

    monkeypatch.setattr(c5a, "_load_paired_records", fail)
    with pytest.raises(c5b.C5BValidationError, match="bad paired manifest"):
        c5b._load_dataset_structure("broken.json")


def test_c5a_exact_four_file_provenance_is_accepted(tmp_path):
    fixture = _valid_c5a_fixture(tmp_path)
    output_dir, kwargs = _provenance_args(fixture)
    provenance = c5b._load_c5a_provenance(output_dir, **kwargs)
    assert provenance.split_path == (output_dir / "split_manifest.json").resolve()
    assert provenance.metric_path == (output_dir / "metric_summary.json").resolve()


@pytest.mark.parametrize(
    ("target", "key", "bad_value", "message"),
    [
        ("split_manifest.json", "schema_version", "bad", "split schema"),
        ("metric_summary.json", "schema_version", "bad", "metric schema"),
        ("metric_summary.json", "run_status", "FAILED", "run_status"),
        ("metric_summary.json", "c5a_gate", "NO-GO", "c5a_gate"),
    ],
)
def test_c5a_schema_status_and_gate_rejection(
    tmp_path, target, key, bad_value, message
):
    fixture = _valid_c5a_fixture(tmp_path)
    output_dir, kwargs = _provenance_args(fixture)
    path = output_dir / target
    value = _load_json(path)
    value[key] = bad_value
    _dump_json(path, value)
    with pytest.raises(c5b.C5BValidationError, match=message):
        c5b._load_c5a_provenance(output_dir, **kwargs)


@pytest.mark.parametrize("target", ["split_manifest.json", "metric_summary.json"])
def test_both_c5a_json_sources_must_match_current_manifest(tmp_path, target):
    fixture = _valid_c5a_fixture(tmp_path)
    output_dir, kwargs = _provenance_args(fixture)
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    path = output_dir / target
    value = _load_json(path)
    value["source_paired_manifest"] = str(other.resolve())
    _dump_json(path, value)
    with pytest.raises(c5b.C5BValidationError, match="current C4"):
        c5b._load_c5a_provenance(output_dir, **kwargs)


def test_c5a_split_inheritance_must_match_recomputation(tmp_path):
    fixture = _valid_c5a_fixture(tmp_path)
    output_dir, kwargs = _provenance_args(fixture)
    path = output_dir / "split_manifest.json"
    value = _load_json(path)
    value["groups"][0]["assignment"] = "HELD-OUT"
    _dump_json(path, value)
    with pytest.raises(c5b.C5BValidationError, match="recomputed frozen split"):
        c5b._load_c5a_provenance(output_dir, **kwargs)


@pytest.mark.parametrize("mode", ["extra", "missing"])
def test_c5a_directory_requires_exact_four_files(tmp_path, mode):
    fixture = _valid_c5a_fixture(tmp_path)
    output_dir, kwargs = _provenance_args(fixture)
    if mode == "extra":
        (output_dir / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        (output_dir / "summary.md").unlink()
    with pytest.raises(c5b.C5BValidationError, match="exactly"):
        c5b._load_c5a_provenance(output_dir, **kwargs)


def test_feature_archive_validation_delegates_existing_schemas(tmp_path, monkeypatch):
    record = c5a.PairRecord(
        sample_id="sample",
        source_image_hash="sha256:" + "a" * 64,
        openvla_feature_path=tmp_path / "a.npz",
        pi05_feature_path=tmp_path / "b.npz",
        task_id=0,
        initial_state_id=0,
        step_id=0,
    )
    calls = []

    def validate(path, *, record, spec):
        calls.append((path, record.sample_id, spec))

    monkeypatch.setattr(c5a, "_validate_archive", validate)
    c5b._validate_feature_archives([record])
    assert calls == [
        (record.openvla_feature_path, "sample", c5a.c2.SPEC),
        (record.pi05_feature_path, "sample", c5a.c3.SPEC),
    ]


def test_position_aligned_token_flattening_and_float64_conversion(tmp_path):
    records = []
    for index in range(2):
        path = tmp_path / f"feature-{index}.npz"
        feature = (np.arange(6).reshape(3, 2) + index * 10).astype(np.float32)
        np.savez_compressed(path, node=feature)
        records.append(
            c5a.PairRecord(
                sample_id=str(index),
                source_image_hash="sha256:" + "a" * 64,
                openvla_feature_path=path,
                pi05_feature_path=path,
                task_id=0,
                initial_state_id=0,
                step_id=index,
            )
        )
    rows = c5b._load_node_rows(
        records,
        np.array([1, 0], dtype=np.int64),
        archive_attribute="openvla_feature_path",
        archive_key="node",
        expected_shape=(3, 2),
    )
    assert rows.dtype == np.float64
    assert rows.tolist() == [
        [10.0, 11.0],
        [12.0, 13.0],
        [14.0, 15.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ]


def test_pca_uses_one_full_svd_and_cutoffs_are_prefixes(monkeypatch):
    rows = _full_rank_rows(1)
    original_svd = np.linalg.svd
    calls = []

    def counting_svd(*args, **kwargs):
        calls.append(kwargs)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counting_svd)
    model, scores = c5b._fit_pca(rows)
    assert calls == [{"full_matrices": False}]
    assert 10 <= model.d_95 <= model.d_99 <= model.full_rank
    expected_99 = (rows - model.mean) @ model.basis_99
    np.testing.assert_allclose(scores, expected_99, atol=1e-12)
    np.testing.assert_allclose(
        scores[:, : model.d_95], expected_99[:, : model.d_95], atol=1e-12
    )


@pytest.mark.parametrize(
    "rows",
    [
        np.ones((20, 12), dtype=np.float64),
        np.column_stack(
            [
                _full_rank_rows(2, rows=30, dimensions=9),
                np.zeros((30, 3), dtype=np.float64),
            ]
        ),
    ],
)
def test_pca_rejects_zero_variance_or_insufficient_numerical_rank(rows):
    with pytest.raises(c5b.C5BValidationError, match="variance|dimension"):
        c5b._fit_pca(rows)


def test_ordinary_cca_whitening_and_train_top5_semantics():
    a = _full_rank_rows(4)
    q, _ = np.linalg.qr(_full_rank_rows(5, rows=12, dimensions=12))
    b = a @ q
    whitening = c5b._prepare_cca(a, b)
    fit = c5b._fit_cca(a, b, whitening)
    np.testing.assert_allclose(fit.sigma[:10], np.ones(10), atol=1e-10)
    train_top5 = float(np.mean(fit.sigma[:5], dtype=np.float64))
    assert train_top5 == pytest.approx(1.0, abs=1e-10)
    assert fit.w_a.shape == (12, 12)
    assert fit.w_b.shape == (12, 12)


def test_cca_rejects_fewer_than_ten_covariance_directions():
    a = _full_rank_rows(6, rows=50, dimensions=9)
    b = _full_rank_rows(7, rows=50, dimensions=9)
    with pytest.raises(c5b.C5BValidationError, match="at least 10"):
        c5b._prepare_cca(a, b)


def test_heldout_direct_pearson_preserves_order_and_sign():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(40, 10))
    b = a.copy()
    b[:, 0] *= -1
    identity = np.eye(10, dtype=np.float64)
    fit = c5b.CCAFit(
        sigma=np.ones(10, dtype=np.float64),
        w_a=identity,
        w_b=identity,
    )
    metrics = c5b._heldout_metrics(a, b, fit)
    assert metrics["top1"] == pytest.approx(-1.0)
    assert metrics["top5mean"] == pytest.approx(0.6)
    assert metrics["top10mean"] == pytest.approx(0.8)


def test_group_block_permutation_preserves_progress_and_token_positions():
    permutation = np.array([1, 2, 0], dtype=np.int64)
    order = c5b._block_row_permutation(permutation, rows_per_group=4)
    assert order.tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]


def test_frozen_null_constants_and_rejection_sampler():
    assert c5b.NULL_REPEATS == 200
    assert c5b.ROOT_RNG_SEED == 17

    class FakeRng:
        def __init__(self):
            self.values = iter(
                (
                    np.array([0, 1, 2], dtype=np.int64),
                    np.array([1, 2, 0], dtype=np.int64),
                )
            )
            self.calls = 0

        def permutation(self, _size):
            self.calls += 1
            return next(self.values)

    rng = FakeRng()
    values = c5b._draw_derangements(rng, 3, repeats=1)
    assert rng.calls == 2
    assert values.dtype == np.int64
    assert values.tolist() == [[1, 2, 0]]


def test_heldout_pca_uses_frozen_train_mean_and_basis():
    train = _full_rank_rows(11)
    heldout = _full_rank_rows(12, rows=20) + 100.0
    model, _ = c5b._fit_pca(train)
    transformed = c5b._transform_pca(model, heldout)
    np.testing.assert_allclose(
        transformed,
        (heldout - np.mean(train, axis=0)) @ model.basis_99,
        atol=1e-12,
    )
    assert not np.allclose(
        transformed,
        (heldout - np.mean(heldout, axis=0)) @ model.basis_99,
    )


def test_null_refits_cca_but_does_not_refit_pca(monkeypatch):
    monkeypatch.setattr(c5b, "NULL_REPEATS", 2)
    train_permutations = np.stack(
        [np.roll(np.arange(40, dtype=np.int64), shift) for shift in (1, 2)]
    )
    heldout_permutations = np.stack(
        [np.roll(np.arange(10, dtype=np.int64), shift) for shift in (1, 2)]
    )
    original_fit = c5b._fit_cca
    fit_calls = []

    def counting_fit(*args, **kwargs):
        fit_calls.append(kwargs.get("permutation"))
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(c5b, "_fit_cca", counting_fit)
    monkeypatch.setattr(
        c5b,
        "_fit_pca",
        lambda *_args, **_kwargs: pytest.fail("PCA must not be refit in null analysis"),
    )
    result, null = c5b._analyze_configuration(
        *_configuration_inputs(),
        train_permutations,
        heldout_permutations,
        rows_per_group=2,
    )
    assert len(fit_calls) == 3
    assert fit_calls[0] is None
    assert all(value is not None for value in fit_calls[1:])
    assert result["train_top5mean"] == pytest.approx(
        np.mean(
            original_fit(
                _configuration_inputs()[0],
                _configuration_inputs()[1],
                c5b._prepare_cca(
                    _configuration_inputs()[0], _configuration_inputs()[1]
                ),
            ).sigma[:5]
        )
    )
    assert all(value.shape == (2,) for value in null.values())


def test_rng_replays_exactly_and_streams_are_independent(monkeypatch):
    monkeypatch.setattr(c5b, "NULL_REPEATS", 5)
    first = c5b._generate_null_permutations()
    second = c5b._generate_null_permutations()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == (5, 40)
    assert first[1].shape == (5, 10)
    assert np.all(first[0] != np.arange(40))
    assert np.all(first[1] != np.arange(10))
    assert not np.array_equal(first[0][:, :10], first[1])


def test_empirical_p_and_null_summary_are_exact(monkeypatch):
    monkeypatch.setattr(c5b, "NULL_REPEATS", 4)
    null = np.array([-1.0, 0.0, 2.0, 4.0], dtype=np.float64)
    summary = c5b._summarize_metric(2.0, null)
    assert summary == {
        "true": 2.0,
        "null_mean": 1.25,
        "null_std": float(np.std(null, ddof=0)),
        "null_median": 1.0,
        "null_q95": float(np.quantile(null, 0.95, method="linear")),
        "true_minus_null_median": 1.0,
        "empirical_p": 0.6,
    }


def test_null_archive_rejects_nonfinite_metrics(monkeypatch):
    monkeypatch.setattr(c5b, "NULL_REPEATS", 2)
    arrays = _null_arrays(2)
    metric_key = next(
        name
        for name in arrays
        if name not in {"train_permutations", "heldout_permutations"}
    )
    arrays[metric_key][0] = np.nan
    with pytest.raises(c5b.C5BValidationError, match="invalid null metric"):
        c5b._validate_null_arrays(arrays)


def test_same_permutation_bank_is_reused_for_both_pairs(tmp_path, monkeypatch):
    records, groups, split = _records_and_split(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    c5a_dir = tmp_path / "c5a"
    c5a_dir.mkdir()
    output = tmp_path / "output"
    train = np.empty((1, 40), dtype=np.int64)
    heldout = np.empty((1, 10), dtype=np.int64)
    provenance = c5b.C5AProvenance(
        split_path=c5a_dir / "split.json",
        metric_path=c5a_dir / "metric.json",
        split_manifest={},
        metric_summary={},
    )
    seen = []

    monkeypatch.setattr(
        c5b,
        "_load_dataset_structure",
        lambda _path: (manifest.resolve(), records, groups, split),
    )
    monkeypatch.setattr(
        c5b, "_load_c5a_provenance", lambda *_args, **_kwargs: provenance
    )
    monkeypatch.setattr(c5b, "_validate_feature_archives", lambda _records: None)
    monkeypatch.setattr(c5b, "_generate_null_permutations", lambda: (train, heldout))

    def analyze(_records, _split, spec, train_bank, heldout_bank):
        seen.append((spec.name, train_bank, heldout_bank))
        pca = {
            spec.openvla_pca_name: _pca_metadata()[spec.openvla_pca_name],
            spec.pi05_pca_name: _pca_metadata()[spec.pi05_pca_name],
        }
        names = {
            f"{spec.name}__99": _fake_results()[f"{spec.name}__99"],
            f"{spec.name}__95": _fake_results()[f"{spec.name}__95"],
        }
        arrays = {
            f"{spec.name}__{cutoff}__heldout_{metric}": np.zeros(1, dtype=np.float64)
            for cutoff in ("99", "95")
            for metric in ("top1", "top5mean", "top10mean")
        }
        return pca, names, arrays

    monkeypatch.setattr(c5b, "_analyze_pair", analyze)
    monkeypatch.setattr(c5b, "_publish_outputs", lambda *_args, **_kwargs: None)
    c5b._run(
        argparse.Namespace(
            paired_manifest=manifest,
            c5a_output_dir=c5a_dir,
            output_dir=output,
        )
    )
    assert [value[0] for value in seen] == ["o2_p2", "o1s_p1"]
    assert all(value[1] is train and value[2] is heldout for value in seen)


@pytest.mark.parametrize(
    ("top5", "p_value", "expected"),
    [(0.1, 0.05, "PASS"), (0.0, 0.01, "FAIL"), (0.1, 0.051, "FAIL")],
)
def test_gate_uses_only_primary_99_heldout_top5(tmp_path, top5, p_value, expected):
    manifest = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    metric_path = tmp_path / "metric.json"
    for path in (manifest, split_path, metric_path):
        path.write_text("{}", encoding="utf-8")
    provenance = c5b.C5AProvenance(split_path, metric_path, {}, {})
    summary = c5b._build_alignment_summary(
        manifest.resolve(),
        provenance,
        _pca_metadata(),
        _fake_results(primary_top5=top5, empirical_p=p_value),
    )
    assert summary["c5b_result"] == expected
    assert summary["representation_stage_result"] == expected


def test_exact_json_and_npz_schemas_and_staged_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(c5b, "NULL_REPEATS", 3)
    fixture = _valid_c5a_fixture(tmp_path)
    manifest, records, groups, split, c5a_dir = fixture
    provenance = c5b._load_c5a_provenance(
        c5a_dir,
        paired_manifest_path=manifest,
        records=records,
        groups=groups,
        split=split,
    )
    split_manifest = c5b._build_split_manifest(
        manifest, provenance, records, groups, split
    )
    summary = c5b._build_alignment_summary(
        manifest, provenance, _pca_metadata(), _fake_results()
    )
    arrays = _null_arrays(3)
    output = tmp_path / "c5b"
    input_before = manifest.read_bytes()
    c5b._publish_outputs(
        output,
        split_manifest=split_manifest,
        alignment_summary=summary,
        null_arrays=arrays,
        summary_markdown=c5b._build_summary_markdown(summary),
    )
    assert {path.name for path in output.iterdir()} == c5b.C5B_FILES
    assert set(_load_json(output / "split_manifest.json")) == {
        "schema_version",
        "source_paired_manifest",
        "source_c5a_split_manifest",
        "split_rule_id",
        "counts",
        "heldout_state_by_task",
        "canonical_group_order",
        "groups",
    }
    assert set(_load_json(output / "alignment_summary.json")) == {
        "schema_version",
        "run_status",
        "source_paired_manifest",
        "source_c5a_metric_summary",
        "source_c5a_split_manifest",
        "c5a_gate",
        "primary_pair",
        "control_pair",
        "pca",
        "cca",
        "null",
        "results",
        "c5b_result",
        "representation_stage_result",
        "heldout_null_limitation",
        "interpretation_boundary",
    }
    with np.load(output / "null_alignment_metrics.npz", allow_pickle=False) as data:
        assert set(data.files) == c5b._expected_null_keys()
        assert data["train_permutations"].dtype == np.int64
        assert data["heldout_permutations"].dtype == np.int64
        assert all(
            data[name].dtype == np.float64
            for name in data.files
            if name not in {"train_permutations", "heldout_permutations"}
        )
    assert manifest.read_bytes() == input_before


def test_output_admission_and_failed_publication_are_safe(tmp_path, monkeypatch):
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(c5b.C5BValidationError, match="must be empty"):
        c5b._validate_output_admission(nonempty)

    monkeypatch.setattr(c5b, "NULL_REPEATS", 2)
    output = tmp_path / "failed"
    original_write = c5b._write_json
    calls = 0

    def fail_second_write(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publication failure")
        original_write(path, value)

    monkeypatch.setattr(c5b, "_write_json", fail_second_write)
    with pytest.raises(c5b.C5BValidationError, match="failed to publish"):
        c5b._publish_outputs(
            output,
            split_manifest={},
            alignment_summary={},
            null_arrays=_null_arrays(2),
            summary_markdown="summary\n",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.*"))
