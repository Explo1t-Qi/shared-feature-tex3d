from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import shared_feature.torch_cca_mapping as target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = (
    PROJECT_ROOT
    / "experiment_inbox"
    / "shared-feature-phase1"
    / "phase1-o2-p2-pi05-torch-v1"
)
ARTIFACT = PHASE1_ROOT / "mapping"
C2_FEATURES = PROJECT_ROOT / "experiment_inbox" / "c5-c2-output" / "features"
C3_FEATURES = PHASE1_ROOT / "c3-pi05-torch" / "features"


@pytest.fixture(scope="module")
def mapping32() -> target.FrozenSharedCCAMapping:
    if not ARTIFACT.is_dir():
        pytest.skip("synchronized Phase 1 artifact absent")
    return target.FrozenSharedCCAMapping.from_artifact(ARTIFACT)


@pytest.fixture(scope="module")
def mapping64() -> target.FrozenSharedCCAMapping:
    if not ARTIFACT.is_dir():
        pytest.skip("synchronized Phase 1 artifact absent")
    return target.FrozenSharedCCAMapping.from_artifact(ARTIFACT, dtype=torch.float64)


def _load_real_feature_batch() -> tuple[np.ndarray, np.ndarray]:
    sample_ids = (
        "libero_spatial__task00__state00__step0008",
        "libero_spatial__task09__state07__step0119",
    )
    o2_values = []
    p2_values = []
    for sample_id in sample_ids:
        with np.load(C2_FEATURES / f"{sample_id}.npz", allow_pickle=False) as archive:
            o2_values.append(np.array(archive["o2_projected"], copy=True))
        with np.load(C3_FEATURES / f"{sample_id}.npz", allow_pickle=False) as archive:
            p2_values.append(np.array(archive["p2_projected"], copy=True))
    return np.stack(o2_values), np.stack(p2_values)


def test_mapping_preserves_batch_tokens_and_autograd(
    mapping32: target.FrozenSharedCCAMapping,
) -> None:
    generator = torch.Generator().manual_seed(41)
    o2 = torch.randn(
        2, 256, target.O2_DIMENSION, generator=generator, requires_grad=True
    )
    p2 = torch.randn(
        2, 256, target.P2_DIMENSION, generator=generator, requires_grad=True
    )

    h_o = mapping32.map_o2(o2)
    h_p = mapping32.map_p2(p2)
    assert h_o.shape == (2, 256, target.CANONICAL_DIMENSION)
    assert h_p.shape == (2, 256, target.CANONICAL_DIMENSION)

    permutation = torch.randperm(256, generator=torch.Generator().manual_seed(7))
    assert torch.equal(mapping32.map_o2(o2[:, permutation]), h_o[:, permutation])
    assert torch.equal(mapping32.map_p2(p2[:, permutation]), h_p[:, permutation])

    loss = h_o.square().mean() + h_p.square().mean()
    loss.backward()
    for gradient in (o2.grad, p2.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    assert tuple(mapping32.parameters()) == ()
    assert set(dict(mapping32.named_buffers())) == {
        "mean_a",
        "mean_b",
        "basis_a",
        "basis_b",
        "w_a",
        "w_b",
        "sigma",
    }
    assert all(not value.requires_grad and value.grad is None for value in mapping32.buffers())


def test_forward_dispatch_and_input_validation(
    mapping32: target.FrozenSharedCCAMapping,
) -> None:
    with pytest.raises(target.FrozenSharedCCAMappingError, match="side"):
        mapping32(torch.zeros(1), side="unknown")
    with pytest.raises(target.FrozenSharedCCAMappingError, match="shape"):
        mapping32.map_o2(torch.zeros(1, 255, target.O2_DIMENSION))
    with pytest.raises(target.FrozenSharedCCAMappingError, match="floating"):
        mapping32.map_p2(
            torch.zeros(1, 256, target.P2_DIMENSION, dtype=torch.int64)
        )


def test_real_artifact_numpy_torch_parity_and_runtime_error(
    mapping32: target.FrozenSharedCCAMapping,
    mapping64: target.FrozenSharedCCAMapping,
    capsys,
) -> None:
    o2, p2 = _load_real_feature_batch()
    with np.load(ARTIFACT / "mapping.npz", allow_pickle=False) as archive:
        reference_o = ((o2.astype(np.float64) - archive["mean_a"]) @ archive["basis_a"]) @ archive["w_a"]
        reference_p = ((p2.astype(np.float64) - archive["mean_b"]) @ archive["basis_b"]) @ archive["w_b"]

    actual_o64 = mapping64.map_o2(torch.from_numpy(o2)).detach().numpy()
    actual_p64 = mapping64.map_p2(torch.from_numpy(p2)).detach().numpy()
    np.testing.assert_allclose(actual_o64, reference_o, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual_p64, reference_p, rtol=2e-13, atol=2e-13)

    actual_o32 = mapping32.map_o2(torch.from_numpy(o2)).detach().numpy()
    actual_p32 = mapping32.map_p2(torch.from_numpy(p2)).detach().numpy()
    combined_reference = np.concatenate((reference_o.ravel(), reference_p.ravel()))
    combined_error = np.concatenate(
        (
            (actual_o32.astype(np.float64) - reference_o).ravel(),
            (actual_p32.astype(np.float64) - reference_p).ravel(),
        )
    )
    metrics = {
        "max_absolute_error": float(np.max(np.abs(combined_error))),
        "mean_absolute_error": float(np.mean(np.abs(combined_error))),
        "relative_l2_error": float(
            np.linalg.norm(combined_error) / np.linalg.norm(combined_reference)
        ),
    }
    print(json.dumps(metrics, sort_keys=True))
    assert metrics["max_absolute_error"] < 7e-6
    assert metrics["mean_absolute_error"] < 4e-7
    assert metrics["relative_l2_error"] < 5e-7
    assert "max_absolute_error" in capsys.readouterr().out


def test_real_artifact_identity_dtype_and_shapes(
    mapping32: target.FrozenSharedCCAMapping,
) -> None:
    assert mapping32.artifact_sha256 == target.AUTHORITATIVE_MAPPING_SHA256
    assert mapping32.materialization_id == target.MATERIALIZATION_ID
    assert mapping32.mean_a.dtype == torch.float32
    assert mapping32.mean_a.shape == (4096,)
    assert mapping32.basis_a.shape == (4096, 1793)
    assert mapping32.w_a.shape == (1793, 262)
    assert mapping32.mean_b.shape == (2048,)
    assert mapping32.basis_b.shape == (2048, 262)
    assert mapping32.w_b.shape == (262, 262)
    assert mapping32.sigma.shape == (262,)

    with pytest.raises(target.FrozenSharedCCAMappingError, match="dtype"):
        target.FrozenSharedCCAMapping.from_artifact(ARTIFACT, dtype=torch.bfloat16)
    with pytest.raises(target.FrozenSharedCCAMappingError, match="SHA-256"):
        target.FrozenSharedCCAMapping.from_artifact(
            ARTIFACT, expected_mapping_sha256="0" * 64
        )


def test_invalid_artifact_and_runtime_arrays_are_rejected(monkeypatch, tmp_path) -> None:
    with pytest.raises(target.FrozenSharedCCAMappingError, match="does not exist"):
        target.FrozenSharedCCAMapping.from_artifact(tmp_path / "missing")

    artifact = tmp_path / "mapping"
    artifact.mkdir()
    np.savez(artifact / "mapping.npz", mean_a=np.zeros(1, dtype=np.float64))
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": target.METADATA_SCHEMA_VERSION,
                "mapping_schema_version": target.MAPPING_SCHEMA_VERSION,
                "materialization_id": target.MATERIALIZATION_ID,
                "mapping_identity": {
                    "canonical_component_count": target.CANONICAL_DIMENSION,
                    "openvla_retained_pca_dimensions": target.O2_PCA_DIMENSION,
                    "pi05_retained_pca_dimensions": target.P2_PCA_DIMENSION,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(target, "_validate_phase1_artifact", lambda path: None)
    digest = target._sha256_file(artifact / "mapping.npz")
    with pytest.raises(target.FrozenSharedCCAMappingError, match="load"):
        target.FrozenSharedCCAMapping.from_artifact(
            artifact, expected_mapping_sha256=digest
        )

    invalid_arrays = {
        name: np.zeros(tuple(1 for _ in shape), dtype=np.float64)
        for name, shape in target._RUNTIME_ARRAY_SHAPES.items()
    }
    np.savez(artifact / "mapping.npz", **invalid_arrays)
    digest = target._sha256_file(artifact / "mapping.npz")
    with pytest.raises(target.FrozenSharedCCAMappingError, match="array shape"):
        target.FrozenSharedCCAMapping.from_artifact(
            artifact, expected_mapping_sha256=digest
        )

    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    metadata["mapping_schema_version"] = "wrong"
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(target.FrozenSharedCCAMappingError, match="mapping schema"):
        target.FrozenSharedCCAMapping.from_artifact(
            artifact, expected_mapping_sha256=digest
        )
