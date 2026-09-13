from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


MATERIALIZATION_ID = "phase1_o2_p2_pi05_torch_v1"
MAPPING_SCHEMA_VERSION = "c5bm_mapping_v1"
METADATA_SCHEMA_VERSION = "phase1_mapping_metadata_v1"
AUTHORITATIVE_MAPPING_SHA256 = (
    "572d4772432025f130ecf0403562bab20a20d4bec008c778985b2b9aee28caec"
)
TOKEN_COUNT = 256
O2_DIMENSION = 4096
P2_DIMENSION = 2048
O2_PCA_DIMENSION = 1793
P2_PCA_DIMENSION = 262
CANONICAL_DIMENSION = 262

_RUNTIME_ARRAY_SHAPES = {
    "mean_a": (O2_DIMENSION,),
    "mean_b": (P2_DIMENSION,),
    "basis_a": (O2_DIMENSION, O2_PCA_DIMENSION),
    "basis_b": (P2_DIMENSION, P2_PCA_DIMENSION),
    "w_a": (O2_PCA_DIMENSION, CANONICAL_DIMENSION),
    "w_b": (P2_PCA_DIMENSION, CANONICAL_DIMENSION),
    "sigma": (CANONICAL_DIMENSION,),
}
_SUPPORTED_DTYPES = {torch.float32, torch.float64}


class FrozenSharedCCAMappingError(ValueError):
    """Raised when the frozen Phase 1 mapping cannot be loaded or applied."""


class FrozenSharedCCAMapping(nn.Module):
    """Differentiable runtime form of the frozen Phase 1 O2/P2 PCA+CCA mapping.

    The saved ``w_a`` and ``w_b`` already include the corresponding CCA whitening
    matrices. Runtime mapping is therefore ``(x - mean) @ basis @ w``; the saved
    ``whitening_a`` and ``whitening_b`` must not be multiplied again.
    """

    def __init__(
        self,
        *,
        mean_a: torch.Tensor,
        mean_b: torch.Tensor,
        basis_a: torch.Tensor,
        basis_b: torch.Tensor,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
        sigma: torch.Tensor,
        artifact_path: Path | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        super().__init__()
        tensors = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "basis_a": basis_a,
            "basis_b": basis_b,
            "w_a": w_a,
            "w_b": w_b,
            "sigma": sigma,
        }
        _validate_tensor_mapping(tensors)
        for name, value in tensors.items():
            self.register_buffer(name, value.contiguous(), persistent=True)
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_sha256
        self.materialization_id = MATERIALIZATION_ID
        self.mapping_schema_version = MAPPING_SCHEMA_VERSION

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        expected_mapping_sha256: str = AUTHORITATIVE_MAPPING_SHA256,
    ) -> FrozenSharedCCAMapping:
        """Load and validate the authoritative Phase 1 mapping.

        Float32 is the normal differentiable runtime policy. Instantiate with
        ``dtype=torch.float64`` for high-precision reference comparisons.
        """

        if dtype not in _SUPPORTED_DTYPES:
            raise FrozenSharedCCAMappingError(
                "mapping dtype must be torch.float32 or torch.float64"
            )
        directory = _resolve_artifact_directory(artifact_dir)
        _validate_phase1_artifact(directory)

        mapping_path = directory / "mapping.npz"
        actual_sha256 = _sha256_file(mapping_path)
        expected_sha256 = _normalize_sha256(expected_mapping_sha256)
        if actual_sha256 != expected_sha256:
            raise FrozenSharedCCAMappingError(
                "mapping.npz SHA-256 differs from the expected authoritative artifact"
            )

        try:
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            with np.load(mapping_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in _RUNTIME_ARRAY_SHAPES}
        except Exception as error:
            raise FrozenSharedCCAMappingError(
                "failed to load Phase 1 mapping arrays or metadata"
            ) from error

        _validate_metadata(metadata)
        _validate_numpy_mapping(arrays)
        tensors = {
            name: torch.from_numpy(np.array(value, copy=True)).to(
                device=device, dtype=dtype
            )
            for name, value in arrays.items()
        }
        return cls(
            **tensors,
            artifact_path=directory,
            artifact_sha256=actual_sha256,
        )

    def map_o2(self, o2: torch.Tensor) -> torch.Tensor:
        """Map ``[B,256,4096]`` OpenVLA O2 tokens to canonical coordinates."""

        return self._map(
            o2,
            mean=self.mean_a,
            basis=self.basis_a,
            canonical=self.w_a,
            label="O2",
        )

    def map_p2(self, p2: torch.Tensor) -> torch.Tensor:
        """Map ``[B,256,2048]`` pi0.5 P2 tokens to canonical coordinates."""

        return self._map(
            p2,
            mean=self.mean_b,
            basis=self.basis_b,
            canonical=self.w_b,
            label="P2",
        )

    def forward(self, features: torch.Tensor, *, side: str) -> torch.Tensor:
        """Map one representation side; ``side`` must be ``"o2"`` or ``"p2"``."""

        if side == "o2":
            return self.map_o2(features)
        if side == "p2":
            return self.map_p2(features)
        raise FrozenSharedCCAMappingError("mapping side must be 'o2' or 'p2'")

    @staticmethod
    def _map(
        value: torch.Tensor,
        *,
        mean: torch.Tensor,
        basis: torch.Tensor,
        canonical: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise FrozenSharedCCAMappingError(f"{label} input must be a torch.Tensor")
        expected = (TOKEN_COUNT, mean.numel())
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise FrozenSharedCCAMappingError(
                f"{label} input must have shape [B,{TOKEN_COUNT},{mean.numel()}]"
            )
        if not value.is_floating_point():
            raise FrozenSharedCCAMappingError(f"{label} input must be floating point")
        if value.device != mean.device:
            raise FrozenSharedCCAMappingError(
                f"{label} input and mapping buffers must be on the same device"
            )
        if mean.dtype not in _SUPPORTED_DTYPES:
            raise FrozenSharedCCAMappingError(
                "mapping buffers must remain float32 or float64"
            )
        runtime_value = value.to(dtype=mean.dtype)
        return ((runtime_value - mean) @ basis) @ canonical

    def extra_repr(self) -> str:
        return (
            f"materialization_id={self.materialization_id!r}, "
            f"dtype={self.mean_a.dtype}, canonical_dimension={self.sigma.numel()}"
        )


def _resolve_artifact_directory(path: str | Path) -> Path:
    try:
        directory = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FrozenSharedCCAMappingError(
            f"Phase 1 mapping artifact does not exist: {path}"
        ) from error
    if not directory.is_dir():
        raise FrozenSharedCCAMappingError(
            f"Phase 1 mapping artifact is not a directory: {directory}"
        )
    return directory


def _validate_phase1_artifact(directory: Path) -> None:
    try:
        from scripts import phase1_pi05_torch_mapping as phase1

        phase1._validate_published_artifact(directory)
    except Exception as error:
        raise FrozenSharedCCAMappingError(
            "Phase 1 authoritative artifact validation failed"
        ) from error


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise FrozenSharedCCAMappingError("invalid Phase 1 metadata schema")
    if metadata.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION:
        raise FrozenSharedCCAMappingError("invalid mapping schema")
    if metadata.get("materialization_id") != MATERIALIZATION_ID:
        raise FrozenSharedCCAMappingError("invalid mapping materialization identity")
    identity = metadata.get("mapping_identity")
    if not isinstance(identity, Mapping) or (
        identity.get("canonical_component_count") != CANONICAL_DIMENSION
        or identity.get("openvla_retained_pca_dimensions") != O2_PCA_DIMENSION
        or identity.get("pi05_retained_pca_dimensions") != P2_PCA_DIMENSION
    ):
        raise FrozenSharedCCAMappingError("invalid Phase 1 mapping dimensions")


def _validate_numpy_mapping(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(_RUNTIME_ARRAY_SHAPES):
        raise FrozenSharedCCAMappingError("runtime mapping arrays are incomplete")
    for name, expected_shape in _RUNTIME_ARRAY_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != expected_shape:
            raise FrozenSharedCCAMappingError(
                f"invalid Phase 1 mapping array shape for {name}: {value.shape}"
            )
        if value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise FrozenSharedCCAMappingError(
                f"invalid Phase 1 mapping array values for {name}"
            )


def _validate_tensor_mapping(tensors: Mapping[str, torch.Tensor]) -> None:
    if set(tensors) != set(_RUNTIME_ARRAY_SHAPES):
        raise FrozenSharedCCAMappingError("runtime mapping tensors are incomplete")
    dtypes = set()
    devices = set()
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise FrozenSharedCCAMappingError(f"mapping tensor is invalid: {name}")
        if tuple(value.shape) != _RUNTIME_ARRAY_SHAPES[name]:
            raise FrozenSharedCCAMappingError(
                f"mapping tensor has an invalid shape: {name}={tuple(value.shape)}"
            )
        if value.dtype not in _SUPPORTED_DTYPES or not torch.isfinite(value).all():
            raise FrozenSharedCCAMappingError(f"mapping tensor is invalid: {name}")
        dtypes.add(value.dtype)
        devices.add(value.device)
    if len(dtypes) != 1 or len(devices) != 1:
        raise FrozenSharedCCAMappingError(
            "mapping tensors must share one supported dtype and device"
        )


def _normalize_sha256(value: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise FrozenSharedCCAMappingError("expected mapping SHA-256 is invalid")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
