from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np


_ARRAY_FIELDS = ("base_rgb_raw", "wrist_rgb_raw", "state")
_METADATA_FIELDS = (
    "sample_id",
    "task_id",
    "initial_state_id",
    "episode_id",
    "step_id",
    "normalized_episode_progress",
    "prompt",
    "episode_success",
)
_ARCHIVE_FIELDS = {"metadata_json", *_ARRAY_FIELDS}


@dataclass
class PilotObservation:
    """One persistent raw-observation sample for Pilot v0.1."""

    sample_id: str
    task_id: str
    initial_state_id: int
    episode_id: int
    step_id: int
    normalized_episode_progress: float
    base_rgb_raw: np.ndarray
    wrist_rgb_raw: np.ndarray
    state: np.ndarray
    prompt: str
    episode_success: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate structural constraints required for safe sample pairing."""
        self._validate_non_empty_string("sample_id", self.sample_id)
        self._validate_non_empty_string("task_id", self.task_id)

        for name in ("initial_state_id", "episode_id", "step_id"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        progress = self.normalized_episode_progress
        if isinstance(progress, (bool, np.bool_)) or not isinstance(progress, Real):
            raise TypeError("normalized_episode_progress must be a real number")
        if not math.isfinite(float(progress)) or not 0 <= progress <= 1:
            raise ValueError("normalized_episode_progress must be within [0, 1]")

        self._validate_image("base_rgb_raw", self.base_rgb_raw)
        self._validate_image("wrist_rgb_raw", self.wrist_rgb_raw)
        self._validate_array("state", self.state)
        if self.state.ndim == 0 or self.state.size == 0:
            raise ValueError("state must have a non-empty shape")

        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if type(self.episode_success) is not bool:
            raise TypeError("episode_success must be a boolean")

    def save(self, path: str | Path) -> None:
        """Save this record as one compressed NumPy archive."""
        self._validate()
        metadata = {name: getattr(self, name) for name in _METADATA_FIELDS}
        metadata["normalized_episode_progress"] = float(
            self.normalized_episode_progress
        )
        metadata_json = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        with Path(path).open("wb") as output:
            np.savez_compressed(
                output,
                metadata_json=np.asarray(metadata_json),
                base_rgb_raw=self.base_rgb_raw,
                wrist_rgb_raw=self.wrist_rgb_raw,
                state=self.state,
            )

    @classmethod
    def load(cls, path: str | Path) -> PilotObservation:
        """Load and validate one record from a compressed NumPy archive."""
        with np.load(Path(path), allow_pickle=False) as archive:
            archive_fields = set(archive.files)
            if archive_fields != _ARCHIVE_FIELDS:
                missing = sorted(_ARCHIVE_FIELDS - archive_fields)
                unexpected = sorted(archive_fields - _ARCHIVE_FIELDS)
                raise ValueError(
                    "invalid observation archive fields: "
                    f"missing={missing}, unexpected={unexpected}"
                )

            metadata_value = archive["metadata_json"]
            if metadata_value.ndim != 0 or metadata_value.dtype.kind != "U":
                raise ValueError("metadata_json must be a Unicode scalar")

            metadata = json.loads(str(metadata_value.item()))
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must encode an object")
            metadata_fields = set(metadata)
            expected_metadata_fields = set(_METADATA_FIELDS)
            if metadata_fields != expected_metadata_fields:
                missing = sorted(expected_metadata_fields - metadata_fields)
                unexpected = sorted(metadata_fields - expected_metadata_fields)
                raise ValueError(
                    "invalid observation metadata fields: "
                    f"missing={missing}, unexpected={unexpected}"
                )

            arrays = {name: archive[name].copy() for name in _ARRAY_FIELDS}

        return cls(**metadata, **arrays)

    @staticmethod
    def _validate_non_empty_string(name: str, value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")

    @classmethod
    def _validate_image(cls, name: str, value: object) -> None:
        cls._validate_array(name, value)
        if value.ndim != 3 or value.size == 0:
            raise ValueError(f"{name} must be a non-empty rank-3 array")

    @staticmethod
    def _validate_array(name: str, value: object) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if value.dtype.hasobject:
            raise ValueError(f"{name} must not use object dtype")
