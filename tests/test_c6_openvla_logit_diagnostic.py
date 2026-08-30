from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import c6_openvla_logit_diagnostic as diagnostic


ACTION_IDS = np.asarray([[11, 12, 13, 14, 15, 16, 17]], dtype=np.int64)


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def __call__(
        self,
        *,
        inputs_embeds,
        attention_mask,
        use_cache,
        return_dict,
    ):
        assert use_cache is False
        assert return_dict is True
        self.calls.append((inputs_embeds.detach().clone(), attention_mask))
        logits = torch.zeros(
            (1, inputs_embeds.shape[1], 32),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )
        visual_marker = inputs_embeds[:, 1:257, :].sum(dtype=torch.float32)
        for position, token_id in enumerate(ACTION_IDS[0]):
            row = logits.shape[1] - 7 + position
            logits[0, row, token_id] = 10.0
            logits[0, row, (token_id + 1) % 32] = 9.0
            logits[0, row, 0] += visual_marker
        return SimpleNamespace(logits=logits)


class _FakeModel:
    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(32, 4096)
        self.language_model = _FakeLanguageModel()

    def get_input_embeddings(self):
        return self.embedding


def _prepared() -> SimpleNamespace:
    model = _FakeModel()
    o2 = torch.zeros((1, 256, 4096), dtype=torch.float32)
    text_embeddings = torch.zeros((1, 3, 4096), dtype=torch.float32)
    return SimpleNamespace(
        o2=o2,
        _runtime=SimpleNamespace(torch=torch),
        _model=model,
        _input_ids=torch.ones((1, 3), dtype=torch.int64),
        _text_embeddings=text_embeddings,
        _multimodal_attention_mask=torch.ones((1, 259), dtype=torch.int64),
    )


def _base_logits() -> np.ndarray:
    logits = np.zeros((7, 20), dtype=np.float32)
    for position in range(7):
        logits[position, position] = 4.0
        logits[position, position + 1] = 3.0
    return logits


def test_aligned_logits_use_clean_prefix_and_supplied_o2() -> None:
    prepared = _prepared()
    clean = diagnostic._aligned_next_token_logits(
        prepared=prepared,
        o2=prepared.o2,
        clean_action_token_ids=ACTION_IDS,
    )
    modified_o2 = prepared.o2.clone()
    modified_o2[0, 0, 0] = 0.25
    modified = diagnostic._aligned_next_token_logits(
        prepared=prepared,
        o2=modified_o2,
        clean_action_token_ids=ACTION_IDS,
    )

    assert clean.shape == (7, 32)
    np.testing.assert_array_equal(np.argmax(clean, axis=1), ACTION_IDS[0])
    assert np.all(modified[:, 0] == clean[:, 0] + 0.25)
    assert len(prepared._model.language_model.calls) == 2
    first_embeddings, first_mask = prepared._model.language_model.calls[0]
    second_embeddings, second_mask = prepared._model.language_model.calls[1]
    assert first_embeddings.shape == second_embeddings.shape == (1, 265, 4096)
    assert first_mask.shape == second_mask.shape == (1, 265)
    torch.testing.assert_close(
        first_embeddings[:, 257:],
        second_embeddings[:, 257:],
    )
    assert not torch.equal(
        first_embeddings[:, 1:257],
        second_embeddings[:, 1:257],
    )


def test_logit_report_records_top2_margins_and_repeatability() -> None:
    clean = _base_logits()
    modified = clean.copy()
    modified[:, 2] += np.float32(0.125)
    report = diagnostic._logit_report(clean, modified, clean.copy())

    assert report["delta_logits_l2"] == pytest.approx(np.sqrt(7) * 0.125)
    assert report["delta_logits_max_abs"] == pytest.approx(0.125)
    assert report["clean_repeat_delta_l2"] == 0.0
    assert report["clean_repeat_delta_max_abs"] == 0.0
    assert report["changed_beyond_clean_repeatability"] is True
    assert len(report["per_position"]) == 7
    first = report["per_position"][0]
    assert first["clean_top1_token_id"] == 0
    assert first["clean_top2_token_id"] == 1
    assert first["clean_top1_top2_margin"] == pytest.approx(1.0)
    assert first["delta_logits_l2"] == pytest.approx(0.125)


def test_diagnostic_cases_do_not_introduce_a_change_threshold() -> None:
    translation = np.zeros(3, dtype=np.float64)
    case_a = diagnostic._diagnostic_case(
        clean_token_ids=ACTION_IDS,
        modified_token_ids=ACTION_IDS.copy(),
        clean_translation=translation,
        modified_translation=translation.copy(),
        logit_report={"changed_beyond_clean_repeatability": True},
    )
    assert case_a[0] == "A"

    changed_rotation = ACTION_IDS.copy()
    changed_rotation[0, 3] += 1
    case_b = diagnostic._diagnostic_case(
        clean_token_ids=ACTION_IDS,
        modified_token_ids=changed_rotation,
        clean_translation=translation,
        modified_translation=translation.copy(),
        logit_report={"changed_beyond_clean_repeatability": True},
    )
    assert case_b[0] == "B"

    case_c = diagnostic._diagnostic_case(
        clean_token_ids=ACTION_IDS,
        modified_token_ids=ACTION_IDS.copy(),
        clean_translation=translation,
        modified_translation=translation.copy(),
        logit_report={"changed_beyond_clean_repeatability": False},
    )
    assert case_c[0] == "C"


def test_changed_translation_is_a_frozen_smoke_mismatch() -> None:
    changed = ACTION_IDS.copy()
    changed[0, 0] += 1
    with pytest.raises(
        diagnostic.OpenVLALogitDiagnosticError,
        match="no longer reproduces",
    ):
        diagnostic._diagnostic_case(
            clean_token_ids=ACTION_IDS,
            modified_token_ids=changed,
            clean_translation=np.zeros(3),
            modified_translation=np.ones(3),
            logit_report={"changed_beyond_clean_repeatability": True},
        )


def test_transactional_publication_and_no_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostic"
    result = {
        "diagnostic_result": "PASS",
        "protocol": {"alpha": 1e-3},
        "observations": [
            {
                "sample_id": "sample",
                "diagnostic_case": "A",
                "actual_relative_o2_perturbation": 1e-3,
                "clean_action_token_ids": ACTION_IDS[0].tolist(),
                "modified_action_token_ids": ACTION_IDS[0].tolist(),
                "logits": {
                    "delta_logits_l2": 1.0,
                    "delta_logits_max_abs": 0.1,
                },
                "interpretation": "changed logits",
            }
        ],
    }
    diagnostic._publish(output_dir, result)
    assert {path.name for path in output_dir.iterdir()} == {
        diagnostic.RESULT_FILENAME,
        diagnostic.SUMMARY_FILENAME,
    }
    assert (
        diagnostic._validate_output_slot(output_dir.parent / "fresh")
        == (output_dir.parent / "fresh").resolve()
    )
    with pytest.raises(
        diagnostic.OpenVLALogitDiagnosticError,
        match="already exists",
    ):
        diagnostic._validate_output_slot(output_dir)
