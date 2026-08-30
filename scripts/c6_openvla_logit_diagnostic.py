from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import c6_openvla_real_smoke as smoke  # noqa: E402
from scripts import c6_real_smoke_summary as common  # noqa: E402
from shared_feature import openvla_intervention as intervention  # noqa: E402


SCHEMA_VERSION = "c6_openvla_logit_diagnostic_v1"
RESULT_FILENAME = "results.json"
SUMMARY_FILENAME = "summary.md"
CHECKPOINT_IDENTITY = common.OPENVLA_CHECKPOINT_IDENTITY
UNNORM_KEY = smoke.UNNORM_KEY
GLOBAL_SEED = smoke.GLOBAL_SEED


class OpenVLALogitDiagnosticError(RuntimeError):
    """Raised when the frozen OpenVLA diagnostic cannot run safely."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenVLA action logits under the frozen C6 O2 perturbations."
        )
    )
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--smoke-results", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=smoke.DEFAULT_TEX3D_OPENVLA_ROOT,
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_output_slot(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise OpenVLALogitDiagnosticError(
            f"diagnostic output directory already exists: {target}"
        )
    if not target.parent.is_dir():
        raise OpenVLALogitDiagnosticError(
            f"diagnostic output parent does not exist: {target.parent}"
        )
    return target


def _load_smoke_results(path: Path, checkpoint: Path) -> tuple[dict[str, Any], str]:
    source = path.expanduser().resolve()
    digest = _sha256_file(source)
    payload = common._load_phase(
        source,
        "c6_openvla_real_smoke_v1",
        "openvla",
    )
    identity = payload["model_identity"]
    if Path(identity["resolved_checkpoint_path"]).resolve() != checkpoint:
        raise OpenVLALogitDiagnosticError(
            "real-smoke and diagnostic checkpoint paths differ"
        )
    if payload["clean_gate"] != "PASS" or payload["model_result"] != "BLOCKED":
        raise OpenVLALogitDiagnosticError(
            "diagnostic requires the frozen OpenVLA clean-PASS/intervention-BLOCKED result"
        )
    for report in payload["observations"]:
        applied = report["intervention"]
        if (
            applied is None
            or applied["actual_modified_differs"] is not True
            or applied["translation_delta_norm"] != 0.0
        ):
            raise OpenVLALogitDiagnosticError(
                "real-smoke does not contain the frozen nonzero-O2/zero-translation case"
            )
    return payload, digest


def _aligned_next_token_logits(
    *,
    prepared: Any,
    o2: Any,
    clean_action_token_ids: np.ndarray,
) -> np.ndarray:
    """Return seven logits rows under one authoritative clean action prefix."""
    torch = prepared._runtime.torch
    token_ids = np.asarray(clean_action_token_ids)
    if token_ids.shape != (1, 7) or token_ids.dtype.kind not in "iu":
        raise OpenVLALogitDiagnosticError(
            "clean action token IDs must have integer shape [1, 7]"
        )
    converted = intervention._validate_and_convert_override(
        o2,
        prepared.o2,
        torch,
    )
    base_embeddings = intervention._multimodal_embeddings(
        prepared._text_embeddings,
        converted,
        torch,
    )
    prefix_ids = torch.as_tensor(
        token_ids[:, :-1],
        dtype=prepared._input_ids.dtype,
        device=prepared._input_ids.device,
    )
    try:
        with torch.inference_mode():
            prefix_embeddings = prepared._model.get_input_embeddings()(prefix_ids)
            inputs_embeds = torch.cat([base_embeddings, prefix_embeddings], dim=1)
            attention_mask = prepared._multimodal_attention_mask
            if attention_mask is not None:
                suffix_mask = torch.ones(
                    (1, prefix_ids.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
            output = prepared._model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits
            if (
                not torch.is_tensor(logits)
                or logits.ndim != 3
                or logits.shape[0] != 1
                or logits.shape[1] != inputs_embeds.shape[1]
                or logits.shape[2] < 2
                or not bool(torch.all(torch.isfinite(logits[:, -7:, :])))
            ):
                raise OpenVLALogitDiagnosticError(
                    "language model returned malformed action-position logits"
                )
            aligned = (
                logits[:, -7:, :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                .copy()
            )
    except OpenVLALogitDiagnosticError:
        raise
    except Exception as error:
        raise OpenVLALogitDiagnosticError(
            "aligned OpenVLA language-model forward failed"
        ) from error
    if aligned.shape[0:2] != (1, 7) or not np.all(np.isfinite(aligned)):
        raise OpenVLALogitDiagnosticError("aligned logits are malformed")
    return aligned[0]


def _top2(row: np.ndarray) -> tuple[int, float, int, float]:
    order = np.argsort(-np.asarray(row), kind="stable")[:2]
    return (
        int(order[0]),
        float(row[order[0]]),
        int(order[1]),
        float(row[order[1]]),
    )


def _logit_report(
    clean_logits: np.ndarray,
    modified_logits: np.ndarray,
    clean_repeat_logits: np.ndarray,
) -> dict[str, Any]:
    arrays = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            clean_logits,
            modified_logits,
            clean_repeat_logits,
        )
    )
    clean, modified, repeated = arrays
    if (
        clean.ndim != 2
        or clean.shape[0] != 7
        or clean.shape[1] < 2
        or modified.shape != clean.shape
        or repeated.shape != clean.shape
        or any(not np.all(np.isfinite(value)) for value in arrays)
    ):
        raise OpenVLALogitDiagnosticError("logit arrays must be finite [7, vocabulary]")
    delta = modified.astype(np.float64) - clean.astype(np.float64)
    repeat_delta = repeated.astype(np.float64) - clean.astype(np.float64)
    per_position = []
    for position in range(7):
        clean_top1, clean_logit1, clean_top2, clean_logit2 = _top2(clean[position])
        modified_top1, modified_logit1, modified_top2, modified_logit2 = _top2(
            modified[position]
        )
        position_delta = delta[position]
        repeat_position_delta = repeat_delta[position]
        per_position.append(
            {
                "position": position,
                "clean_top1_token_id": clean_top1,
                "clean_top1_logit": clean_logit1,
                "clean_top2_token_id": clean_top2,
                "clean_top2_logit": clean_logit2,
                "clean_top1_top2_margin": clean_logit1 - clean_logit2,
                "modified_top1_token_id": modified_top1,
                "modified_top1_logit": modified_logit1,
                "modified_top2_token_id": modified_top2,
                "modified_top2_logit": modified_logit2,
                "modified_top1_top2_margin": modified_logit1 - modified_logit2,
                "top1_changed": modified_top1 != clean_top1,
                "delta_logits_l2": float(np.linalg.norm(position_delta)),
                "delta_logits_max_abs": float(
                    np.max(np.abs(position_delta), initial=0.0)
                ),
                "clean_repeat_delta_l2": float(np.linalg.norm(repeat_position_delta)),
                "clean_repeat_delta_max_abs": float(
                    np.max(np.abs(repeat_position_delta), initial=0.0)
                ),
            }
        )
    delta_l2 = float(np.linalg.norm(delta))
    delta_max = float(np.max(np.abs(delta), initial=0.0))
    repeat_l2 = float(np.linalg.norm(repeat_delta))
    repeat_max = float(np.max(np.abs(repeat_delta), initial=0.0))
    return {
        "comparison_semantics": (
            "teacher-forced next-token logits at all seven action positions; "
            "both O2 conditions use the authoritative clean action-token prefix"
        ),
        "logits_dtype_after_capture": "float32",
        "vocabulary_size": int(clean.shape[1]),
        "delta_logits_l2": delta_l2,
        "delta_logits_max_abs": delta_max,
        "clean_repeat_delta_l2": repeat_l2,
        "clean_repeat_delta_max_abs": repeat_max,
        "changed_beyond_clean_repeatability": bool(delta_max > repeat_max),
        "per_position": per_position,
    }


def _diagnostic_case(
    *,
    clean_token_ids: np.ndarray,
    modified_token_ids: np.ndarray,
    clean_translation: np.ndarray,
    modified_translation: np.ndarray,
    logit_report: dict[str, Any],
) -> tuple[str, str]:
    clean_ids = np.asarray(clean_token_ids)
    modified_ids = np.asarray(modified_token_ids)
    if clean_ids.shape != (1, 7) or modified_ids.shape != (1, 7):
        raise OpenVLALogitDiagnosticError(
            "free-running token IDs must have shape [1, 7]"
        )
    translations_equal = bool(
        np.array_equal(
            np.asarray(clean_translation),
            np.asarray(modified_translation),
        )
    )
    ids_equal = bool(np.array_equal(clean_ids, modified_ids))
    translation_ids_equal = bool(np.array_equal(clean_ids[:, :3], modified_ids[:, :3]))
    if not ids_equal:
        if not translations_equal:
            raise OpenVLALogitDiagnosticError(
                "rerun no longer reproduces the frozen zero-translation response"
            )
        return (
            "B",
            (
                "free-running action tokens changed while decoded translation remained "
                "unchanged"
                + (
                    " and translation token IDs remained unchanged"
                    if translation_ids_equal
                    else ""
                )
            ),
        )
    if logit_report["changed_beyond_clean_repeatability"]:
        return (
            "A",
            "aligned logits changed beyond clean repeatability but action token IDs did not",
        )
    return (
        "C",
        "aligned logit changes did not exceed observed clean-repeatability noise",
    )


def _require_smoke_reproduction(
    *,
    prior: dict[str, Any],
    perturbation: dict[str, Any],
    clean_result: Any,
    modified_result: Any,
) -> None:
    previous = prior["intervention"]
    for key in (
        "clean_feature_norm",
        "intended_delta_norm",
        "actual_delta_norm",
        "intended_relative_perturbation",
        "actual_relative_perturbation",
    ):
        if not np.isclose(
            perturbation[key],
            previous[key],
            rtol=0,
            atol=1e-12,
        ):
            raise OpenVLALogitDiagnosticError(
                f"diagnostic did not reproduce real-smoke perturbation metric {key}"
            )
    comparisons = (
        (clean_result.unnormalized_action[0, :3], previous["clean_translation"]),
        (
            modified_result.unnormalized_action[0, :3],
            previous["modified_translation"],
        ),
    )
    if any(
        not np.allclose(left, right, rtol=0, atol=smoke.CLEAN_ATOL)
        for left, right in comparisons
    ):
        raise OpenVLALogitDiagnosticError(
            "diagnostic did not reproduce real-smoke translations"
        )


def _summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# C6 OpenVLA Minimal Token/Logit Diagnostic",
        "",
        f"- Diagnostic result: {result['diagnostic_result']}",
        "- Alignment: teacher-forced logits with the authoritative clean token prefix",
        f"- Alpha: {result['protocol']['alpha']}",
        "",
    ]
    for report in result["observations"]:
        lines.extend(
            [
                f"## {report['sample_id']}",
                "",
                f"- Case: {report['diagnostic_case']}",
                f"- Relative O2 perturbation: {report['actual_relative_o2_perturbation']}",
                f"- Clean tokens: {report['clean_action_token_ids']}",
                f"- Modified tokens: {report['modified_action_token_ids']}",
                f"- Logit delta L2: {report['logits']['delta_logits_l2']}",
                f"- Logit delta max abs: {report['logits']['delta_logits_max_abs']}",
                f"- Interpretation: {report['interpretation']}",
                "",
            ]
        )
    lines.append(
        "This diagnostic does not search perturbations and does not establish C6-B policy relevance."
    )
    lines.append("")
    return "\n".join(lines)


def _publish(output_dir: Path, result: dict[str, Any]) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        common.write_json_atomic(staging / RESULT_FILENAME, result)
        common.write_text_atomic(staging / SUMMARY_FILENAME, _summary_markdown(result))
        if {path.name for path in staging.iterdir()} != {
            RESULT_FILENAME,
            SUMMARY_FILENAME,
        }:
            raise OpenVLALogitDiagnosticError("staged diagnostic output is incomplete")
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _validate_output_slot(args.output_dir)
    checkpoint, tex3d_root = smoke._validate_runtime_paths(args)
    smoke_payload, smoke_hash_before = _load_smoke_results(
        args.smoke_results,
        checkpoint,
    )
    reference_repository_commit = common.git_commit(tex3d_root.parent)
    if reference_repository_commit != smoke_payload["reference_repository_commit"]:
        raise OpenVLALogitDiagnosticError(
            "Tex3D OpenVLA source commit differs from the real-smoke runtime"
        )
    observations = common.load_frozen_observations(args.collection_manifest)
    runtime = smoke._load_runtime(tex3d_root)
    if not runtime.torch.cuda.is_available():
        raise OpenVLALogitDiagnosticError("CUDA is unavailable in the OpenVLA runtime")

    runtime.set_seed_everywhere(GLOBAL_SEED)
    model_config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    model = runtime.get_model(model_config)
    processor = runtime.get_processor(model_config)
    runtime.validate_model(model)
    prior_by_id = {
        report["sample_id"]: report for report in smoke_payload["observations"]
    }
    reports = []
    for loaded in observations:
        policy_observation = runtime.build_policy_observation(loaded.record)
        prepared = runtime.prepare_context(
            model=model,
            processor=processor,
            observation=policy_observation,
            task_description=loaded.record.prompt,
            pretrained_checkpoint=CHECKPOINT_IDENTITY,
            unnorm_key=UNNORM_KEY,
            center_crop=True,
        )
        reference = runtime.run_reference(prepared=prepared)
        clean_result = runtime.continue_from_o2(prepared=prepared, o2=prepared.o2)
        if not smoke._clean_report(reference, clean_result)["pass"]:
            raise OpenVLALogitDiagnosticError(
                f"clean equivalence failed for {loaded.identity.sample_id}"
            )

        clean_host = (
            prepared.o2.detach()
            .to(device="cpu", dtype=runtime.torch.float32)
            .numpy()
            .copy()
        )
        intended_delta, intended_modified = common.intended_perturbation(
            clean_host,
            seed=loaded.identity.direction_seeds["openvla"],
        )
        actual_modified = runtime.torch.as_tensor(
            intended_modified,
            dtype=prepared.o2.dtype,
            device=prepared.o2.device,
        )
        actual_host = (
            actual_modified.detach()
            .to(device="cpu", dtype=runtime.torch.float32)
            .numpy()
            .copy()
        )
        perturbation = common.perturbation_metrics(
            clean_host,
            intended_delta,
            actual_host,
        )
        modified_result = runtime.continue_from_o2(
            prepared=prepared,
            o2=actual_modified,
        )
        prior = prior_by_id[loaded.identity.sample_id]
        _require_smoke_reproduction(
            prior=prior,
            perturbation=perturbation,
            clean_result=clean_result,
            modified_result=modified_result,
        )

        clean_logits = _aligned_next_token_logits(
            prepared=prepared,
            o2=prepared.o2,
            clean_action_token_ids=clean_result.action_token_ids,
        )
        clean_repeat_logits = _aligned_next_token_logits(
            prepared=prepared,
            o2=prepared.o2,
            clean_action_token_ids=clean_result.action_token_ids,
        )
        modified_logits = _aligned_next_token_logits(
            prepared=prepared,
            o2=actual_modified,
            clean_action_token_ids=clean_result.action_token_ids,
        )
        logits = _logit_report(clean_logits, modified_logits, clean_repeat_logits)
        aligned_clean_top1 = np.asarray(
            [value["clean_top1_token_id"] for value in logits["per_position"]]
        )
        if not np.array_equal(aligned_clean_top1, clean_result.action_token_ids[0]):
            raise OpenVLALogitDiagnosticError(
                "teacher-forced logit positions do not align with clean generation"
            )
        diagnostic_case, interpretation = _diagnostic_case(
            clean_token_ids=clean_result.action_token_ids,
            modified_token_ids=modified_result.action_token_ids,
            clean_translation=clean_result.unnormalized_action[0, :3],
            modified_translation=modified_result.unnormalized_action[0, :3],
            logit_report=logits,
        )
        reports.append(
            {
                "sample_id": loaded.identity.sample_id,
                "source_image_hash": loaded.identity.source_image_hash,
                "direction_seed": loaded.identity.direction_seeds["openvla"],
                "actual_relative_o2_perturbation": perturbation[
                    "actual_relative_perturbation"
                ],
                "actual_o2_delta_norm": perturbation["actual_delta_norm"],
                "clean_action_token_ids": clean_result.action_token_ids[0].tolist(),
                "modified_action_token_ids": (
                    modified_result.action_token_ids[0].tolist()
                ),
                "clean_unnormalized_translation": (
                    clean_result.unnormalized_action[0, :3].tolist()
                ),
                "modified_unnormalized_translation": (
                    modified_result.unnormalized_action[0, :3].tolist()
                ),
                "logits": logits,
                "diagnostic_case": diagnostic_case,
                "interpretation": interpretation,
            }
        )

    cases = {report["diagnostic_case"] for report in reports}
    diagnostic_result = "NEEDS FOLLOW-UP" if "C" in cases else "PASS"
    smoke_source = args.smoke_results.expanduser().resolve()
    if _sha256_file(smoke_source) != smoke_hash_before:
        raise OpenVLALogitDiagnosticError(
            "source real-smoke artifact changed during run"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_status": "COMPLETED",
        "diagnostic_result": diagnostic_result,
        "repository_commit": common.git_commit(PROJECT_ROOT),
        "reference_repository_commit": reference_repository_commit,
        "model_identity": {
            "scientific_checkpoint": CHECKPOINT_IDENTITY,
            "resolved_checkpoint_path": str(checkpoint),
            "unnorm_key": UNNORM_KEY,
            "center_crop": True,
            "do_sample": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
        },
        "source_real_smoke": {
            "path": str(smoke_source),
            "content_hash": smoke_hash_before,
            "repository_commit": smoke_payload["repository_commit"],
        },
        "runtime_versions": runtime.versions,
        "protocol": {
            "batch_size": 1,
            "global_seed": GLOBAL_SEED,
            "alpha": common.ALPHA,
            "direction_rng": common.RNG_IMPLEMENTATION,
            "direction_dtype": "float32",
            "free_running_generation": "max_new_tokens=7, do_sample=False",
            "logit_alignment": "teacher-forced authoritative clean action-token prefix",
            "logit_change_rule": (
                "intervention max-absolute logit delta exceeds observed clean-repeat max"
            ),
        },
        "observations": reports,
    }
    _publish(output_dir, result)
    return {
        "status": "C6 OpenVLA token/logit diagnostic — COMPLETED",
        "diagnostic_result": diagnostic_result,
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
