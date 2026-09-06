"""UP action-relevant pilot：先 screen 冻结候选，再 validate 读取独立轨迹。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import up_concept_pilot as pilot  # noqa: E402
from shared_feature.pilot_observation import PilotObservation  # noqa: E402
from shared_feature.up_concept import Moments, Neuron, down_projections, ffn_hooks, random_controls  # noqa: E402
from shared_feature.up_action_screen import (  # noqa: E402
    action_mapping, check_decoding, rank_candidates, validation_summary, z_readout,
)

SCHEMA = "up_action_relevant_v1"
SOURCE_FILES = ("protocol.json", "results.json", "candidates.json", "checkpoint_hashes.json")
FROZEN_FILES = ("protocol.json", "checkpoint_hashes.json", "frozen.json", "results.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_identities(root: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: pilot.sha256_file(root / name) for name in names}


def load_validation(manifest: Path, excluded: Sequence[pilot.Sample]) -> list[pilot.Sample]:
    """读取显式验证 manifest；拒绝旧 state/image/sample 身份，不自动扫描或挑选帧。

    provenance 由采集者提供；代码只能核验声明与文件身份，无法证明未见过这些数据。
    """
    data = read_json(manifest)
    if data.get("schema") != "up_action_validation_observations_v1":
        raise ValueError("unsupported validation manifest schema")
    provenance = data.get("provenance", {})
    if (provenance.get("checkpoint_identity") != pilot.smoke.CHECKPOINT_IDENTITY
            or provenance.get("task_suite") != "libero_spatial" or provenance.get("task_id") != 0
            or not isinstance(provenance.get("collection_description"), str)
            or not provenance["collection_description"].strip()
            or provenance.get("not_used_for_selection") is not True):
        raise ValueError("validation collection provenance is incomplete")
    records = data.get("observations")
    if not isinstance(records, list) or not records:
        raise ValueError("validation observations must be an explicit nonempty list")
    old_states = {s.record.initial_state_id for s in excluded}
    old_samples = {s.record.sample_id for s in excluded}
    old_pixels = {hashlib.sha256(s.record.base_rgb_raw.tobytes()).hexdigest() for s in excluded}
    samples = []
    seen_ids, seen_steps, seen_images = set(), set(), set()
    for entry in records:
        path = Path(entry["path"])
        path = (path if path.is_absolute() else manifest.parent / path).resolve()
        digest = pilot.sha256_file(path)
        if digest != entry.get("sha256"):
            raise ValueError(f"validation observation hash mismatch: {path}")
        record = PilotObservation.load(path)
        if (record.task_id != "0" or not record.sample_id.startswith("libero_spatial__task00__")
                or record.prompt.casefold().strip().rstrip(".") != pilot.TASK0_PROMPT
                or not record.episode_success or record.base_rgb_raw.shape != (512, 512, 3)):
            raise ValueError("validation requires successful task0 raw 512x512 observations")
        pixel_hash = hashlib.sha256(record.base_rgb_raw.tobytes()).hexdigest()
        identity = (record.initial_state_id, record.step_id)
        if (record.initial_state_id in old_states or record.sample_id in old_samples
                or pixel_hash in old_pixels or record.sample_id in seen_ids
                or identity in seen_steps or pixel_hash in seen_images):
            raise ValueError("validation reuses old or duplicate observation identity")
        seen_ids.add(record.sample_id)
        seen_steps.add(identity)
        seen_images.add(pixel_hash)
        samples.append(pilot.Sample(path, record, digest, "sha256:" + pixel_hash))
    states = sorted({s.record.initial_state_id for s in samples})
    if len(states) < 2:
        raise ValueError("validation needs at least two new trajectory groups")
    for state in states:
        group = [s.record for s in samples if s.record.initial_state_id == state]
        if len(group) != 4 or len({r.episode_id for r in group}) != 1:
            raise ValueError("each validation group needs four frames from one episode")
    return sorted(samples, key=lambda s: (s.record.initial_state_id, s.record.step_id))


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[pilot.Sample], list[pilot.Sample]]:
    info, calibration, old_evaluation = pilot.preflight(args)
    source = read_json(args.v1_dir / "protocol.json")
    result = read_json(args.v1_dir / "results.json")
    if (result.get("engineering_status") != "COMPLETE" or (args.v1_dir / "failure.json").exists()
            or source["checkpoint_identity"] != pilot.smoke.CHECKPOINT_IDENTITY):
        raise ValueError("v1 source must be a completed OpenVLA UP run")
    actual = [(s.record.sample_id, s.archive_sha256) for s in calibration + old_evaluation]
    recorded = [(s["sample_id"], s["archive_sha256"]) for s in source["calibration"] + source["evaluation"]]
    if actual != recorded:
        raise ValueError("calibration/development observations differ from completed v1 run")
    info.update(schema=SCHEMA, stage=args.stage, v1_dir=str(args.v1_dir),
                v1_hashes=file_identities(args.v1_dir, SOURCE_FILES))
    info.pop("evaluation")
    info["excluded_development"] = [pilot.sample_dict(s) for s in old_evaluation]
    info["protocol"] = {"k": 10, "screen_offsets": [-1.0, 1.0], "validation_alphas": [1.0, 2.0],
                        "score": "minimum across trajectories of plus and reversed-minus frame medians",
                        "eligibility_floor": 1e-8, "validation": "independent new states only",
                        "screen_readout": "clean-prefix conditional expected normalized z",
                        "intervention_scope": "all forward tokens", "rollout": False}
    validation = []
    if args.stage == "validate":
        if args.selection_dir is None or args.validation_manifest is None:
            raise ValueError("validate requires --selection-dir and --validation-manifest")
        selected_result = read_json(args.selection_dir / "results.json")
        selected_protocol = read_json(args.selection_dir / "protocol.json")
        if (selected_result.get("selection_status") != "FROZEN"
                or selected_result.get("engineering_status") != "COMPLETE"
                or (args.selection_dir / "failure.json").exists()
                or selected_protocol.get("schema") != SCHEMA
                or selected_protocol.get("stage") != "screen"
                or selected_protocol["v1_hashes"] != info["v1_hashes"]
                or selected_protocol["repository"]["head"] != info["repository"]["head"]):
            raise ValueError("selection must be frozen with the same code and v1 source")
        if selected_result["frozen_sha256"] != pilot.sha256_file(args.selection_dir / "frozen.json"):
            raise ValueError("frozen selection hash mismatch")
        validation = load_validation(args.validation_manifest, calibration + old_evaluation)
        info.update(selection_hashes=file_identities(args.selection_dir, FROZEN_FILES),
                    selection_dir=str(args.selection_dir), validation_manifest=str(args.validation_manifest),
                    validation_manifest_sha256=pilot.sha256_file(args.validation_manifest),
                    validation=[pilot.sample_dict(s) for s in validation])
    elif args.validation_manifest or args.selection_dir:
        raise ValueError("screen must not access a validation manifest or selection directory")
    return info, calibration, validation


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """复用真实 v1 runtime；不替换 checkpoint、tokenizer 或模型参数。"""
    pilot.prepare_tex3d_import_path(args.tex3d_openvla_root)
    runtime = pilot.smoke._load_runtime(args.tex3d_openvla_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for screen/validate")
    runtime.set_seed_everywhere(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = SimpleNamespace(model_family="openvla", pretrained_checkpoint=str(args.pretrained_checkpoint),
                          load_in_8bit=False, load_in_4bit=False,
                          unnorm_key=pilot.smoke.UNNORM_KEY, center_crop=True)
    model, processor = runtime.get_model(cfg), runtime.get_processor(cfg)
    runtime.validate_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return runtime, model, processor


def calibrate(runtime: Any, model: Any, processor: Any, samples: Sequence[pilot.Sample],
              neurons: Sequence[Neuron], ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """只读原校准集的 generation activation，收集 [neurons] population std。"""
    modules, moments, records = down_projections(model), Moments(), []
    for sample in samples:
        prepared = pilot.prepare(runtime, model, processor, sample)
        clean = runtime.run_reference(prepared=prepared)
        check_decoding(clean, ids, values)
        with ffn_hooks(modules, neurons, moments=moments):
            recorded = runtime.run_reference(prepared=prepared)
        equality = pilot.require_equal(clean, recorded, "screen calibration hook")
        records.append({**pilot.sample_dict(sample), "action": pilot.action_dict(clean), "equivalence": equality})
    if moments.count < 2:
        raise ValueError("insufficient calibration")
    # 零 std 候选不能接受规定的尺度干预，显式标记无效，保留其统计。
    std = np.sqrt(moments.m2 / moments.count)
    if not np.isfinite(std).all():
        raise ValueError("nonfinite calibration std")
    return std, records


def shift_norm(modules: Sequence[Any], neurons: Sequence[Neuron], offsets: np.ndarray) -> float:
    """按层合计理想 FFN 输出增量平方范数，用于组合干预预算匹配。"""
    total = 0.0
    for layer in sorted({n.layer for n in neurons}):
        positions = [i for i, n in enumerate(neurons) if n.layer == layer]
        weight = modules[layer].weight[:, [neurons[i].index for i in positions]].detach().float()
        vector = weight @ torch.as_tensor(offsets[positions], device=weight.device, dtype=weight.dtype)
        total += float(vector.double().square().sum())
    norm = float(np.sqrt(total))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("zero/nonfinite FFN shift norm")
    return norm


def screen(args: argparse.Namespace, runtime: Any, model: Any, processor: Any,
           calibration: list[pilot.Sample]) -> dict[str, Any]:
    candidates = read_json(args.v1_dir / "candidates.json")["candidates"]
    neurons = [Neuron(**c["neuron"]) for c in candidates]
    if len(neurons) < 10 or len(set(neurons)) != len(neurons):
        raise ValueError("candidate pool must contain at least ten unique neurons")
    modules = down_projections(model)
    ids, values = action_mapping(model)
    std_values, clean_records = calibrate(runtime, model, processor, calibration, neurons, ids, values)
    std = dict(zip(neurons, std_values.tolist()))
    valid = [n for n in neurons if std[n] > 0]
    pilot.write_json(args.output_dir / "calibration.json", {
        "neurons": [asdict(n) for n in neurons], "std": std_values.tolist(),
        "invalid_zero_std": [asdict(n) for n in neurons if std[n] <= 0], "clean_records": clean_records})
    evidence = args.output_dir / "screen_logits"
    evidence.mkdir()
    rows = []
    for sample in calibration:
        prepared = pilot.prepare(runtime, model, processor, sample)
        clean = runtime.run_reference(prepared=prepared)
        pilot.require_equal(clean, runtime.continue_from_o2(prepared=prepared, o2=prepared.o2), "screen continuation")
        base = pilot.logits(prepared, clean)
        repeat = pilot.logits(prepared, clean)
        if not np.array_equal(base, repeat):
            raise ValueError("clean screen logits are not repeatable")
        baseline = z_readout(base, ids, values)
        modified_z = []
        for i, neuron in enumerate(valid):
            readings, hooks, z_arrays = [], [], []
            for sign in (1, -1):
                with ffn_hooks(modules, [neuron], offsets=np.array([sign * std[neuron]])) as applied:
                    modified = pilot.logits(prepared, clean)
                readings.append(z_readout(modified, ids, values))
                hooks.append(applied)
                z_arrays.append(modified[2])
            # 检查每个候选的 hook 移除后回到同一 clean-prefix 计算。
            if not np.array_equal(base, pilot.logits(prepared, clean)):
                raise ValueError("screen logits failed restoration")
            modified_z.append(z_arrays)
            rows.append({"sample_id": sample.record.sample_id, "state_id": sample.record.initial_state_id,
                         "layer": neuron.layer, "index": neuron.index,
                         "plus_delta_z": readings[0]["expected_normalized_z"] - baseline["expected_normalized_z"],
                         "minus_delta_z": readings[1]["expected_normalized_z"] - baseline["expected_normalized_z"],
                         "clean": baseline, "plus": readings[0], "minus": readings[1], "hooks": hooks})
            if (i + 1) % 25 == 0:
                print(f"screen {sample.record.sample_id}: {i + 1}/{len(valid)}", flush=True)
        pilot.require_equal(clean, runtime.run_reference(prepared=prepared), "screen native restoration")
        np.savez_compressed(evidence / f"{sample.record.sample_id}.npz", clean_logits=base,
                            candidate_z_logits=np.asarray(modified_z),
                            neurons=np.array([[n.layer, n.index] for n in valid]),
                            signs=np.array([1, -1]), action_ids=ids, action_values=values)
        pilot.write_json(args.output_dir / "screen_rows.json", rows)
    ranked = rank_candidates(rows)
    pilot.write_json(args.output_dir / "ranking.json", ranked)
    if ranked["selection_status"] != "FROZEN":
        return {"selection_status": "INSUFFICIENT_CANDIDATES", "eligible_count": ranked["eligible_count"]}
    selected = [Neuron(**n) for n in ranked["selected"]]
    lexical = neurons[:10]
    sets = {"lexical": lexical, "selected": selected}
    for target, group in list(sets.items()):
        controls = random_controls(group, [m.in_features for m in modules], neurons, repeats=3, seed=7)
        sets.update({f"random_{target}_{i}": control for i, control in enumerate(controls)})
    all_neurons = sorted(set(n for group in sets.values() for n in group), key=lambda n: (n.layer, n.index))
    combined_std, combined_records = calibrate(runtime, model, processor, calibration, all_neurons, ids, values)
    if np.any(combined_std <= 0):
        raise ValueError("frozen combination contains zero-std neurons; no resampling")
    std = dict(zip(all_neurons, combined_std.tolist()))
    offsets = {name: np.array([std[n] for n in sets[name]], dtype=np.float32) for name in ("lexical", "selected")}
    budget = shift_norm(modules, lexical, offsets["lexical"])
    offsets["selected"] *= budget / shift_norm(modules, selected, offsets["selected"])
    for target in ("lexical", "selected"):
        scale = float(offsets[target][0] / std[sets[target][0]])
        for i in range(3):
            name = f"random_{target}_{i}"
            offsets[name] = pilot.matched_offsets(modules, sets[target], sets[name], std) * scale
    frozen = {"schema": SCHEMA, "action_ids": ids.tolist(), "action_values": values.tolist(),
              "sets": {name: [asdict(n) for n in group] for name, group in sets.items()},
              "offsets": {name: offset.tolist() for name, offset in offsets.items()},
              "clean_neurons": [asdict(n) for n in all_neurons], "clean_std": combined_std.tolist(),
              "clean_records": combined_records, "ideal_shift_budget": budget,
              "ideal_shift_norms": {name: shift_norm(modules, sets[name], offsets[name]) for name in sets}}
    pilot.write_json(args.output_dir / "frozen.json", frozen)
    return {"selection_status": "FROZEN", "eligible_count": ranked["eligible_count"],
            "frozen_sha256": pilot.sha256_file(args.output_dir / "frozen.json")}


def validate(args: argparse.Namespace, runtime: Any, model: Any, processor: Any,
             samples: list[pilot.Sample]) -> dict[str, Any]:
    """只读取已冻结集合/offsets，不在验证观测上估计 std、排序或调整强度。"""
    frozen = read_json(args.selection_dir / "frozen.json")
    ids, values = action_mapping(model)
    if ids.tolist() != frozen["action_ids"] or values.tolist() != frozen["action_values"]:
        raise ValueError("model action mapping changed since screening")
    conditions = {name: ([Neuron(**n) for n in group], np.asarray(frozen["offsets"][name], dtype=np.float32))
                  for name, group in frozen["sets"].items()}
    all_neurons = sorted({n for group, _ in conditions.values() for n in group}, key=lambda n: (n.layer, n.index))
    modules = down_projections(model)
    evidence = args.output_dir / "logits"
    evidence.mkdir()
    rows = []
    for sample in samples:
        prepared = pilot.prepare(runtime, model, processor, sample)
        check_decoding(runtime.run_reference(prepared=prepared), ids, values)
        paired = pilot.evaluate_sample(runtime, prepared, modules, all_neurons, conditions, sample, evidence)
        with np.load(evidence / f"{sample.record.sample_id}__clean.npz", allow_pickle=False) as archive:
            clean = z_readout(archive["clean_logits"], ids, values)
        for row in paired:
            filename = f"{sample.record.sample_id}__{row['condition']}__{row['alpha']:g}.npz"
            with np.load(evidence / filename, allow_pickle=False) as archive:
                modified = z_readout(archive["logits"], ids, values)
            row.update(clean_z_readout=clean, modified_z_readout=modified,
                       delta_expected_z=modified["expected_normalized_z"] - clean["expected_normalized_z"])
        rows.extend(paired)
        pilot.write_json(args.output_dir / "paired_results.json", rows)
    summary = validation_summary(rows)
    pilot.write_json(args.output_dir / "summary.json", summary)
    return {"validation_status": "DESCRIPTIVE_ONLY_REQUIRES_REVIEW", "paired_rows": len(rows), "summary": summary}


def run(args: argparse.Namespace, info: dict[str, Any], calibration: list[pilot.Sample],
        validation: list[pilot.Sample]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        pilot.write_json(args.output_dir / "protocol.json", info)
        hashes = pilot.checkpoint_hashes(args.pretrained_checkpoint)
        if hashes != read_json(args.v1_dir / "checkpoint_hashes.json"):
            raise ValueError("checkpoint differs from completed v1 source")
        if args.stage == "validate" and hashes != read_json(args.selection_dir / "checkpoint_hashes.json"):
            raise ValueError("checkpoint differs from frozen selection")
        pilot.write_json(args.output_dir / "checkpoint_hashes.json", hashes)
        runtime, model, processor = load_model(args)
        result = (screen(args, runtime, model, processor, calibration) if args.stage == "screen"
                  else validate(args, runtime, model, processor, validation))
        for sample in calibration + validation:
            if pilot.sha256_file(sample.path) != sample.archive_sha256:
                raise ValueError("source observation changed during run")
        if pilot.sha256_file(args.collection_manifest) != info["manifest_sha256"]:
            raise ValueError("source collection manifest changed")
        if file_identities(args.v1_dir, SOURCE_FILES) != info["v1_hashes"]:
            raise ValueError("v1 source changed during run")
        if pilot.repository_identity(ROOT) != info["repository"]:
            raise ValueError("project code changed during run")
        if pilot.repository_identity(args.tex3d_openvla_root.parent) != info["tex3d_repository"]:
            raise ValueError("Tex3D code changed during run")
        if args.stage == "validate":
            if (file_identities(args.selection_dir, FROZEN_FILES) != info["selection_hashes"]
                    or pilot.sha256_file(args.validation_manifest) != info["validation_manifest_sha256"]):
                raise ValueError("frozen selection or validation manifest changed during run")
        pilot.write_json(args.output_dir / "results.json", {
            "schema": SCHEMA, "stage": args.stage, "engineering_status": "COMPLETE",
            "scientific_status": "DESCRIPTIVE_ONLY_REQUIRES_REVIEW", "runtime": runtime.versions,
            "rollout_executed": False, **result})
        print(f"{args.stage.upper()} COMPLETE: {args.output_dir}", flush=True)
    except Exception as error:
        pilot.write_json(args.output_dir / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "validate"))
    for name in ("collection-manifest", "pretrained-checkpoint", "tex3d-openvla-root", "output-dir", "v1-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    info, calibration, validation = preflight(args)
    if args.preflight_only:
        # 真正导入完整预处理链，但不加载模型。早于权重 hash 暴露 import 错误。
        pilot.smoke._load_runtime(args.tex3d_openvla_root)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("PREFLIGHT PASSED: runtime imports only; no model or output directory")
        return 0
    run(args, info, calibration, validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
