"""运行 UP 单概念的冻结 observation pilot，不采集轨迹、不优化纹理。

复用 C6 已验证的 OpenVLA 输入/原生 generation 和 clean-prefix logits 接口。
候选只由 checkpoint 的词汇投影决定。task 0 前三组用于 clean 尺度校准，
后两组用于干预比较；历史 C5 split 和产物完全不改动。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _full_feature_extraction_common as collection  # noqa: E402
from scripts import c6_openvla_logit_diagnostic as diagnostic  # noqa: E402
from scripts import c6_openvla_real_smoke as smoke  # noqa: E402
from shared_feature.pilot_observation import PilotObservation  # noqa: E402
from shared_feature.up_concept import (  # noqa: E402
    Moments, Neuron, candidate_dict, discover_candidates, down_projections,
    exact_up_ids, ffn_hooks, paired_contrasts, random_controls, summarize_rows,
)

SCHEMA = "openvla_up_concept_pilot_v1"
K = 10
ALPHAS = (1.0, 2.0)
RANDOM_SEED = 7
RANDOM_REPEATS = 3
TASK0_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


@dataclass(frozen=True)
class Sample:
    path: Path
    record: PilotObservation
    archive_sha256: str
    image_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """每阶段结束写入小 JSON；不把非有限浮点默默编码为 NaN。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def repository_identity(path: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("diff", "HEAD", "--name-only"):
        raise ValueError(f"tracked changes in repository: {path}")
    return {"path": str(path.resolve()), "head": git("rev-parse", "HEAD")}


def load_samples(manifest: Path) -> tuple[list[Sample], list[Sample]]:
    """沿用完整 collection 验证，选择 task00 五组（每组四帧），不按结果换样本。"""
    source = collection.load_source_collection(manifest)
    samples: list[Sample] = []
    for item in source.records:
        if not item.sample_id.startswith("libero_spatial__task00__"):
            continue
        record = PilotObservation.load(item.resolved_observation_path)
        if record.prompt.casefold().strip().rstrip(".") != TASK0_PROMPT:
            raise ValueError(f"unexpected task 0 instruction: {record.prompt!r}")
        samples.append(Sample(item.resolved_observation_path, record,
                              sha256_file(item.resolved_observation_path), item.source_image_hash))
    samples.sort(key=lambda s: (s.record.initial_state_id, s.record.step_id))
    groups = sorted({s.record.initial_state_id for s in samples})
    if len(groups) != 5 or len(samples) != 20 or any(
        sum(s.record.initial_state_id == g for s in samples) != 4 for g in groups
    ):
        raise ValueError("task 0 requires five complete four-observation groups")
    return ([s for s in samples if s.record.initial_state_id in groups[:3]],
            [s for s in samples if s.record.initial_state_id in groups[3:]])


def sample_dict(sample: Sample) -> dict[str, Any]:
    r = sample.record
    return {"sample_id": r.sample_id, "state_id": r.initial_state_id, "step_id": r.step_id,
            "normalized_episode_progress": r.normalized_episode_progress,
            "prompt": r.prompt, "archive_path": str(sample.path),
            "archive_sha256": sample.archive_sha256, "image_sha256": sample.image_sha256}


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[Sample], list[Sample]]:
    """无模型加载、无输出目录创建；只核对文件、collection 和已知 source 环境。"""
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists (no overwrite/resume): {output}")
    checkpoint, tex3d = smoke._validate_runtime_paths(args)
    if not (tex3d / "experiments/robot/openvla_utils.py").is_file():
        raise FileNotFoundError(f"not a Tex3D OpenVLA root: {tex3d}")
    prepare_tex3d_import_path(tex3d)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "tokenizers", "numpy")}
    required = {"torch": "2.2.0+cu121", "torchvision": "0.17.0+cu121",
                "transformers": "4.40.1", "tokenizers": "0.19.1"}
    if any(versions[k] != v for k, v in required.items()):
        raise ValueError(f"source environment differs from authoritative versions: {versions}")
    calibration, evaluation = load_samples(args.collection_manifest)
    own = repository_identity(PROJECT_ROOT)
    if args.expected_head and own["head"] != args.expected_head:
        raise ValueError(f"HEAD mismatch: expected {args.expected_head}, found {own['head']}")
    info = {"schema": SCHEMA, "repository": own,
            "tex3d_repository": repository_identity(tex3d.parent),
            "checkpoint": str(checkpoint), "checkpoint_identity": smoke.CHECKPOINT_IDENTITY,
            "versions": versions, "manifest": str(args.collection_manifest.resolve()),
            "manifest_sha256": sha256_file(args.collection_manifest),
            "calibration": [sample_dict(s) for s in calibration],
            "evaluation": [sample_dict(s) for s in evaluation],
            "protocol": {"concept": "up", "k": K, "alphas": ALPHAS,
                         "random_repeats": RANDOM_REPEATS, "random_seed": RANDOM_SEED,
                         "scope": "all_forward_tokens_prefill_and_decode",
                         "intervention": "activation + alpha * clean_population_std",
                         "metric": "decoded translation component 2 (action z)",
                         "world_up_interpretation": "requires deployed OSC_POSE convention confirmation",
                         "rollout": False, "scientific_gate": "descriptive_only"}}
    return info, calibration, evaluation


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    """记录本地权重及配置内容身份；只读流式 hash，不载入第二份权重。"""
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and
                   (p.suffix in {".json", ".safetensors", ".bin", ".model", ".py"}))
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise FileNotFoundError("no checkpoint weight shards found")
    result = {}
    for p in files:
        print(f"hash checkpoint {p.name}", flush=True)
        result[p.name] = sha256_file(p)
    return result


def prepare_tex3d_import_path(tex3d_openvla_root: Path) -> None:
    """兼容 Tex3D 旧式 sibling imports used by ``openvla_utils``.

    The deployed Tex3D tree keeps ``openvla_model_inputs.py`` and
    ``openvla_policy_view.py`` beside ``openvla_utils.py`` but imports them as
    top-level modules.  Adding that directory explicitly keeps the runtime
    import deterministic without copying or modifying the Tex3D checkout.
    """
    robot_root = tex3d_openvla_root / "experiments" / "robot"
    if not (robot_root / "openvla_model_inputs.py").is_file():
        raise FileNotFoundError(
            f"Tex3D sibling-import module not found: {robot_root / 'openvla_model_inputs.py'}"
        )
    source = str(robot_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def action_dict(action: Any) -> dict[str, Any]:
    return {key: np.asarray(getattr(action, key)).tolist() for key in
            ("action_token_ids", "normalized_action", "unnormalized_action", "deployed_action")}


def require_equal(clean: Any, other: Any, label: str) -> dict[str, Any]:
    report = smoke._clean_report(clean, other)
    # C6 helper 的命名/结果结构保持原样；不以近似动作一致替代 token exact match。
    if not report["pass"]:
        raise ValueError(f"clean equivalence failed: {label}: {report}")
    return report


def prepare(runtime: Any, model: Any, processor: Any, sample: Sample) -> Any:
    if sha256_file(sample.path) != sample.archive_sha256:
        raise ValueError(f"observation changed after preflight: {sample.path}")
    return runtime.prepare_context(
        model=model, processor=processor,
        observation=runtime.build_policy_observation(sample.record),
        task_description=sample.record.prompt, pretrained_checkpoint=smoke.CHECKPOINT_IDENTITY,
        unnorm_key=smoke.UNNORM_KEY, center_crop=True,
    )


def logits(prepared: Any, clean: Any) -> np.ndarray:
    return diagnostic._aligned_next_token_logits(
        prepared=prepared, o2=prepared.o2, clean_action_token_ids=clean.action_token_ids)


def matched_offsets(
    modules: Sequence[Any], up: Sequence[Neuron], random: Sequence[Neuron],
    std: dict[Neuron, float],
) -> np.ndarray:
    """逐层匹配 alpha=1 时的理想 FFN residual-shift L2，降低 value-vector 范数混杂。

    原始方向仍是各 neuron 自身的 positive clean std。因 BF16 舍入，实际 shift
    另行记录；此匹配不声称两个集合具有相同 downstream Jacobian。
    """
    result = np.asarray([std[n] for n in random], dtype=np.float32)
    for layer in sorted({n.layer for n in up}):
        u = [n for n in up if n.layer == layer]
        positions = [i for i, n in enumerate(random) if n.layer == layer]
        r = [random[i] for i in positions]
        weight = modules[layer].weight.detach().float()
        uv = weight[:, [n.index for n in u]] @ torch.tensor([std[n] for n in u], device=weight.device, dtype=weight.dtype)
        rv = weight[:, [n.index for n in r]] @ torch.tensor([std[n] for n in r], device=weight.device, dtype=weight.dtype)
        u_norm, r_norm = float(uv.norm()), float(rv.norm())
        if not np.isfinite([u_norm, r_norm]).all() or min(u_norm, r_norm) <= 0:
            raise ValueError("cannot match zero/nonfinite residual shift")
        result[positions] *= u_norm / r_norm
    if not np.isfinite(result).all():
        raise ValueError("invalid matched random offsets")
    return result


def evaluate_sample(
    runtime: Any, prepared: Any, modules: Sequence[Any], all_neurons: Sequence[Neuron],
    conditions: dict[str, tuple[list[Neuron], np.ndarray]], sample: Sample, output: Path,
) -> list[dict[str, Any]]:
    """原生 greedy action 是行为读出，统一 clean prefix logits 仅为边界诊断。"""
    clean = runtime.run_reference(prepared=prepared)
    require_equal(clean, runtime.continue_from_o2(prepared=prepared, o2=prepared.o2),
                  "native generation vs clean continuation")
    with ffn_hooks(modules, all_neurons):
        recorded = runtime.run_reference(prepared=prepared)
    require_equal(clean, recorded, "record-only hook")
    with ffn_hooks(modules, all_neurons, offsets=np.zeros(len(all_neurons), dtype=np.float32)):
        noop = runtime.run_reference(prepared=prepared)
    require_equal(clean, noop, "zero-offset hook")
    clean_logits = logits(prepared, clean)
    repeat_logits = logits(prepared, clean)
    np.savez_compressed(output / f"{sample.record.sample_id}__clean.npz",
                        clean_logits=clean_logits, repeat_logits=repeat_logits,
                        action_token_ids=clean.action_token_ids)
    rows = []
    for name, (neurons, offsets) in conditions.items():
        for alpha in ALPHAS:
            with ffn_hooks(modules, neurons, offsets=alpha * offsets) as applied:
                modified = runtime.run_reference(prepared=prepared)
            with ffn_hooks(modules, neurons, offsets=alpha * offsets) as diagnostic_applied:
                modified_logits = logits(prepared, clean)
            restored = runtime.run_reference(prepared=prepared)
            require_equal(clean, restored, f"restore after {name}/{alpha}")
            report = diagnostic._logit_report(clean_logits, modified_logits, repeat_logits)
            delta = (modified.deployed_action[0, :3] - clean.deployed_action[0, :3]).tolist()
            row = {**sample_dict(sample), "condition": name, "alpha": alpha,
                   "clean_action": action_dict(clean), "modified_action": action_dict(modified),
                   "delta_translation": delta,
                   "token_hamming": int(np.count_nonzero(clean.action_token_ids != modified.action_token_ids)),
                   "gripper_changed": bool(clean.deployed_action[0, 6] != modified.deployed_action[0, 6]),
                   "logit_max_abs": report["delta_logits_max_abs"], "logit_report": report,
                   "applied_generation": applied, "applied_teacher_forced": diagnostic_applied,
                   "clean_equivalence": "PASS", "hook_restoration": "PASS"}
            np.savez_compressed(output / f"{sample.record.sample_id}__{name}__{alpha:g}.npz",
                                logits=modified_logits, action_token_ids=modified.action_token_ids)
            rows.append(row)
            print(f"{sample.record.sample_id} {name} alpha={alpha:g} delta_z={delta[2]:.8g}", flush=True)
    return rows


def run(args: argparse.Namespace, info: dict[str, Any], calibration: list[Sample],
        evaluation: list[Sample]) -> None:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", info)
    try:
        write_json(output / "checkpoint_hashes.json", checkpoint_hashes(args.pretrained_checkpoint))
        tex3d_openvla_root = args.tex3d_openvla_root.resolve()
        prepare_tex3d_import_path(tex3d_openvla_root)
        runtime = smoke._load_runtime(tex3d_openvla_root)
        torch = runtime.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; preflight does not require a GPU")
        runtime.set_seed_everywhere(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_cfg = SimpleNamespace(model_family="openvla", pretrained_checkpoint=str(args.pretrained_checkpoint),
                                    load_in_8bit=False, load_in_4bit=False,
                                    unnorm_key=smoke.UNNORM_KEY, center_crop=True)
        model = runtime.get_model(model_cfg)
        processor = runtime.get_processor(model_cfg)
        runtime.validate_model(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        modules = down_projections(model)
        candidates = discover_candidates(model, processor.tokenizer, batch_size=args.projection_batch_size)
        write_json(output / "candidates.json", {"up_token_ids": exact_up_ids(processor.tokenizer,
                    model.language_model.lm_head.weight.shape[0]),
                    "ranking": "up probability mass descending, best up rank, layer, index",
                    "candidates": [candidate_dict(c) for c in candidates]})
        if len(candidates) < K:
            raise ValueError(f"only {len(candidates)} eligible up neurons; require {K}; no fallback")
        up = [c.neuron for c in candidates[:K]]
        controls = random_controls(up, [m.in_features for m in modules], [c.neuron for c in candidates],
                                   repeats=RANDOM_REPEATS, seed=RANDOM_SEED)
        sets = {"up": up, **{f"random_{i}": group for i, group in enumerate(controls)}}
        all_neurons = sorted(set(n for group in sets.values() for n in group), key=lambda n: (n.layer, n.index))
        write_json(output / "selected.json", {key: [asdict(n) for n in group] for key, group in sets.items()})
        moments = Moments()
        clean_records = []
        for sample in calibration:
            prepared = prepare(runtime, model, processor, sample)
            clean = runtime.run_reference(prepared=prepared)
            with ffn_hooks(modules, all_neurons, moments=moments):
                recorded = runtime.run_reference(prepared=prepared)
            report = require_equal(clean, recorded, "calibration record hook")
            clean_records.append({**sample_dict(sample), "action": action_dict(clean), "equivalence": report})
            print(f"calibration {sample.record.sample_id}", flush=True)
        std_values = moments.std()
        std = dict(zip(all_neurons, std_values.tolist()))
        conditions = {"up": (up, np.asarray([std[n] for n in up], dtype=np.float32))}
        for name, group in sets.items():
            if name != "up":
                conditions[name] = (group, matched_offsets(modules, up, group, std))
        write_json(output / "calibration.json", {"count_token_positions": moments.count,
                    "neurons": [asdict(n) for n in all_neurons], "mean": moments.mean.tolist(),
                    "population_std": std_values.tolist(), "clean_records": clean_records,
                    "condition_offsets": {name: offsets.tolist() for name, (_, offsets) in conditions.items()}})
        evidence = output / "logits"
        evidence.mkdir()
        rows = []
        for sample in evaluation:
            prepared = prepare(runtime, model, processor, sample)
            rows.extend(evaluate_sample(runtime, prepared, modules, all_neurons, conditions, sample, evidence))
            write_json(output / "paired_results.json", rows)
        summary = summarize_rows(rows)
        contrasts = paired_contrasts(rows)
        # 完成前检查所有 source 输入与 tracked code 身份，失败不发布 COMPLETE。
        for sample in calibration + evaluation:
            if sha256_file(sample.path) != sample.archive_sha256:
                raise ValueError("source observation changed during run")
        if sha256_file(args.collection_manifest) != info["manifest_sha256"]:
            raise ValueError("source collection manifest changed during run")
        if repository_identity(PROJECT_ROOT) != info["repository"]:
            raise ValueError("project revision changed during run")
        if repository_identity(args.tex3d_openvla_root.resolve().parent) != info["tex3d_repository"]:
            raise ValueError("Tex3D revision changed during run")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        result = {"schema": SCHEMA, "engineering_status": "COMPLETE",
                    "scientific_status": "DESCRIPTIVE_ONLY_REQUIRES_REVIEW", "runtime": runtime.versions,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "attention_implementation": getattr(model.language_model.config, "_attn_implementation", None),
                    "generation_config": model.language_model.generation_config.to_dict(),
                    "summary": summary, "paired_contrasts": contrasts,
                    "paired_rows": len(rows), "rollout_executed": False}
        lines = ["# UP concept pilot", "", "Engineering: COMPLETE. Scientific result: descriptive only.",
                 "", "Primary readout: decoded action z, not measured robot displacement.",
                 "Evaluation contains two trajectory groups; no significance/transfer claim.", "",
                 "| condition | alpha | mean delta z | positive groups | token-change observations |",
                 "|---|---:|---:|---:|---:|"]
        lines += [f"| {s['condition']} | {s['alpha']:g} | {s['mean_delta_z']:.8g} | "
                  f"{s['positive_groups']}/{s['trajectory_groups']} | {s['action_changed_observations']} |"
                  for s in summary]
        lines += ["", "UP minus mean random control (paired within each observation):", ""]
        lines += [f"- alpha={c['alpha']:g}: {c['mean_up_minus_random_mean']:.8g}; "
                  f"positive contrast groups {c['positive_contrast_groups']}/{c['trajectory_groups']}"
                  for c in contrasts]
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 最后发布完成标记；任一上游证据/摘要写入失败均不会留下 COMPLETE。
        write_json(output / "results.json", result)
        print(f"UP PILOT COMPLETE: {output}", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "error": str(error),
                                             "traceback": traceback.format_exc()})
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--tex3d-openvla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head")
    parser.add_argument("--projection-batch-size", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.projection_batch_size < 1:
        parser.error("--projection-batch-size must be positive")
    for name in ("collection_manifest", "pretrained_checkpoint", "tex3d_openvla_root", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    info, calibration, evaluation = preflight(args)
    if args.preflight_only:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print("PREFLIGHT PASSED: no model loaded, no output directory created")
        return 0
    run(args, info, calibration, evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
