"""UP runner 的身份、边界与整条 CPU 合成执行验证。"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import up_concept_pilot as pilot
from shared_feature.pilot_observation import PilotObservation
from shared_feature.up_concept import Candidate, Neuron


def samples(root: Path) -> list[pilot.Sample]:
    result = []
    for state in range(5):
        for step in (1, 4, 7, 9):
            record = PilotObservation(
                sample_id=f"libero_spatial__task00__state{state:02d}__step{step:04d}",
                task_id="0", initial_state_id=state, episode_id=state, step_id=step,
                normalized_episode_progress=step / 10, prompt=pilot.TASK0_PROMPT,
                base_rgb_raw=np.ones((2, 2, 3), dtype=np.uint8) * state,
                wrist_rgb_raw=np.zeros((2, 2, 3), dtype=np.uint8),
                state=np.arange(8, dtype=np.float32), episode_success=True,
            )
            path = root / f"{record.sample_id}.npz"
            record.save(path)
            result.append(pilot.Sample(path, record, pilot.sha256_file(path), "image-hash"))
    return result


def test_selection_preserves_complete_groups_and_progress(tmp_path, monkeypatch):
    records = samples(tmp_path)
    entries = [SimpleNamespace(sample_id=s.record.sample_id, resolved_observation_path=s.path,
                               source_image_hash=s.image_sha256) for s in reversed(records)]
    monkeypatch.setattr(pilot.collection, "load_source_collection", lambda _: SimpleNamespace(records=entries))
    calibration, evaluation = pilot.load_samples(tmp_path / "manifest.json")
    assert len(calibration) == 12 and len(evaluation) == 8
    assert {s.record.initial_state_id for s in calibration} == {0, 1, 2}
    assert {s.record.initial_state_id for s in evaluation} == {3, 4}
    assert [s.record.step_id for s in evaluation[:4]] == [1, 4, 7, 9]
    entries.pop()
    with pytest.raises(ValueError, match="five complete"):
        pilot.load_samples(tmp_path / "manifest.json")


def test_preflight_creates_no_outputs_and_never_loads_model(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    tex3d = tmp_path / "tex3d/openvla"
    (tex3d / "experiments/robot").mkdir(parents=True)
    (tex3d / "experiments/robot/openvla_utils.py").write_text("")
    monkeypatch.setattr(pilot.smoke, "_validate_runtime_paths", lambda _: (checkpoint, tex3d))
    monkeypatch.setattr(pilot, "load_samples", lambda _: ([], []))
    monkeypatch.setattr(pilot, "repository_identity", lambda _: {"head": "a" * 40})
    versions = {"torch": "2.2.0+cu121", "torchvision": "0.17.0+cu121", "transformers": "4.40.1",
                "tokenizers": "0.19.1", "numpy": "1.26.4"}
    monkeypatch.setattr(pilot.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(pilot.smoke, "_load_runtime", lambda _: pytest.fail("model runtime called"))
    argv = ["--collection-manifest", str(manifest), "--pretrained-checkpoint", str(checkpoint),
            "--tex3d-openvla-root", str(tex3d), "--output-dir", str(tmp_path / "new-parent/out"),
            "--expected-head", "a" * 40, "--preflight-only"]
    assert pilot.main(argv) == 0
    assert not (tmp_path / "new-parent").exists()
    versions["transformers"] = "4.53.2"
    with pytest.raises(ValueError, match="authoritative"):
        pilot.main(argv)
    versions["transformers"] = "4.40.1"
    (tmp_path / "new-parent/out").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        pilot.main(argv)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Linear(32, 4, bias=False)
        with torch.no_grad(): self.down.weight.fill_(0.1)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=self.down))]),
            lm_head=SimpleNamespace(weight=torch.zeros(12, 4)),
            config=SimpleNamespace(_attn_implementation="synthetic"),
            generation_config=SimpleNamespace(to_dict=lambda: {"use_cache": True}),
        )


def fake_runtime(model: FakeModel):
    def prepare_context(**kwargs):
        return SimpleNamespace(_model=model, o2=torch.zeros(1), marker=float(kwargs["observation"]["marker"]))
    def forward(prepared):
        x = torch.arange(96, dtype=torch.float32).reshape(1, 3, 32) / 100 + prepared.marker
        with torch.inference_mode():
            first = model.down(x)
            second = model.down(x[:, :1] + 0.1)
        return float(first.mean() + second.mean())
    def reference(*, prepared):
        marker = forward(prepared)
        action = np.zeros((1, 7)); action[0, 2] = round(marker, 4)
        ids = np.ones((1, 7), dtype=np.int64); ids[0, 2] = int(round(marker * 1000))
        return SimpleNamespace(action_token_ids=ids, normalized_action=action.copy(),
                               unnormalized_action=action.copy(), deployed_action=action.copy())
    def logits(prepared, clean):
        marker = forward(prepared)
        result = np.tile(np.arange(12, dtype=np.float32), (7, 1))
        result[:, 3] += marker
        return result
    runtime = SimpleNamespace(
        torch=torch, get_model=lambda _: model, get_processor=lambda _: SimpleNamespace(tokenizer=None),
        validate_model=lambda _: None, set_seed_everywhere=lambda _: None,
        prepare_context=prepare_context,
        build_policy_observation=lambda r: {"marker": r.initial_state_id / 100},
        run_reference=reference, continue_from_o2=lambda *, prepared, o2: reference(prepared=prepared),
        versions={"torch": "synthetic-cpu"},
    )
    return runtime, logits


def setup_run(tmp_path, monkeypatch):
    records = samples(tmp_path)
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    model = FakeModel()
    runtime, logits = fake_runtime(model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # no actual GPU call in this fake
    monkeypatch.setattr(pilot.smoke, "_load_runtime", lambda _: runtime)
    monkeypatch.setattr(pilot, "logits", logits)
    monkeypatch.setattr(pilot, "checkpoint_hashes", lambda _: {"weights": "synthetic"})
    monkeypatch.setattr(pilot, "repository_identity", lambda _: {"head": "a" * 40})
    monkeypatch.setattr(pilot, "exact_up_ids", lambda *a: [1])
    candidates = [Candidate(Neuron(0, i), 1.0, 1, (1,), ("up",)) for i in range(10)]
    monkeypatch.setattr(pilot, "discover_candidates", lambda *a, **kw: candidates)
    args = SimpleNamespace(output_dir=tmp_path / "out", pretrained_checkpoint=tmp_path / "checkpoint",
                           tex3d_openvla_root=tmp_path / "tex3d/openvla", collection_manifest=manifest,
                           projection_batch_size=128)
    info = {"repository": {"head": "a" * 40}, "tex3d_repository": {"head": "a" * 40},
            "manifest_sha256": pilot.sha256_file(manifest)}
    return args, info, records, model


def test_complete_cpu_pipeline_writes_reviewable_evidence(tmp_path, monkeypatch):
    args, info, records, model = setup_run(tmp_path, monkeypatch)
    pilot.run(args, info, records[:12], records[12:])
    result = json.loads((args.output_dir / "results.json").read_text())
    assert result["engineering_status"] == "COMPLETE"
    assert result["scientific_status"] == "DESCRIPTIVE_ONLY_REQUIRES_REVIEW"
    assert result["paired_rows"] == 64
    assert not result["rollout_executed"]
    assert len(list((args.output_dir / "logits").glob("*.npz"))) == 72
    assert not model.down._forward_pre_hooks
    rows = json.loads((args.output_dir / "paired_results.json").read_text())
    assert all(row["hook_restoration"] == "PASS" for row in rows)
    assert all(row["applied_generation"]["changed_values"] > 0 for row in rows)
    assert all(s["trajectory_groups"] == 2 for s in result["summary"])
    assert (args.output_dir / "summary.csv").is_file()
    assert (args.output_dir / "summary.md").is_file()
    with pytest.raises(FileExistsError):
        pilot.run(args, info, records[:12], records[12:])


def test_candidate_shortage_keeps_evidence_and_never_publishes_complete(tmp_path, monkeypatch):
    args, info, records, model = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot, "discover_candidates", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="no fallback"):
        pilot.run(args, info, records[:12], records[12:])
    assert (args.output_dir / "candidates.json").is_file()
    assert (args.output_dir / "failure.json").is_file()
    assert not (args.output_dir / "results.json").exists()
    assert not model.down._forward_pre_hooks


def test_failure_inside_intervention_cleans_hooks_and_stops(tmp_path, monkeypatch):
    args, info, records, model = setup_run(tmp_path, monkeypatch)
    base_logits = pilot.logits
    def failing_logits(prepared, clean):
        if model.down._forward_pre_hooks:
            raise RuntimeError("intentional diagnostic failure")
        return base_logits(prepared, clean)
    monkeypatch.setattr(pilot, "logits", failing_logits)
    with pytest.raises(RuntimeError, match="intentional"):
        pilot.run(args, info, records[:12], records[12:])
    assert not model.down._forward_pre_hooks
    assert (args.output_dir / "failure.json").is_file()
    assert not (args.output_dir / "results.json").exists()


def test_summary_write_failure_does_not_publish_complete(tmp_path, monkeypatch):
    args, info, records, model = setup_run(tmp_path, monkeypatch)
    original = Path.write_text
    def fail_summary(path, *a, **kw):
        if path.name == "summary.md":
            raise OSError("injected summary write failure")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "write_text", fail_summary)
    with pytest.raises(OSError, match="summary write"):
        pilot.run(args, info, records[:12], records[12:])
    assert not (args.output_dir / "results.json").exists()
    assert (args.output_dir / "failure.json").is_file()
    assert not model.down._forward_pre_hooks


def test_shell_preflight_uses_no_gpu_and_does_not_create_output(tmp_path):
    root = tmp_path / "repo with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "shared_feature").mkdir()
    source = Path(pilot.__file__).with_name("up_concept_server_run.sh")
    script = root / "scripts/up_concept_server_run.sh"
    script.write_text(source.read_text())
    (root / "scripts/up_concept_pilot.py").write_text("# stub")
    (root / "shared_feature/up_concept.py").write_text("# stub")
    for command in (["git", "init", "-q", str(root)],
                    ["git", "-C", str(root), "add", "."],
                    ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                     "commit", "-qm", "fixture"]):
        subprocess.run(command, check=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    python = tmp_path / "python stub"
    python.write_text('#!/usr/bin/env bash\n[[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "$CUDA_VISIBLE_DEVICES" ]] || exit 91\n[[ " ${*} " == *" --preflight-only "* ]] || exit 92\necho PREFLIGHT_STUB\n')
    python.chmod(0o755)
    tex3d = tmp_path / "tex3d"; (tex3d / "openvla").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    env = {**os.environ, "OPENVLA_PY": str(python), "TEX3D_ROOT": str(tex3d),
           "OPENVLA_CKPT": str(checkpoint), "COLLECTION_MANIFEST": str(manifest),
           "UP_OUTPUT_ROOT": str(tmp_path / "output"), "UP_RUN_ID": "trial"}
    env.pop("GPU_ID", None)
    done = subprocess.run(["bash", str(script), head, "--preflight-only"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "PREFLIGHT_STUB" in done.stdout
    assert not (tmp_path / "output").exists()
    missing_gpu = subprocess.run(["bash", str(script), head, "--run"], env=env, capture_output=True, text=True)
    assert missing_gpu.returncode != 0 and "GPU_ID" in missing_gpu.stderr
    assert not (tmp_path / "output").exists()
    wrong_head = subprocess.run(["bash", str(script), "0" * 40], env=env, capture_output=True, text=True)
    assert wrong_head.returncode != 0 and "HEAD" in wrong_head.stderr
    (tmp_path / "output/trial").mkdir(parents=True)
    repeated = subprocess.run(["bash", str(script), head], env=env, capture_output=True, text=True)
    assert repeated.returncode != 0 and "refusing resume" in repeated.stderr
