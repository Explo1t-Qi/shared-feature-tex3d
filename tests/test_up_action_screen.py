"""检查方向读出、稳健筛选、冻结验证边界及真实 hook 的合成运行。"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import up_action_screen as runner
from shared_feature.up_action_screen import action_mapping, check_decoding, rank_candidates, z_readout
from shared_feature.up_concept import Neuron
from shared_feature.pilot_observation import PilotObservation


def mapping_model():
    bins = np.linspace(-1, 1, 6)
    return SimpleNamespace(bins=bins, bin_centers=(bins[:-1] + bins[1:]) / 2, vocab_size=20)


def test_mapping_preserves_duplicate_endpoint_and_real_decode_direction():
    ids, values = action_mapping(mapping_model())
    assert ids.tolist() == list(range(14, 20))
    np.testing.assert_allclose(values, [.8, .8, .4, 0, -.4, -.8], atol=1e-15)
    logits = np.zeros((7, 24))
    baseline = z_readout(logits, ids, values)
    assert baseline["action_probability_mass"] == pytest.approx(6 / 24)
    logits[2, 14] = 4
    assert z_readout(logits, ids, values)["expected_normalized_z"] > baseline["expected_normalized_z"]
    logits[2, 14] = 0
    logits[2, 19] = 4
    assert z_readout(logits, ids, values)["expected_normalized_z"] < baseline["expected_normalized_z"]
    clean = SimpleNamespace(action_token_ids=np.full((1, 7), 14), normalized_action=np.full((1, 7), .8))
    check_decoding(clean, ids, values)
    clean.normalized_action[:] = -.8
    with pytest.raises(ValueError, match="disagrees"):
        check_decoding(clean, ids, values)


def score_rows(index=0, plus=.02, minus=-.01):
    return [dict(layer=0, index=index, state_id=state, sample_id=f"{state}/{frame}",
                 plus_delta_z=plus, minus_delta_z=minus)
            for state in range(3) for frame in range(4)]


def test_score_rejects_single_frame_outlier_unidirectional_and_one_bad_trajectory():
    good = score_rows()
    outlier = score_rows(1, 0, 0)
    for row in outlier:
        if row["sample_id"].endswith("/0"):
            row.update(plus_delta_z=100, minus_delta_z=-100)
    wrong = score_rows(2, .1, .1)
    bad_group = score_rows(3)
    for row in bad_group:
        if row["state_id"] == 2:
            row["plus_delta_z"] = -.01
    result = rank_candidates(good + outlier + wrong + bad_group, k=1)
    assert result["selected"] == [{"layer": 0, "index": 0}]
    assert result["eligible_count"] == 1
    assert rank_candidates(good)["selection_status"] == "INSUFFICIENT_CANDIDATES"
    with pytest.raises(ValueError, match="duplicate"):
        rank_candidates(good + [good[0]])
    with pytest.raises(ValueError, match="different"):
        rank_candidates(good + score_rows(1)[:-1])


def make_samples(root, states):
    samples = []
    for state in states:
        for step in range(4):
            record = PilotObservation(
                sample_id=f"libero_spatial__task00__state{state:02d}__step{step:04d}", task_id="0",
                initial_state_id=state, episode_id=state, step_id=step,
                normalized_episode_progress=step / 4, prompt=runner.pilot.TASK0_PROMPT,
                base_rgb_raw=np.full((512, 512, 3), state * 4 + step, dtype=np.uint8),
                wrist_rgb_raw=np.zeros((512, 512, 3), dtype=np.uint8), state=np.ones(8), episode_success=True)
            path = root / (record.sample_id + ".npz")
            record.save(path)
            samples.append(runner.pilot.Sample(path, record, runner.pilot.sha256_file(path), "hash"))
    return samples


def validation_manifest(root, samples):
    data = {"schema": "up_action_validation_observations_v1",
            "provenance": {"checkpoint_identity": runner.pilot.smoke.CHECKPOINT_IDENTITY,
                           "task_suite": "libero_spatial", "task_id": 0,
                           "collection_description": "synthetic test", "not_used_for_selection": True},
            "observations": [{"path": s.path.name, "sha256": s.archive_sha256} for s in samples]}
    path = root / "validation.json"
    runner.pilot.write_json(path, data)
    return path


def test_validation_rejects_old_states_duplicates_and_hash_tampering(tmp_path):
    old = make_samples(tmp_path, [0, 1, 2, 3, 4])
    new = make_samples(tmp_path, [5, 6])
    path = validation_manifest(tmp_path, new)
    assert len(runner.load_validation(path, old)) == 8
    with pytest.raises(ValueError, match="old or duplicate"):
        runner.load_validation(validation_manifest(tmp_path, old[:4] + new[4:]), old)
    with pytest.raises(ValueError, match="old or duplicate"):
        runner.load_validation(validation_manifest(tmp_path, new + new[:1]), old)
    path = validation_manifest(tmp_path, new)
    data = runner.read_json(path)
    data["observations"][0]["sha256"] = "wrong"
    runner.pilot.write_json(path, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_validation(path, old)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Linear(64, 3, bias=False)
        with torch.no_grad():
            self.down.weight.fill_(.1)
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=self.down))]))
        self.bins = np.linspace(-1, 1, 6)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.vocab_size = 20


def synthetic_screen(tmp_path, monkeypatch, signal=True):
    samples = make_samples(tmp_path, [0, 1, 2])
    model = TinyModel()
    out = tmp_path / "screen"
    out.mkdir()
    source = tmp_path / "v1"
    source.mkdir()
    runner.pilot.write_json(source / "candidates.json", {"candidates": [{"neuron": {"layer": 0, "index": i}} for i in range(12)]})
    args = SimpleNamespace(output_dir=out, v1_dir=source)
    def forward(prepared):
        with torch.no_grad():
            x = torch.arange(128).float().reshape(1, 2, 64) / 128
            return float(model.down(x).mean())
    def reference(*, prepared):
        forward(prepared)
        action = np.full((1, 7), model.bin_centers[3])
        return SimpleNamespace(action_token_ids=np.full((1, 7), 16), normalized_action=action,
                               unnormalized_action=action, deployed_action=action)
    def logits(prepared, clean):
        value = forward(prepared)
        result = np.zeros((7, 24))
        result[2, 14] = value if signal else 0
        return result
    monkeypatch.setattr(runner.pilot, "prepare", lambda *a: SimpleNamespace(o2=None))
    monkeypatch.setattr(runner.pilot, "logits", logits)
    monkeypatch.setattr(runner.pilot, "require_equal", lambda a, b, label: {"pass": True})
    runtime = SimpleNamespace(run_reference=reference, continue_from_o2=lambda **kw: reference(prepared=kw["prepared"]))
    return args, runtime, model, samples


def test_real_hooks_screen_freezes_only_calibration_evidence(tmp_path, monkeypatch):
    args, runtime, model, samples = synthetic_screen(tmp_path, monkeypatch)
    result = runner.screen(args, runtime, model, None, samples)
    assert result["selection_status"] == "FROZEN"
    frozen = runner.read_json(args.output_dir / "frozen.json")
    assert len(frozen["sets"]["selected"]) == 10
    assert len(frozen["sets"]) == 8
    norms = list(frozen["ideal_shift_norms"].values())
    np.testing.assert_allclose(norms, norms[0], rtol=1e-5)
    assert len(runner.read_json(args.output_dir / "screen_rows.json")) == 12 * 12
    assert not model.down._forward_pre_hooks
    assert not (args.output_dir / "results.json").exists()  # only orchestrator publishes COMPLETE


def test_no_signal_preserves_negative_result_without_forced_top10(tmp_path, monkeypatch):
    args, runtime, model, samples = synthetic_screen(tmp_path, monkeypatch, signal=False)
    result = runner.screen(args, runtime, model, None, samples)
    assert result["selection_status"] == "INSUFFICIENT_CANDIDATES"
    assert not (args.output_dir / "frozen.json").exists()
    assert not model.down._forward_pre_hooks


def test_screen_failure_during_intervention_removes_hooks(tmp_path, monkeypatch):
    args, runtime, model, samples = synthetic_screen(tmp_path, monkeypatch)
    original = runner.pilot.logits
    def failing(prepared, clean):
        if model.down._forward_pre_hooks:
            raise RuntimeError("injected failure")
        return original(prepared, clean)
    monkeypatch.setattr(runner.pilot, "logits", failing)
    with pytest.raises(RuntimeError, match="injected"):
        runner.screen(args, runtime, model, None, samples)
    assert not model.down._forward_pre_hooks
    assert not (args.output_dir / "frozen.json").exists()


def test_validation_reads_frozen_offsets_without_fitting(tmp_path, monkeypatch):
    args, runtime, model, samples = synthetic_screen(tmp_path, monkeypatch)
    runner.screen(args, runtime, model, None, samples)
    args.selection_dir = args.output_dir
    before = runner.pilot.sha256_file(args.selection_dir / "frozen.json")
    args.output_dir = tmp_path / "validate"
    args.output_dir.mkdir()
    new = make_samples(tmp_path, [5, 6])
    monkeypatch.setattr(runner, "calibrate", lambda *a: pytest.fail("validation must not calibrate"))
    monkeypatch.setattr(runner, "rank_candidates", lambda *a: pytest.fail("validation must not rank"))
    result = runner.validate(args, runtime, model, None, new)
    assert result["paired_rows"] == 128
    assert len(result["summary"]["conditions"]) == 16
    assert len(result["summary"]["paired_random_contrasts"]) == 4
    assert len(list((args.output_dir / "logits").glob("*.npz"))) == 136
    assert runner.pilot.sha256_file(args.selection_dir / "frozen.json") == before
    assert not model.down._forward_pre_hooks


def test_run_failure_keeps_failure_record_and_never_complete(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    args = SimpleNamespace(output_dir=output, pretrained_checkpoint=tmp_path,
                           v1_dir=tmp_path, stage="screen")
    monkeypatch.setattr(runner.pilot, "checkpoint_hashes", lambda _: {"weights": "new"})
    monkeypatch.setattr(runner, "read_json", lambda _: {"weights": "different"})
    monkeypatch.setattr(runner, "load_model", lambda _: pytest.fail("must reject before model loading"))
    with pytest.raises(ValueError, match="checkpoint differs"):
        runner.run(args, {}, [], [])
    assert (output / "failure.json").is_file()
    assert not (output / "results.json").exists()
    with pytest.raises(FileExistsError):
        runner.run(args, {}, [], [])


def test_preflight_screen_does_not_read_independent_data(tmp_path, monkeypatch):
    samples = make_samples(tmp_path, [0, 1, 2, 3, 4])
    source = tmp_path / "v1"
    source.mkdir()
    proto = {"checkpoint_identity": runner.pilot.smoke.CHECKPOINT_IDENTITY,
             "calibration": [runner.pilot.sample_dict(s) for s in samples[:12]],
             "evaluation": [runner.pilot.sample_dict(s) for s in samples[12:]]}
    runner.pilot.write_json(source / "protocol.json", proto)
    runner.pilot.write_json(source / "results.json", {"engineering_status": "COMPLETE"})
    for name in ("candidates.json", "checkpoint_hashes.json"):
        runner.pilot.write_json(source / name, {})
    args = SimpleNamespace(v1_dir=source, stage="screen", validation_manifest=None, selection_dir=None)
    monkeypatch.setattr(runner.pilot, "preflight", lambda _: ({"evaluation": []}, samples[:12], samples[12:]))
    monkeypatch.setattr(runner, "load_validation", lambda *a: pytest.fail("screen read independent data"))
    info, calibration, validation = runner.preflight(args)
    assert len(calibration) == 12 and validation == []
    assert "evaluation" not in info
    args.validation_manifest = tmp_path / "do-not-read.json"
    with pytest.raises(ValueError, match="must not access"):
        runner.preflight(args)


def test_bash_preflight_quotes_paths_requires_gpu_and_frozen_inputs(tmp_path):
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / "shared_feature").mkdir()
    source = Path(__file__).resolve().parents[1]
    for name in ("scripts/up_action_server_run.sh", "scripts/up_action_screen.py", "shared_feature/up_action_screen.py"):
        shutil.copyfile(source / name, repo / name)
    for command in (["init", "-q"], ["add", "."],
                    ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(repo), *command], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    stub = tmp_path / "fake python"
    stub.write_text("#!/usr/bin/env bash\nset -eu\n"
                    "[[ ${CUDA_VISIBLE_DEVICES} == '' ]]\n"
                    "[[ ${@: -1} == --preflight-only ]]\n"
                    "printf 'PREFLIGHT_STUB_OK\\n'\n", encoding="utf-8")
    stub.chmod(0o755)
    output = tmp_path / "output with spaces"
    env = {k: v for k, v in os.environ.items()
           if k not in {"GPU_ID", "UP_RUN_ID", "UP_SELECTION_DIR", "UP_VALIDATION_MANIFEST"}}
    env.update(OPENVLA_PY=str(stub), UP_OUTPUT_ROOT=str(output), TEX3D_ROOT=str(tmp_path / "tex3d space"))
    def call(stage="screen", mode="--preflight-only", sha=head):
        return subprocess.run(["bash", str(repo / "scripts/up_action_server_run.sh"), sha, stage, mode],
                              env=env, text=True, capture_output=True)
    assert call().returncode == 0
    assert not output.exists()
    failed = call(mode="--run")
    assert failed.returncode != 0 and "GPU_ID" in failed.stderr
    assert not output.exists()
    assert "HEAD mismatch" in call(sha="0" * 40).stderr
    assert "UP_SELECTION_DIR" in call(stage="validate").stderr


def test_validation_preflight_rejects_modified_frozen_selection(tmp_path, monkeypatch):
    samples = make_samples(tmp_path, [0, 1, 2, 3, 4])
    source, selection = tmp_path / "v1", tmp_path / "selection"
    source.mkdir()
    selection.mkdir()
    runner.pilot.write_json(source / "protocol.json", {
        "checkpoint_identity": runner.pilot.smoke.CHECKPOINT_IDENTITY,
        "calibration": [runner.pilot.sample_dict(s) for s in samples[:12]],
        "evaluation": [runner.pilot.sample_dict(s) for s in samples[12:]]})
    runner.pilot.write_json(source / "results.json", {"engineering_status": "COMPLETE"})
    for name in ("candidates.json", "checkpoint_hashes.json"):
        runner.pilot.write_json(source / name, {})
    runner.pilot.write_json(selection / "protocol.json", {
        "schema": runner.SCHEMA, "stage": "screen", "repository": {"head": "same"},
        "v1_hashes": runner.file_identities(source, runner.SOURCE_FILES)})
    runner.pilot.write_json(selection / "frozen.json", {"original": True})
    runner.pilot.write_json(selection / "results.json", {
        "selection_status": "FROZEN", "engineering_status": "COMPLETE",
        "frozen_sha256": runner.pilot.sha256_file(selection / "frozen.json")})
    runner.pilot.write_json(selection / "frozen.json", {"original": False})
    args = SimpleNamespace(v1_dir=source, stage="validate", selection_dir=selection,
                           validation_manifest=tmp_path / "must-not-read.json")
    monkeypatch.setattr(runner.pilot, "preflight", lambda _: (
        {"evaluation": [], "repository": {"head": "same"}}, samples[:12], samples[12:]))
    monkeypatch.setattr(runner, "load_validation", lambda *a: pytest.fail("read validation before hash check"))
    with pytest.raises(ValueError, match="frozen selection hash"):
        runner.preflight(args)
