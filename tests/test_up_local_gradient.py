"""CPU 工程诊断：真实 broadcast hook/autograd/读出路径，非 OpenVLA 科学结果。"""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import up_local_gradient as runner
from shared_feature import up_local_gradient as local
from shared_feature.up_concept import Neuron, down_projections


def test_broadcast_gradient_sums_all_tokens_and_repeated_calls():
    module = torch.nn.Linear(2, 1, bias=False)
    module.weight.requires_grad_(False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[2., -3.]]))
    delta = torch.zeros(2, requires_grad=True)
    x = torch.ones(1, 4, 2)
    with local.broadcast_hooks([module], {0: delta}) as stats:
        objective = module(x).sum() + module(x[:, :1]).sum()
        gradient, = torch.autograd.grad(objective, [delta])
    torch.testing.assert_close(gradient, torch.tensor([10., -15.]))
    assert gradient.shape == (2,) and stats['0']['calls'] == 2
    assert torch.equal(x, torch.ones_like(x))
    assert not module._forward_pre_hooks
    with pytest.raises(ValueError, match='delta must'):
        with local.broadcast_hooks([module], {0: torch.zeros(4, 2)}):
            module(x)
    assert not module._forward_pre_hooks
    with pytest.raises(ValueError, match='missing'):
        with local.broadcast_hooks([module], {0: delta}):
            pass
    assert not module._forward_pre_hooks


def test_later_hook_preserves_earlier_delta_gradient():
    modules = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
    for module, weight in zip(modules, [2., 3.]):
        with torch.no_grad():
            module.weight.fill_(weight)
        module.requires_grad_(False)
    deltas = {i: torch.zeros(1, requires_grad=True) for i in range(2)}
    with local.broadcast_hooks(modules, deltas):
        objective = modules[1](modules[0](torch.ones(1, 4, 1))).sum()
        gradients = torch.autograd.grad(objective, list(deltas.values()))
    assert [float(g) for g in gradients] == [24., 12.]


def test_measures_actual_bfloat16_residual_and_zero_rounding():
    module = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2))
    x = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    offsets = {0: np.array([1e-5, 0], dtype=np.float32)}
    with torch.no_grad(), local.broadcast_hooks([module], runner.arrays_to_tensors([module], offsets), measure=True) as stats:
        module(x)
    assert local.residual_norm([module], offsets) > 0
    assert local.measured_norm(stats) == 0  # native rounding erased the intended shift
    with torch.no_grad(), local.broadcast_hooks([module], {0: torch.tensor([.1, -.2])}, measure=True) as stats:
        actual = module(x) - x
    assert local.measured_norm(stats) == pytest.approx(float(actual.float().norm(dim=-1).mean()))
    assert stats['0']['token_rows'] == 3
    assert not module._forward_hooks and not module._forward_pre_hooks


def test_directional_D_fixed_masks_and_boundaries():
    ids, values = np.array([2, 3, 4, 5]), np.array([1., 1., 0., -1.])
    logits = torch.zeros(7, 6, requires_grad=True)
    d = local.directional_D(logits, ids, values, 0.)
    assert float(d) == pytest.approx(np.log(2))  # duplicate endpoint preserved
    grad, = torch.autograd.grad(d, [logits])
    torch.testing.assert_close(grad[2], torch.tensor([0., 0., .5, .5, 0., -1.]))
    assert local.directional_D(logits, ids, values, -1.) is None
    assert local.directional_D(logits, ids, values, 1.) is None


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, boundary=False):
        super().__init__()
        self.ups = torch.nn.ModuleList([torch.nn.Linear(4, 20, bias=False) for _ in range(2)])
        self.downs = torch.nn.ModuleList([torch.nn.Linear(20, 4, bias=False) for _ in range(2)])
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=m)) for m in self.downs])
        self.lm_head = torch.nn.Linear(4, 24)
        with torch.no_grad():
            self.lm_head.weight.mul_(.2)
            self.lm_head.bias.fill_(-4.)
            self.lm_head.bias[14:20] = 0.
            self.lm_head.bias[19 if boundary else 17] = 1.

    def forward(self, inputs_embeds, **kwargs):
        x = inputs_embeds
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        for up, down in zip(self.ups, self.downs):
            # causal token mixing makes early-token gradients contribute to z
            mixed = x.cumsum(dim=1) / denominator
            x = x + down(torch.tanh(up(mixed)))
        return SimpleNamespace(logits=self.lm_head(x))


class TinyModel(torch.nn.Module):
    def __init__(self, boundary=False):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = torch.nn.Embedding(24, 4)
        self.language_model = TinyLanguageModel(boundary)
        self.bins = np.linspace(-1, 1, 6)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.vocab_size = 20
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding


def synthetic_fixture(monkeypatch, tmp_path, boundary=False):
    model = TinyModel(boundary)
    with torch.inference_mode():
        text = model.embedding(torch.tensor([[1, 2, 3]]))
        o2 = torch.zeros(1, 2, 4)
        mask = torch.ones(1, 5, dtype=torch.long)
        input_ids = torch.tensor([[1, 2, 3]])
    prepared = SimpleNamespace(_model=model, _text_embeddings=text, o2=o2,
                              _multimodal_attention_mask=mask, _input_ids=input_ids,
                              _runtime=SimpleNamespace(torch=torch))
    ids, values = runner.action_mapping(model)
    lookup = dict(zip(ids.tolist(), values.tolist()))

    def reference(*, prepared):
        with torch.inference_mode():
            base = torch.cat([text[:, :1], o2, text[:, 1:]], dim=1)
            tokens = []
            for _ in range(7):
                token = int(model.language_model(inputs_embeds=base).logits[0, -1].argmax())
                tokens.append(token)
                base = torch.cat([base, model.embedding(torch.tensor([[token]]))], dim=1)
            normalized = np.array([[lookup[t] for t in tokens]])
        return SimpleNamespace(action_token_ids=np.array([tokens]), normalized_action=normalized,
                               unnormalized_action=normalized.copy(), deployed_action=normalized.copy())

    runtime = SimpleNamespace(run_reference=reference, continue_from_o2=lambda **kw: reference(prepared=kw['prepared']))
    # Retain the historical aligned-logit builder; only relax its fixed O2 size for this tiny model.
    monkeypatch.setattr(runner.pilot.diagnostic.intervention, '_validate_o2', lambda value, torch: None)
    monkeypatch.setattr(runner.pilot, 'prepare', lambda *args: prepared)
    record = SimpleNamespace(sample_id='synthetic_boundary' if boundary else 'synthetic_interior', initial_state_id=0,
                             step_id=0, normalized_episode_progress=0., prompt='synthetic CPU fixture')
    sample = SimpleNamespace(record=record, path=tmp_path / 'synthetic.npz', archive_sha256='synthetic', image_sha256='synthetic')
    candidates = [Neuron(l, i) for i in range(8) for l in range(2)]
    modules = down_projections(model)
    calibration = {'neurons': [dict(layer=n.layer, index=n.index) for n in candidates], 'std': [.1] * len(candidates)}
    lexical = runner.make_reference(modules, candidates, calibration)
    norms = local.column_norms(modules, [0, 1])
    return runtime, model, prepared, sample, candidates, lexical, norms, ids, values


def finite_difference_diagnostic(monkeypatch, tmp_path):
    runtime, model, prepared, _, candidates, lexical, norms, ids, values = synthetic_fixture(monkeypatch, tmp_path)
    clean = runtime.run_reference(prepared=prepared)
    baseline = runner.pilot.logits(prepared, clean)
    anchor = clean.normalized_action[0, 2]
    modules = down_projections(model)
    gradients, _ = runner.gradient_at_zero(prepared, clean, modules, [0, 1], ids, values, anchor, baseline)
    candidate, support = local.sparse_direction(gradients, norms, candidates)
    broad, broad_support = local.sparse_direction(gradients, norms, None, counts=dict(Counter(n.layer for n in support)))
    budget = local.residual_norm(modules, lexical)
    results = []
    for name, offsets in [('candidate', candidate), ('broad', broad)]:
        direction = local.normalize(modules, offsets, budget)
        assert sum(np.count_nonzero(v) for v in direction.values()) == 10
        assert local.residual_norm(modules, direction) == pytest.approx(budget, rel=1e-5)
        prediction = local.prediction(gradients, direction)
        assert prediction > 0
        d0 = float(local.directional_D(torch.from_numpy(baseline), ids, values, anchor))
        pair = []
        for sign in (1, -1):
            with torch.no_grad(), local.broadcast_hooks(modules, runner.arrays_to_tensors(modules, direction, sign * .05)):
                dz = float(local.directional_D(local.aligned_logits(prepared, clean.action_token_ids), ids, values, anchor)) - d0
            assert sign * dz > 0
            pair.append(dz)
        finite = (pair[0] - pair[1]) / .1
        assert finite == pytest.approx(prediction, rel=.005, abs=1e-6)
        results.append(dict(pool=name, predicted_slope=prediction, central_difference_slope=finite,
                            relative_error=abs(finite / prediction - 1), plus_delta_D=pair[0], minus_delta_D=pair[1]))
    assert set(support) <= set(candidates)
    assert Counter(n.layer for n in support) == Counter(n.layer for n in broad_support)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    return results


def test_inference_tensor_cloning_and_finite_difference(monkeypatch, tmp_path):
    finite_difference_diagnostic(monkeypatch, tmp_path)


@pytest.mark.parametrize('boundary', [False, True])
def test_complete_observation_native_readouts_restoration_and_boundary(monkeypatch, tmp_path, boundary):
    runtime, model, _, sample, candidates, lexical, norms, ids, values = synthetic_fixture(monkeypatch, tmp_path, boundary)
    if boundary:
        monkeypatch.setattr(runner, 'gradient_at_zero', lambda *args: pytest.fail('boundary must not construct gradients'))
    evidence = runner.evaluate_observation(runtime, model, None, sample, candidates, lexical, norms, ids, values, tmp_path)
    assert len(evidence['rows']) == (4 if boundary else 12)
    assert not evidence['unavailable_probes']
    assert evidence['shared_direction_constructed'] is False
    for row in evidence['rows']:
        assert len(row['native_action_delta_all7']) == 7 and len(row['other6_delta']) == 6
        assert row['original_interface_equivalence'] == row['restoration'] == 'PASS'
        assert all(s['calls'] == 7 for s in row['native_hooks'].values())
        assert row['actual_teacher_forced_to_intended_ratio'] == pytest.approx(1., rel=.002)
        if boundary:
            assert row['delta_D'] is None and row['predicted_delta_D'] is None
    if not boundary:
        for entry in runner.summarize([evidence]):
            if entry['condition'] != 'lexical':
                assert entry['D_both_count'] == 1
    assert not list(tmp_path.glob('*frozen*'))
    with np.load(tmp_path / (sample.record.sample_id + '.npz')) as arrays:
        assert any(k.startswith('local_gradient_') for k in arrays.files) != boundary
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in down_projections(model))


def test_zero_gradient_is_unavailable_not_a_negative_scientific_conclusion(monkeypatch, tmp_path):
    runtime, model, _, sample, candidates, lexical, norms, ids, values = synthetic_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, 'gradient_at_zero', lambda *args: ({l: np.zeros(20) for l in [0, 1]}, {}))
    evidence = runner.evaluate_observation(runtime, model, None, sample, candidates, lexical, norms, ids, values, tmp_path)
    assert set(evidence['unavailable_probes']) == {'candidate_gradient', 'broad_gradient'}
    assert {r['condition'] for r in evidence['rows']} == {'lexical'}
    assert all(r['valid_observations'] == 0 for r in runner.summarize([evidence]) if r['condition'] != 'lexical')


def test_failure_persists_reason_without_complete_result(monkeypatch, tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / 'output', pretrained_checkpoint=tmp_path)
    def fail(_):
        raise RuntimeError('synthetic checkpoint failure')
    monkeypatch.setattr(runner.pilot, 'checkpoint_hashes', fail)
    with pytest.raises(RuntimeError, match='synthetic checkpoint'):
        runner.run(args, {}, [])
    assert (args.output_dir / 'failure.json').is_file()
    assert not (args.output_dir / 'results.json').exists()


def test_preflight_reuses_only_original_calibration_and_rejects_changed_source(monkeypatch, tmp_path):
    sample = SimpleNamespace(record=SimpleNamespace(sample_id='old'), archive_sha256='same')
    args = SimpleNamespace(screen_dir=tmp_path / 'screen', v1_dir=tmp_path / 'v1')
    args.screen_dir.mkdir()
    args.v1_dir.mkdir()
    for name, data in {'protocol.json': {'schema': runner.screen.SCHEMA, 'v1_hashes': {'fixed': 'hash'}},
                       'results.json': {'engineering_status': 'COMPLETE', 'stage': 'screen'},
                       'calibration.json': {'clean_records': [{'sample_id': 'old', 'archive_sha256': 'same'}]},
                       'checkpoint_hashes.json': {}}.items():
        runner.pilot.write_json(args.screen_dir / name, data)
    runner.pilot.write_json(args.v1_dir / 'candidates.json', {'candidates': [{'neuron': {'layer': 0, 'index': 1}}]})
    def prior_preflight(args):
        assert args.stage == 'screen' and args.validation_manifest is None and args.selection_dir is None
        return {'v1_hashes': {'fixed': 'hash'}}, [sample], []
    monkeypatch.setattr(runner.screen, 'preflight', prior_preflight)
    info, samples = runner.preflight(args)
    assert samples == [sample] and info['protocol']['scope_experiment'] is False
    runner.pilot.write_json(args.screen_dir / 'calibration.json', {'clean_records': [{'sample_id': 'new', 'archive_sha256': 'same'}]})
    with pytest.raises(ValueError, match='observations changed'):
        runner.preflight(args)
