"""实验 2：逐帧广播增量的 local gradient positive control，无共享方向学习。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import up_action_screen as screen  # noqa: E402
from scripts import up_concept_pilot as pilot  # noqa: E402
from shared_feature.up_action_screen import action_mapping, check_decoding, z_readout  # noqa: E402
from shared_feature.up_concept import Neuron, down_projections, ffn_hooks  # noqa: E402
from shared_feature import up_local_gradient as local  # noqa: E402

SCHEMA = 'up_local_gradient_diagnostic_v1'
STEPS = (0.05, 0.1)
SCREEN_FILES = ('protocol.json', 'results.json', 'calibration.json', 'checkpoint_hashes.json')


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[pilot.Sample]]:
    # 只复用已验证的旧数据身份核对，不调用 screen 或读取任何新验证数据。
    args.stage, args.selection_dir, args.validation_manifest = 'screen', None, None
    info, samples, _ = screen.preflight(args)
    result = screen.read_json(args.screen_dir / 'results.json')
    source = screen.read_json(args.screen_dir / 'protocol.json')
    if (result.get('engineering_status') != 'COMPLETE' or result.get('stage') != 'screen'
            or source.get('schema') != screen.SCHEMA or (args.screen_dir / 'failure.json').exists()
            or source['v1_hashes'] != info['v1_hashes']):
        raise ValueError('screen source must match completed v1 identities')
    stats = screen.read_json(args.screen_dir / 'calibration.json')
    if [(r['sample_id'], r['archive_sha256']) for r in stats['clean_records']] != [
            (s.record.sample_id, s.archive_sha256) for s in samples]:
        raise ValueError('screen calibration observations changed')
    candidates = screen.read_json(args.v1_dir / 'candidates.json')['candidates']
    layers = sorted({c['neuron']['layer'] for c in candidates})
    info.update(schema=SCHEMA, stage='local_gradient_diagnostic',
                screen_dir=str(args.screen_dir), screen_hashes=screen.file_identities(args.screen_dir, SCREEN_FILES))
    info['protocol'] = {'scope': 'one delta per neuron broadcast to all tokens of every forward call',
                        'observations': '12 existing calibration/development frames only', 'k': 10,
                        'steps': list(STEPS), 'signs': [1, -1], 'candidate_count': len(candidates),
                        'broader_pool_layers': layers, 'direction': 'per_observation_only',
                        'ranking': 'abs(g)/column_norm', 'coefficients': 'g/column_norm_squared',
                        'gradient_pair_layer_counts': 'broad matches candidate local support allocation',
                        'budget': 'sqrt(sum_layer ||W delta||^2), baseline lexical clean std',
                        'boundary': 'no D or gradient probes; lexical and native readouts retained',
                        'no_shared_direction': True, 'scope_experiment': False, 'rollout': False}
    return info, samples


def arrays_to_tensors(modules: Sequence[Any], offsets: dict[int, np.ndarray], scale: float = 1.) -> dict[int, torch.Tensor]:
    return {layer: torch.as_tensor(value * scale, device=modules[layer].weight.device, dtype=torch.float32)
            for layer, value in offsets.items()}


def readout(logits: np.ndarray, ids: np.ndarray, values: np.ndarray, anchor: float) -> dict[str, Any]:
    result = z_readout(logits, ids, values)
    d = local.directional_D(torch.from_numpy(logits), ids, values, anchor)
    token = int(logits[2].argmax())
    lookup = dict(zip(ids.tolist(), values.tolist()))
    result.update(D=None if d is None else float(d), z_argmax_token=token,
                  z_argmax_value=lookup.get(token), z_argmax_is_action=token in lookup)
    return result


def make_reference(modules: Sequence[Any], candidates: list[Neuron], calibration: dict[str, Any]) -> dict[int, np.ndarray]:
    std = {Neuron(**n): float(v) for n, v in zip(calibration['neurons'], calibration['std'])}
    offsets = {l: np.zeros(modules[l].in_features, dtype=np.float32) for l in {n.layer for n in candidates[:10]}}
    for n in candidates[:10]:
        if not np.isfinite(std[n]) or std[n] <= 0:
            raise ValueError('invalid lexical reference std')
        offsets[n.layer][n.index] = std[n]
    return offsets


def gradient_at_zero(prepared: Any, clean: Any, modules: Sequence[Any], layers: list[int],
                     ids: np.ndarray, values: np.ndarray, anchor: float,
                     baseline: np.ndarray) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """只对 delta 求导，冻结模型；zero-hook 必须与普通 clean-prefix logits 精确一致。"""
    with torch.enable_grad():
        delta = {l: torch.zeros(modules[l].in_features, device=modules[l].weight.device,
                                dtype=torch.float32, requires_grad=True) for l in layers}
        with local.broadcast_hooks(modules, delta) as stats:
            output = local.aligned_logits(prepared, clean.action_token_ids)
            if not np.array_equal(output.detach().cpu().numpy(), baseline):
                raise ValueError('zero-delta autograd forward differs from clean logits')
            objective = local.directional_D(output, ids, values, anchor)
            if objective is None:
                raise ValueError('boundary observation must not construct a gradient')
            gradients = torch.autograd.grad(objective, [delta[l] for l in layers])
        result = {l: g.detach().float().cpu().numpy().copy() for l, g in zip(layers, gradients)}
    if any(not np.isfinite(g).all() for g in result.values()):
        raise ValueError('nonfinite gradient')
    if any(p.grad is not None for p in prepared._model.parameters()):
        raise ValueError('model parameters accumulated gradients')
    return result, stats


def evaluate_observation(runtime: Any, model: Any, processor: Any, sample: pilot.Sample,
                         candidates: list[Neuron], lexical: dict[int, np.ndarray], norms: dict[int, np.ndarray],
                         ids: np.ndarray, values: np.ndarray, output: Path) -> dict[str, Any]:
    modules = down_projections(model)
    prepared = pilot.prepare(runtime, model, processor, sample)
    clean = runtime.run_reference(prepared=prepared)
    check_decoding(clean, ids, values)
    anchor = float(clean.normalized_action[0, 2])
    baseline = pilot.logits(prepared, clean)
    if not np.array_equal(baseline, pilot.logits(prepared, clean)):
        raise ValueError('clean logits not repeatable')
    with torch.no_grad():
        direct = local.aligned_logits(prepared, clean.action_token_ids).cpu().numpy()
    if not np.array_equal(direct, baseline):
        raise ValueError('autograd-capable input builder differs from original diagnostic')
    baseline_readout = readout(baseline, ids, values, anchor)
    pilot.require_equal(clean, runtime.continue_from_o2(prepared=prepared, o2=prepared.o2), 'clean continuation')
    zero = arrays_to_tensors(modules, {l: np.zeros_like(v) for l, v in lexical.items()})
    with torch.no_grad(), local.broadcast_hooks(modules, zero, measure=True) as zero_stats:
        noop = runtime.run_reference(prepared=prepared)
    pilot.require_equal(clean, noop, 'zero broadcast native generation')
    if local.measured_norm(zero_stats) != 0:
        raise ValueError('zero delta changed native FFN output')
    budget = local.residual_norm(modules, lexical)
    directions = {'lexical': lexical}
    evidence: dict[str, Any] = {**pilot.sample_dict(sample), 'anchor_normalized_z': anchor,
        'clean': baseline_readout, 'clean_action': pilot.action_dict(clean), 'zero_hook': zero_stats,
        'local_only': True, 'shared_direction_constructed': False, 'reference_budget': budget,
        'gradient_status': 'BOUNDARY_D_UNDEFINED' if baseline_readout['D'] is None else 'COMPUTED',
        'probe_supports': {}, 'unavailable_probes': {}, 'rows': []}
    gradients = None
    arrays = {'clean_logits': baseline, 'action_ids': ids, 'action_values': values}
    if baseline_readout['D'] is not None:
        gradients, grad_stats = gradient_at_zero(prepared, clean, modules, sorted(norms), ids, values, anchor, baseline)
        evidence['gradient_hook_calls'] = grad_stats
        for l, g in gradients.items():
            arrays[f'local_gradient_layer_{l}'] = g
        try:
            candidate, support = local.sparse_direction(gradients, norms, candidates)
            directions['candidate_gradient'] = local.normalize(modules, candidate, budget)
            evidence['probe_supports']['candidate_gradient'] = [[n.layer, n.index] for n in support]
            try:
                broad, broad_support = local.sparse_direction(gradients, norms, None, counts=dict(Counter(n.layer for n in support)))
                directions['broad_gradient'] = local.normalize(modules, broad, budget)
                evidence['probe_supports']['broad_gradient'] = [[n.layer, n.index] for n in broad_support]
            except ValueError as error:
                evidence['unavailable_probes']['broad_gradient'] = str(error)
        except ValueError as error:
            evidence['unavailable_probes'].update(candidate_gradient=str(error), broad_gradient='no candidate layer allocation; matched comparison unavailable')
    for name, offsets in directions.items():
        intended_norm = local.residual_norm(modules, offsets)
        if not np.isclose(intended_norm, budget, rtol=1e-5, atol=1e-10):
            raise ValueError('probe residual budget mismatch')
        slope = None if gradients is None else local.prediction(gradients, offsets)
        if name != 'lexical' and (slope is None or slope <= 0):
            raise ValueError('gradient probe is not predicted to ascend D')
        evidence['probe_supports'][name] = [[l, int(i)] for l, v in sorted(offsets.items()) for i in np.flatnonzero(v)]
        if len(evidence['probe_supports'][name]) != 10:
            raise ValueError('probe must retain exactly 10 nonzero neuron increments')
        for l, v in offsets.items():
            arrays[f'{name}_local_offset_layer_{l}'] = v
        for step in STEPS:
            for sign in (1, -1):
                delta = arrays_to_tensors(modules, offsets, sign * step)
                with torch.no_grad(), local.broadcast_hooks(modules, delta, measure=True) as logit_stats:
                    changed = local.aligned_logits(prepared, clean.action_token_ids).cpu().numpy()
                with torch.no_grad(), local.broadcast_hooks(modules, delta, measure=True) as native_stats:
                    action = runtime.run_reference(prepared=prepared)
                pilot.require_equal(clean, runtime.run_reference(prepared=prepared), 'native hook restoration')
                if not np.array_equal(baseline, pilot.logits(prepared, clean)):
                    raise ValueError('teacher-forced hook restoration failed')
                # Check numerical equivalence with the prior fixed-offset steering interface.
                neurons = [Neuron(l, int(i)) for l, v in sorted(offsets.items()) for i in np.flatnonzero(v)]
                sparse = np.array([float(delta[n.layer][n.index]) for n in neurons], dtype=np.float32)
                with ffn_hooks(modules, neurons, offsets=sparse):
                    old = pilot.logits(prepared, clean)
                if not np.array_equal(old, changed):
                    raise ValueError('new broadcast hook differs from original fixed-offset interface')
                modified = readout(changed, ids, values, anchor)
                delta_D = None if modified['D'] is None else modified['D'] - baseline_readout['D']
                predicted = None if slope is None else sign * step * slope
                changes = (action.deployed_action - clean.deployed_action)[0].tolist()
                row = {'condition': name, 'step': step, 'sign': sign,
                       'predicted_delta_D': predicted, 'delta_D': delta_D,
                       'first_order_error': None if predicted is None else delta_D - predicted,
                       'first_order_ratio': None if predicted is None or abs(predicted) < 1e-12 else delta_D / predicted,
                       'delta_expected_z': modified['expected_normalized_z'] - baseline_readout['expected_normalized_z'],
                       'teacher_forced': modified,
                       'z_argmax_token_changed': modified['z_argmax_token'] != baseline_readout['z_argmax_token'],
                       'z_argmax_bin_change': (None if modified['z_argmax_value'] is None or baseline_readout['z_argmax_value'] is None
                                               else modified['z_argmax_value'] - baseline_readout['z_argmax_value']),
                       'native_action': pilot.action_dict(action), 'native_action_delta_all7': changes,
                       'native_decoded_delta_z': changes[2], 'other6_indices': [0, 1, 3, 4, 5, 6],
                       'other6_delta': [changes[i] for i in [0, 1, 3, 4, 5, 6]],
                       'intended_per_token_residual_norm': step * intended_norm,
                       'actual_teacher_forced_residual_rms': local.measured_norm(logit_stats),
                       'actual_native_residual_rms': local.measured_norm(native_stats),
                       'actual_teacher_forced_to_intended_ratio': local.measured_norm(logit_stats) / (step * intended_norm),
                       'actual_native_to_intended_ratio': local.measured_norm(native_stats) / (step * intended_norm),
                       'teacher_forced_hooks': logit_stats, 'native_hooks': native_stats,
                       'restoration': 'PASS', 'original_interface_equivalence': 'PASS'}
                evidence['rows'].append(row)
                arrays[f'{name}_{step:g}_{sign}_logits'] = changed
                print(f'{sample.record.sample_id} {name} step={sign * step:g} delta_D={delta_D} native_delta_z={changes[2]}', flush=True)
    np.savez_compressed(output / f'{sample.record.sample_id}.npz', **arrays)
    pilot.write_json(output / f'{sample.record.sample_id}.json', evidence)
    return evidence


def summarize(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """报告逐帧局部探针，不把每帧不同的方向解释为共同方向。"""
    result = []
    for name in ('lexical', 'candidate_gradient', 'broad_gradient'):
        for step in STEPS:
            pairs = []
            for obs in observations:
                rows = [r for r in obs['rows'] if r['condition'] == name and r['step'] == step]
                if len(rows) != 2 or rows[0]['delta_D'] is None:
                    continue
                plus, minus = next(r for r in rows if r['sign'] == 1), next(r for r in rows if r['sign'] == -1)
                pairs.append({'sample_id': obs['sample_id'], 'state_id': obs['state_id'],
                              'D_both': plus['delta_D'] > 1e-8 and minus['delta_D'] < -1e-8,
                              'native_z_both': plus['native_decoded_delta_z'] > 1e-8 and minus['native_decoded_delta_z'] < -1e-8,
                              'plus_D': plus['delta_D'], 'minus_D': minus['delta_D'],
                              'plus_native_z': plus['native_decoded_delta_z'], 'minus_native_z': minus['native_decoded_delta_z']})
            result.append({'condition': name, 'step': step, 'valid_observations': len(pairs),
                           'D_both_count': sum(p['D_both'] for p in pairs),
                           'native_z_both_count': sum(p['native_z_both'] for p in pairs), 'pairs': pairs})
    return result


def run(args: argparse.Namespace, info: dict[str, Any], samples: list[pilot.Sample]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        pilot.write_json(args.output_dir / 'protocol.json', info)
        hashes = pilot.checkpoint_hashes(args.pretrained_checkpoint)
        if hashes != screen.read_json(args.v1_dir / 'checkpoint_hashes.json') or hashes != screen.read_json(args.screen_dir / 'checkpoint_hashes.json'):
            raise ValueError('checkpoint identity mismatch')
        pilot.write_json(args.output_dir / 'checkpoint_hashes.json', hashes)
        runtime, model, processor = screen.load_model(args)
        modules = down_projections(model)
        candidates = [Neuron(**r['neuron']) for r in screen.read_json(args.v1_dir / 'candidates.json')['candidates']]
        if len(set(candidates)) != len(candidates) or len(candidates) != 232:
            raise ValueError('expected the fixed 232-candidate source pool')
        lexical = make_reference(modules, candidates, screen.read_json(args.screen_dir / 'calibration.json'))
        layers = sorted({n.layer for n in candidates})
        norms = local.column_norms(modules, layers)
        ids, values = action_mapping(model)
        evidence_dir = args.output_dir / 'observations'
        evidence_dir.mkdir()
        observations = []
        saved_clean = {r['sample_id']: r['action'] for r in screen.read_json(args.screen_dir / 'calibration.json')['clean_records']}
        for sample in samples:
            evidence = evaluate_observation(runtime, model, processor, sample, candidates, lexical, norms, ids, values, evidence_dir)
            if evidence['clean_action'] != saved_clean[sample.record.sample_id]:
                raise ValueError('clean generation differs from prior screen calibration')
            observations.append(evidence)
        summary = summarize(observations)
        pilot.write_json(args.output_dir / 'summary.json', summary)
        for sample in samples:
            if pilot.sha256_file(sample.path) != sample.archive_sha256:
                raise ValueError('source observation changed')
        if (screen.file_identities(args.v1_dir, screen.SOURCE_FILES) != info['v1_hashes']
                or screen.file_identities(args.screen_dir, SCREEN_FILES) != info['screen_hashes']
                or pilot.sha256_file(args.collection_manifest) != info['manifest_sha256']
                or pilot.repository_identity(ROOT) != info['repository']
                or pilot.repository_identity(args.tex3d_openvla_root.parent) != info['tex3d_repository']):
            raise ValueError('source or code identity changed during diagnostic')
        pilot.write_json(args.output_dir / 'results.json', {
            'schema': SCHEMA, 'engineering_status': 'COMPLETE', 'scientific_status': 'LOCAL_DIAGNOSTIC_ONLY',
            'runtime': runtime.versions, 'observation_count': len(observations),
            'boundary_count': sum(o['gradient_status'] == 'BOUNDARY_D_UNDEFINED' for o in observations),
            'unavailable_probes': {o['sample_id']: o['unavailable_probes'] for o in observations if o['unavailable_probes']},
            'new_neuron_set_published': False, 'shared_direction_constructed': False,
            'scope_experiment': False, 'rollout_executed': False, 'summary': summary})
        print(f'LOCAL DIAGNOSTIC COMPLETE: {args.output_dir}', flush=True)
    except Exception as error:
        pilot.write_json(args.output_dir / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-manifest', 'pretrained-checkpoint', 'tex3d-openvla-root', 'output-dir', 'v1-dir', 'screen-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    info, samples = preflight(args)
    if args.preflight_only:
        pilot.smoke._load_runtime(args.tex3d_openvla_root)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print('PREFLIGHT PASSED: imports only, no model/output created')
        return 0
    run(args, info, samples)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
