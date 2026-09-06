"""验证关键数值语义：候选排名、随机对照、原生 dtype 干预、统计分组。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shared_feature.up_concept import (
    Moments, Neuron, discover_candidates, exact_up_ids, ffn_hooks,
    paired_contrasts, random_controls, rank_projection_rows, summarize_rows,
)
from scripts.up_concept_pilot import matched_offsets


def test_exact_up_excludes_subwords_and_out_of_vocabulary():
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "b": 1, "c": 2, "d": 20},
        decode=lambda ids: {0: "up", 1: " UP ", 2: "upcoming", 20: "up"}[ids[0]],
    )
    assert exact_up_ids(tokenizer, 3) == [0, 1]


def test_projection_rank_matches_full_sort_with_ties():
    scores = torch.tensor([[0., 3., 3., 2.], [3., 2., 1., 0.], [0., 0., 0., 0.]])
    rows = rank_projection_rows(scores, [2, 1], top_k=2)
    assert [r[0] for r in rows] == [0, 1, 2]
    for row, mass, rank, ids in rows:
        order = torch.argsort(scores[row], descending=True, stable=True).tolist()
        assert rank == min(order.index(1), order.index(2)) + 1
        assert ids == tuple(order[:2])
        assert mass == pytest.approx(float(scores[row].softmax(-1)[[1, 2]].sum()))
    assert rank_projection_rows(scores[1:2], [3], top_k=2) == []


def test_projection_uses_weight_columns_and_excludes_non_up_rows():
    class Tokenizer:
        def __len__(self): return 12
        def get_vocab(self): return {str(i): i for i in range(12)}
        def decode(self, ids): return "up" if ids[0] == 11 else str(ids[0])
    down = torch.nn.Linear(2, 12, bias=False)
    head = torch.nn.Linear(12, 12, bias=False)
    with torch.no_grad():
        down.weight[:, 0] = torch.arange(12)
        down.weight[:, 1] = -torch.arange(12)
        head.weight.copy_(torch.eye(12))
    model = SimpleNamespace(language_model=SimpleNamespace(
        lm_head=head, model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=down))])))
    result = discover_candidates(model, Tokenizer(), batch_size=1)
    assert len(result) == 1
    assert result[0].neuron == Neuron(0, 0)
    assert result[0].top_token_ids[0] == 11


def test_random_controls_match_layers_and_exclude_all_semantic_candidates():
    selected = [Neuron(0, 1), Neuron(1, 2), Neuron(1, 3)]
    excluded = selected + [Neuron(0, 2), Neuron(1, 4)]
    controls = random_controls(selected, [20, 20], excluded)
    assert controls == random_controls(selected, [20, 20], excluded)
    assert len(controls) == 3
    for group in controls:
        assert [n.layer for n in group] == [0, 1, 1]
        assert not set(group) & set(excluded)
        assert len(set(group)) == 3
    with pytest.raises(ValueError, match="not enough"):
        random_controls([Neuron(0, 0)], [1], [Neuron(0, 0)])


def test_streaming_moments_equal_concatenated_population_std():
    values = np.random.default_rng(4).normal(size=(21, 5)) + 10000
    moments = Moments()
    moments.add(values[:7]); moments.add(values[7:9]); moments.add(values[9:])
    np.testing.assert_allclose(moments.std(), values.std(axis=0), rtol=1e-10)
    assert moments.count == 21
    with pytest.raises(ValueError):
        Moments().std()
    constant = Moments(); constant.add(np.ones((3, 1)))
    with pytest.raises(ValueError, match="zero"):
        constant.std()


def test_hooks_modify_only_selected_inputs_and_restore_without_inplace_writes():
    module = torch.nn.Linear(4, 3, bias=False)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    original = x.clone(); clean = module(x).detach()
    with ffn_hooks([module], [Neuron(0, 2)], offsets=np.array([0.5])) as stats:
        output = module(x)
    expected = x.clone(); expected[..., 2] += 0.5
    torch.testing.assert_close(output, module(expected))
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(module(x), clean, rtol=0, atol=0)
    assert stats["changed_values"] == 2
    assert stats["actual_delta_sq"] == pytest.approx(0.5)
    assert stats["ffn_shift_sq"] == pytest.approx(float((output.detach() - clean).double().square().sum()))
    assert not module._forward_pre_hooks


@pytest.mark.parametrize("offsets", [None, np.zeros(1)])
def test_noop_hook_retains_exact_forward(offsets):
    module = torch.nn.Linear(4, 2, bias=False)
    x = torch.randn(1, 3, 4)
    clean = module(x)
    with ffn_hooks([module], [Neuron(0, 1)], offsets=offsets):
        observed = module(x)
    torch.testing.assert_close(observed, clean, atol=0, rtol=0)


def test_hooks_cleanup_on_exception_and_missed_forward():
    module = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(RuntimeError, match="intentional"):
        with ffn_hooks([module], [Neuron(0, 1)]):
            raise RuntimeError("intentional")
    assert not module._forward_pre_hooks
    with pytest.raises(ValueError, match="never called"):
        with ffn_hooks([module], [Neuron(0, 1)]):
            pass
    assert not module._forward_pre_hooks


def test_bfloat16_reports_applied_not_intended_delta():
    module = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    x = torch.ones(1, 1, 2, dtype=torch.bfloat16) * 100
    with ffn_hooks([module], [Neuron(0, 0)], offsets=np.array([0.001])) as stats:
        module(x)
    assert stats["changed_values"] == 0
    assert stats["actual_delta_sq"] == 0


def test_calibration_combines_layers_in_declared_neuron_order():
    modules = [torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(3, 2, bias=False)]
    neurons = [Neuron(1, 2), Neuron(0, 1)]
    values = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    moments = Moments()
    with ffn_hooks(modules, neurons, moments=moments):
        for x in (values, values[:, :1] + 10):
            modules[0](x); modules[1](x + 20)
    joined = torch.cat([values, values[:, :1] + 10], dim=1).numpy()[0]
    expected = np.column_stack([joined[:, 2] + 20, joined[:, 1]])
    np.testing.assert_allclose(moments.mean, expected.mean(0))
    np.testing.assert_allclose(moments.std(), expected.std(0), rtol=1e-6)


def test_control_offsets_match_each_layer_residual_norm():
    modules = [torch.nn.Linear(4, 3, bias=False), torch.nn.Linear(4, 3, bias=False)]
    up = [Neuron(0, 0), Neuron(1, 0)]
    random = [Neuron(0, 2), Neuron(1, 2)]
    std = {n: float(i + 1) for i, n in enumerate(up + random)}
    offsets = matched_offsets(modules, up, random, std)
    for i, module in enumerate(modules):
        target = module.weight[:, 0] * std[up[i]]
        actual = module.weight[:, 2] * offsets[i]
        assert float(target.norm()) == pytest.approx(float(actual.norm()), rel=1e-6)


def test_summary_weights_trajectory_groups_equally():
    rows = [{"condition": "up", "alpha": 1., "state_id": state,
             "delta_translation": [0, 0, dz], "token_hamming": 0, "logit_max_abs": 1.}
            for state, dz in [(1, 2.), (1, 2.), (2, -1.)]]
    summary = summarize_rows(rows)[0]
    assert summary["mean_delta_z"] == 0.5  # frame mean would be 1.0
    assert summary["trajectory_groups"] == 2
    assert summary["positive_groups"] == 1


def test_paired_contrast_requires_complete_controls_and_equal_group_weights():
    rows = []
    for sample_id, state, up_value in [("a", 0, 3.), ("b", 0, 3.), ("c", 1, 0.)]:
        for condition in ("up", "random_0", "random_1", "random_2"):
            rows.append({"sample_id": sample_id, "state_id": state, "condition": condition,
                         "alpha": 1., "delta_translation": [0, 0, up_value if condition == "up" else 1.]})
    report = paired_contrasts(rows)[0]
    assert report["mean_up_minus_random_mean"] == 0.5
    assert report["positive_contrast_groups"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        paired_contrasts(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_contrasts(rows + [rows[0]])
