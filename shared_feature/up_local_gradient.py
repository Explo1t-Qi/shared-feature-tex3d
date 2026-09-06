"""逐 observation 的广播增量梯度诊断；不学习跨观测方向或修改权重。"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from shared_feature.up_concept import Neuron


def aligned_logits(prepared: Any, tokens: np.ndarray) -> torch.Tensor:
    """保留 autograd 的 clean-prefix forward，返回 [7,V] FP32 logits。

    上游视觉/文字特征在 inference_mode 中产生；在当前普通模式 clone，
    防止 inference tensors 被下游 backward 保存。模型参数保持冻结。
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("gradient forward must run outside inference_mode")
    tokens = np.asarray(tokens)
    if tokens.shape != (1, 7) or tokens.dtype.kind not in "iu":
        raise ValueError("expected clean action tokens [1,7]")
    model = prepared._model
    with torch.no_grad():
        text = prepared._text_embeddings.detach().clone()
        visual = prepared.o2.detach().clone()
        prefix = torch.as_tensor(tokens[:, :-1], device=text.device, dtype=torch.long)
        prefix_embeddings = model.get_input_embeddings()(prefix).detach().clone()
        inputs = torch.cat([text[:, :1], visual, text[:, 1:], prefix_embeddings], dim=1)
        mask = prepared._multimodal_attention_mask
        if mask is not None:
            mask = torch.cat([mask.detach().clone(), mask.new_ones((1, 6))], dim=1)
    output = model.language_model(inputs_embeds=inputs, attention_mask=mask, use_cache=False, return_dict=True)
    logits = output.logits[0, -7:].float()
    if logits.shape[0] != 7 or not torch.isfinite(logits).all():
        raise ValueError("invalid action-position logits")
    return logits


def directional_D(logits: torch.Tensor, ids: np.ndarray, values: np.ndarray, anchor: float) -> torch.Tensor | None:
    """固定 clean bin 掩码；边界帧返回 None，不构造替代梯度。"""
    up, down = ids[values > anchor], ids[values < anchor]
    if not len(up) or not len(down):
        return None
    z = logits[2].double()
    return torch.logsumexp(z[torch.as_tensor(up, device=z.device)], dim=0) - torch.logsumexp(
        z[torch.as_tensor(down, device=z.device)], dim=0)


@contextmanager
def broadcast_hooks(modules: Sequence[Any], deltas: dict[int, torch.Tensor], *,
                    measure: bool = False) -> Iterator[dict[str, Any]]:
    """每层 delta [neurons] 广播到 [B,T,neurons]，所有调用复用同一增量。

    不 detach activation，保留早期 delta 经后续层传播的完整梯度。
    measure 仅用于 no-grad 测量：额外 F.linear 得到同一上游输入未加 delta
    时的输出，直接度量 native dtype 下的真实局部 FFN 输出变化。
    """
    if not deltas:
        raise ValueError("empty intervention")
    if measure and torch.is_grad_enabled():
        raise ValueError("measurement requires no_grad or inference_mode")
    handles, baselines = [], {}
    stats: dict[str, Any] = {}

    def before(layer: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            x = args[0]
            delta = deltas[layer]
            if x.ndim != 3 or delta.shape != (module.in_features,) or delta.device != x.device:
                raise ValueError("delta must be [neurons] on the activation device")
            changed = (x.float() + delta.view(1, 1, -1)).to(x.dtype)
            if not torch.isfinite(changed).all():
                raise ValueError("nonfinite intervention activation")
            item = stats[str(layer)]
            item['calls'] += 1
            if measure:
                baselines[layer] = F.linear(x, module.weight, module.bias)
                actual = changed.float() - x.float()
                item['token_rows'] += x.shape[0] * x.shape[1]
                item['changed_values'] += int(torch.count_nonzero(actual))
                item['actual_delta_sq'] += float(actual.double().square().sum())
            return (changed, *args[1:])
        return hook

    def after(layer: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
            difference = output.float() - baselines.pop(layer).float()
            stats[str(layer)]['actual_residual_shift_sq'] += float(difference.double().square().sum())
        return hook

    try:
        for layer, delta in sorted(deltas.items()):
            if not 0 <= layer < len(modules) or not torch.isfinite(delta).all():
                raise ValueError("invalid layer/delta")
            stats[str(layer)] = {'calls': 0, 'token_rows': 0, 'changed_values': 0,
                                 'actual_delta_sq': 0., 'actual_residual_shift_sq': 0.}
            handles.append(modules[layer].register_forward_pre_hook(before(layer)))
            if measure:
                handles.append(modules[layer].register_forward_hook(after(layer)))
        yield stats
        if any(s['calls'] == 0 for s in stats.values()) or baselines:
            raise ValueError("missing or incomplete intervention hook calls")
    finally:
        for handle in handles:
            handle.remove()


def residual_norm(modules: Sequence[Any], offsets: dict[int, np.ndarray]) -> float:
    """理想每 token 预算 sqrt(sum_layer ||W_down delta||²)，仅计算非零列。"""
    squared = 0.
    for layer, delta in offsets.items():
        indices = np.flatnonzero(delta)
        if not len(indices):
            continue
        w = modules[layer].weight[:, indices.tolist()].detach().float()
        v = w @ torch.as_tensor(delta[indices], dtype=w.dtype, device=w.device)
        squared += float(v.double().square().sum())
    return float(np.sqrt(squared))


def column_norms(modules: Sequence[Any], layers: Sequence[int]) -> dict[int, np.ndarray]:
    """FP32 分块计算 value-vector 范数，避免复制整层 FP32 权重。"""
    result = {}
    for layer in layers:
        weight = modules[layer].weight.detach()
        chunks = [weight[:, start:start + 256].float().norm(dim=0).cpu().numpy()
                  for start in range(0, weight.shape[1], 256)]
        result[layer] = np.concatenate(chunks)
        if not np.isfinite(result[layer]).all() or np.any(result[layer] <= 0):
            raise ValueError("zero/nonfinite value-vector column norm")
    return result


def sparse_direction(gradients: dict[int, np.ndarray], norms: dict[int, np.ndarray],
                     candidates: Sequence[Neuron] | None, *, counts: dict[int, int] | None = None,
                     k: int = 10) -> tuple[dict[int, np.ndarray], list[Neuron]]:
    """仅为当前帧构造 k-sparse ascent probe，不发布可复用的 neuron 集合。

    排序 |g_i|/||W_i||；系数 g_i/||W_i||²，再由调用方匹配精确组合范数。
    broad probe 按 candidate probe 的逐层数量选择，控制实际干预层分布。
    """
    pool = candidates if candidates is not None else [Neuron(l, i) for l, g in sorted(gradients.items()) for i in range(len(g))]
    ranked = sorted((n for n in pool if np.isfinite(gradients[n.layer][n.index]) and gradients[n.layer][n.index] != 0),
                    key=lambda n: (-abs(float(gradients[n.layer][n.index])) / float(norms[n.layer][n.index]), n.layer, n.index))
    if counts is None:
        chosen = ranked[:k]
    else:
        chosen = [n for layer, count in sorted(counts.items())
                  for n in [item for item in ranked if item.layer == layer][:count]]
        if Counter(n.layer for n in chosen) != counts:
            raise ValueError("not enough nonzero gradients for matched layer allocation")
    if len(chosen) != k:
        raise ValueError("not enough nonzero gradients for k-sparse probe")
    offsets = {layer: np.zeros_like(gradients[layer], dtype=np.float64) for layer in sorted({n.layer for n in chosen})}
    for n in chosen:
        offsets[n.layer][n.index] = float(gradients[n.layer][n.index]) / float(norms[n.layer][n.index]) ** 2
    # Common rescaling prevents float32 underflow without changing direction.
    scale = max(float(np.max(np.abs(v))) for v in offsets.values())
    return {l: (v / scale).astype(np.float32) for l, v in offsets.items()}, chosen


def normalize(modules: Sequence[Any], offsets: dict[int, np.ndarray], budget: float) -> dict[int, np.ndarray]:
    norm = residual_norm(modules, offsets)
    if not np.isfinite(norm) or norm <= 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("invalid residual shift normalization")
    return {l: (v * (budget / norm)).astype(np.float32) for l, v in offsets.items()}


def prediction(gradients: dict[int, np.ndarray], offsets: dict[int, np.ndarray]) -> float:
    return sum(float(np.asarray(gradients[l], dtype=np.float64) @ v.astype(np.float64)) for l, v in offsets.items())


def measured_norm(stats: dict[str, Any]) -> float:
    """各层 token rows 均方后求和开方；可比较不同 forward token 数量。"""
    return float(np.sqrt(sum(s['actual_residual_shift_sq'] / s['token_rows'] for s in stats.values())))
