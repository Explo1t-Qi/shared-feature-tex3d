"""UP pilot 的候选、FFN 干预及配对汇总。

参考 Häon et al. (CoRL 2025) 的 value-vector → vocabulary 方法；本模块
独立实现按词汇投影选候选，以及按 clean 标准差增量干预（不是作者的常数覆盖）。
只操作 OpenVLA Llama FFN down_proj 的输入 [B, tokens, neurons]。
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class Neuron:
    layer: int
    index: int


@dataclass(frozen=True)
class Candidate:
    neuron: Neuron
    up_mass: float
    best_up_rank: int
    top_token_ids: tuple[int, ...]
    top_tokens: tuple[str, ...]


def candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate)


def down_projections(model: Any) -> list[Any]:
    """返回原生 down_proj；从权重读取每层宽度，不硬编码 11008。"""
    modules = [layer.mlp.down_proj for layer in model.language_model.model.layers]
    if not modules or any(m.weight.ndim != 2 or m.bias is not None for m in modules):
        raise ValueError("UP pilot requires nonempty, bias-free Llama down_proj layers")
    return modules


def exact_up_ids(tokenizer: Any, vocabulary_size: int) -> list[int]:
    """仅使用单 token 解码后 strip/casefold 等于 up 的词，不匹配 upcoming。"""
    ids = sorted(set(int(i) for i in tokenizer.get_vocab().values()))
    result = [i for i in ids if 0 <= i < vocabulary_size
              and tokenizer.decode([i]).strip().casefold() == "up"]
    if not result:
        raise ValueError("tokenizer has no exact up token")
    return result


def rank_projection_rows(
    logits: torch.Tensor, up_ids: Sequence[int], top_k: int = 10,
) -> list[tuple[int, float, int, tuple[int, ...]]]:
    """对 [neurons, vocab] FP32 logits 排名；仅为合格行排序完整词表。

    best rank 使用 logit 降序、token ID 升序的确定性规则。分数是 up 变体
    softmax 概率质量，不使用 observation、action 或干预结果挑选 neuron。
    """
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("projection logits must be finite [neurons, vocabulary]")
    if not up_ids or min(up_ids) < 0 or max(up_ids) >= logits.shape[1]:
        raise ValueError("invalid up token IDs")
    ordered_ids = sorted(set(up_ids))
    up_logits = logits[:, ordered_ids]
    best, offset = up_logits.max(dim=1)
    best_ids = torch.tensor(ordered_ids, device=logits.device)[offset]
    vocab_ids = torch.arange(logits.shape[1], device=logits.device)
    ranks = 1 + (logits > best[:, None]).sum(dim=1)
    ranks += ((logits == best[:, None]) & (vocab_ids[None, :] < best_ids[:, None])).sum(dim=1)
    masses = torch.exp(torch.logsumexp(up_logits, dim=1) - torch.logsumexp(logits, dim=1))
    result = []
    for row in torch.where(ranks <= top_k)[0].tolist():
        top = torch.argsort(logits[row], descending=True, stable=True)[:top_k].tolist()
        result.append((row, float(masses[row]), int(ranks[row]), tuple(top)))
    return result


def discover_candidates(model: Any, tokenizer: Any, *, batch_size: int = 128) -> list[Candidate]:
    """按层/小批投影，避免保存 [全部 neurons, vocabulary] 巨型矩阵。"""
    if batch_size < 1:
        raise ValueError("projection batch size must be positive")
    head = model.language_model.lm_head.weight.detach()
    up_ids = exact_up_ids(tokenizer, head.shape[0])
    candidates: list[Candidate] = []
    with torch.inference_mode():
        head_fp32 = head.float()
        for layer, module in enumerate(down_projections(model)):
            print(f"projection layer {layer + 1}/{len(down_projections(model))}", flush=True)
            for start in range(0, module.in_features, batch_size):
                # PyTorch weight [hidden, neurons] 的列即论文中的 value vectors。
                vectors = module.weight[:, start:start + batch_size].T.float()
                logits = vectors @ head_fp32.T
                for row, mass, rank, ids in rank_projection_rows(logits, up_ids):
                    labels = tuple(tokenizer.decode([i]) if i < len(tokenizer)
                                   else f"<vocab:{i}>" for i in ids)
                    candidates.append(Candidate(Neuron(layer, start + row), mass, rank, ids, labels))
    return sorted(candidates, key=lambda c: (-c.up_mass, c.best_up_rank, c.neuron.layer, c.neuron.index))


def random_controls(
    selected: Sequence[Neuron], widths: Sequence[int], excluded: Sequence[Neuron],
    *, repeats: int = 3, seed: int = 7,
) -> list[list[Neuron]]:
    """固定 PCG64，各随机集合匹配逐层数量，排除所有 up 合格候选。"""
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected neurons must be nonempty and unique")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    counts = Counter(n.layer for n in selected)
    blocked = set(excluded) | set(selected)
    for neuron in blocked:
        if not 0 <= neuron.layer < len(widths) or not 0 <= neuron.index < widths[neuron.layer]:
            raise ValueError("neuron index outside model")
    rng = np.random.Generator(np.random.PCG64(seed))
    result: list[list[Neuron]] = []
    for _ in range(repeats):
        group = []
        for layer, count in sorted(counts.items()):
            pool = [i for i in range(widths[layer]) if Neuron(layer, i) not in blocked]
            if len(pool) < count:
                raise ValueError(f"not enough random control neurons at layer {layer}")
            group.extend(Neuron(layer, int(i)) for i in sorted(rng.choice(pool, count, replace=False)))
        result.append(group)
    return result


class Moments:
    """在线合并 clean activation moments；不保存全层 activation。"""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def add(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or not x.shape[0] or not np.isfinite(x).all():
            raise ValueError("moments require finite nonempty [positions, neurons]")
        count, mean = x.shape[0], x.mean(axis=0)
        m2 = ((x - mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        if mean.shape != self.mean.shape:
            raise ValueError("calibration neuron shape changed")
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def std(self) -> np.ndarray:
        if self.count < 2 or self.m2 is None:
            raise ValueError("insufficient calibration values")
        result = np.sqrt(self.m2 / self.count)
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("zero/nonfinite calibration std; do not silently rescale")
        return result


@contextmanager
def ffn_hooks(
    modules: Sequence[Any], neurons: Sequence[Neuron], *,
    offsets: np.ndarray | None = None, moments: Moments | None = None,
) -> Iterator[dict[str, Any]]:
    """只替换 down_proj 输入的 clone；finally 移除所有 hook。

    offsets [selected_neurons] 是 FP32 定义的固定增量，运算后转回模型 dtype。
    作用域为所有 forward token（prefill + autoregressive decode；诊断另行注册）。
    moments 采集时先按调用序号对齐各层，再合并为同一 neuron 顺序。
    """
    if not neurons or len(set(neurons)) != len(neurons):
        raise ValueError("hook neurons must be nonempty and unique")
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float32)
        if offsets.shape != (len(neurons),) or not np.isfinite(offsets).all():
            raise ValueError("offsets must be finite [selected_neurons]")
    locations: dict[int, list[tuple[int, int]]] = {}
    for column, neuron in enumerate(neurons):
        if not 0 <= neuron.layer < len(modules) or not 0 <= neuron.index < modules[neuron.layer].in_features:
            raise ValueError("hook neuron outside model")
        locations.setdefault(neuron.layer, []).append((column, neuron.index))
    stats: dict[str, Any] = {"calls": {}, "changed_values": 0, "actual_delta_sq": 0.0,
                             "ffn_shift_sq": 0.0, "selected_values": 0}
    captured: dict[int, list[np.ndarray]] = {layer: [] for layer in locations}
    handles = []

    def make_hook(layer: int, entries: list[tuple[int, int]]) -> Any:
        columns, indices = zip(*entries)

        def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            x = args[0]
            if x.ndim != 3 or x.shape[-1] != module.in_features:
                raise ValueError("FFN input must have shape [B, tokens, neurons]")
            values = x[..., list(indices)].detach().float()
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite FFN activation")
            stats["calls"][str(layer)] = stats["calls"].get(str(layer), 0) + 1
            if moments is not None:
                captured[layer].append(values.reshape(-1, len(indices)).cpu().numpy().copy())
            if offsets is None or not np.any(offsets):
                return None  # no-op 路径返回原对象，不重算 F.linear。
            intended = torch.as_tensor(offsets[list(columns)], device=x.device)
            modified = (values + intended).to(dtype=x.dtype)
            if not torch.isfinite(modified).all():
                raise ValueError("native-dtype FFN intervention overflowed")
            actual = modified.float() - values
            stats["changed_values"] += int(torch.count_nonzero(actual))
            stats["selected_values"] += actual.numel()
            stats["actual_delta_sq"] += float(actual.double().square().sum())
            # 每次实际 native-dtype 增量所产生的 FFN residual shift。
            shift = actual @ module.weight[:, list(indices)].detach().float().T
            stats["ffn_shift_sq"] += float(shift.double().square().sum())
            result = x.clone()
            result[..., list(indices)] = modified
            return (result, *args[1:])
        return hook

    try:
        for layer, entries in locations.items():
            handles.append(modules[layer].register_forward_pre_hook(make_hook(layer, entries)))
        yield stats
        if any(stats["calls"].get(str(layer), 0) == 0 for layer in locations):
            raise ValueError("some requested FFN hooks were never called")
        if moments is not None:
            counts = {len(values) for values in captured.values()}
            if len(counts) != 1:
                raise ValueError("calibration forward call counts differ across layers")
            for call in range(next(iter(counts))):
                rows = {values[call].shape[0] for values in captured.values()}
                if len(rows) != 1:
                    raise ValueError("calibration token positions differ across layers")
                joined = np.empty((next(iter(rows)), len(neurons)), dtype=np.float64)
                for layer, entries in locations.items():
                    joined[:, [column for column, _ in entries]] = captured[layer][call]
                moments.add(joined)
    finally:
        for handle in handles:
            handle.remove()


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """先在 trajectory 内平均，再跨组汇总；不把帧当作独立统计样本。"""
    result: list[dict[str, Any]] = []
    keys = sorted({(r["condition"], r["alpha"]) for r in rows})
    for condition, alpha in keys:
        subset = [r for r in rows if (r["condition"], r["alpha"]) == (condition, alpha)]
        groups = sorted({r["state_id"] for r in subset})
        means = [float(np.mean([r["delta_translation"][2] for r in subset if r["state_id"] == g]))
                 for g in groups]
        result.append({"condition": condition, "alpha": alpha, "observations": len(subset),
                       "trajectory_groups": len(groups), "group_mean_delta_z": dict(zip(groups, means)),
                       "mean_delta_z": float(np.mean(means)),
                       "positive_groups": sum(x > 1e-8 for x in means),
                       "action_changed_observations": sum(r["token_hamming"] > 0 for r in subset),
                       "max_teacher_forced_logit_delta": max(r["logit_max_abs"] for r in subset)})
    return result


def paired_contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐 observation 配对 UP 与三个随机集合，再对 trajectory 等权汇总。"""
    results: list[dict[str, Any]] = []
    for alpha in sorted({row["alpha"] for row in rows}):
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            if row["alpha"] != alpha:
                continue
            sample = pairs.setdefault(row["sample_id"], {})
            if row["condition"] in sample:
                raise ValueError("duplicate sample/condition/alpha row")
            sample[row["condition"]] = row
        by_group: dict[int, list[float]] = {}
        for sample in pairs.values():
            if set(sample) != {"up", "random_0", "random_1", "random_2"}:
                raise ValueError("incomplete paired random controls")
            state = sample["up"]["state_id"]
            if any(row["state_id"] != state for row in sample.values()):
                raise ValueError("paired state identities differ")
            random_mean = np.mean([sample[f"random_{i}"]["delta_translation"][2] for i in range(3)])
            difference = float(sample["up"]["delta_translation"][2] - random_mean)
            by_group.setdefault(state, []).append(difference)
        means = {str(state): float(np.mean(values)) for state, values in sorted(by_group.items())}
        results.append({"alpha": alpha, "group_up_minus_random_mean": means,
                        "mean_up_minus_random_mean": float(np.mean(list(means.values()))),
                        "positive_contrast_groups": sum(value > 1e-8 for value in means.values()),
                        "trajectory_groups": len(means)})
    return results
