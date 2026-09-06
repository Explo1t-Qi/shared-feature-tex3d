"""UP action-relevant 筛选的概率读出与轨迹级评分，不运行模型。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def action_mapping(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """由实际模型构造 token IDs→归一化动作值，保留最高 bin 的重复端点。"""
    bins = np.asarray(model.bins, dtype=np.float64)
    centers = np.asarray(model.bin_centers, dtype=np.float64)
    if (bins.ndim != 1 or len(bins) < 3 or centers.shape != (len(bins) - 1,)
            or not np.isfinite(bins).all() or not np.all(np.diff(bins) > 0)
            or not np.allclose(centers, (bins[:-1] + bins[1:]) / 2, rtol=0, atol=1e-12)):
        raise ValueError("unsupported model action bins")
    vocabulary = int(model.vocab_size)
    ids = np.arange(vocabulary - len(bins), vocabulary, dtype=np.int64)
    if ids[0] < 0:
        raise ValueError("action vocabulary is smaller than bins")
    values = centers[np.clip(vocabulary - ids - 1, 0, len(centers) - 1)]
    return ids, values


def z_readout(logits: np.ndarray, ids: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """输入 [7,V] clean-prefix logits，返回 action 条件期望和全词表 action 概率质量。

    只解释第三个动作分量；不把 softmax 期望当成真实 greedy 动作或位移。
    """
    logits = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(ids)
    values = np.asarray(values, dtype=np.float64)
    if (logits.ndim != 2 or logits.shape[0] != 7 or not np.isfinite(logits).all()
            or ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids) or values.shape != ids.shape
            or not np.isfinite(values).all() or ids.min() < 0 or ids.max() >= logits.shape[1]):
        raise ValueError("invalid logits/action mapping")
    full = logits[2]
    log_total = np.logaddexp.reduce(full)
    action = full[ids]
    log_action = np.logaddexp.reduce(action)
    probabilities = np.exp(action - log_action)
    return {"expected_normalized_z": float(probabilities @ values),
            "action_probability_mass": float(np.exp(log_action - log_total))}


def check_decoding(clean: Any, ids: np.ndarray, values: np.ndarray) -> None:
    """核对七维 clean 解码值，防止 token 映射方向或 off-by-one 错误。"""
    tokens = np.asarray(clean.action_token_ids).reshape(-1)
    lookup = dict(zip(ids.tolist(), values.tolist()))
    if tokens.shape != (7,) or any(int(t) not in lookup for t in tokens):
        raise ValueError("clean generation contains non-action tokens")
    decoded = [lookup[int(t)] for t in tokens]
    if not np.allclose(decoded, np.asarray(clean.normalized_action).reshape(-1), rtol=0, atol=1e-10):
        raise ValueError("live model action mapping disagrees with clean decode")


def rank_candidates(rows: Sequence[dict[str, Any]], *, k: int = 10) -> dict[str, Any]:
    """每轨迹分别取 plus 和 -minus 的四帧中位数，以全部轨迹的最弱值排名。

    全部三条轨迹两个方向均 >1e-8 才合格，避免单帧大值主导；不根据结果改符号。
    """
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["layer"], row["index"]), []).append(row)
    ranking = []
    expected_samples = None
    for (layer, index), observations in grouped.items():
        identities = [(r["state_id"], r["sample_id"]) for r in observations]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate candidate observation")
        if expected_samples is None:
            expected_samples = set(identities)
        if set(identities) != expected_samples:
            raise ValueError("candidates have different calibration samples")
        states = sorted({r["state_id"] for r in observations})
        if len(states) != 3 or any(sum(r["state_id"] == s for r in observations) != 4 for s in states):
            raise ValueError("screen requires three complete four-frame groups")
        scores = []
        for state in states:
            subset = [r for r in observations if r["state_id"] == state]
            plus = np.asarray([r["plus_delta_z"] for r in subset], dtype=float)
            minus = -np.asarray([r["minus_delta_z"] for r in subset], dtype=float)
            if not np.isfinite(plus).all() or not np.isfinite(minus).all():
                raise ValueError("nonfinite candidate response")
            scores.append({"state_id": state, "plus_median": float(np.median(plus)),
                           "minus_reversed_median": float(np.median(minus)),
                           "plus_mean": float(plus.mean()), "minus_reversed_mean": float(minus.mean())})
        score = min(min(s["plus_median"], s["minus_reversed_median"]) for s in scores)
        ranking.append({"neuron": {"layer": layer, "index": index}, "score": score,
                        "eligible": score > 1e-8, "groups": scores})
    ranking.sort(key=lambda r: (-r["score"], r["neuron"]["layer"], r["neuron"]["index"]))
    eligible = [r for r in ranking if r["eligible"]]
    return {"ranking": ranking, "eligible_count": len(eligible),
            "selected": [r["neuron"] for r in eligible[:k]] if len(eligible) >= k else [],
            "selection_status": "FROZEN" if len(eligible) >= k else "INSUFFICIENT_CANDIDATES"}


def validation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """验证期按轨迹等权，分别汇总概率期望、实际 decoded z 和匹配随机差值。"""
    summaries = []
    for name, alpha in sorted({(r["condition"], r["alpha"]) for r in rows}):
        subset = [r for r in rows if (r["condition"], r["alpha"]) == (name, alpha)]
        states = sorted({r["state_id"] for r in subset})
        groups = [{"state_id": state,
                   "delta_expected_z": float(np.mean([r["delta_expected_z"] for r in subset if r["state_id"] == state])),
                   "delta_decoded_z": float(np.mean([r["delta_translation"][2] for r in subset if r["state_id"] == state]))}
                  for state in states]
        summaries.append({"condition": name, "alpha": alpha, "groups": groups,
                          "mean_delta_expected_z": float(np.mean([g["delta_expected_z"] for g in groups])),
                          "mean_delta_decoded_z": float(np.mean([g["delta_decoded_z"] for g in groups]))})
    contrasts = []
    for alpha in sorted({r["alpha"] for r in rows}):
        for target in ("lexical", "selected"):
            groups: dict[int, list[tuple[float, float]]] = {}
            for sample in sorted({r["sample_id"] for r in rows}):
                pair = [r for r in rows if r["sample_id"] == sample and r["alpha"] == alpha]
                conditions = {r["condition"]: r for r in pair}
                required = [target] + [f"random_{target}_{i}" for i in range(3)]
                if len(conditions) != len(pair) or any(c not in conditions for c in required):
                    raise ValueError("incomplete or duplicate validation pair")
                values = [conditions[c] for c in required]
                if len({v["state_id"] for v in values}) != 1:
                    raise ValueError("paired validation state mismatch")
                groups.setdefault(values[0]["state_id"], []).append((
                    values[0]["delta_expected_z"] - float(np.mean([v["delta_expected_z"] for v in values[1:]])),
                    values[0]["delta_translation"][2] - float(np.mean([v["delta_translation"][2] for v in values[1:]]))))
            contrasts.append({"target": target, "alpha": alpha,
                              "groups": [{"state_id": s, "expected_contrast": float(np.mean(v, axis=0)[0]),
                                          "decoded_contrast": float(np.mean(v, axis=0)[1])}
                                         for s, v in sorted(groups.items())]})
    return {"conditions": summaries, "paired_random_contrasts": contrasts}
