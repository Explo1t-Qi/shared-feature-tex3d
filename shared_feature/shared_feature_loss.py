from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional


TOKEN_COUNT = 256
CANONICAL_DIMENSION = 262
DEFAULT_EPSILON = 1e-12
_SUPPORTED_DTYPES = {torch.float32, torch.float64}


class SharedFeatureLossError(ValueError):
    """Raised when the frozen v1 shared-feature objective receives invalid input."""


@dataclass(frozen=True)
class SharedFeatureLossResult:
    loss: torch.Tensor
    shared_mse: torch.Tensor
    o2_mse: torch.Tensor
    p2_mse: torch.Tensor
    displacement_cosine_mean: torch.Tensor
    o2_to_p2_mse_ratio: torch.Tensor


def shared_feature_loss(
    h_o_adv: torch.Tensor,
    h_p_adv: torch.Tensor,
    h_o_clean: torch.Tensor,
    h_p_clean: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> SharedFeatureLossResult:
    """Compute the frozen v1 fused canonical-space displacement objective.

    Clean tensors are detached inside this function so they remain fixed references
    even if a caller accidentally supplies tensors that require gradients. The
    adversarial tensors remain connected to their upstream autograd graphs.
    """

    values = {
        "h_o_adv": h_o_adv,
        "h_p_adv": h_p_adv,
        "h_o_clean": h_o_clean,
        "h_p_clean": h_p_clean,
    }
    _validate_inputs(values, epsilon=epsilon)

    clean_o = h_o_clean.detach()
    clean_p = h_p_clean.detach()
    delta_o = h_o_adv - clean_o
    delta_p = h_p_adv - clean_p

    shared_clean = (clean_o + clean_p) / 2
    shared_adv = (h_o_adv + h_p_adv) / 2
    shared_mse = (shared_adv - shared_clean).square().mean()
    o2_mse = delta_o.square().mean()
    p2_mse = delta_p.square().mean()

    flat_o = delta_o.flatten(start_dim=1)
    flat_p = delta_p.flatten(start_dim=1)
    cosine_per_sample = functional.cosine_similarity(
        flat_o,
        flat_p,
        dim=1,
        eps=epsilon,
    )
    cosine_mean = cosine_per_sample.mean()
    ratio = o2_mse / (p2_mse + epsilon)
    loss = -shared_mse

    diagnostics = {
        "loss_shared": loss,
        "shared_mse": shared_mse,
        "o2_mse": o2_mse,
        "p2_mse": p2_mse,
        "displacement_cosine_mean": cosine_mean,
        "o2_to_p2_mse_ratio": ratio,
    }
    if any(
        value.ndim != 0 or not bool(torch.isfinite(value))
        for value in diagnostics.values()
    ):
        raise SharedFeatureLossError("shared-feature loss produced invalid diagnostics")

    return SharedFeatureLossResult(
        loss=loss,
        shared_mse=shared_mse,
        o2_mse=o2_mse,
        p2_mse=p2_mse,
        displacement_cosine_mean=cosine_mean,
        o2_to_p2_mse_ratio=ratio,
    )


def _validate_inputs(values: dict[str, torch.Tensor], *, epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise SharedFeatureLossError("epsilon must be a finite positive scalar")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise SharedFeatureLossError("epsilon must be a finite positive scalar")

    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise SharedFeatureLossError(f"{name} must be a torch.Tensor")
        if value.ndim != 3 or tuple(value.shape[1:]) != (
            TOKEN_COUNT,
            CANONICAL_DIMENSION,
        ):
            raise SharedFeatureLossError(
                f"{name} must have shape [B,{TOKEN_COUNT},{CANONICAL_DIMENSION}]"
            )
        if value.shape[0] <= 0:
            raise SharedFeatureLossError(f"{name} must have a non-empty batch")
        if value.dtype not in _SUPPORTED_DTYPES:
            raise SharedFeatureLossError(f"{name} must use float32 or float64")

    reference = values["h_o_adv"]
    for name, value in values.items():
        if value.shape != reference.shape:
            raise SharedFeatureLossError("all shared-feature inputs must have one shape")
        if value.device != reference.device:
            raise SharedFeatureLossError("all shared-feature inputs must use one device")
        if value.dtype != reference.dtype:
            raise SharedFeatureLossError("all shared-feature inputs must use one dtype")
        if not bool(torch.isfinite(value).all()):
            raise SharedFeatureLossError(f"{name} contains non-finite values")
