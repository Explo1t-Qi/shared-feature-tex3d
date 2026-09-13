from __future__ import annotations

import pytest
import torch

from shared_feature.shared_feature_loss import (
    SharedFeatureLossError,
    shared_feature_loss,
)


SHAPE = (2, 256, 262)


def _random(seed: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(SHAPE, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_v1_equation_and_scalar_diagnostics_match_direct_reference() -> None:
    h_o_adv = _random(1)
    h_p_adv = _random(2)
    h_o_clean = _random(3)
    h_p_clean = _random(4)

    result = shared_feature_loss(h_o_adv, h_p_adv, h_o_clean, h_p_clean)
    shared_adv = (h_o_adv + h_p_adv) / 2
    shared_clean = (h_o_clean + h_p_clean) / 2
    expected = (shared_adv - shared_clean).square().mean()

    torch.testing.assert_close(result.shared_mse, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.loss, -expected, rtol=0, atol=0)
    torch.testing.assert_close(
        result.o2_mse,
        (h_o_adv - h_o_clean).square().mean(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.p2_mse,
        (h_p_adv - h_p_clean).square().mean(),
        rtol=0,
        atol=0,
    )
    delta_o = h_o_adv - h_o_clean
    delta_p = h_p_adv - h_p_clean
    decomposition = (
        result.o2_mse + result.p2_mse + 2 * (delta_o * delta_p).mean()
    ) / 4
    torch.testing.assert_close(result.shared_mse, decomposition)
    torch.testing.assert_close(
        result.o2_to_p2_mse_ratio,
        result.o2_mse / (result.p2_mse + 1e-12),
        rtol=0,
        atol=0,
    )
    assert all(
        value.shape == ()
        for value in (
            result.loss,
            result.shared_mse,
            result.o2_mse,
            result.p2_mse,
            result.displacement_cosine_mean,
            result.o2_to_p2_mse_ratio,
        )
    )


def test_zero_displacement_is_finite_and_zero() -> None:
    clean_o = _random(5)
    clean_p = _random(6)
    result = shared_feature_loss(clean_o.clone(), clean_p.clone(), clean_o, clean_p)

    for value in (
        result.loss,
        result.shared_mse,
        result.o2_mse,
        result.p2_mse,
        result.displacement_cosine_mean,
        result.o2_to_p2_mse_ratio,
    ):
        assert torch.isfinite(value)
        assert value == 0


def test_same_direction_displacements_have_unit_cosine() -> None:
    clean = torch.zeros(SHAPE, dtype=torch.float64)
    displacement = _random(7)
    result = shared_feature_loss(displacement, displacement, clean, clean)

    torch.testing.assert_close(
        result.displacement_cosine_mean,
        torch.tensor(1.0, dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(result.shared_mse, displacement.square().mean())


def test_opposite_displacements_distinguish_shared_from_native_ensemble() -> None:
    clean = torch.zeros(SHAPE, dtype=torch.float64)
    displacement = _random(8)
    result = shared_feature_loss(displacement, -displacement, clean, clean)
    native_ensemble = (result.o2_mse + result.p2_mse) / 2

    assert native_ensemble > 0
    assert result.o2_mse > 0
    assert result.p2_mse > 0
    assert result.shared_mse == 0
    assert result.loss == 0
    torch.testing.assert_close(
        result.displacement_cosine_mean,
        torch.tensor(-1.0, dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )


def test_cosine_is_computed_per_sample_before_batch_mean() -> None:
    clean = torch.zeros(SHAPE, dtype=torch.float64)
    delta_o = torch.ones(SHAPE, dtype=torch.float64)
    delta_p = delta_o.clone()
    delta_o[1] *= 2
    delta_p[1] *= -2

    result = shared_feature_loss(delta_o, delta_p, clean, clean)
    torch.testing.assert_close(
        result.displacement_cosine_mean,
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )


def test_autograd_reaches_only_adversarial_inputs() -> None:
    h_o_adv = _random(9, dtype=torch.float32).requires_grad_(True)
    h_p_adv = _random(10, dtype=torch.float32).requires_grad_(True)
    h_o_clean = _random(11, dtype=torch.float32).requires_grad_(True)
    h_p_clean = _random(12, dtype=torch.float32).requires_grad_(True)

    result = shared_feature_loss(h_o_adv, h_p_adv, h_o_clean, h_p_clean)
    result.loss.backward()

    for value in (h_o_adv, h_p_adv):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    assert h_o_clean.grad is None
    assert h_p_clean.grad is None


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (torch.zeros(2, 255, 262), "shape"),
        (torch.zeros(2, 256, 261), "shape"),
        (torch.zeros(0, 256, 262), "non-empty"),
        (torch.zeros(SHAPE, dtype=torch.int64), "float32 or float64"),
        (torch.zeros(SHAPE, dtype=torch.float64), "one dtype"),
    ),
)
def test_invalid_shape_batch_and_dtype_are_rejected(replacement, message) -> None:
    values = [torch.zeros(SHAPE) for _ in range(4)]
    values[1] = replacement
    with pytest.raises(SharedFeatureLossError, match=message):
        shared_feature_loss(*values)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf")))
def test_nonfinite_inputs_are_rejected(bad_value: float) -> None:
    values = [torch.zeros(SHAPE) for _ in range(4)]
    values[2][0, 0, 0] = bad_value
    with pytest.raises(SharedFeatureLossError, match="non-finite"):
        shared_feature_loss(*values)


@pytest.mark.parametrize("epsilon", (0.0, -1.0, float("nan"), True))
def test_invalid_epsilon_is_rejected(epsilon) -> None:
    values = [torch.zeros(SHAPE) for _ in range(4)]
    with pytest.raises(SharedFeatureLossError, match="epsilon"):
        shared_feature_loss(*values, epsilon=epsilon)


def test_non_tensor_and_device_mismatch_are_rejected() -> None:
    values = [torch.zeros(SHAPE) for _ in range(4)]
    values[0] = None
    with pytest.raises(SharedFeatureLossError, match="torch.Tensor"):
        shared_feature_loss(*values)

    if torch.cuda.is_available():
        values = [torch.zeros(SHAPE) for _ in range(4)]
        values[-1] = values[-1].cuda()
        with pytest.raises(SharedFeatureLossError, match="one device"):
            shared_feature_loss(*values)
