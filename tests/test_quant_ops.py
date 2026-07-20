"""INT8 fake-quantization tests (for Conv1/Conv3, CLAUDE.md 2.2).

The prologue/epilogue layers are int8, not binary. During QAT we keep latent
full-precision weights and fake-quantize in the forward pass: per-output-
channel symmetric scale, round via STE. What must hold:

1. Quantized weights sit exactly on an integer grid of <= 2^(bits-1)-1 steps.
2. Gradients pass through round() (STE), and latent weights stay continuous.
3. Quantization error is bounded by half a step.
"""

from __future__ import annotations

import torch

from models.quant_ops import QuantConv1d, round_ste


def test_round_ste_forward_is_round() -> None:
    x = torch.tensor([-1.6, -0.4, 0.5, 2.3])
    assert torch.equal(round_ste(x), torch.tensor([-2.0, -0.0, 0.0, 2.0]).round())
    assert torch.equal(round_ste(x), torch.round(x))


def test_round_ste_gradient_is_identity() -> None:
    x = torch.tensor([-1.6, 0.4, 2.3], requires_grad=True)
    round_ste(x).sum().backward()
    assert torch.equal(x.grad, torch.ones(3))


def test_quant_weights_live_on_integer_grid() -> None:
    torch.manual_seed(0)
    conv = QuantConv1d(16, 32, 11, bits=8)
    q = conv.quant_weight()
    s = conv.weight_scale()                      # [out, 1, 1]
    steps = q / s
    assert torch.allclose(steps, steps.round(), atol=1e-4)
    assert steps.abs().max() <= 127.0 + 1e-4


def test_quant_error_is_bounded_by_half_step() -> None:
    torch.manual_seed(1)
    conv = QuantConv1d(8, 8, 3, bits=8)
    err = (conv.quant_weight() - conv.weight).abs()
    assert torch.all(err <= conv.weight_scale() * 0.5 + 1e-6)


def test_per_channel_scale_shape_and_positivity() -> None:
    conv = QuantConv1d(4, 6, 5, bits=8)
    s = conv.weight_scale()
    assert s.shape == (6, 1, 1)
    assert torch.all(s > 0)


def test_gradient_reaches_latent_weights() -> None:
    conv = QuantConv1d(4, 8, 3, padding=1, bits=8)
    x = torch.randn(2, 4, 16)
    conv(x).pow(2).mean().backward()
    assert conv.weight.grad is not None
    assert torch.any(conv.weight.grad != 0)
    # latent weights untouched by the forward pass: still fully continuous
    # (random init makes every element distinct; the quant grid would collapse
    # them onto <= 255 shared values per channel)
    w = conv.weight.detach()
    assert w.unique().numel() == w.numel()
    steps = w / conv.weight_scale()
    assert not torch.allclose(steps, steps.round(), atol=1e-6)


def test_binary_input_is_fine_for_int8_conv() -> None:
    """Conv1 consumes the AFE's {-1,+1} image with int8 weights."""
    conv = QuantConv1d(16, 32, 11, stride=2, padding=5, bits=8)
    x = torch.randint(0, 2, (2, 16, 128)).float() * 2 - 1
    out = conv(x)
    assert out.shape == (2, 32, 64)
    assert out.unique().numel() > 2              # continuous feature space
