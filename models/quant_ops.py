"""INT8 fake-quantization for the non-binary layers (Conv1/Conv3).

CLAUDE.md 2.2 puts int8 weights on the prologue/epilogue. During QAT we keep
latent full-precision weights and fake-quantize on the forward pass:

    scale_o = max|W_o| / 127          (per output channel, symmetric)
    W_q     = clamp(round(W / scale), -127, 127) * scale

round() has zero gradient, so it goes through an identity-STE, same recipe as
the binary layers. `fixed` precision (Conv4) intentionally has no module here:
it trains in full precision and becomes fixed-point only at export, because
its output feeds the classifier head where CLAUDE.md forbids precision loss
during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        return grad_out


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """round() forward, identity backward."""
    return _RoundSTE.apply(x)


class QuantConv1d(nn.Conv1d):
    """1D conv with per-output-channel symmetric int8 fake-quantized weights."""

    def __init__(self, *args, bits: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bits < 2:
            raise ValueError(f"bits must be >= 2, got {bits}")
        self.bits = bits
        self.qmax = float(2 ** (bits - 1) - 1)   # 127 for int8

    def weight_scale(self) -> torch.Tensor:
        """Per-output-channel step size, detached (scale is not learned)."""
        s = self.weight.detach().abs().amax(dim=(1, 2), keepdim=True) / self.qmax
        return s.clamp_min(1e-8)

    def quant_weight(self) -> torch.Tensor:
        s = self.weight_scale()
        q = round_ste(self.weight / s).clamp(-self.qmax, self.qmax)
        return q * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.quant_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, fake_quant=int{self.bits}"


class ActQuant(nn.Module):
    """Per-tensor fake-quantized ACTIVATIONS (model.act_bits > 0).

    QuantConv1d quantizes weights only, so an "int8" network still trains on
    float activations and its accuracy is an upper bound -- the hardware has
    to cut every activation to a fixed width. This closes that gap for the
    comparison runs (W8A8 when act_bits=8).

        scale = running_max / qmax          (EMA of max|x|, like BN's stats)
        x_q   = clamp(round(x / scale), lo, qmax) * scale

    `signed` False for post-ReLU tensors (0..2^bits-1), True for the raw
    depthwise output between dw and pw (-qmax..qmax). Gradient: identity
    inside the range, zero where clamped (the usual QAT straight-through).
    The scale is a buffer, not learned -- frozen in eval, as BN is.
    """

    def __init__(self, bits: int, *, signed: bool,
                 momentum: float = 0.1) -> None:
        super().__init__()
        if bits < 2:
            raise ValueError(f"act bits must be >= 2, got {bits}")
        self.bits, self.signed, self.momentum = bits, signed, momentum
        self.qmax = float(2 ** (bits - 1) - 1 if signed else 2 ** bits - 1)
        self.lo = -self.qmax if signed else 0.0
        self.register_buffer("running_max", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            m = x.detach().abs().amax().float()
            if self.running_max == 0:                # first batch seeds it
                self.running_max.copy_(m)
            else:
                self.running_max.mul_(1 - self.momentum).add_(self.momentum * m)
        s = (self.running_max / self.qmax).clamp_min(1e-8).to(x.dtype)
        return round_ste(x / s).clamp(self.lo, self.qmax) * s

    def extra_repr(self) -> str:
        return f"bits={self.bits}, signed={self.signed}"

