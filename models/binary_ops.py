"""Binary primitives: sign-STE and ±1 convolutions.

Everything here is written so that the software forward pass is *bit-exact*
with the XNOR/popcount datapath the FPGA will run (Cerutti section IV-B):

    a, w in {-1,+1}
    a*w                       == XNOR(a_hat, w_hat)      with x_hat = (x+1)/2
    sum_n a_n * w_n           == 2 * popcount(XNOR) - N

We compute it with `F.conv1d` in float because that is fast and
autograd-friendly, but the *values* are exactly the integers the hardware
produces -- `tests/test_binary_ops.py` asserts this against a bool/popcount
reference implementation.

Two deliberate deviations, both confined to training:

* Weights are stored full-precision (latent/shadow weights) and binarized in
  the forward pass only. This is standard QAT: the optimizer needs somewhere
  continuous to accumulate small updates, otherwise nothing ever flips.
* `scale=True` multiplies each output channel by alpha = mean(|W_o|)
  (XNOR-Net). This breaks exact ±1-ness, but the following BatchNorm absorbs
  any positive per-channel scale, so it vanishes when BN is fused into the
  integer threshold at export time (Cerutti eq. 3). Pass `scale=False` when
  the raw integer accumulator value matters.

Zero padding note: a padded 0 is not a ±1 value, so the popcount identity
holds only on the valid region. That is fine -- 0 contributes 0 to the
accumulator, which is what zero padding means in hardware too.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# straight-through estimator
# --------------------------------------------------------------------------- #
class _SignSTE(torch.autograd.Function):
    """sign() forward, clipped identity backward.

    d/dx sign(x) is 0 almost everywhere, which would zero out every gradient
    upstream of a binary layer. The standard fix (Hubara et al. / Courbariaux)
    is to pretend the forward was hardtanh: pass the gradient through where
    |x| <= clip and drop it outside, so latent weights that have already
    saturated stop being pushed further out.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, clip: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.clip = clip
        # torch.sign(0) == 0, which is not a valid binary value; map 0 -> +1.
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_out * (x.abs() <= ctx.clip).to(grad_out.dtype), None


def sign_ste(x: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """Binarize to {-1,+1} with a straight-through gradient."""
    return _SignSTE.apply(x, clip)


def binary_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """±1 dot product, computed the way the hardware does it.

    Provided as executable documentation of `2*popcount(XNOR) - N`; the model
    itself uses conv1d, which the tests prove equivalent.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    xnor = ~((a > 0) ^ (b > 0))
    return 2 * xnor.sum() - a.numel()


# --------------------------------------------------------------------------- #
# binary convolutions
# --------------------------------------------------------------------------- #
class BinaryConv1d(nn.Conv1d):
    """1D convolution with binary weights, XNOR/popcount-equivalent.

    Inputs are expected to already be in {-1,+1} (the AFE binarizes the input;
    every interior layer ends in a sign activation). Nothing enforces that at
    runtime -- it would cost a full tensor scan per forward -- but the
    popcount equivalence only holds when it is true.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        scale: bool = True,
        ste_clip: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.scale = scale
        self.ste_clip = ste_clip

    def binary_weight(self) -> torch.Tensor:
        """The weight actually used in the forward pass."""
        w = sign_ste(self.weight, self.ste_clip)
        if self.scale:
            # alpha_o = mean |W_o|, one positive scalar per output channel.
            alpha = self.weight.detach().abs().mean(
                dim=tuple(range(1, self.weight.dim())), keepdim=True
            )
            w = w * alpha
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x, self.binary_weight(), self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, binary=True, scale={self.scale}"


class DepthwiseBinaryConv1d(BinaryConv1d):
    """Per-channel temporal convolution -- the 'time' half of a TCS block."""

    def __init__(self, channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = False,
                 scale: bool = True, ste_clip: float = 1.0) -> None:
        super().__init__(
            channels, channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=channels, bias=bias, scale=scale, ste_clip=ste_clip,
        )


class PointwiseBinaryConv1d(BinaryConv1d):
    """1x1 channel-mixing convolution -- the 'channel' half of a TCS block."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False,
                 scale: bool = True, ste_clip: float = 1.0) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size=1,
            bias=bias, scale=scale, ste_clip=ste_clip,
        )


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    """Padding that keeps the time axis length unchanged at stride 1."""
    return ((kernel_size - 1) * dilation) // 2
