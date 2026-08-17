"""Fold BatchNorm into an integer threshold (Cerutti eq. 3).

Every `sign(BN(acc))` in the network becomes one integer comparison in RTL.
That is the whole point: BN needs a multiply, a divide and two adds per element
in software, but on the accumulator side it collapses to `acc >= T`, because
sign() only cares which side of zero BN lands on.

The derivation, once, since everything here depends on it.

BN in eval is an affine map: BN(y) = g'(y - mu) + b, with g' = gamma/sqrt(var+eps).
The conv before it produces an INTEGER accumulator n (for binary layers
n = 2*popcount(XNOR) - N), optionally scaled by a positive per-output-channel
alpha (XNOR-Net), so y = alpha * n. Then

    sign(BN(y)) > 0
      <=>  g'(alpha*n - mu) + b > 0
      <=>  g'*alpha*n > g'*mu - b

alpha > 0 always (it is mean|W|), so the inequality flips exactly when g' < 0:

    g' > 0:   n >  mu/alpha - b/(g'*alpha)  =: T
    g' < 0:   n <  T

which is `sign_out = sgn(g') * sgn(n - T)`. Two numbers per output channel:
an integer threshold and a one-bit polarity. No multiplier survives.

Integerization: n is an integer, so `n > T` is `n >= floor(T) + 1`, and
`n < T` is `n <= ceil(T) - 1`. Both are exact -- no rounding error is
introduced, which is why the fused model reproduces the float model bit for
bit (verified in tests/test_export.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FusedThreshold:
    """`sign(BN(alpha*n))` reduced to one integer compare per output channel.

    out[c] = +1 if (n[c] >= t[c]) == take_ge[c] else -1

    `t` is int64 and `take_ge` is bool, one of each per channel. RTL needs
    nothing else -- no gamma, beta, mean, var, alpha.
    """

    t: torch.Tensor           # [C] int64
    take_ge: torch.Tensor     # [C] bool -- True: fire when n >= t, else n <= t

    def apply(self, n: torch.Tensor) -> torch.Tensor:
        """Reference implementation of what the RTL comparator does."""
        t = self.t.view(1, -1, *([1] * (n.dim() - 2))).to(n.device)
        ge = self.take_ge.view(1, -1, *([1] * (n.dim() - 2))).to(n.device)
        fired = torch.where(ge, n >= t, n <= t)
        return torch.where(fired, 1.0, -1.0).to(n.dtype)


def fuse_bn_to_threshold(bn: nn.BatchNorm1d,
                         alpha: Optional[torch.Tensor] = None
                         ) -> FusedThreshold:
    """Collapse `sign(bn(alpha * n))` into an integer threshold per channel.

    alpha is the XNOR-Net per-output-channel weight scale (None or all-ones if
    the conv ran unscaled). It must be strictly positive -- it is mean|W| by
    construction, and a non-positive value would silently flip the comparison,
    so it is checked rather than assumed.
    """
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BN has no running stats; fuse only a model in eval "
                         "mode that has seen data (track_running_stats=True)")

    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    gamma = (bn.weight.detach().double() if bn.weight is not None
             else torch.ones_like(mu))
    beta = (bn.bias.detach().double() if bn.bias is not None
            else torch.zeros_like(mu))

    g = gamma / torch.sqrt(var + bn.eps)          # g' in the docstring
    a = (torch.ones_like(mu) if alpha is None
         else alpha.detach().double().reshape(-1))
    if bool((a <= 0).any()):
        raise ValueError("alpha must be strictly positive (it is mean|W|); "
                         "a non-positive scale would flip the comparison")

    # g' == 0 kills the channel: BN output is the constant beta, so the sign is
    # fixed. Encode it as an unreachable threshold rather than dividing by zero.
    dead = g == 0
    g_safe = torch.where(dead, torch.ones_like(g), g)
    t_real = mu / a - beta / (g_safe * a)

    take_ge = g > 0
    t_int = torch.where(take_ge,
                        torch.floor(t_real) + 1,      # n > t  <=>  n >= floor+1
                        torch.ceil(t_real) - 1)       # n < t  <=>  n <= ceil-1

    if bool(dead.any()):
        # constant output sign(beta): +1 -> always fire, -1 -> never.
        always = beta > 0
        big = torch.full_like(t_int, float(-(2 ** 40)))
        never = torch.full_like(t_int, float(2 ** 40))
        t_int = torch.where(dead, torch.where(always, big, never), t_int)
        take_ge = torch.where(dead, torch.ones_like(take_ge), take_ge)

    return FusedThreshold(t=t_int.to(torch.int64), take_ge=take_ge)


def bn_affine(bn: nn.Module) -> tuple:
    """BN in eval as a plain affine map: (g, b) with BN(y) = g*y + b, [C] float64.

    The same g' as above, but kept instead of divided away. The tail needs this
    because it has no sign() to fold into: `conv2_pw` and `conv3` end in relu, so
    their BN survives as arithmetic and export/tailfmt.py turns it into an
    integer gain and offset. An nn.Identity BN (the final stage) is the identity
    map, reported as g=1, b=0 rather than special-cased at every call site.
    """
    if isinstance(bn, nn.Identity):
        return None, None
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BN has no running stats; fuse only a model in eval "
                         "mode that has seen data (track_running_stats=True)")
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    gamma = (bn.weight.detach().double() if bn.weight is not None
             else torch.ones_like(mu))
    beta = (bn.bias.detach().double() if bn.bias is not None
            else torch.zeros_like(mu))
    g = gamma / torch.sqrt(var + bn.eps)
    return g, beta - g * mu


def conv_alpha(conv: nn.Module) -> Optional[torch.Tensor]:
    """The per-output-channel scale a BinaryConv1d applied, or None.

    Mirrors BinaryConv1d.binary_weight(): alpha_o = mean|W_o| when scale=True.
    Kept here so export never has to guess whether a layer was scaled.
    """
    if not getattr(conv, "scale", False):
        return None
    w = conv.weight.detach()
    return w.abs().mean(dim=tuple(range(1, w.dim())), keepdim=False)


def binary_accumulator(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The INTEGER accumulator a binary conv produces, before any alpha.

    Software computes sign(W) (*) x directly; hardware computes
    2*popcount(XNOR) - N. They are equal when x is +-1, which is exactly the
    equivalence tests/test_binary_ops.py pins down. This returns the software
    form so export can derive thresholds without simulating bit packing.
    """
    import torch.nn.functional as F
    w = torch.sign(conv.weight.detach())
    w = torch.where(w == 0, torch.ones_like(w), w)     # sign(0) -> +1
    return F.conv1d(x, w, None, conv.stride, conv.padding, conv.dilation,
                    conv.groups)
