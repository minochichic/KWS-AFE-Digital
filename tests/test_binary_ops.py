"""Binary layer tests (FIRST_TASK.md step 2).

Two things must hold before any of the model code is worth trusting:

1. Our ±1 convolution is *numerically identical* to what the FPGA will do with
   XNOR + popcount, i.e. `dot = 2*popcount(xnor(a,b)) - N`. If this drifts, the
   trained software model and the exported hardware disagree silently.
2. Gradients survive the sign() non-linearity via the STE. sign' is 0 almost
   everywhere, so without the STE the whole binary stack trains nothing.
"""

from __future__ import annotations

import pytest
import torch

from models.binary_ops import (
    BinaryConv1d,
    DepthwiseBinaryConv1d,
    PointwiseBinaryConv1d,
    binary_dot,
    sign_ste,
)


# --------------------------------------------------------------------------- #
# reference implementation: what the hardware actually computes
# --------------------------------------------------------------------------- #
def xnor_popcount_conv1d(x_pm1: torch.Tensor, w_pm1: torch.Tensor,
                         groups: int = 1) -> torch.Tensor:
    """Bit-domain reference for a ±1 conv, using only bools and popcount.

    Mirrors Cerutti section IV-B: -1 is stored as 0, multiply becomes XNOR,
    accumulate becomes popcount, and the ±1-domain result is 2*popcount - N.
    No padding -- zero padding is not a ±1 value, so the identity only holds
    on the valid region.
    """
    xh = x_pm1 > 0                       # {-1,+1} -> {0,1}
    wh = w_pm1 > 0
    B, Cin, L = xh.shape
    Cout, Cin_g, K = wh.shape
    Lout = L - K + 1
    out = torch.zeros(B, Cout, Lout, dtype=torch.long)

    patches = xh.unfold(2, K, 1)         # [B, Cin, Lout, K]
    N = Cin_g * K                        # taps accumulated per output
    out_per_group = Cout // groups

    for o in range(Cout):
        g = o // out_per_group
        sl = slice(g * Cin_g, (g + 1) * Cin_g)
        xnor = ~(patches[:, sl] ^ wh[o].unsqueeze(0).unsqueeze(2))
        out[:, o, :] = 2 * xnor.sum(dim=(1, 3)) - N
    return out


# --------------------------------------------------------------------------- #
# 1. the core identity
# --------------------------------------------------------------------------- #
def test_binary_dot_matches_popcount_formula() -> None:
    torch.manual_seed(0)
    a = torch.randint(0, 2, (256,)) * 2 - 1
    b = torch.randint(0, 2, (256,)) * 2 - 1

    expected = (a * b).sum()
    popcount = (~((a > 0) ^ (b > 0))).sum()
    assert 2 * popcount - a.numel() == expected
    assert binary_dot(a, b) == expected


def test_binary_dot_extremes() -> None:
    ones = torch.ones(64)
    assert binary_dot(ones, ones) == 64            # all agree
    assert binary_dot(ones, -ones) == -64          # all disagree


@pytest.mark.parametrize("kernel", [1, 3, 13])
@pytest.mark.parametrize("cin,cout", [(4, 8), (16, 16)])
def test_binary_conv1d_matches_xnor_popcount(kernel: int, cin: int,
                                             cout: int) -> None:
    """The whole point: our conv == the hardware's XNOR/popcount pipeline."""
    torch.manual_seed(kernel + cin)
    conv = BinaryConv1d(cin, cout, kernel, padding=0, scale=False, bias=False)

    x = (torch.randint(0, 2, (2, cin, 40)).float() * 2 - 1)
    with torch.no_grad():
        got = conv(x)
        ref = xnor_popcount_conv1d(x, conv.binary_weight())

    assert torch.equal(got, ref.float())


def test_depthwise_matches_xnor_popcount() -> None:
    torch.manual_seed(1)
    ch = 16
    conv = DepthwiseBinaryConv1d(ch, kernel_size=13, padding=0, scale=False)

    x = (torch.randint(0, 2, (2, ch, 40)).float() * 2 - 1)
    with torch.no_grad():
        got = conv(x)
        ref = xnor_popcount_conv1d(x, conv.binary_weight(), groups=ch)

    assert conv.groups == ch
    assert torch.equal(got, ref.float())


def test_pointwise_matches_xnor_popcount() -> None:
    torch.manual_seed(2)
    conv = PointwiseBinaryConv1d(16, 32, scale=False)

    x = (torch.randint(0, 2, (2, 16, 20)).float() * 2 - 1)
    with torch.no_grad():
        got = conv(x)
        ref = xnor_popcount_conv1d(x, conv.binary_weight())

    assert conv.kernel_size == (1,)
    assert torch.equal(got, ref.float())


def test_binary_conv_output_parity_is_structured() -> None:
    """2*popcount - N has the same parity as N, for every output element.

    A cheap invariant that catches accidental float contamination of the
    accumulator (e.g. a stray scale factor sneaking in).
    """
    conv = BinaryConv1d(8, 4, 5, padding=0, scale=False, bias=False)
    x = (torch.randint(0, 2, (1, 8, 30)).float() * 2 - 1)
    with torch.no_grad():
        out = conv(x)
    N = 8 * 5
    assert torch.all((out.long() - N) % 2 == 0)


# --------------------------------------------------------------------------- #
# 2. straight-through estimator
# --------------------------------------------------------------------------- #
def test_sign_ste_forward_is_hard_sign() -> None:
    x = torch.tensor([-2.0, -0.3, 0.0, 0.4, 5.0])
    out = sign_ste(x)
    assert torch.equal(out, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))
    assert set(out.unique().tolist()) <= {-1.0, 1.0}


def test_sign_ste_gradient_flows() -> None:
    """Without the STE this gradient would be exactly zero everywhere."""
    x = torch.tensor([-0.5, 0.2, 0.9], requires_grad=True)
    sign_ste(x).sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad != 0)
    assert torch.equal(x.grad, torch.ones(3))      # identity inside the clip


def test_sign_ste_gradient_is_clipped_outside_window() -> None:
    """hardtanh STE: latent weights far from 0 stop receiving gradient."""
    x = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    sign_ste(x, clip=1.0).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_sign_ste_respects_custom_clip() -> None:
    x = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    sign_ste(x, clip=4.0).sum().backward()
    assert torch.equal(x.grad, torch.ones(4))


def test_gradient_reaches_latent_weights_through_conv() -> None:
    conv = BinaryConv1d(4, 8, 3, padding=1, scale=False)
    x = (torch.randint(0, 2, (2, 4, 16)).float() * 2 - 1)
    conv(x).pow(2).mean().backward()

    assert conv.weight.grad is not None
    assert torch.any(conv.weight.grad != 0)


def test_latent_weights_stay_full_precision() -> None:
    """QAT keeps a real-valued shadow weight; only the forward pass binarizes."""
    conv = BinaryConv1d(4, 8, 3, scale=False)
    before = conv.weight.detach().clone()

    x = (torch.randint(0, 2, (2, 4, 16)).float() * 2 - 1)
    conv(x).sum().backward()

    assert torch.equal(conv.weight.detach(), before)   # forward is side-effect free
    assert conv.weight.detach().unique().numel() > 2   # still continuous
    assert set(conv.binary_weight().unique().tolist()) <= {-1.0, 1.0}


def test_latent_weights_actually_update_under_sgd() -> None:
    """End-to-end: STE gradients must move the latent weights, and eventually
    flip a sign -- otherwise the binary net can never change."""
    torch.manual_seed(3)
    conv = BinaryConv1d(4, 4, 3, padding=1, scale=False)
    opt = torch.optim.SGD(conv.parameters(), lr=0.5)

    x = (torch.randint(0, 2, (8, 4, 16)).float() * 2 - 1)
    target = torch.randn(8, 4, 16)

    signs_before = conv.binary_weight().clone()
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.mse_loss(conv(x), target).backward()
        opt.step()

    assert not torch.equal(conv.binary_weight(), signs_before)


# --------------------------------------------------------------------------- #
# 3. weight scaling (XNOR-Net alpha)
# --------------------------------------------------------------------------- #
def test_scaling_is_per_output_channel() -> None:
    conv = BinaryConv1d(4, 8, 3, scale=True)
    w = conv.binary_weight()
    # Each output channel is alpha_o * (±1), so |w| is constant within a channel.
    per_channel = w.abs().flatten(1)
    assert torch.allclose(per_channel.min(dim=1).values,
                          per_channel.max(dim=1).values)
    assert per_channel.min(dim=1).values.unique().numel() > 1  # differs across o


def test_scaling_preserves_sign_pattern() -> None:
    """alpha > 0, so scaling must not change which taps are +1 vs -1."""
    torch.manual_seed(4)
    conv = BinaryConv1d(4, 8, 3, scale=True)
    assert torch.equal(torch.sign(conv.binary_weight()),
                       torch.sign(conv.weight.detach()).sign())


def test_unscaled_weights_are_exactly_pm1() -> None:
    conv = BinaryConv1d(4, 8, 3, scale=False)
    assert set(conv.binary_weight().unique().tolist()) <= {-1.0, 1.0}
