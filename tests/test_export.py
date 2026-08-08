"""Export tests: the fused integer model must match the float model exactly.

These are the tests the RTL will be verified against, so they pin the two
things RTL can silently get wrong -- the direction of a threshold comparison
and the bit order of a packed word.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from export.fuse import (FusedThreshold, binary_accumulator, conv_alpha,
                         fuse_bn_to_threshold)
from export.pack import (WORD_BITS, pack_pm1, quantize_int8, to_hex_words,
                         unpack_pm1)
from models.binary_ops import PointwiseBinaryConv1d, sign_ste


def _trained_bn(ch: int, seed: int = 0) -> nn.BatchNorm1d:
    """A BN with non-trivial running stats, gamma and beta of both signs."""
    torch.manual_seed(seed)
    bn = nn.BatchNorm1d(ch)
    with torch.no_grad():
        bn.weight.copy_(torch.randn(ch))          # gamma: mixed signs
        bn.bias.copy_(torch.randn(ch))
        bn.running_mean.copy_(torch.randn(ch) * 5)
        bn.running_var.copy_(torch.rand(ch) * 4 + 0.1)
    return bn.eval()


# --------------------------------------------------------------------------- #
# 1. BN -> integer threshold is EXACT
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scaled", [False, True])
def test_fused_threshold_matches_sign_of_bn(scaled: bool) -> None:
    """The whole export rests on this: one integer compare reproduces
    sign(BN(alpha*n)) for every integer accumulator value."""
    ch = 12
    bn = _trained_bn(ch)
    alpha = (torch.rand(ch) + 0.5) if scaled else None

    n = torch.arange(-60, 61).view(1, 1, -1).expand(1, ch, -1).float().clone()
    n = n + torch.arange(ch).view(1, -1, 1)        # de-align the channels
    y = n * (alpha.view(1, -1, 1) if scaled else 1.0)

    want = torch.where(bn(y) > 0, 1.0, -1.0)
    got = fuse_bn_to_threshold(bn, alpha).apply(n)
    assert torch.equal(got, want)


def test_fused_threshold_handles_negative_gamma() -> None:
    """gamma < 0 flips the inequality. If export got this wrong the model would
    still 'work' -- just with those channels inverted -- so it is pinned."""
    bn = _trained_bn(6)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5]))
    f = fuse_bn_to_threshold(bn)
    assert torch.equal(f.take_ge, torch.tensor([True, False, True, False,
                                                True, False]))
    n = torch.arange(-40, 41).view(1, 1, -1).expand(1, 6, -1).float()
    assert torch.equal(f.apply(n), torch.where(bn(n) > 0, 1.0, -1.0))


def test_fused_threshold_handles_dead_channel() -> None:
    """gamma == 0 makes BN a constant; the channel must be pinned on or off
    rather than dividing by zero."""
    bn = _trained_bn(2)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([0.0, 0.0]))
        bn.bias.copy_(torch.tensor([3.0, -3.0]))   # +1 always / -1 always
    f = fuse_bn_to_threshold(bn)
    n = torch.randn(1, 2, 50) * 100
    out = f.apply(n)
    assert torch.all(out[:, 0] == 1.0)
    assert torch.all(out[:, 1] == -1.0)


def test_fuse_rejects_non_positive_alpha() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fuse_bn_to_threshold(_trained_bn(3), torch.tensor([1.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# 2. a real binary conv + BN, end to end
# --------------------------------------------------------------------------- #
def test_binary_layer_fuses_exactly() -> None:
    """conv -> BN -> sign, as the network actually runs it, against
    integer accumulator -> threshold, as the RTL will run it."""
    torch.manual_seed(0)
    conv = PointwiseBinaryConv1d(16, 24, scale=True)
    bn = _trained_bn(24, seed=1)
    x = torch.where(torch.rand(4, 16, 30) > 0.5, 1.0, -1.0)

    want = sign_ste(bn(conv(x)), 1.0)                       # software path
    n = binary_accumulator(conv, x)                          # integer path
    got = fuse_bn_to_threshold(bn, conv_alpha(conv)).apply(n)
    assert torch.equal(got, want)


def test_accumulator_is_integral_and_matches_popcount_form() -> None:
    """n must be an integer and equal 2*popcount(XNOR) - N, since that is what
    the RTL computes. A non-integer here means alpha leaked in."""
    torch.manual_seed(0)
    conv = PointwiseBinaryConv1d(32, 8, scale=True)
    x = torch.where(torch.rand(2, 32, 5) > 0.5, 1.0, -1.0)
    n = binary_accumulator(conv, x)
    assert torch.equal(n, n.round())

    w = torch.sign(conv.weight.detach()).squeeze(-1)         # [8, 32]
    agree = (w.unsqueeze(0).unsqueeze(-1) == x.unsqueeze(1)).sum(dim=2)
    assert torch.equal(n, (2 * agree - 32).float())


# --------------------------------------------------------------------------- #
# 3. bit packing -- order is the classic silent bug
# --------------------------------------------------------------------------- #
def test_pack_is_lsb_first() -> None:
    """word[i] in Verilog must be arr[i] in Python. An asymmetric pattern
    catches a reversal that a symmetric one would hide."""
    x = torch.full((WORD_BITS,), -1.0)
    x[0] = 1.0                                   # only element 0 -> only bit 0
    assert int(pack_pm1(x).words[0]) == 1
    x = torch.full((WORD_BITS,), -1.0)
    x[3] = 1.0
    assert int(pack_pm1(x).words[0]) == 0b1000


def test_pack_roundtrip_including_partial_word() -> None:
    torch.manual_seed(0)
    for n in (1, 31, 32, 33, 64, 100):
        x = torch.where(torch.rand(3, 5, n) > 0.5, 1.0, -1.0)
        p = pack_pm1(x)
        assert p.n_valid == n
        assert p.n_words == (n + 31) // 32
        assert torch.equal(unpack_pm1(p), x)


def test_pack_pads_with_zeros_which_decode_as_minus_one() -> None:
    """The tail is real -1s, not don't-cares. RTL must mask it before popcount
    or every partial word adds a bias, so the padding value is pinned."""
    x = torch.ones(33)                            # 1 valid bit in word 1
    p = pack_pm1(x)
    assert int(p.words[1]) == 1                   # bit 0 set, bits 1..31 zero
    assert p.n_valid == 33


def test_pack_rejects_non_pm1() -> None:
    with pytest.raises(ValueError, match="expected a"):
        pack_pm1(torch.tensor([0.0, 1.0, -1.0]))


def test_hex_words_are_eight_digits() -> None:
    x = torch.where(torch.rand(64) > 0.5, 1.0, -1.0)
    hx = to_hex_words(pack_pm1(x))
    assert len(hx) == 2 and all(len(h) == 8 for h in hx)


# --------------------------------------------------------------------------- #
# 4. int8 quantization
# --------------------------------------------------------------------------- #
def test_int8_quantization_is_symmetric_and_close() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 8, 5) * 0.3
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8
    assert int(q.abs().max()) <= 127
    assert scale.shape == (16,)
    err = (q.float() * scale.view(-1, 1, 1) - w).abs().max()
    assert float(err) < float(w.abs().max()) / 100
