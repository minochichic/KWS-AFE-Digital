"""The tail's fixed-point fold, checked without torch.

export/tailfmt.py is the one place emit.py, golden.py and the RTL agree on what
"8.6" means, so a bug here is a bug in all three at once and shows up as a
fraction of a percent of accuracy with nothing in the logs. These tests check
the fold against float arithmetic done independently, at the actual scales the
network uses.
"""
from __future__ import annotations

import random

import pytest

from export.tailfmt import (FMT_CONV2_PW, FMT_CONV3, FMT_CONV4, FRAC_BITS,
                            GAIN_BITS, AffineFold, FixedFormat,
                            acc_bits_for_real_input, fold_affine,
                            pooled_argmax, round_shift, signed_bits,
                            _ceil_log2)

torch_missing = False
try:
    import torch  # noqa: F401
except ImportError:
    torch_missing = True


# --------------------------------------------------------------------------- #
# formats
# --------------------------------------------------------------------------- #

def test_the_three_formats_are_the_ones_the_sweep_confirmed():
    assert (FMT_CONV2_PW.int_bits, FMT_CONV2_PW.frac_bits) == (8, 6)
    assert (FMT_CONV3.int_bits, FMT_CONV3.frac_bits) == (5, 6)
    assert (FMT_CONV4.int_bits, FMT_CONV4.frac_bits) == (8, 6)
    assert FMT_CONV2_PW.bits == 14 and FMT_CONV3.bits == 11
    assert FRAC_BITS == 6


def test_conv4_range_covers_the_measured_logits():
    """ranges.json measured conv4 at [-68.1, 26.1]. int_bits includes the sign,
    so 8.6 reaches -128..+127.98 and the negative side is what sizes it."""
    assert FMT_CONV4.to_real(FMT_CONV4.lo) <= -68.1
    assert FMT_CONV4.to_real(FMT_CONV4.hi) >= 26.1
    # one bit less would not: 7.6 stops at -64
    assert FixedFormat(7).to_real(FixedFormat(7).lo) > -68.1


def test_clamp_saturates_rather_than_wrapping():
    f = FMT_CONV3
    assert f.clamp(f.hi + 1) == f.hi
    assert f.clamp(f.lo - 1) == f.lo
    assert f.clamp(0) == 0


def test_step_is_the_grid_the_sweep_measured():
    assert FMT_CONV3.step == pytest.approx(0.015625)     # frac=6 row in README


# --------------------------------------------------------------------------- #
# round_shift
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("x,s,want", [
    (0, 4, 0),
    (7, 1, 4),        # 3.5 -> 4, half UP
    (-7, 1, -3),      # -3.5 -> -3, half up means toward +inf
    (8, 1, 4),
    (5, 0, 5),        # shift 0 is the identity, not a rounding
])
def test_round_shift_is_half_up_in_both_signs(x, s, want):
    assert round_shift(x, s) == want


def test_round_shift_never_drifts_more_than_half_an_lsb():
    rng = random.Random(7)
    for _ in range(2000):
        x = rng.randint(-(1 << 30), 1 << 30)
        s = rng.randint(1, 20)
        assert abs(round_shift(x, s) - x / float(1 << s)) <= 0.5


def test_ceil_log2_handles_the_sub_unit_gains_the_tail_actually_has():
    """BN gains times a binary alpha land well below 1, and if _ceil_log2 were
    only right for x >= 1 the shift would come out too small there -- exactly
    the case that quantizes the gain."""
    assert _ceil_log2(1.0) == 0
    assert _ceil_log2(1.5) == 1
    assert _ceil_log2(2.0) == 1
    assert _ceil_log2(0.5) == -1
    assert _ceil_log2(0.4) == -1
    assert _ceil_log2(0.25) == -2
    assert _ceil_log2(0.003) == -8
    for e in range(-20, 12):
        x = 2.0 ** e
        assert _ceil_log2(x) == e, x


# --------------------------------------------------------------------------- #
# the fold itself
# --------------------------------------------------------------------------- #

def _realistic(n=128, seed=3):
    """Gains and biases at the scale the tail actually has them.

    conv2_pw's gain is gamma' * alpha: BN gains are O(0.1-1) and a binary
    layer's alpha is O(0.01-0.1), so the product is O(1e-3) -- three orders
    below 1, which is the regime where a naive shift choice loses precision.
    """
    rng = random.Random(seed)
    gain = [rng.uniform(2e-4, 8e-3) * rng.choice([1.0, 1.0, 1.0, -1.0])
            for _ in range(n)]
    bias = [rng.uniform(-3.0, 3.0) for _ in range(n)]
    return gain, bias


def test_the_fold_reproduces_the_float_result_on_the_output_grid():
    gain, bias = _realistic()
    f = fold_affine("conv2_pw", gain, bias, FMT_CONV2_PW, relu=True)
    rng = random.Random(11)
    for _ in range(4000):
        ch = rng.randrange(len(gain))
        acc = rng.randint(-64, 64)                # binary_pw acc, 8 bits signed
        got = f.apply(ch, acc)
        want_real = f.apply_real(ch, acc)
        want = max(0, int(round(want_real * (1 << FRAC_BITS))))
        # one LSB of slack: round-half-up vs round-half-even on exact ties
        assert abs(got - want) <= 1, (ch, acc, got, want, want_real)


# (name, format, accumulator bound). The accumulator bound is what the layer
# can actually reach: conv2_pw sums 64 binary terms; conv3 sums 128 int8 terms
# against an 8.6 relu output; conv4 sums 128 against a 5.6 one.
TAIL_SITES = [
    ("conv2_pw", FMT_CONV2_PW, 1, 64),
    ("conv3", FMT_CONV3, 2, 128 * 127 * FMT_CONV2_PW.hi),
    ("conv4", FMT_CONV4, 3, 128 * 127 * FMT_CONV3.hi),
]


@pytest.mark.parametrize("name,fmt,seed,acc_max", TAIL_SITES)
def test_the_fold_adds_far_less_than_one_output_lsb(name, fmt, seed, acc_max):
    """The metric that matters, and the one this file originally got wrong.

    Relative gain error is largest on the quietest channel, which is also the
    channel with the smallest output swing, so it overstates the damage. What
    is observable is error in LSBs of the output grid -- and that has to be a
    small fraction of one, because a whole LSB at frac=6 is the step the sweep
    measured as costing 0.0pp.
    """
    gain, bias = _realistic(seed=seed)
    f = fold_affine(name, gain, bias, fmt, relu=True)
    err = f.max_output_error_lsb(acc_max)
    assert err < 0.25, (name, f.shift, err, f.max_rel_gain_error())


@pytest.mark.parametrize("name,fmt,seed,acc_max", TAIL_SITES)
def test_no_channel_is_deleted_by_the_fold(name, fmt, seed, acc_max):
    """A gain that rounds to zero removes a channel from the network, and
    nothing downstream can tell that from a channel the training killed."""
    gain, bias = _realistic(seed=seed)
    f = fold_affine(name, gain, bias, fmt, relu=True)
    assert f.dead_gains() == []


def test_the_gain_stays_inside_the_width_it_was_given():
    gain, bias = _realistic()
    f = fold_affine("conv2_pw", gain, bias, FMT_CONV2_PW, relu=True)
    assert f.gain_bits_used() <= GAIN_BITS, f.gain_bits_used()
    # and it USES the width -- a shift far too small would also pass the line
    # above while quantizing the gain to nothing
    assert f.gain_bits_used() >= GAIN_BITS - 2, f.gain_bits_used()


def test_the_quietest_channel_still_gets_real_bits():
    """The shift is shared, so the spread in the trained BN gains decides how
    many bits the quiet end gets. Three orders of spread costs ~10 bits; if the
    quiet end fell to 4 or 5 the fold would start being visible there while
    every aggregate number still looked fine."""
    gain, bias = _realistic()
    f = fold_affine("conv2_pw", gain, bias, FMT_CONV2_PW, relu=True)
    assert f.quietest_gain_bits() >= 10, (f.quietest_gain_bits(), f.shift)


def test_a_tiny_gain_does_not_collapse_to_zero():
    """A near-dead channel is where a shared shift would hurt first. It gets
    fewer bits than the loud channels, which is correct -- it contributes less
    -- but it must not round to A=0, which would delete the channel."""
    gain = [1e-2, 1e-3, 1e-4, 1e-5]
    f = fold_affine("t", gain, [0.0] * 4, FMT_CONV2_PW, relu=False)
    assert f.dead_gains() == []
    assert f.max_output_error_lsb(64) < 0.01


def test_the_output_error_bound_takes_the_reachable_one():
    """conv3's accumulator bound alone says 124 LSB, which is unreachable: a
    channel cannot sit at 2^27 of accumulator and inside an 11-bit output at the
    same time. Taking the min of the two bounds is what makes the number honest,
    and taking only the accumulator one is what would send this chasing a
    31-bit shift for nothing."""
    gain, bias = _realistic(seed=2)
    f = fold_affine("conv3", gain, bias, FMT_CONV3, relu=True)
    acc_max = 128 * 127 * FMT_CONV2_PW.hi
    naive = (0.5 * acc_max + 0.5) / float(1 << f.shift)
    assert naive > 1.0, naive                       # the misleading bound
    assert f.max_output_error_lsb(acc_max) < naive  # the reachable one wins


def test_relu_is_applied_before_the_clamp_not_after():
    """Order matters at the negative end: clamping first would map a big
    negative to lo and then relu would lift it to 0 anyway -- same answer -- but
    clamping a positive overflow AFTER relu is what keeps the wire in range."""
    f = fold_affine("t", [1.0], [0.0], FixedFormat(3), relu=True)
    assert f.apply(0, -1000) == 0
    assert f.apply(0, 1000) == FixedFormat(3).hi


def test_a_pinned_shift_is_honoured():
    """Re-exports must be able to reproduce an existing ROM bit for bit."""
    gain, bias = _realistic()
    a = fold_affine("t", gain, bias, FMT_CONV3, relu=True)
    b = fold_affine("t", gain, bias, FMT_CONV3, relu=True, shift=a.shift)
    assert (a.gain, a.bias, a.shift) == (b.gain, b.bias, b.shift)


def test_mismatched_gain_and_bias_lengths_are_refused():
    with pytest.raises(ValueError):
        fold_affine("t", [1.0, 2.0], [0.0], FMT_CONV3, relu=True)


# --------------------------------------------------------------------------- #
# accumulator widths
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(torch_missing, reason="export.ranges imports torch")
def test_signed_bits_matches_the_rule_ranges_py_uses():
    """tailfmt has its own copy because it must run without torch. Two copies
    of a width rule that disagree would size a ROM one way and a wire the
    other, so they are tied together here."""
    from export.ranges import signed_bits as ref
    for lo, hi in [(0, 0), (-1, 1), (-128, 127), (-64, 64), (0, 8191),
                   (-8192, 8191), (-1, 0), (-3, 200)]:
        assert signed_bits(lo, hi) == ref(lo, hi), (lo, hi)


def test_conv3_accumulator_is_the_28_bits_the_readme_claims():
    """conv3: 128 terms of int8 weight times an 8.6 relu output.
    128 * 127 * 8191 = 133,193,536 -> 28 bits signed."""
    b = acc_bits_for_real_input(128, 127, FMT_CONV2_PW, in_nonneg=True)
    assert b == 28, b


def test_conv4_accumulator_width():
    """conv4: 128 terms of int8-scaled weight times a 5.6 relu output."""
    b = acc_bits_for_real_input(128, 127, FMT_CONV3, in_nonneg=True)
    assert b == 25, b


def test_the_bound_is_reachable_so_it_is_not_slack():
    """The extremes really do occur together: max weight, max activation, all
    terms agreeing. If the bound were loose this would fail and the width would
    be a guess rather than a proof."""
    n, w, fmt = 8, 127, FMT_CONV3
    reach = n * w * fmt.hi
    assert acc_bits_for_real_input(n, w, fmt) == signed_bits(-reach, reach)
    assert reach <= (1 << (signed_bits(-reach, reach) - 1)) - 1


# --------------------------------------------------------------------------- #
# the pooling head
# --------------------------------------------------------------------------- #

def test_pooling_never_divides_and_still_agrees_with_the_mean():
    rng = random.Random(5)
    for _ in range(500):
        frames = [[rng.randint(-8192, 8191) for _ in range(12)]
                  for _ in range(64)]
        means = [sum(f[c] for f in frames) / 64.0 for c in range(12)]
        assert pooled_argmax(frames) == max(range(12), key=lambda c: means[c])


def test_pooling_breaks_ties_to_the_lower_index_like_torch():
    frames = [[5, 5, 1]]
    assert pooled_argmax(frames) == 0


def test_pooling_refuses_empty_and_ragged_input():
    with pytest.raises(ValueError):
        pooled_argmax([])
    with pytest.raises(ValueError):
        pooled_argmax([[1, 2], [1]])


# --------------------------------------------------------------------------- #
# the constants have to survive a ROM word
# --------------------------------------------------------------------------- #

def test_the_shift_is_capped_so_the_offset_fits_a_rom_word():
    """The bug this test exists for.

    The shift used to be chosen from the gain alone, which wants it as large as
    possible. But B = bias * 2^frac * 2^shift grows with it, and conv3 at
    shift=28 needed 35 bits -- while every ROM this project writes is 32-bit
    hex. The emitter masked with 0xFFFFFFFF and produced a different,
    valid-looking number, and nothing downstream noticed, because emit and
    golden both work from Python ints, which have no width.
    """
    # a small gain wants a large shift; a large offset cannot afford one.
    # (At bias 2.5 the two constraints happen to land on the same shift, which
    # is why the first version of this test proved nothing.)
    gain = [3e-3] * 8
    bias = [50.0] * 8
    f = fold_affine("conv3", gain, bias, FMT_CONV3, relu=True)
    assert f.fits_word(32)
    assert f.bias_bits() <= 32, f.bias_bits()

    # the offset, not the gain, is what limited it: given a wider word the
    # shift goes back up
    wide = fold_affine("conv3", gain, bias, FMT_CONV3, relu=True, word_bits=64)
    assert wide.shift > f.shift, (wide.shift, f.shift)
    assert not wide.fits_word(32), "the wide fold is exactly what used to ship"


def test_a_layer_whose_gain_binds_first_keeps_its_full_shift():
    """The cap must not fire when it is not needed -- a tiny bias should leave
    the gain-driven shift alone, or every layer pays for conv3's problem."""
    gain = [3e-3] * 8
    bias = [1e-4] * 8
    f32 = fold_affine("t", gain, bias, FMT_CONV3, relu=True)
    f64 = fold_affine("t", gain, bias, FMT_CONV3, relu=True, word_bits=64)
    assert f32.shift == f64.shift


def test_capping_the_shift_does_not_break_the_fold():
    """Fewer bits of headroom is fine; a wrong answer is not."""
    gain, bias = _realistic()
    bias = [b * 40.0 for b in bias]       # force the cap to bind hard
    f = fold_affine("t", gain, bias, FMT_CONV3, relu=True)
    assert f.fits_word(32)
    assert f.dead_gains() == []
    assert f.max_output_error_lsb(1 << 20) < 0.25, f.max_output_error_lsb(1 << 20)


def test_fits_word_actually_rejects_an_oversized_constant():
    f = fold_affine("t", [1e-3], [1.0], FMT_CONV3, relu=True)
    assert f.fits_word(32)
    f.bias = [1 << 40]
    assert not f.fits_word(32)
    f.bias = [0]
    f.gain = [-(1 << 40)]
    assert not f.fits_word(32)


def test_an_impossible_fold_raises_rather_than_being_written():
    """A pinned shift bypasses the cap, so the constructor still has to check.
    Re-exporting with a pinned shift from an older run is exactly how a ROM
    that used to fit stops fitting."""
    with pytest.raises(ValueError, match="more than 32 bits"):
        fold_affine("t", [1e-3], [3.0], FMT_CONV3, relu=True, shift=40)
