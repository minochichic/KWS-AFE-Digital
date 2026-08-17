"""The tail's fixed-point contract: how BN, weight scales and relu become integers.

The three binary blocks end in a compare, so their BN vanishes (export/fuse.py).
The tail cannot do that -- nothing downstream is a sign(), so there is no
threshold to fold into and the BN has to survive as arithmetic. This module is
where the arithmetic is pinned down, once, so that emit.py, golden.py and the
RTL all mean the same thing by "8.6".

WHAT THE TAIL ACTUALLY COMPUTES

    conv2_pw : binary acc n     -> real  gamma'*(alpha*n) + beta'      -> relu
    conv3    : int8  acc        -> real  gamma'*(s_o/2^F)*acc + beta'  -> relu
    conv4    : fixed acc        -> real  (s4/2^F)*acc + b_o            -> logits

Every one of them is AFFINE IN AN INTEGER ACCUMULATOR. That is the whole reason
the tail is cheap: the per-channel BN scale, the binary layer's alpha, and the
int8 layer's per-channel weight scale are all positive constants multiplying the
same accumulator, so they collapse into ONE integer gain per output channel --

    Y = (A_o * acc + B_o + half) >> S        (then relu, then clamp)

-- and the only multiplier in the tail is A_o * acc. 128 of those per frame at
10 Hz, against 64 DSP48s that are otherwise idle.

WHY THE SHIFT IS PER LAYER AND THE GAIN IS PER CHANNEL. A per-channel shift
would mean a variable shifter per channel in RTL; a per-layer shift is a
constant wire. Precision is not lost by sharing it, because the gains within a
layer differ by a couple of bits at most and A is given ~18 bits to work with,
which is far finer than the 1/64 output grid it lands on.

WHY ROUND-HALF-UP AND NOT torch.round. `(x + half) >>> S` is one adder and a
wire; torch.round is half-to-even, which differs only on exact ties and would
cost a parity test in hardware. Ties land 1 LSB apart on a grid where the
measured accuracy cost of the whole 6th fractional bit is 0.0pp, so the
difference cannot matter -- but it is a real difference, and golden.py compares
against THIS function rather than against torch so the RTL has one target.

Nothing here imports torch: the fold is integer arithmetic and the RTL's
reference model has to run on a machine that only has the exported .hex files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# The measured formats, docs in rtl/README.md 3-07. frac=6 everywhere: the
# sweep put the accuracy cliff at frac=3 and the 1% logit-margin quantile asked
# for 5, so 6 is one bit of headroom over the tighter of the two estimates.
FRAC_BITS = 6

# How many bits the integer gain A gets. Sized by the QUIETEST channel, not the
# loudest: the shift is shared, so a layer whose per-channel gains span three
# orders of magnitude gives its smallest gain ~10 fewer bits than its largest.
# 22 leaves the quiet end around 12-13 bits, which puts the fold's error a few
# hundredths of an output LSB even there (see max_output_error_lsb).
#
# It is not free-but-why-not: A*acc is a real multiplier, and conv3's
# accumulator is 28 bits, so 22 bits of gain means a 50-bit product -- two
# DSP48s instead of one. At 128 multiplies per frame and 10 Hz that is
# irrelevant next to being able to say the fold is invisible.
GAIN_BITS = 22


# --------------------------------------------------------------------------- #
# formats
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FixedFormat:
    """A two's-complement fixed-point wire. int_bits INCLUDES the sign bit."""
    int_bits: int
    frac_bits: int = FRAC_BITS

    @property
    def bits(self) -> int:
        return self.int_bits + self.frac_bits

    @property
    def lo(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def hi(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def step(self) -> float:
        return 1.0 / float(1 << self.frac_bits)

    def clamp(self, v: int) -> int:
        return self.lo if v < self.lo else (self.hi if v > self.hi else v)

    def to_real(self, v: int) -> float:
        return float(v) / float(1 << self.frac_bits)

    def __str__(self) -> str:
        return f"{self.int_bits}.{self.frac_bits}"


# The three points, from the measured ranges x guard 1.25 (runs/<tag>/ranges.json).
# conv4 is asymmetric ([-68.1, 26.1]) and sized on the negative side, because
# two's complement cannot be asymmetric.
FMT_CONV2_PW = FixedFormat(8)      # 14 bits, relu output so never negative
FMT_CONV3    = FixedFormat(5)      # 11 bits, relu output
FMT_CONV4    = FixedFormat(8)      # 14 bits, signed logits


# --------------------------------------------------------------------------- #
# the fold
# --------------------------------------------------------------------------- #

def _ceil_log2(x: float) -> int:
    """Smallest e with x <= 2^e, for x > 0."""
    e = 0
    if x >= 1.0:
        while (1 << e) < x:
            e += 1
        return e
    while x <= 1.0 / float(1 << (e + 1)):
        e += 1
    return -e


def round_shift(x: int, s: int) -> int:
    """Round-half-up by an arithmetic right shift. s >= 0.

    Python's >> on negatives floors, which is what an arithmetic shift does in
    hardware, so adding half first gives round-half-up for both signs -- the
    same result the RTL gets from `(x + half) >>> s`.
    """
    if s <= 0:
        return x
    return (x + (1 << (s - 1))) >> s


@dataclass
class AffineFold:
    """Y = clamp(relu?( (A[o]*acc + B[o] + half) >> shift ))  in `out`.

    `gain_real`/`bias_real` are kept alongside the integers so a caller can
    check the fold rather than trust it (see `max_rel_gain_error`).
    """
    name: str
    gain: List[int]
    bias: List[int]
    shift: int
    out: FixedFormat
    relu: bool
    gain_real: List[float] = field(default_factory=list)
    bias_real: List[float] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return len(self.gain)

    def apply(self, ch: int, acc: int) -> int:
        y = round_shift(self.gain[ch] * acc + self.bias[ch], self.shift)
        if self.relu and y < 0:
            y = 0
        return self.out.clamp(y)

    def apply_real(self, ch: int, acc: int) -> float:
        """What the float model computes, for comparison."""
        y = self.gain_real[ch] * float(acc) + self.bias_real[ch]
        return max(0.0, y) if self.relu else y

    # -- self-checks ------------------------------------------------------- #
    def max_rel_gain_error(self) -> float:
        """Worst relative error in A/2^shift against the real gain.

        A diagnostic, NOT the thing to hold a bound on. It is largest for the
        quietest channel, which is also the channel whose output swing is
        smallest, so a bad-looking number here can still be far below one LSB
        of anything observable. `max_output_error_lsb` is the metric.
        """
        worst = 0.0
        scale = float(1 << self.shift) * float(1 << self.out.frac_bits)
        for a, g in zip(self.gain, self.gain_real):
            if g == 0.0:
                continue
            worst = max(worst, abs(float(a) / scale - g) / abs(g))
        return worst

    def max_output_error_lsb(self, acc_absmax: int) -> float:
        """Worst error the fold adds, in LSBs of the output format.

        THIS is the number that matters, and getting to it took getting the
        metric wrong once. Two bounds apply and the true error is under both:

          A. rounding A and B costs at most half a unit each, and they are
             divided by 2^shift, so the error is (0.5*|acc| + 0.5) / 2^shift.
             Grows with the accumulator.
          B. the error is `rel_gain_error * output`, and any output beyond the
             format's range clamps -- where both the exact and folded results
             clamp to the same number and the error is zero. So it is also
             bounded by rel_gain_error * out.hi.

        Bound A alone looks alarming for conv3, whose accumulator reaches 2^27:
        it says 124 LSB. That bound is unreachable, because a channel cannot sit
        at 2^27 of accumulator AND inside an 11-bit output. Bound B is what
        applies there. Reporting the min is the honest answer; reporting only A
        would have sent this design chasing a 31-bit shift.

        The final `>> shift` also rounds, but that half-LSB is inherent to
        landing on the grid, not something the fold added.
        """
        a = (0.5 * float(acc_absmax) + 0.5) / float(1 << self.shift)
        b = self.max_rel_gain_error() * float(self.out.hi)
        return min(a, b)

    def gain_bits_used(self) -> int:
        """Bits the LARGEST gain needs, sign included."""
        m = max((abs(a) for a in self.gain), default=0)
        return 0 if m == 0 else m.bit_length() + 1        # +1 for the sign

    def quietest_gain_bits(self) -> int:
        """Bits the SMALLEST nonzero gain gets. The shift is shared, so this is
        what a pathological spread in the trained BN would show up as, and it is
        worth printing at export time rather than discovering later."""
        vals = [abs(a) for a in self.gain if a != 0]
        return 0 if not vals else min(vals).bit_length() + 1

    def dead_gains(self) -> List[int]:
        """Channels whose gain rounded to zero, i.e. deleted by the fold."""
        return [o for o, (a, g) in enumerate(zip(self.gain, self.gain_real))
                if a == 0 and g != 0.0]


def fold_affine(name: str, gain_real: Sequence[float], bias_real: Sequence[float],
                out: FixedFormat, relu: bool,
                gain_bits: int = GAIN_BITS,
                shift: Optional[int] = None) -> AffineFold:
    """Turn a real affine map on an integer accumulator into integers.

    The output is `Y = round((gain*acc + bias) * 2^frac)`, so both constants
    absorb the output's 2^frac and then a shared 2^shift of headroom:

        A = round(gain * 2^frac * 2^shift)
        B = round(bias * 2^frac * 2^shift)

    `shift` is picked so the largest |A| just fits in `gain_bits`. Passing it
    explicitly is for pinning a layer's ROM across re-exports.
    """
    if len(gain_real) != len(bias_real):
        raise ValueError(f"{name}: {len(gain_real)} gains vs "
                         f"{len(bias_real)} biases")
    g2 = [g * float(1 << out.frac_bits) for g in gain_real]
    peak = max((abs(g) for g in g2), default=0.0)
    if shift is None:
        if peak == 0.0:
            shift = 0
        else:
            shift = gain_bits - 1 - _ceil_log2(peak)
            if shift < 0:
                shift = 0
    k = float(1 << shift)
    return AffineFold(
        name=name,
        gain=[int(round(g * k)) for g in g2],
        bias=[int(round(b * float(1 << out.frac_bits) * k)) for b in bias_real],
        shift=shift, out=out, relu=relu,
        gain_real=list(gain_real), bias_real=list(bias_real))


# --------------------------------------------------------------------------- #
# accumulator widths
# --------------------------------------------------------------------------- #

def signed_bits(lo: int, hi: int) -> int:
    """Bits for a two's-complement value in [lo, hi]. Same rule as ranges.py."""
    span = max(hi + 1, -lo, 1)
    b = 1
    while (1 << b) < span:
        b += 1
    return 1 + max(1, b)


def acc_bits_for_real_input(n_terms: int, w_absmax: int,
                            in_fmt: FixedFormat,
                            in_nonneg: bool = True) -> int:
    """Width of sum(w_i * x_i) where x is `in_fmt` and |w| <= w_absmax.

    A real bound, not a guess: the weights are integers known at export time
    and the input is a clamped fixed-point wire, so the product of the extremes
    IS the extreme. `in_nonneg` is true wherever the previous stage ends in relu
    -- it does not narrow the width (the weights still have both signs) but it
    is recorded because it is the reason no negative input term exists.
    """
    x_max = in_fmt.hi
    x_min = 0 if in_nonneg else in_fmt.lo
    reach = n_terms * w_absmax * max(x_max, abs(x_min))
    return signed_bits(-reach, reach)


# --------------------------------------------------------------------------- #
# the pooling head
# --------------------------------------------------------------------------- #

def pooled_argmax(logits_per_frame: Sequence[Sequence[int]]) -> int:
    """argmax over time-averaged logits, without ever dividing.

    adaptive_avg_pool1d divides every class by the same T, and argmax does not
    care about a shared positive factor, so the divider is not built. The sum is
    T times wider -- 6 bits at T=64 -- which is cheaper than a divider and
    exact, where the divider would not be.

    Ties go to the lower class index, matching torch.argmax.
    """
    if not logits_per_frame:
        raise ValueError("no frames to pool")
    n = len(logits_per_frame[0])
    tot = [0] * n
    for fr in logits_per_frame:
        if len(fr) != n:
            raise ValueError("ragged logits")
        for c, v in enumerate(fr):
            tot[c] += v
    best = 0
    for c in range(1, n):
        if tot[c] > tot[best]:
            best = c
    return best
