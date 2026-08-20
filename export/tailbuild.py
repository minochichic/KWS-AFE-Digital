"""Read the tail out of a trained model, once, for both emit and golden.

export/tailfmt.py owns the fixed-point arithmetic and stays torch-free so the
RTL's reference model can run without it. This module is the other half: it
looks at the trained modules and works out what each tail layer's gain, offset,
accumulator width and integer weights actually are.

WHY IT IS NOT INSIDE emit.py. emit writes the ROMs and golden writes the vectors
the testbench compares against, and if each derived the fold itself they could
disagree -- the ROM saying one thing and the expected output another. Every
symptom of that is a testbench that fails in a way pointing at the RTL. The
same failure mode as the layer-NAME drift that tests/test_golden.py exists to
catch, so it gets the same treatment: one source, two consumers.

WHAT THE TAIL IS. A stage is in the tail when its activation is not `sign`,
because that is exactly when its BN has no compare to fold into
(export/fuse.py). With the stage list in configs/base.yaml that is three
layers, and they differ in what sits in front of the accumulator:

    conv2_pw   binary weights, alpha           gain = bn_g * alpha
    conv3      int8 weights, per-ch scale      gain = bn_g * scale / 2^F_in
    conv4      fixed weights, shared scale     gain = 2^-(F + F_in), no BN

THE INPUT'S OWN SCALE IS THE EASY THING TO DROP. conv3's accumulator sums int8
weights against integers that represent `x * 2^F`, so the accumulator is 2^F too
large and the gain has to carry a matching 2^-F. Forget it and conv3's output is
64x too big -- which, after a relu and a clamp, does not look like a scale error.
It looks like a dead network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from export.fuse import bn_affine, conv_alpha
from export.tailfmt import (FMT_CONV2_PW, FMT_CONV3, FMT_CONV4, FRAC_BITS,
                            ROM_WORD_BITS, AffineFold, FixedFormat,
                            acc_bits_for_real_input, fold_affine, signed_bits)
from models.binary_ops import BinaryConv1d
from models.binary_matchboxnet import BinaryMatchboxNet, PlainStage
from models.quant_ops import QuantConv1d

# Which format each tail site lands in, keyed by the name emit gives the layer.
# A stage rename raises a KeyError here rather than silently leaving the tail
# unquantized.
TAIL_FORMATS: Dict[str, FixedFormat] = {
    "conv2_pw": FMT_CONV2_PW,
    "conv3": FMT_CONV3,
    "conv4": FMT_CONV4,
}


@dataclass
class TailSite:
    """One tail layer, with everything both emit and golden need."""

    name: str
    kind: str                             # "bn_relu" | "logits"
    conv: nn.Module
    fold: AffineFold
    out_fmt: FixedFormat
    acc_bits: int
    in_fmt: Optional[FixedFormat] = None  # None where the input is +-1
    weights: Optional[torch.Tensor] = None   # integer weights this layer owns
    weight_absmax: int = 0
    weight_bits: int = 0
    n_terms: int = 0
    meta: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


def fixed_weights(conv: nn.Module) -> torch.Tensor:
    """conv4's weights on the SHARED 1/2^FRAC grid the accuracy sweep validated.

    Not per-channel int8, although that would give finer weights. The accuracy
    this format is allowed to claim is what experiments/tail_fixedpoint.py
    measured (0.8445 at frac=6), and that sweep rounded conv4's weights onto a
    single fixed-point grid at the same frac_bits. Per-channel scales here would
    be a different quantization from the one that was measured, so the 0.8445
    would no longer certify it.

    It is also the cheaper option, which is a coincidence and not the reason: a
    shared 2^-F on the weights and 2^-F on the activations make conv4's gain
    exactly 2^-2F, so its multiply degenerates into a shift.
    """
    w = conv.weight.detach().double()
    return torch.round(w * float(1 << FRAC_BITS)).to(torch.int64)


def int8_weights(conv: QuantConv1d) -> tuple:
    """(q, per-output-channel scale), exactly as the module quantized in training.

    Deliberately not export/pack.quantize_int8: that clamps amax before dividing
    while QuantConv1d clamps the scale after, and the two disagree only on a
    degenerate channel. Ask the module.
    """
    s = conv.weight_scale()
    q = torch.round(conv.weight.detach() / s).clamp(-conv.qmax, conv.qmax)
    return q.to(torch.int64), s.detach().double().reshape(-1)


def _bn_relu_site(name: str, conv: nn.Module, bn: nn.Module,
                  in_fmt: Optional[FixedFormat], n_terms_binary: int) -> TailSite:
    g, b = bn_affine(bn)
    if g is None:
        raise ValueError(f"{name}: activation is relu but BN is Identity")
    out_fmt = TAIL_FORMATS[name]

    if isinstance(conv, BinaryConv1d):
        a = conv_alpha(conv)
        alpha = (torch.ones_like(g) if a is None
                 else a.detach().double().reshape(-1))
        gain = (g * alpha).tolist()
        acc_bits = signed_bits(-n_terms_binary, n_terms_binary)
        site_kw = dict(n_terms=n_terms_binary,
                       meta={"source": "gain = bn_g * alpha, binary accumulator"})
        w = None
        absmax = wbits = 0
    elif isinstance(conv, QuantConv1d):
        if in_fmt is None:
            raise ValueError(f"{name}: int8 conv with no known input format")
        q, s = int8_weights(conv)
        gain = (g * s / float(1 << in_fmt.frac_bits)).tolist()
        absmax = int(q.abs().max())
        terms = conv.in_channels * conv.kernel_size[0]
        acc_bits = acc_bits_for_real_input(terms, absmax, in_fmt, in_nonneg=True)
        w, wbits = q, 8
        site_kw = dict(n_terms=terms,
                       meta={"source": "gain = bn_g * int8_scale / 2^in_frac"})
    else:
        raise NotImplementedError(f"{name}: relu on {type(conv).__name__}")

    return TailSite(name=name, kind="bn_relu", conv=conv,
                    fold=fold_affine(name, gain, b.tolist(), out_fmt, relu=True),
                    out_fmt=out_fmt, acc_bits=acc_bits, in_fmt=in_fmt,
                    weights=w, weight_absmax=absmax, weight_bits=wbits,
                    **site_kw)


def _logits_site(name: str, conv: nn.Module,
                 in_fmt: Optional[FixedFormat]) -> TailSite:
    """conv4: fixed weights x fixed activations, the conv's own bias, no BN.

    PlainStage sets bn=Identity and bias=True on the final stage, so the affine
    map is entirely the conv bias plus a gain that is the SAME for every class.
    It has to be the same: the pooled outputs are compared against each other by
    argmax, and a per-class gain would reorder them.
    """
    if in_fmt is None:
        raise ValueError(f"{name}: no known input format")
    out_fmt = TAIL_FORMATS[name]
    q = fixed_weights(conv)
    absmax = int(q.abs().max())
    n_out = conv.out_channels
    terms = conv.in_channels * conv.kernel_size[0]

    gain = [1.0 / float(1 << (FRAC_BITS + in_fmt.frac_bits))] * n_out
    bias = ([float(v) for v in conv.bias.detach().double()]
            if conv.bias is not None else [0.0] * n_out)
    fold = fold_affine(name, gain, bias, out_fmt, relu=False)
    if len(set(fold.gain)) != 1:
        raise ValueError(f"{name}: per-class gains {sorted(set(fold.gain))}; "
                         f"argmax compares these outputs directly, so they must "
                         f"share a scale")
    return TailSite(
        name=name, kind="logits", conv=conv, fold=fold, out_fmt=out_fmt,
        acc_bits=acc_bits_for_real_input(terms, absmax, in_fmt, in_nonneg=True),
        in_fmt=in_fmt, weights=q, weight_absmax=absmax,
        weight_bits=signed_bits(-absmax, absmax), n_terms=terms,
        meta={"source": f"gain = 2^-({FRAC_BITS}+{in_fmt.frac_bits}), uniform "
                        f"across classes; bias = the conv's own bias",
              "pool": "sum over T frames then argmax; the divide is not built"})


def tail_plan(model: BinaryMatchboxNet) -> List[TailSite]:
    """Every layer whose BN survives, in dataflow order, with formats threaded.

    The format threading is the part that has to be in dataflow order: a layer's
    gain divides out its INPUT's fixed-point scale, so it needs to know which
    format the layer before it landed in.
    """
    sites: List[TailSite] = []
    in_fmt: Optional[FixedFormat] = None
    for st in model.cfg.stages:
        mod = model.stages[st.name]
        if not isinstance(mod, PlainStage) or mod.activation == "sign":
            # a sign() stage fuses, and a TCS block is all sign()
            continue
        if mod.dw is not None:                       # separable: pw is the conv
            name, conv = f"{st.name}_pw", mod.pw
        else:
            name, conv = st.name, mod.conv

        if mod.activation == "relu":
            n_terms = (conv.in_channels // conv.groups) * conv.kernel_size[0]
            s = _bn_relu_site(name, conv, mod.bn, in_fmt, n_terms)
        elif mod.activation == "none":
            s = _logits_site(name, conv, in_fmt)
        else:
            raise NotImplementedError(f"{name}: activation {mod.activation!r}")
        sites.append(s)
        in_fmt = s.out_fmt
    return sites


# Where to draw the line, and why there are two lines.
#
# `err` is how far the fold moves a value, in LSBs of that site's own output
# grid, and it works out to about 0.5 * out.hi / |A| once the clamp binds -- so
# it is set by the QUIETEST gain's bit count, not by the shift.
#
# One whole LSB is the natural failure point, from measurement rather than
# taste: experiments/tail_fixedpoint.py ran frac=4, a grid FOUR TIMES COARSER
# than the frac=6 being shipped, and scored 0.8447 against 0.8445 for float.
# A systematic 4x coarsening of every value costs 0.0pp, so a fold error below
# one LSB of the finer grid is strictly under the measured noise floor.
#
# 0.25 is not a failure, it is where the number stops being obviously
# negligible and becomes worth reading. xl_g12 sits at 0.009.
WARN_LSB = 0.25
FAIL_LSB = 1.0


def check_site(s: TailSite, warn_lsb: float = WARN_LSB,
               fail_lsb: float = FAIL_LSB) -> Optional[str]:
    """Refuse a fold that would cost accuracy; report one that might.

    The failures here are invisible downstream: a gain that rounded to zero
    deletes a channel, an oversized constant is masked into a different number,
    and a starved gain moves every value a little. None trips a clamp or an
    assertion -- the accuracy just comes out lower, with nothing in the log.

    Returns a warning string, or None. Raises only past `fail_lsb`.
    """
    if not s.fold.fits_word(ROM_WORD_BITS):
        raise ValueError(
            f"{s.name}: a constant does not fit a {ROM_WORD_BITS}-bit ROM word "
            f"at shift={s.fold.shift}. The emitter masks with 0xFFFFFFFF, so "
            f"this would be written as a different, valid-looking number and "
            f"nothing downstream of the .hex would notice.")
    if s.fold.dead_gains():
        raise ValueError(
            f"{s.name}: gain rounded to zero on channels "
            f"{s.fold.dead_gains()[:8]} -- the fold would delete them. "
            f"Raise export.tailfmt.GAIN_BITS.")

    err = s.fold.max_output_error_lsb((1 << (s.acc_bits - 1)) - 1)
    if err <= warn_lsb:
        return None

    quiet, loud = s.fold.quietest_gain_bits(), s.fold.gain_bits_used()
    which = s.fold.limiting_constraint()
    # Describe what the numbers say rather than a fixed story. A spread is one
    # way to starve the quiet end; an offset that caps the shift for every
    # channel at once is another, and they need different answers.
    cause = (f"gains span {quiet}..{loud} bits, so the shared shift cannot "
             f"serve both ends" if loud - quiet >= 4 else
             f"every gain is small ({quiet}..{loud} bits)")
    remedy = ("raise export.tailfmt.GAIN_BITS -- the gain is what caps the "
              "shift here" if which == "gain" else
              "the OFFSET caps the shift, so GAIN_BITS will not help; a wider "
              "ROM word would, or a BN whose beta is smaller relative to its "
              "gamma")
    msg = (f"{s.name}: the fold moves the result by {err:.3f} LSB of its own "
           f"output grid (shift={s.fold.shift}, {cause}). {remedy}.")
    if err > fail_lsb:
        raise ValueError(
            msg + f" Past {fail_lsb} LSB this is a whole grid step and is "
                  f"refused.")
    return msg


def apply_site(s: TailSite, acc: torch.Tensor) -> torch.Tensor:
    """The fold, vectorized over [N, C, T], as integers. What RTL computes.

    Kept beside the fold rather than in golden.py so there is one definition of
    "what the hardware does" for the whole tail. AffineFold.apply is the
    per-scalar version and this must agree with it; tests/test_tailbuild.py
    checks that on real tensors.
    """
    if acc.dim() != 3:
        raise ValueError(f"{s.name}: expected [N, C, T], got {tuple(acc.shape)}")
    a = torch.tensor(s.fold.gain, dtype=torch.int64).view(1, -1, 1)
    b = torch.tensor(s.fold.bias, dtype=torch.int64).view(1, -1, 1)
    n = acc.to(torch.int64)
    y = a * n + b
    if s.fold.shift > 0:
        # round-half-up, matching tailfmt.round_shift and `(x + half) >>> s`.
        # torch's >> on int64 is arithmetic, so negatives floor -- which is what
        # makes adding half a round rather than a truncate.
        y = (y + (1 << (s.fold.shift - 1))) >> s.fold.shift
    if s.fold.relu:
        y = torch.clamp_min(y, 0)
    return torch.clamp(y, s.out_fmt.lo, s.out_fmt.hi)
