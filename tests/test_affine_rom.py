"""The tail's epilogue, computed from the exported files alone.

This is the same relationship kws_affine implements, done in Python from the
.hex files and the manifest -- no torch, no simulator. It closes the gap that
let a real bug through: emit and golden both worked from Python ints, which have
no width, so a constant too large for a 32-bit ROM word passed every check and
then got masked to a different number on the way to disk. Reading the file back
is the only test that could have caught it.

It is also the bisect for the Verilog. If this passes and tb_affine fails, the
arithmetic and the ROM format are right and the fault is in kws_affine.v --
widths, pipeline, addressing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parents[1] / "rtl" / "gen" / "xl_g12"
GOLD = GEN / "golden"

pytestmark = pytest.mark.skipif(
    not (GEN / "manifest.json").is_file()
    or not (GOLD / "conv2_pw_out.hex").is_file(),
    reason="tail not exported yet (see rtl/gen/README.md)")


def _words(path):
    return [int(w, 16) for w in Path(path).read_text().split()]


def _signed(v, bits=32):
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


@pytest.fixture(scope="module")
def man():
    return json.loads((GEN / "manifest.json").read_text())


@pytest.fixture(scope="module")
def sites(man):
    return {s["name"]: s for s in man["tail"]["sites"]}


def _fmt(s):
    i, f = (int(v) for v in s["out_format"].split("."))
    bits = i + f
    return bits, -(1 << (bits - 1)), (1 << (bits - 1)) - 1


TAIL = ["conv2_pw", "conv3", "conv4"]


@pytest.mark.parametrize("name", TAIL)
def test_the_rom_reproduces_the_golden_output(name, man, sites):
    """(A*acc + B + half) >> shift, then relu, then saturate.

    Constants read from the ROM file rather than recomputed, so this checks the
    file the hardware will actually load.
    """
    s = sites[name]
    n = s["n_out"]
    rom = _words(GEN / f"{name}_bn.hex")
    assert len(rom) == 2 * n, f"{len(rom)} words for {n} channels"
    gain = [_signed(v) for v in rom[:n]]
    bias = [_signed(v) for v in rom[n:]]

    acc = [_signed(v) for v in _words(GOLD / f"{name}_acc.hex")]
    want = [_signed(v) for v in _words(GOLD / f"{name}_out.hex")]
    assert len(acc) == len(want)
    assert len(acc) % n == 0

    bits, lo, hi = _fmt(s)
    shift, relu = s["shift"], s["relu"]
    half = 1 << (shift - 1) if shift > 0 else 0

    bad = 0
    for i, (a, w) in enumerate(zip(acc, want)):
        # golden order is clip-major, then out_ch, then frame:
        # i = clip*n*T + ch*T + t, so i//T = clip*n + ch and the clip term is a
        # multiple of n
        ch = (i // 64) % n
        y = (gain[ch] * a + bias[ch] + half) >> shift
        if relu and y < 0:
            y = 0
        y = lo if y < lo else (hi if y > hi else y)
        if y != w:
            bad += 1
            if bad <= 5:
                print(f"{name} i={i} ch={ch} acc={a}: got {y} want {w}")
    assert bad == 0, f"{bad}/{len(acc)} values disagree"


@pytest.mark.parametrize("name", TAIL)
def test_every_constant_fits_a_rom_word(name, man, sites):
    """The bug, pinned at the file level.

    A 32-bit hex word holds [-2^31, 2^31). The offset B = bias * 2^frac *
    2^shift grows with the shift, and a shift chosen from the gain alone pushed
    conv3's offset to 35 bits -- which `int(b) & 0xFFFFFFFF` wrote as a
    different, entirely plausible number.
    """
    s = sites[name]
    rom = _words(GEN / f"{name}_bn.hex")
    assert all(0 <= w < (1 << 32) for w in rom)
    n = s["n_out"]
    gain = [_signed(v) for v in rom[:n]]
    bias = [_signed(v) for v in rom[n:]]
    assert max(abs(v) for v in gain).bit_length() + 1 <= s["gain_bits"]
    assert max(abs(v) for v in bias).bit_length() + 1 <= s["bias_bits"]
    assert s["bias_bits"] <= 32, s["bias_bits"]
    assert s["gain_bits"] <= 32, s["gain_bits"]


@pytest.mark.parametrize("name", TAIL)
def test_no_gain_is_zero(name, sites):
    """A gain that rounded to zero deletes a channel, and downstream cannot
    tell that from a channel the training killed."""
    s = sites[name]
    rom = _words(GEN / f"{name}_bn.hex")
    assert all(_signed(v) != 0 for v in rom[:s["n_out"]])


@pytest.mark.parametrize("name", TAIL)
def test_the_output_stays_inside_its_declared_format(name, sites):
    s = sites[name]
    bits, lo, hi = _fmt(s)
    vals = [_signed(v) for v in _words(GOLD / f"{name}_out.hex")]
    assert all(lo <= v <= hi for v in vals)
    if s["relu"]:
        assert all(v >= 0 for v in vals), "relu output cannot be negative"


@pytest.mark.parametrize("name", TAIL)
def test_the_accumulator_stays_inside_its_declared_width(name, sites):
    """acc_bits is a bound, so the data must never reach past it. If it does,
    the width is not a bound and the datapath wraps."""
    s = sites[name]
    lim = 1 << (s["acc_bits"] - 1)
    vals = [_signed(v) for v in _words(GOLD / f"{name}_acc.hex")]
    assert all(-lim <= v < lim for v in vals), (
        f"{name}: |acc| reached {max(abs(v) for v in vals)}, "
        f"bound is {lim - 1}")


def test_conv4_shares_one_gain_across_the_classes(sites):
    """argmax compares the twelve pooled outputs directly, so a per-class gain
    would reorder them."""
    n = sites["conv4"]["n_out"]
    rom = _words(GEN / "conv4_bn.hex")
    assert len({_signed(v) for v in rom[:n]}) == 1


def test_the_pooled_argmax_matches_the_recorded_prediction(sites):
    """The head, end to end: sum each class over time, take the largest.

    No divide by T -- it is the same positive factor on every class.
    """
    n = sites["conv4"]["n_out"]
    vals = [_signed(v) for v in _words(GOLD / "conv4_out.hex")]
    T = 64
    n_clip = len(vals) // (n * T)
    got = []
    for c in range(n_clip):
        tot = [sum(vals[(c * n + k) * T + t] for t in range(T))
               for k in range(n)]
        got.append(max(range(n), key=lambda k: tot[k]))
    want = [int(v) for v in (GOLD / "predictions_fixed.txt").read_text().split()]
    assert got == want
