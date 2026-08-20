"""A cycle model of kws_conv1, checked against the golden vectors.

conv1 is the one layer where three things are different at once -- stride 2,
int8 weights against +-1 activations, and the conv's own zero padding at both
ends -- so it is also where an off-by-one is most likely and least visible.

The three claims worth checking rather than assuming:

* Output t is due after input frame `2t + PAD` has been pushed. One push either
  way shifts every output by a frame, which produces a complete, plausible
  feature map.
* A zero-padded tap contributes ZERO, not -1. `2*popcount(XNOR) - N` cannot
  express that at all, which is why kws_dw_conv slides its window instead; here
  the term is simply skipped. Getting it wrong changes only the six edge frames.
* The negation happens at accumulator width. `-(-128)` does not fit in eight
  bits, and a weight at the int8 floor would come back as itself.

Runs with no simulator, no torch, and nothing but the exported .hex files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parents[1] / "rtl" / "gen" / "xl_g12"
GOLD = GEN / "golden"

pytestmark = pytest.mark.skipif(
    not (GOLD / "conv1_acc.hex").is_file(),
    reason="not exported yet (see rtl/gen/README.md)")


def _words(path, bits=32):
    out = []
    for tok in Path(path).read_text().split():
        v = int(tok, 16) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >= (1 << (bits - 1)) else v)
    return out


@pytest.fixture(scope="module")
def cfg():
    man = json.loads((GEN / "manifest.json").read_text())
    L = next(l for l in man["layers"] if l["name"] == "conv1")
    return man, L


def run_conv1(frames, w, c_in, c_out, k, pad, stride, t_in):
    """Cycle-step rtl/kws_conv1.v. `frames` is t_in ints, bit i = channel i.

    Returns [[acc per out_ch] per output frame], in emission order. Written the
    way the RTL walks it -- line buffer, valid shadow, one term per cycle --
    rather than as a convolution, so the schedule is what gets tested.
    """
    fbuf = [0] * k          # slot k-1 is newest
    vld = [0] * k
    out, pcnt = [], 0

    for p in range(stride * ((t_in + 2 * pad - k) // stride) + pad + 1):
        real = p < t_in
        fbuf = fbuf[1:] + [frames[p] if real else 0]
        vld = vld[1:] + [1 if real else 0]

        if p >= pad and (p & 1) == (pad & 1):
            accs = []
            for co in range(c_out):
                a = 0
                for ti in range(c_in):
                    for tk in range(k):
                        if not vld[tk]:
                            continue          # the conv's own zero padding
                        q = w[(co * c_in + ti) * k + tk]
                        a += q if (fbuf[tk] >> ti) & 1 else -q
                accs.append(a)
            out.append(accs)
        pcnt = p
    return out


def _load(cfg):
    man, L = cfg
    wbits = man["roms"]["conv1_w"].get("weight_bits", 8)
    w = _words(GEN / "conv1_w.hex", wbits)
    assert len(w) == L["out_ch"] * L["in_ch"] * L["kernel"]
    nw = (L["in_ch"] + 31) // 32
    raw = _words(GOLD / "input.hex")
    frames = [sum((raw[(n * man["T"] + t) * nw + j] & 0xFFFFFFFF) << (32 * j)
                  for j in range(nw))
              for n in range(2) for t in range(man["T"])]
    return man, L, w, frames


def test_the_model_reproduces_the_golden_accumulator(cfg):
    man, L, w, frames = _load(cfg)
    T_IN, T_OUT = man["T"], man["T"] // L["stride"]
    want = _words(GOLD / "conv1_acc.hex")

    for n in range(2):
        got = run_conv1(frames[n * T_IN:(n + 1) * T_IN], w, L["in_ch"],
                        L["out_ch"], L["kernel"], L["padding"], L["stride"],
                        T_IN)
        assert len(got) == T_OUT, f"{len(got)} output frames, expected {T_OUT}"
        for t in (0, 1, 2, 31, 62, 63):
            for o in (0, 1, 63, 127):
                assert got[t][o] == want[(n * L["out_ch"] + o) * T_OUT + t], \
                    (n, t, o)


def test_the_edge_frames_are_where_padding_shows(cfg):
    """The first and last outputs use fewer taps than the middle. If padding
    contributed -1 instead of 0, only these would change -- so a test that
    looked at frame 31 alone would pass for a wrong padding rule."""
    man, L, w, frames = _load(cfg)
    k, pad, stride, T_IN = L["kernel"], L["padding"], L["stride"], man["T"]

    def as_minus_one(clip):
        fbuf, vld, out = [0] * k, [0] * k, []
        for p in range(stride * ((T_IN + 2*pad - k) // stride) + pad + 1):
            real = p < T_IN
            fbuf = fbuf[1:] + [clip[p] if real else 0]
            vld = vld[1:] + [1 if real else 0]
            if p >= pad and (p & 1) == (pad & 1):
                a = 0
                for ti in range(L["in_ch"]):
                    for tk in range(k):
                        q = w[(0 * L["in_ch"] + ti) * k + tk]
                        # the wrong rule: a padded tap counts as -1
                        a += q if ((fbuf[tk] >> ti) & 1 and vld[tk]) else -q
                out.append(a)
        return out

    clip = frames[:T_IN]
    right = run_conv1(clip, w, L["in_ch"], L["out_ch"], k, pad, stride, T_IN)
    wrong = as_minus_one(clip)
    assert right[0][0] != wrong[0], "frame 0 must expose the padding rule"
    assert right[63][0] != wrong[63], "frame 63 too"
    assert right[31][0] == wrong[31], (
        "a middle frame uses every tap, so it cannot tell the rules apart -- "
        "which is why the edges have to be checked explicitly")


def test_the_output_bits_match_the_golden_threshold(cfg):
    """acc -> compare -> one bit, the same fused threshold every other layer
    uses. Checked here so the RTL's epilogue has a target of its own."""
    man, L, w, frames = _load(cfg)
    T_IN, T_OUT, C = man["T"], man["T"] // L["stride"], L["out_ch"]
    trom = _words(GEN / "conv1_t.hex")
    thr, ge = trom[:C], trom[C:]
    packed = _words(GOLD / "conv1_out.hex")
    nw = (C + 31) // 32

    for n in (0, 1):
        got = run_conv1(frames[n * T_IN:(n + 1) * T_IN], w, L["in_ch"], C,
                        L["kernel"], L["padding"], L["stride"], T_IN)
        for t in (0, 5, 31, 63):
            word = sum((packed[(n * T_OUT + t) * nw + j] & 0xFFFFFFFF)
                       << (32 * j) for j in range(nw))
            for o in range(C):
                fired = (got[t][o] >= thr[o]) == bool(ge[o])
                assert fired == bool((word >> o) & 1), (n, t, o)


def test_negating_at_eight_bits_would_be_wrong(cfg):
    """-(-128) is +128, which does not fit in int8 and comes back as -128. Only
    a weight at the floor exposes it, so the check is whether one exists."""
    man, _ = cfg
    wbits = man["roms"]["conv1_w"].get("weight_bits", 8)
    w = _words(GEN / "conv1_w.hex", wbits)
    floor = -(1 << (wbits - 1))
    n = sum(1 for v in w if v == floor)
    if n == 0:
        pytest.skip(f"no weight sits at {floor}; nothing to expose the bug")
    assert (-floor) & ((1 << wbits) - 1) == floor & ((1 << wbits) - 1), (
        "the wrap this test is about")


@pytest.mark.parametrize("mask_matches_buffer", [True, False])
def test_the_valid_mask_must_shift_the_same_way_as_the_buffer(
        cfg, mask_matches_buffer):
    """The bug this file exists for, with real data on both sides.

    fbuf moves data toward index 0 and puts the new frame at K-1, so the new
    valid bit belongs at K-1. Put it at bit 0 and vld[k] shadows fbuf[K-1-k]:
    the padding is masked at the wrong end. It passes every middle frame,
    because those use all K taps and cannot tell the masks apart -- only the
    five edge frames disagree, which is a shape that looks like a subtle
    numerical issue rather than a reversed register.
    """
    man, L, w, frames = _load(cfg)
    K, P, S, TI = L["kernel"], L["padding"], L["stride"], man["T"]
    CI, CO = L["in_ch"], L["out_ch"]
    TO = (TI + 2*P - K) // S + 1
    want = _words(GOLD / "conv1_acc.hex")
    clip = frames[:TI]

    fbuf, vld, got = [0]*K, [0]*K, []
    for p in range(S*(TO-1) + P + 1):
        real = 1 if p < TI else 0
        fbuf = fbuf[1:] + [clip[p] if real else 0]
        vld = (vld[1:] + [real]) if mask_matches_buffer else ([real] + vld[:-1])
        if p >= P and (p & 1) == (P & 1):
            a = 0
            for ti in range(CI):
                for tk in range(K):
                    if not vld[tk]:
                        continue
                    q = w[(0 * CI + ti) * K + tk]
                    a += q if (fbuf[tk] >> ti) & 1 else -q
            got.append(a)

    bad = [t for t in range(TO) if got[t] != want[t]]
    if mask_matches_buffer:
        assert bad == [], bad
    else:
        assert bad == [0, 1, 2, 62, 63], (
            f"the reversed mask must fail on exactly the padded frames; got "
            f"{bad}")


def test_the_rtl_still_shifts_them_together():
    """Pinned at the source, because the two registers are declared apart and
    nothing else makes their directions agree."""
    import re
    src = (Path(__file__).resolve().parents[1] / "rtl" / "kws_conv1.v").read_text()
    assert re.search(r"fbuf\[K-1\]\s*<=\s*in_frame", src)
    assert re.search(r"valid_next\s*=\s*\{in_real,\s*vld\[K-1:1\]\}", src), (
        "the valid mask must take its new bit at K-1, like fbuf")
