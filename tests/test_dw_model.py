"""A reference model of kws_dw_conv, checked against the golden vectors.

This exists because the RTL and the thing it is supposed to compute can fail
independently, and separating them is most of the debugging. If this test
passes and the Verilog testbench fails, the algorithm is right and the fault is
in the Verilog -- widths, FSM, timing. If this test fails, no amount of staring
at the RTL will help.

It is also the only check of the dw datapath that runs without a simulator, so
it works on a machine with neither iverilog nor torch: everything it needs is
already committed under rtl/gen/.

The model is deliberately written the way the hardware works, not the way
PyTorch does -- line buffer, valid shift register, trailing-zero shift,
popcount, integer threshold. A numpy convolution would agree with the golden
vectors while saying nothing about whether the RTL's approach is sound.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parents[1] / "rtl" / "gen" / "xl_g12"
LAYER = "b1_s0_dw"

pytestmark = pytest.mark.skipif(
    not (GEN / "manifest.json").is_file(),
    reason="rtl/gen/xl_g12 not exported (see rtl/gen/README.md)")


def _words(rel):
    return [int(w, 16) for w in (GEN / rel).read_text().split()]


def _frames(rel, n_clip, T, nw):
    """Frame-major hex -> [clip][t] as one wide integer, channel c in bit c."""
    w = _words(rel)
    return [[sum(w[(n * T + t) * nw + j] << (32 * j) for j in range(nw))
             for t in range(T)] for n in range(n_clip)]


@pytest.fixture(scope="module")
def setup():
    man = json.loads((GEN / "manifest.json").read_text())
    lay = {l["name"]: l for l in man["layers"]}[LAYER]
    g = json.loads((GEN / "golden" / "golden.json").read_text())
    n_clip, C, T = g["files"][f"{LAYER}_out"]["shape"]
    nw = C // 32
    trom = _words(f"{LAYER}_t.hex")
    return {
        "C": C, "K": lay["kernel"], "PAD": lay["padding"], "T": T,
        "n_clip": n_clip, "nw": nw,
        "inp": _frames("golden/conv1_out.hex", n_clip, T, nw),
        "exp": _frames(f"golden/{LAYER}_out.hex", n_clip, T, nw),
        "w": _words(f"{LAYER}_w.hex"),
        "thr": [t - (1 << 32) if t >= 1 << 31 else t for t in trom[:C]],
        "ge": [bool(v) for v in trom[C:]],
    }


def run_layer(s):
    """Exactly what kws_dw_conv does, frame by frame."""
    C, K, PAD, T = s["C"], s["K"], s["PAD"], s["T"]
    mask_k = (1 << K) - 1
    out_all = []
    for n in range(s["n_clip"]):
        fbuf, valid, outs = [0] * K, 0, {}
        # T real pushes, then PAD flush pushes to drain the tail
        for i in range(T + PAD):
            real = i < T
            fbuf = fbuf[1:] + [s["inp"][n][i] if real else 0]
            valid = ((valid >> 1) | (int(real) << (K - 1))) & mask_k
            if not (valid >> (K - 1 - PAD)) & 1:
                continue                       # centre tap is padding: no output
            shift = (valid & -valid).bit_length() - 1     # trailing zeros
            n_valid = bin(valid).count("1")
            frame = 0
            for c in range(C):
                taps = sum(((fbuf[j] >> c) & 1) << j for j in range(K))
                # the same shift on both, or taps stop meeting their weights
                a = taps >> shift
                w = (s["w"][c] & mask_k) >> shift
                p = bin(~(a ^ w) & ((1 << n_valid) - 1)).count("1")
                acc = 2 * p - n_valid
                if (acc >= s["thr"][c]) == s["ge"][c]:
                    frame |= 1 << c
            outs[i - PAD] = frame
        out_all.append([outs[t] for t in range(T)])
    return out_all


def test_model_reproduces_the_golden_output(setup):
    got = run_layer(setup)
    bad = [(n, t) for n in range(setup["n_clip"]) for t in range(setup["T"])
           if got[n][t] != setup["exp"][n][t]]
    assert not bad, f"{len(bad)} frames differ, first at clip/t {bad[:3]}"


def test_edges_really_are_edges(setup):
    """The whole padding argument, restated as something that can fail.

    If padding were treated as -1 rather than as absent, every frame would use
    all K taps and the edge frames would stop being special.
    """
    K, PAD, T = setup["K"], setup["PAD"], setup["T"]
    acc = _words(f"golden/{LAYER}_acc.hex")
    acc = [v - (1 << 32) if v >= 1 << 31 else v for v in acc]
    C = setup["C"]
    for t in range(T):
        n_valid = K - max(0, PAD - t) - max(0, (t + PAD) - (T - 1))
        peak = max(abs(acc[c * T + t]) for c in range(C))
        assert peak <= n_valid, f"t={t}: |acc|={peak} exceeds {n_valid} taps"
        # 2P-N fixes the parity, which catches an off-by-one that a magnitude
        # bound would let through
        assert (n_valid - peak) % 2 == 0, f"t={t}: parity {peak} vs {n_valid}"
    assert sum(1 for t in range(T)
               if K - max(0, PAD - t) - max(0, (t + PAD) - (T - 1)) < K) == 2 * PAD


def test_shifting_only_the_activation_would_be_wrong(setup):
    """Pin the trap, so a 'simplification' that drops the weight shift fails.

    Values stay plausible under that bug -- still a sum of +-1 -- and only the
    2*PAD edge frames change, which is exactly why it needs a test rather than
    an inspection.
    """
    s = dict(setup)
    C, K, PAD, T = s["C"], s["K"], s["PAD"], s["T"]
    mask_k = (1 << K) - 1
    n, i = 0, PAD            # first output frame, t=0: the deepest shift
    fbuf, valid = [0] * K, 0
    for j in range(i + 1):
        fbuf = fbuf[1:] + [s["inp"][n][j]]
        valid = ((valid >> 1) | (1 << (K - 1))) & mask_k
    shift = (valid & -valid).bit_length() - 1
    n_valid = bin(valid).count("1")
    assert shift == PAD and n_valid == K - PAD

    wrong = 0
    for c in range(C):
        taps = sum(((fbuf[j] >> c) & 1) << j for j in range(K))
        w_full = s["w"][c] & mask_k
        good = bin(~((taps >> shift) ^ (w_full >> shift))
                   & ((1 << n_valid) - 1)).count("1")
        bug = bin(~((taps >> shift) ^ w_full)
                  & ((1 << n_valid) - 1)).count("1")     # weight not shifted
        wrong += int(good != bug)
    assert wrong > 0, ("dropping the weight shift changed nothing on this "
                       "frame -- the test cannot detect the bug")


# --------------------------------------------------------------------------- #
# pointwise: the same split, one layer up. Its input is the depthwise output,
# so a mismatch here cannot be blamed on the activations reaching it.
# --------------------------------------------------------------------------- #

PW = "b1_s0_pw"


@pytest.fixture(scope="module")
def pw_setup():
    g = json.loads((GEN / "golden" / "golden.json").read_text())
    n_clip, c_in, T = g["files"][f"{LAYER}_out"]["shape"]
    _, c_out, _ = g["files"][f"{PW}_out"]["shape"]
    trom = _words(f"{PW}_t.hex")
    return {
        "c_in": c_in, "c_out": c_out, "T": T, "n_clip": n_clip,
        "nw": c_in // 32,
        "inp": _frames(f"golden/{LAYER}_out.hex", n_clip, T, c_in // 32),
        "exp": _frames(f"golden/{PW}_out.hex", n_clip, T, c_out // 32),
        "w": _words(f"{PW}_w.hex"),
        "thr": [v - (1 << 32) if v >= 1 << 31 else v for v in trom[:c_out]],
        "ge": [bool(v) for v in trom[c_out:]],
    }


def test_pointwise_model_reproduces_the_golden_output(pw_setup):
    """No line buffer, no shift, no per-frame n_valid.

    k=1 means the taps run along channels, not time, and the channel axis is
    the whole vector rather than a window -- so none of the edge machinery the
    depthwise module needs applies here. That absence is the thing worth
    pinning: if a future refactor shares edge handling between the two, this
    test says pointwise never wanted it.
    """
    s = pw_setup
    c_in, c_out, nw = s["c_in"], s["c_out"], s["nw"]
    bad = []
    for n in range(s["n_clip"]):
        for t in range(s["T"]):
            a = s["inp"][n][t]
            frame = 0
            for o in range(c_out):
                w = sum(s["w"][o * nw + j] << (32 * j) for j in range(nw))
                p = bin(~(a ^ w) & ((1 << c_in) - 1)).count("1")
                if (2 * p - c_in >= s["thr"][o]) == s["ge"][o]:
                    frame |= 1 << o
            if frame != s["exp"][n][t]:
                bad.append((n, t))
    assert not bad, f"{len(bad)} frames differ, first at clip/t {bad[:3]}"


# --------------------------------------------------------------------------- #
# the residual block. The add happens in the INTEGER domain, before any
# threshold, and the block applies exactly one threshold afterwards.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def blk_setup():
    g = json.loads((GEN / "golden" / "golden.json").read_text())

    def fr(key):
        n, C, T = g["files"][key]["shape"]
        return _frames(f"golden/{g['files'][key]['file']}", n, T, C // 32), C, T

    x, c_in, T = fr("conv1_out")
    y1, c_mid, _ = fr("b1_s1_dw_out")
    exp, c_out, _ = fr("b1_add_out")
    trom = _words("b1_add_t.hex")
    return {
        "x": x, "y1": y1, "exp": exp, "T": T, "n_clip": g["n_clips"],
        "c_in": c_in, "c_mid": c_mid, "c_out": c_out,
        "w_skip": _words("b1_skip_w.hex"), "w_pw": _words("b1_s1_pw_w.hex"),
        "thr": [v - (1 << 32) if v >= 1 << 31 else v for v in trom[:c_out]],
        "ge": [bool(v) for v in trom[c_out:]],
    }


def _dot(a, w_words, n_in, o, nw):
    w = sum(w_words[o * nw + j] << (32 * j) for j in range(nw))
    p = bin(~(a ^ w) & ((1 << n_in) - 1)).count("1")
    return 2 * p - n_in


def test_block_model_reproduces_the_golden_output(blk_setup):
    s = blk_setup
    nws, nwp = s["c_in"] // 32, s["c_mid"] // 32
    bad = []
    for n in range(s["n_clip"]):
        for t in range(s["T"]):
            frame = 0
            for o in range(s["c_out"]):
                a_sk = _dot(s["x"][n][t], s["w_skip"], s["c_in"], o, nws)
                a_pw = _dot(s["y1"][n][t], s["w_pw"], s["c_mid"], o, nwp)
                if (a_pw + a_sk >= s["thr"][o]) == s["ge"][o]:
                    frame |= 1 << o
            if frame != s["exp"][n][t]:
                bad.append((n, t))
    assert not bad, f"{len(bad)} frames differ, first at clip/t {bad[:3]}"


def test_residual_must_be_added_before_the_threshold(blk_setup):
    """Pin the one structural claim, so a rewrite that thresholds first fails.

    Thresholding each path and combining +-1 outputs is a different network,
    and it produces perfectly plausible bits -- there is no error, only worse
    accuracy. It has to fail as a test or it will not fail at all.
    """
    s = blk_setup
    nws, nwp = s["c_in"] // 32, s["c_mid"] // 32
    n, t = 0, 32
    differ = 0
    for o in range(s["c_out"]):
        a_sk = _dot(s["x"][n][t], s["w_skip"], s["c_in"], o, nws)
        a_pw = _dot(s["y1"][n][t], s["w_pw"], s["c_mid"], o, nwp)
        correct = (a_pw + a_sk >= s["thr"][o]) == s["ge"][o]
        # the wrong version: threshold each path, then OR the +-1 results
        wrong = (((a_pw >= s["thr"][o]) == s["ge"][o]) or
                 ((a_sk >= s["thr"][o]) == s["ge"][o]))
        differ += int(correct != wrong)
    assert differ > 0, ("thresholding before the add changed nothing on this "
                        "frame -- the test cannot detect the mistake")


# --------------------------------------------------------------------------- #
# dilation: conv2_dw spreads 29 taps over 57 slots
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dil_setup():
    man = json.loads((GEN / "manifest.json").read_text())
    lay = {l["name"]: l for l in man["layers"]}["conv2_dw"]
    g = json.loads((GEN / "golden" / "golden.json").read_text())
    n_clip, C, T = g["files"]["conv2_dw_out"]["shape"]
    nw = C // 32
    trom = _words("conv2_dw_t.hex")
    return {
        "C": C, "K": lay["kernel"], "PAD": lay["padding"],
        "DIL": lay["dilation"], "T": T, "n_clip": n_clip, "nw": nw,
        "inp": _frames("golden/b3_add_out.hex", n_clip, T, nw),
        "exp": _frames("golden/conv2_dw_out.hex", n_clip, T, nw),
        "w": _words("conv2_dw_w.hex"),
        "thr": [t - (1 << 32) if t >= 1 << 31 else t for t in trom[:C]],
        "ge": [bool(v) for v in trom[C:]],
    }


def run_dilated(s):
    """kws_dw_conv with DIL > 1.

    The buffer holds SPAN slots and the K taps sit at slot j*DIL. Everything
    after the gather is unchanged, because the shift and the mask work on the
    K TAPS -- feed them the SPAN-wide valid and they would count slots the
    kernel never reads.
    """
    C, K, PAD, DIL, T = s["C"], s["K"], s["PAD"], s["DIL"], s["T"]
    SPAN = (K - 1) * DIL + 1
    mask_k = (1 << K) - 1
    out_all = []
    for n in range(s["n_clip"]):
        fbuf, valid, outs = [0] * SPAN, 0, {}
        for i in range(T + PAD):
            real = i < T
            fbuf = fbuf[1:] + [s["inp"][n][i] if real else 0]
            valid = ((valid >> 1) | (int(real) << (SPAN - 1))) & ((1 << SPAN) - 1)
            if not (valid >> (SPAN - 1 - PAD)) & 1:
                continue
            tvld = sum(((valid >> (j * DIL)) & 1) << j for j in range(K))
            shift = (tvld & -tvld).bit_length() - 1
            n_valid = bin(tvld).count("1")
            frame = 0
            for c in range(C):
                taps = sum(((fbuf[j * DIL] >> c) & 1) << j for j in range(K))
                a = taps >> shift
                w = (s["w"][c] & mask_k) >> shift
                p = bin(~(a ^ w) & ((1 << n_valid) - 1)).count("1")
                acc = 2 * p - n_valid
                if (acc >= s["thr"][c]) == s["ge"][c]:
                    frame |= 1 << c
            outs[i - PAD] = frame
        out_all.append([outs[t] for t in range(T)])
    return out_all


def test_the_dilated_model_reproduces_the_golden_output(dil_setup):
    got = run_dilated(dil_setup)
    bad = [(n, t) for n in range(dil_setup["n_clip"])
           for t in range(dil_setup["T"]) if got[n][t] != dil_setup["exp"][n][t]]
    assert not bad, f"{len(bad)} frames differ, first at clip/t {bad[:3]}"


def test_the_span_is_wider_than_the_kernel(dil_setup):
    s = dil_setup
    assert s["DIL"] == 2 and s["K"] == 29
    assert (s["K"] - 1) * s["DIL"] + 1 == 57, "29 taps over 57 slots"
    assert s["PAD"] == 28
    # the centre slot must be one the kernel actually reads
    assert ((57 - 1 - s["PAD"]) % s["DIL"]) == 0


def test_gathering_valid_at_the_wrong_stride_would_be_caught(dil_setup):
    """The mistake this gather exists to avoid: feeding the shift and the mask
    the raw SPAN-wide valid instead of the K taps. It counts slots the kernel
    never reads, so n_valid comes out roughly twice too large."""
    s = dil_setup
    C, K, PAD, DIL, T = s["C"], s["K"], s["PAD"], s["DIL"], s["T"]
    SPAN = (K - 1) * DIL + 1
    valid = 0
    for i in range(PAD + 1):                      # fill to the first output
        valid = ((valid >> 1) | (1 << (SPAN - 1))) & ((1 << SPAN) - 1)
    tvld = sum(((valid >> (j * DIL)) & 1) << j for j in range(K))
    assert bin(valid).count("1") != bin(tvld).count("1"), (
        "at the edge the slot count and the tap count must differ, or this "
        "test proves nothing")
    assert bin(tvld).count("1") <= K
