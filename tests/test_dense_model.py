"""A cycle model of kws_dense_conv, checked against the golden accumulators.

Same split as tests/test_dw_model.py: if this passes and tb_dense_conv fails,
the algorithm and the schedule are right and the fault is in the Verilog. If
this fails, no amount of staring at waveforms will help.

Two things here are worth modelling rather than assuming, and both are about the
pipeline rather than the arithmetic:

* The weight ROM is 128 Kbit for conv3, far past distributed RAM, so it reads
  synchronously and the counters run one term AHEAD of the accumulator. An
  off-by-one there pairs every weight with the wrong activation, and the result
  is a plausible-looking number.
* The accumulator is published and cleared in the same cycle. The next
  channel's first term has to land on zero, not on the sum that was just sent.

Runs with no simulator, no torch, and nothing but the exported .hex files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parents[1] / "rtl" / "gen" / "xl_g12"
GOLD = GEN / "golden"

def _ready():
    """The weight ROMs must DECLARE their width, not have it guessed.

    A two's-complement weight file only decodes correctly at exactly the width
    it was written for -- `fb` is -5 in eight bits and 251 in thirty-two. So
    this skips rather than assuming 8, which would be right today and silently
    wrong after any change to conv4's weight range.
    """
    if not (GOLD / "conv3_acc.hex").is_file():
        return False
    man = json.loads((GEN / "manifest.json").read_text())
    return all("weight_bits" in man["roms"].get(f"{n}_w", {})
               for n in ("conv3", "conv4"))


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="re-export needed: the weight ROMs do not declare weight_bits")

T, CLIPS = 64, 2


def _words(path, bits=32):
    out = []
    for tok in Path(path).read_text().split():
        v = int(tok, 16) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >= (1 << (bits - 1)) else v)
    return out


@pytest.fixture(scope="module")
def man():
    return json.loads((GEN / "manifest.json").read_text())


def run_dense(act, w, c_in, c_out, acc_bits):
    """Cycle-step the FSM in rtl/kws_dense_conv.v. Returns [(ch, acc)] in order.

    Transcribed from the RTL rather than written as a dot product, so that the
    pipeline depth and the emit-and-clear are actually exercised. A numpy
    matmul would agree with the golden vectors while saying nothing about
    whether the schedule is sound.
    """
    lim = 1 << (acc_bits - 1)
    run, ci, co, wa = True, 0, 0, 0
    vB = lastB = False
    coB, wq, xq = 0, 0, 0
    acc, out = 0, []

    for _ in range(c_in * c_out + 8):
        # ---- stage B consumes what stage A fetched last cycle ------------- #
        if vB:
            s = acc + wq * xq
            assert -lim <= s < lim, f"accumulator {s} escaped {acc_bits} bits"
            if lastB:
                out.append((coB, s))
                acc = 0
            else:
                acc = s
        # ---- stage A addresses, and hands stage B its operands ------------ #
        nvB = run
        if run:
            wq, xq, lastB, coB = w[wa], act[ci], ci == c_in - 1, co
            wa += 1
            if ci == c_in - 1:
                ci = 0
                if co == c_out - 1:
                    run = False
                else:
                    co += 1
            else:
                ci += 1
        vB = nvB
        if not run and not vB:
            break
    return out


DENSE = [("conv3", "conv2_pw_out", 128, 128), ("conv4", "conv3_out", 128, 12)]


@pytest.mark.parametrize("name,src,c_in,c_out", DENSE)
def test_the_model_reproduces_the_golden_accumulator(name, src, c_in, c_out, man):
    wbits = man["roms"][f"{name}_w"]["weight_bits"]
    w = _words(GEN / f"{name}_w.hex", wbits)
    x = _words(GOLD / f"{src}.hex")
    want = _words(GOLD / f"{name}_acc.hex")
    acc_bits = next(s for s in man["tail"]["sites"]
                    if s["name"] == name)["acc_bits"]
    assert len(w) == c_out * c_in

    for n in range(CLIPS):
        for t in (0, 1, 17, 63):
            frame = [x[(n * c_in + i) * T + t] for i in range(c_in)]
            got = run_dense(frame, w, c_in, c_out, acc_bits)
            assert len(got) == c_out, f"{len(got)} channels emitted"
            for o, (ch, acc) in enumerate(got):
                assert ch == o, f"emitted channel {ch} in position {o}"
                assert acc == want[(n * c_out + o) * T + t], (name, n, t, o)


@pytest.mark.parametrize("name,src,c_in,c_out", DENSE)
def test_the_channels_come_out_in_order_and_exactly_once(name, src, c_in,
                                                         c_out, man):
    """The consumer is kws_affine, which indexes its ROM by the channel it is
    handed. A repeated or skipped channel would apply the wrong constants."""
    wbits = man["roms"][f"{name}_w"]["weight_bits"]
    w = _words(GEN / f"{name}_w.hex", wbits)
    got = run_dense([1] * c_in, w, c_in, c_out, 32)
    assert [ch for ch, _ in got] == list(range(c_out))


def test_the_accumulator_is_cleared_between_channels(man):
    """Emit and clear happen in the same cycle. If the clear were a cycle late
    the next channel would start from the previous channel's total -- which
    looks like a plausible number, not like a bug."""
    c_in, c_out = 4, 3
    w = [1] * (c_in * c_out)
    got = run_dense([10, 20, 30, 40], w, c_in, c_out, 32)
    assert [acc for _, acc in got] == [100, 100, 100], got


def test_a_wrong_pipeline_depth_would_be_caught(man):
    """The check that the golden comparison is a real one.

    Pairing each weight with the activation one step off produces a different
    answer on real data; if it did not, the test above would pass for a broken
    schedule.
    """
    wbits = man["roms"]["conv3_w"]["weight_bits"]
    w = _words(GEN / "conv3_w.hex", wbits)
    x = _words(GOLD / "conv2_pw_out.hex")
    frame = [x[i * T] for i in range(128)]
    right = run_dense(frame, w, 128, 128, 32)
    skewed = run_dense(frame[1:] + frame[:1], w, 128, 128, 32)
    assert right != skewed
