"""The discretisation rule, on the case CLAUDE.md 2.8 spells out.

CLAUDE.md section 6 asks for exactly this: the rule pulled out as a pure
function and fed the 2.8 table, because the full pipeline runs at 10 ms STFT
frames and cannot be handed a sub-millisecond pulse train at all. This is the
only place the sticky OR is checked against real event timing.

It is also the only check that the file we ask the analog side for turns into
the file the testbench reads. Both ends of that are contracts with someone
else -- a CSV a colleague produces and a $readmemh layout the Verilog fixes --
so neither can be adjusted later to make a mismatch go away.
"""
from __future__ import annotations

import pytest

from experiments.trace_to_frames import (N_CH, NATIVE_T, PAD_LEFT, T,
                                         frames_from_trace, pad, read_trace,
                                         to_hex, write_vectors)


def rows(*ev):
    """(time_ms, channel, value) -> the (time_s, channel, value) rows sorted."""
    return sorted(((t / 1000.0, c, v) for t, c, v in ev), key=lambda r: r[0])


def test_the_claude_md_2_8_table() -> None:
    """Fig.1's 20 ms: pulses at 1.2, 4.6, 13.4 and 18.1 ms, 10 ms windows.

    | window | time     | pulses inside | max |
    |      0 | 0-10 ms  | 1.2, 4.6      |   1 |
    |      1 | 10-20 ms | 13.4, 18.1    |   1 |

    Four pulses in, two bits out. The count, the arrival times and the widths
    are all gone -- one pulse would have given the same answer. That loss is
    the rule, not a shortcoming of this implementation.
    """
    ev = []
    for t in (1.2, 4.6, 13.4, 18.1):
        ev += [(t, 0, 1), (t + 0.2, 0, 0)]     # sub-ms pulses
    fr = frames_from_trace(rows(*ev))
    assert fr[0] & 1, "window 0 saw pulses at 1.2 and 4.6 ms"
    assert fr[1] & 1, "window 1 saw pulses at 13.4 and 18.1 ms"
    assert all(f == 0 for f in fr[2:]), "nothing fired after 20 ms"


def test_one_pulse_and_four_pulses_are_the_same_bit() -> None:
    """The information the rule throws away, stated as a test so that a future
    `count` or `mean` reduce cannot be introduced without this failing."""
    four = frames_from_trace(rows(*[e for t in (1.2, 4.6, 8.0, 9.5)
                                    for e in ((t, 0, 1), (t + 0.2, 0, 0))]))
    one = frames_from_trace(rows((1.2, 0, 1), (1.4, 0, 0)))
    assert four[0] == one[0] == 1


def test_a_run_spanning_windows_marks_every_one_it_touches() -> None:
    """High from 5 ms to 35 ms is four windows, not two endpoints."""
    fr = frames_from_trace(rows((5.0, 2, 1), (35.0, 2, 0)))
    assert [bool(fr[w] & (1 << 2)) for w in range(5)] == \
           [True, True, True, True, False]


def test_a_run_still_high_at_the_end_reaches_the_last_frame() -> None:
    """No falling edge in the trace must not mean the channel goes quiet -- the
    natural shape of a clip that ends mid-word."""
    fr = frames_from_trace(rows((995.0, 15, 1)))
    assert fr[NATIVE_T - 1] == 1 << 15


def test_a_channel_is_low_until_its_first_row() -> None:
    fr = frames_from_trace(rows((50.0, 7, 1), (55.0, 7, 0)))
    assert all(f == 0 for f in fr[:5])
    assert fr[5] == 1 << 7


def test_a_repeated_level_is_not_a_transition() -> None:
    """A trace that restates 1 while already high must not restart the run, or
    the time before the restatement is lost."""
    a = frames_from_trace(rows((5.0, 0, 1), (35.0, 0, 0)))
    b = frames_from_trace(rows((5.0, 0, 1), (15.0, 0, 1), (25.0, 0, 1),
                               (35.0, 0, 0)))
    assert a == b


def test_row_order_in_the_file_does_not_matter(tmp_path) -> None:
    """We are asking someone else to produce this file. Per-channel blocks are
    as natural an export as time order, and both must read the same."""
    body = [(0.0050, 0, 1), (0.0350, 0, 0), (0.0100, 5, 1), (0.0200, 5, 0)]
    head = "channel,time_s,value\n"
    f1 = tmp_path / "time.csv"
    f1.write_text(head + "".join(f"{c},{t},{v}\n" for t, c, v in
                                 sorted(body, key=lambda r: r[0])))
    f2 = tmp_path / "chan.csv"
    f2.write_text(head + "".join(f"{c},{t},{v}\n" for t, c, v in
                                 sorted(body, key=lambda r: (r[1], r[0]))))
    assert frames_from_trace(read_trace(f1)) == frames_from_trace(read_trace(f2))


def test_t0_shifts_the_whole_grid(tmp_path) -> None:
    """A settling period before the audio is the common case, and getting t0
    wrong slides every frame without any other symptom."""
    late = rows((105.0, 3, 1), (115.0, 3, 0))
    assert frames_from_trace(late)[10] == 1 << 3
    assert frames_from_trace(late, t0=0.100)[0] == 1 << 3


@pytest.mark.parametrize("bad,msg", [
    ("channel,time_s\n0,0.1\n", "value"),
    ("channel,time_s,value\n16,0.1,1\n", "channel"),
    ("channel,time_s,value\n0,0.1,2\n", "not 0 or 1"),
])
def test_a_malformed_file_is_refused_by_name(tmp_path, bad, msg) -> None:
    """This file comes from outside the repo, so its errors must name what is
    wrong rather than produce frames that are quietly incomplete."""
    p = tmp_path / "bad.csv"
    p.write_text(bad)
    with pytest.raises(ValueError, match=msg):
        read_trace(p)


def test_padding_matches_the_layout_the_testbench_reads() -> None:
    fr = [1] * NATIVE_T
    p = pad(fr)
    assert len(p) == T
    assert p[:PAD_LEFT] == [0] * PAD_LEFT
    assert p[PAD_LEFT:PAD_LEFT + NATIVE_T] == fr
    assert p[PAD_LEFT + NATIVE_T:] == [0] * (T - PAD_LEFT - NATIVE_T)


def test_hex_is_one_word_per_frame_with_channel_c_at_bit_c() -> None:
    """The $readmemh layout tb_top fixes: one 32-bit word per frame, low 16
    bits used, ch0 the least significant."""
    frames = pad([0] * NATIVE_T)
    frames[PAD_LEFT] = (1 << 0) | (1 << 3) | (1 << 15)
    lines = to_hex(frames).strip().splitlines()
    assert len(lines) == T
    assert lines[PAD_LEFT] == f"{0x8009:08x}"
    assert lines[0] == "00000000", "padding is -1, which packs as a zero word"


def test_vectors_dir_is_what_run_tb_needs(tmp_path) -> None:
    frames = pad([0] * NATIVE_T)
    write_vectors(frames, tmp_path / "vec", want=9)
    assert len((tmp_path / "vec" / "input.hex").read_text()
               .strip().splitlines()) == T
    assert (tmp_path / "vec" / "predictions_fixed.txt").read_text() == "9\n"
    vh = (tmp_path / "vec" / "paths.vh").read_text()
    assert "`define KWS_GOLD_CLIPS   1" in vh
    assert "KWS_GOLD_INPUT" in vh and "KWS_GOLD_PREDICTIONS_FIXED" in vh
    # the per-layer vectors describe OUR bits, so they must not be claimed here
    assert "CONV1_ACC" not in vh and "B1_S0_DW_OUT" not in vh


def test_channels_are_independent() -> None:
    """Sixteen channels, sixteen separate runs -- a shared `level` list would
    show up as one channel closing another's pulse."""
    fr = frames_from_trace(rows(*[(5.0, c, 1) for c in range(N_CH)],
                                *[(5.0 + c, c, 0) for c in range(N_CH)]))
    assert fr[0] == (1 << N_CH) - 1
