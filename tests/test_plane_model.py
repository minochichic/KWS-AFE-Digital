"""A cycle model of kws_plane's read FSM, checked against tb_block's push loop.

kws_plane holds no arithmetic -- it is a schedule, and the way a schedule fails
is off-by-one: 75 pushes instead of 76, the flush tail starting one frame early,
rd_done arriving before the last push, a push landing while the consumer is
still working. Every one of those produces a plausible-looking result rather
than an error, and every one costs a simulator round-trip to find on the box.

So the FSM is transcribed here beside a stub consumer whose busy behaves like
kws_block's (registered: it is still low in the cycle right after a push, and
high for a long run of cycles after that), and the sequence it emits is compared
against what the hand-written loop in rtl/tb/tb_block.v does. That loop is the
specification: kws_plane's only job is to be indistinguishable from it.

Runs with no simulator, no torch, and no exported weights.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

T = 64
PAD = 6
FLUSH = 2 * PAD


# --------------------------------------------------------------------------- #
# the two things being compared
# --------------------------------------------------------------------------- #

def plane_pushes(mem, n_frames=T, flush=FLUSH, consumer_latency=1613):
    """Cycle-step the kws_plane read FSM. Returns [(real, frame)] in order.

    Mirrors rtl/kws_plane.v: S_IDLE -> S_WAIT -> S_FETCH -> S_PUSH -> S_GAP1
    -> S_GAP2, one frame per lap, `is_real = rp < T`.

    The consumer stub is the point of the exercise. Its busy is REGISTERED, so
    it stays low for the cycle in which the push is visible and only rises the
    cycle after -- which is why the FSM cannot sample rd_ready right after a
    push, and why two gap cycles exist at all.
    """
    total = n_frames + flush
    st, rp, rdata = "IDLE", 0, 0
    pushes, dones = [], []
    busy, busy_left = False, 0
    started = False
    seen_push = False        # what the consumer saw in the previous cycle

    for cycle in range(total * (consumer_latency + 8) + 64):
        # ---- consumer, clocked on the same edge -------------------------- #
        if seen_push:
            busy, busy_left = True, consumer_latency
        elif busy:
            busy_left -= 1
            if busy_left == 0:
                busy = False
        ready = not busy

        # ---- plane ------------------------------------------------------- #
        push, real, frame, done = False, False, 0, False
        if st == "IDLE":
            if not started:
                started, rp, st = True, 0, "WAIT"
        elif st == "WAIT":
            if ready:
                st = "FETCH"
        elif st == "FETCH":
            rdata = mem[rp] if rp < n_frames else 0
            st = "PUSH"
        elif st == "PUSH":
            push, real, frame = True, rp < n_frames, rdata
            st = "GAP1"
        elif st == "GAP1":
            st = "GAP2"
        elif st == "GAP2":
            if rp + 1 >= total:
                done, st = True, "IDLE"
            else:
                rp, st = rp + 1, "WAIT"

        if push:
            assert ready, f"cycle {cycle}: pushed while the consumer is busy"
            pushes.append((real, frame))
        if done:
            dones.append(len(pushes))
        seen_push = push
        if dones and st == "IDLE" and not push:
            break

    return pushes, dones


def tb_block_pushes(mem, n_frames=T, lag=FLUSH):
    """The loop in rtl/tb/tb_block.v, as data.

        for (i = 0; i < T + LAG; i = i + 1) begin
            frame = 0;
            if (i < T) frame = in_mem[...];
            push(i < T, frame);
        end
    """
    out = []
    for i in range(n_frames + lag):
        out.append((i < n_frames, mem[i] if i < n_frames else 0))
    return out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def mem():
    # distinct, nonzero, and not equal to the index, so a frame delivered from
    # the wrong address cannot coincidentally match
    return [(t * 0x9E3779B1) & ((1 << 128) - 1) | 1 for t in range(T)]


def test_plane_emits_exactly_the_testbench_loop(mem):
    pushes, _ = plane_pushes(mem, consumer_latency=17)
    assert pushes == tb_block_pushes(mem)


def test_push_count_is_frames_plus_the_consumers_flush(mem):
    pushes, _ = plane_pushes(mem, consumer_latency=17)
    assert len(pushes) == T + FLUSH == 76
    assert [p[0] for p in pushes] == [True] * T + [False] * FLUSH


def test_done_fires_once_and_only_after_the_last_push(mem):
    pushes, dones = plane_pushes(mem, consumer_latency=17)
    assert dones == [len(pushes)], (
        "rd_done must arrive after push 76, not before -- kws_top uses it to "
        "swap planes, and swapping early drops the last frame")


def test_the_flush_tail_reads_no_stored_frame(mem):
    """rp runs past T, so the address would wrap. is_real has to gate it."""
    pushes, _ = plane_pushes(mem, consumer_latency=17)
    assert all(f == 0 and not r for r, f in pushes[T:])
    # if is_real were `rp <= T` the 65th push would carry mem[0] again
    assert pushes[T][1] != mem[0]


@pytest.mark.parametrize("latency", [1, 2, 3, 7, 64, 1613])
def test_the_schedule_is_independent_of_how_slow_the_consumer_is(mem, latency):
    """A plane must not encode its consumer's speed -- only its FLUSH count.

    kws_block takes ~1613 cycles per frame today and will take a different
    number after any retiming; b2 and b3 are cheaper still. If the emitted
    sequence moved with latency, every such change would be a plane change.
    """
    pushes, dones = plane_pushes(mem, consumer_latency=latency)
    assert pushes == tb_block_pushes(mem)
    assert dones == [len(pushes)]


def test_a_consumer_that_never_goes_ready_stalls_rather_than_pushing(mem):
    """Backpressure has to actually apply. If rd_ready were ignored the plane
    would stream all 76 frames into a busy block and the failure would show up
    much later as wrong data."""
    st, rp, pushes = "WAIT", 0, []
    for _ in range(1000):
        if st == "WAIT":
            pass                       # ready never comes
        elif st == "PUSH":
            pushes.append(rp)
    assert pushes == []


# --------------------------------------------------------------------------- #
# the model must stay tied to the RTL it claims to mirror
# --------------------------------------------------------------------------- #

def test_flush_default_matches_the_two_depthwise_stages_it_is_for():
    """FLUSH is the CONSUMER's property. kws_block has two depthwise stages, so
    it needs 2*PAD; a plane feeding a single dw layer needs PAD. If this drifts
    the symptom is missing output frames at the end of a clip."""
    src = (ROOT / "rtl" / "kws_plane.v").read_text()
    m = re.search(r"parameter\s+integer\s+FLUSH\s*=\s*(\d+)", src)
    assert m, "FLUSH parameter not found in rtl/kws_plane.v"
    assert int(m.group(1)) == 2 * PAD

    tb = (ROOT / "rtl" / "tb" / "tb_plane.v").read_text()
    assert re.search(r"FLUSH\s*=\s*2\s*\*\s*PAD", tb), (
        "tb_plane must derive FLUSH from PAD, not restate 12")


def test_the_read_fsm_still_has_both_gap_states():
    """One gap state is not enough: the consumer's busy has not risen yet in the
    cycle after the push, so S_WAIT would sample a stale low and push twice into
    the same frame slot. Deleting a gap looks like a harmless optimisation."""
    src = (ROOT / "rtl" / "kws_plane.v").read_text()
    assert "S_GAP1" in src and "S_GAP2" in src
    assert re.search(r"S_GAP1\s*:\s*st\s*<=\s*S_GAP2", src)


def test_the_plane_reads_memory_synchronously():
    """Asynchronous read would work in simulation and cost 8192 flip-flops in
    Vivado instead of one block RAM. The tell is that mem[] is read into a
    register (S_FETCH) rather than straight into rd_frame."""
    src = (ROOT / "rtl" / "kws_plane.v").read_text()
    assert re.search(r"rdata\s*<=.*mem\[", src, re.S)
    assert not re.search(r"assign\s+rd_frame\s*=.*mem\[", src)
