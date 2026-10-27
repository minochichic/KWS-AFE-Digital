"""The STM32 firmware's decision logic, checked against the simulator.

firmware/stm32_wake/core/wake_core.c is what the board runs;
experiments/eval_trigger.py::simulate_triggers is what measured the rule's
recall, duty and false alarms (docs/wake_stm32_plan.md 2.3). If the two
disagree on a single frame, those numbers are not the board's numbers. So the
C is compiled here with the Mac's cc, fed the same random streams as the
simulator, and every capture -- where it starts, where it ends, all 100 frames
-- must match.

Several policies are compiled (-D overrides in wake_cfg.h), not only the
shipped one, so a bug that happens to cancel out at k1/d4/run1/pre30 has
nowhere to hide. Builds with AddressSanitizer + UBSan when the compiler can.

Also: wake_cfg.h's defaults must equal FIRMWARE_POLICY, and link.c must make
the same bytes as export/wake_link.py.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.eval_trigger import (
    BG_SHIFT,
    FIRMWARE_POLICY,
    NATIVE,
    WARMUP,
    TriggerPolicy,
    simulate_triggers,
)
from export.wake_link import Result, crc8, frame_packet, result_packet

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "stm32_wake"
CORE = FW / "core"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")

STRICT = ["-std=c99", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{CORE}"]


def _build(tmp: Path, name: str, srcs, defines=()) -> Path:
    exe = tmp / name
    base = [CC, *STRICT, *[f"-D{d}" for d in defines], *map(str, srcs), "-o", str(exe)]
    san = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-g"]
    if subprocess.run(base[:1] + san + base[1:], capture_output=True).returncode != 0:
        subprocess.run(base, check=True, capture_output=True)
    return exe


# --------------------------------------------------------------------------- #
# constants: one rule, written twice
# --------------------------------------------------------------------------- #
def _cfg_defaults() -> dict:
    txt = (CORE / "wake_cfg.h").read_text()
    return {m.group(1): int(m.group(2), 0) for m in
            re.finditer(r"#define\s+WAKE_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", txt)}


def test_cfg_matches_the_simulated_policy():
    d = _cfg_defaults()
    p = FIRMWARE_POLICY
    assert (d["K"], d["DELTA"], d["RUN"], d["PRE"], d["BUSY"]) == \
        (p.k, p.delta, p.run, p.pre, p.busy)
    assert d["COUNT_MASK"] == 0xFFFF and p.hot >= 1.0       # every channel counted
    assert d["BG_SHIFT"] == BG_SHIFT and (1 << d["BG_SHIFT"]) == WARMUP
    assert d["NATIVE"] == NATIVE


def test_clip_length_is_the_exports():
    """WAKE_NATIVE is the model's native frame count; the FPGA drops a packet
    whose N differs (rtl/kws_link_rx.v)."""
    vh = ROOT / "rtl" / "gen" / "bd_base" / "parameters.vh"
    if not vh.is_file():
        pytest.skip("no bd_base export")
    m = re.search(r"`define\s+KWS_NATIVE_T\s+(\d+)", vh.read_text())
    assert m and int(m.group(1)) == _cfg_defaults()["NATIVE"]


# --------------------------------------------------------------------------- #
# wake_core vs simulate_triggers
# --------------------------------------------------------------------------- #
def _random_stream(rng: np.random.Generator) -> np.ndarray:
    """Noise floors from silent to 90% on, words of 5..90 frames anywhere --
    inside the warm-up, during a capture, back to back."""
    n = int(rng.integers(150, 700))
    floor = rng.uniform(0, 0.9, 16) * rng.choice([0.0, 0.05, 0.3, 1.0])
    s = (rng.random((n, 16)) < floor).astype(np.uint8)
    for _ in range(int(rng.integers(0, 6))):
        a, length = int(rng.integers(0, n)), int(rng.integers(5, 90))
        ch = rng.random(16) < rng.uniform(0.1, 0.9)
        seg = s[a:a + length]
        seg[:, ch] |= (rng.random((len(seg), int(ch.sum()))) < 0.8).astype(np.uint8)
    return s


def _words(s: np.ndarray) -> np.ndarray:
    return (s.astype(np.uint32) << np.arange(16, dtype=np.uint32)).sum(axis=1)


def _run_c(exe: Path, words_list) -> list:
    text = "".join(f"S {len(w)}\n" + " ".join(f"{x:x}" for x in w) + "\n"
                   for w in words_list)
    out = subprocess.run([str(exe)], input=text, capture_output=True, text=True,
                         check=True).stdout.split("\n")
    streams, cur, i = [], [], 0
    while i < len(out):
        line = out[i]
        if line.startswith("C "):
            _, a, b = line.split()
            clip = [int(x, 16) for x in out[i + 1].split()[1:]]
            cur.append((int(a), int(b), clip))
            i += 2
            continue
        if line == "E":
            streams.append(cur)
            cur = []
        i += 1
    return streams


POLICIES = {
    # name: (TriggerPolicy, COUNT_MASK)
    "shipped":       (FIRMWARE_POLICY, 0xFFFF),
    "k3_run2_pre10": (TriggerPolicy(k=3, delta=0, run=2, pre=10, busy=5), 0xFFFF),
    "d2_run3_pre20": (TriggerPolicy(k=1, delta=2, run=3, pre=20, busy=0), 0xFFFF),
    "d4_mask_ch15":  (TriggerPolicy(k=2, delta=4, run=1, pre=30, busy=8), 0x7FFF),
    "m1_pure_edge":  (TriggerPolicy(k=1, delta=0, run=1, pre=0, busy=3), 0xFFFF),
}


@pytest.mark.parametrize("name", list(POLICIES))
def test_core_matches_simulator_frame_for_frame(tmp_path, name):
    pol, mask = POLICIES[name]
    defines = [] if name == "shipped" else [
        f"WAKE_K={pol.k}", f"WAKE_DELTA={pol.delta}", f"WAKE_RUN={pol.run}",
        f"WAKE_PRE={pol.pre}", f"WAKE_BUSY={pol.busy}",
        f"WAKE_COUNT_MASK={mask:#x}u"]
    exe = _build(tmp_path, "wake_sim",
                 [CORE / "wake_core.c", FW / "host" / "wake_sim.c"], defines)

    rng = np.random.default_rng(20261027 + len(name))
    streams = [_random_stream(rng) for _ in range(1500)]
    counted = np.array([(mask >> k) & 1 for k in range(16)], dtype=bool)
    words = [_words(s) for s in streams]
    got = _run_c(exe, words)
    assert len(got) == len(streams)

    n_caps = 0
    for i, (s, w, g) in enumerate(zip(streams, words, got)):
        want = simulate_triggers(s, pol, counted, scan_end=len(s))
        assert [(c.start, c.end) for c in want] == [(a, b) for a, b, _ in g], \
            f"stream {i}: captures differ"
        for c, (_, _, clip) in zip(want, g):
            assert clip == w[c.start:c.end].tolist(), f"stream {i}: clip content"
        n_caps += len(want)
    assert n_caps > 300, "the streams should trigger often enough to test anything"


def test_silence_never_captures(tmp_path):
    exe = _build(tmp_path, "wake_sim", [CORE / "wake_core.c", FW / "host" / "wake_sim.c"])
    assert _run_c(exe, [np.zeros(1000, dtype=np.uint32)]) == [[]]


def test_ring_supplies_the_frames_before_the_onset(tmp_path):
    exe = _build(tmp_path, "wake_sim", [CORE / "wake_core.c", FW / "host" / "wake_sim.c"])
    w = np.arange(400, dtype=np.uint32) & 0x0001     # ch0 flickers: floor ~0.5
    w[200] = 0x3F00                                    # 6 more channels: sound
    (a, b, clip), = _run_c(exe, [w])[0]
    assert (a, b) == (200 - FIRMWARE_POLICY.pre, 200 - FIRMWARE_POLICY.pre + NATIVE)
    assert clip == w[a:b].tolist() and clip[FIRMWARE_POLICY.pre] == 0x3F00


# --------------------------------------------------------------------------- #
# link.c vs export/wake_link.py
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def link_cli(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("link")
    return _build(tmp, "link_cli", [CORE / "link.c", FW / "host" / "link_cli.c"])


def _cli(exe, *args, stdin=None) -> str:
    return subprocess.run([str(exe), *args], input=stdin, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_crc8_check_value(link_cli):
    assert _cli(link_cli, "crc", *[f"{b:x}" for b in b"123456789"]) == "f4"
    assert int(_cli(link_cli, "crc", "a5", "64", "0", "80"), 16) == crc8(bytes([0xA5, 0x64, 0, 0x80]))


@pytest.mark.parametrize("n", [1, 100, 255])
def test_frame_packet_bytes_match_python(link_cli, n):
    rng = np.random.default_rng(n)
    frames = rng.integers(0, 1 << 16, n).tolist()
    got = bytes.fromhex(_cli(link_cli, "pkt", stdin=f"{n} " + " ".join(f"{f:x}" for f in frames)))
    assert got == frame_packet(frames)


@pytest.mark.parametrize("margin", [0, 1, -1, 27918, -8612, 2**31 - 1, -2**31])
def test_result_parse_matches_python(link_cli, margin):
    pkt = result_packet(Result(11, 5, margin))
    rc, c, k, m = _cli(link_cli, "res", *[f"{b:x}" for b in pkt]).split()
    assert (int(rc), int(c), int(k), int(m)) == (0, 11, 5, margin)


def test_result_parse_rejects_corruption(link_cli):
    pkt = bytearray(result_packet(Result(5, 5, 100)))
    bad_crc = bytes(pkt[:3]) + bytes([pkt[3] ^ 1]) + bytes(pkt[4:])
    assert _cli(link_cli, "res", *[f"{b:x}" for b in bad_crc]).split()[0] == "-2"
    bad_sync = b"\x00" + bytes(pkt[1:])
    assert _cli(link_cli, "res", *[f"{b:x}" for b in bad_sync]).split()[0] == "-1"


# --------------------------------------------------------------------------- #
# replay clips (wake plan step 3)
# --------------------------------------------------------------------------- #
SELFTEST = ROOT / "rtl" / "gen" / "bd_base" / "selftest"


@pytest.mark.skipif(not (SELFTEST / "input.hex").is_file(), reason="no self-test export")
def test_replay_header_compiles_and_carries_the_vectors(tmp_path):
    from export.wake_link import native_frames
    from export.wake_replay import render

    expect = [int(x) for x in (SELFTEST / "predictions_fixed.txt").read_text().split()]
    hdr = tmp_path / "replay_clips.h"
    hdr.write_text(render(SELFTEST / "input.hex", expect, 3, "test"))
    prog = tmp_path / "p.c"
    prog.write_text('#include <stdio.h>\n#include "replay_clips.h"\n'
                    "int main(void){unsigned c,i;for(c=0;c<REPLAY_N;c++){"
                    "unsigned long s=0;for(i=0;i<REPLAY_FRAMES;i++)s+=replay_frames[c][i]*(i+1);"
                    'printf("%lu %u\\n",s,replay_expect[c]);}return 0;}\n')
    exe = tmp_path / "p"
    subprocess.run([CC, "-std=c99", "-Wall", "-Werror", f"-I{tmp_path}", str(prog),
                    "-o", str(exe)], check=True, capture_output=True)
    got = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split("\n")
    for c in range(3):
        f = native_frames(SELFTEST / "input.hex", c)
        assert got[c] == f"{sum(x * (i + 1) for i, x in enumerate(f))} {expect[c]}"
