"""A comparator trace from the analog side -> the tensor the network eats.

This is the sticky-OR framing of rtl/kws_frame_ctrl.v, in Python, so a SPICE
run can be turned into an input.hex without a board or a simulator.

It also DEFINES the format we are asking the analog side for, which is a
sharper way to ask than describing one: produce this CSV and this script eats
it.

    channel,time_s,value
    0,0.001234,1
    0,0.001901,0
    3,0.002011,1
    ...

One row per TRANSITION, not per sample. Every channel is assumed 0 before its
first row. That is the natural output of a SPICE run and it is small -- a second
of speech is a few thousand rows, where uniform sampling at any useful rate is
millions.

WHAT COMES OUT IS NOT AUTOMATICALLY CLASSIFIABLE. The weights were trained on
our software AFE (docs/WHITEPAPER.md 1.1), so feeding them a pattern from a
different front end gives a confident wrong answer, not an error. The first job
for a real trace is to DIFF it against our simulation of the same audio, which
is what --compare does.

    python -m experiments.trace_to_frames --trace cmp.csv --out cmp_input.hex
    python -m experiments.trace_to_frames --trace cmp.csv --compare runs/xl_g12 \\
           --wav path/to/the_same_clip.wav
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

FRAME_MS = 10.0
NATIVE_T = 100
T = 128
PAD_LEFT = 14
N_CH = 16
WORD_BITS = 32


def read_trace(path: Path, n_ch: int = N_CH) -> List[Tuple[float, int, int]]:
    """[(time_s, channel, value)] sorted by time. Tolerates any row order."""
    rows: List[Tuple[float, int, int]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        need = {"channel", "time_s", "value"}
        if not need <= set(r.fieldnames or []):
            raise ValueError(f"{path}: need columns {sorted(need)}, "
                             f"got {r.fieldnames}")
        for i, row in enumerate(r, 2):
            c = int(row["channel"])
            if not 0 <= c < n_ch:
                raise ValueError(f"{path}:{i}: channel {c} outside 0..{n_ch-1}")
            v = int(row["value"])
            if v not in (0, 1):
                raise ValueError(f"{path}:{i}: value {v} is not 0 or 1")
            rows.append((float(row["time_s"]), c, v))
    rows.sort(key=lambda t: t[0])
    return rows


def frames_from_trace(rows, t0: float = 0.0, n_ch: int = N_CH,
                      frame_ms: float = FRAME_MS,
                      native_t: int = NATIVE_T) -> List[int]:
    """Sticky OR per window. Returns `native_t` frames, bit c = channel c.

    The same rule as the hardware and as the training pipeline: a window is 1
    if the comparator was high at ANY point inside it (CLAUDE.md 2.8). A
    transition to 1 marks its window and every window it stays high through.
    """
    out = [0] * native_t
    level = [0] * n_ch
    last_t = [t0] * n_ch
    win = frame_ms / 1000.0

    def mark(c: int, a: float, b: float) -> None:
        """channel c was high over [a, b) -- set every window it touches."""
        lo = max(0, int((a - t0) // win))
        hi = min(native_t - 1, int((b - t0 - 1e-12) // win))
        for w in range(lo, hi + 1):
            if 0 <= w < native_t:
                out[w] |= 1 << c

    for t, c, v in rows:
        if v == level[c]:
            continue
        if v == 0:                      # a high run just ended
            mark(c, last_t[c], t)
        last_t[c] = t
        level[c] = v
    end = t0 + native_t * win
    for c in range(n_ch):
        if level[c]:
            mark(c, last_t[c], end)
    return out


def pad(frames: List[int], t: int = T, pad_left: int = PAD_LEFT) -> List[int]:
    """100 -> 128 with -1 on both sides, which packs as an all-zero word."""
    right = t - pad_left - len(frames)
    if right < 0:
        raise ValueError(f"{len(frames)} frames do not fit in T={t} "
                         f"with pad_left={pad_left}")
    return [0] * pad_left + list(frames) + [0] * right


def to_hex(frames: List[int], n_ch: int = N_CH,
           word_bits: int = WORD_BITS) -> str:
    """The layout export/golden.py writes: one word per (clip, frame)."""
    nw = (n_ch + word_bits - 1) // word_bits
    lines = []
    for fr in frames:
        for j in range(nw):
            lines.append(f"{(fr >> (j * word_bits)) & 0xFFFFFFFF:08x}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write an input.hex here")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="the trace time that is frame 0 (seconds)")
    ap.add_argument("--compare", type=Path,
                    help="runs/<tag> -- diff against our AFE on the same audio")
    ap.add_argument("--wav", type=Path, help="the clip the trace came from")
    args = ap.parse_args()

    rows = read_trace(args.trace)
    if not rows:
        raise SystemExit(f"{args.trace} has no transitions")
    span = rows[-1][0] - rows[0][0]
    print(f"{len(rows)} transitions over {span * 1000:.1f} ms, "
          f"channels {min(r[1] for r in rows)}..{max(r[1] for r in rows)}")

    frames = pad(frames_from_trace(rows, t0=args.t0))
    on = [sum((f >> c) & 1 for f in frames) for c in range(N_CH)]
    print(f"\nframes high per channel (of {NATIVE_T} real):")
    print("  " + "  ".join(f"{c}:{n}" for c, n in enumerate(on)))
    dead = [c for c, n in enumerate(on) if n == 0]
    if dead:
        print(f"  channels that never fire: {dead}  <- a threshold too high, "
              f"or a channel mapping that is off")

    if args.out:
        args.out.write_text(to_hex(frames))
        print(f"\nwrote {args.out}")

    if args.compare:
        if not args.wav:
            raise SystemExit("--compare needs --wav: the point is to run OUR "
                             "front end on the SAME audio")
        _compare(frames, args.compare, args.wav)


def _compare(theirs: List[int], run: Path, wav: Path) -> None:
    """Our AFE on the same clip, bit for bit against theirs.

    This is the measurement that decides whether the trained weights mean
    anything on their circuit. A high disagreement is not a bug in either side
    -- it says the network has to be retrained on data that matches the circuit
    (the Phase B plan), which is a training job, not an RTL one.
    """
    import torch
    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    import torchaudio

    cfg = load_config(str(run / "config.yaml"))
    afe = AFEFrontend(cfg.afe).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    load_afe_state(afe, ck["afe"])

    w, sr = torchaudio.load(str(wav))
    if sr != cfg.afe.sample_rate:
        w = torchaudio.functional.resample(w, sr, cfg.afe.sample_rate)
    with torch.no_grad():
        x = afe(w[:1], target_T=cfg.model.T)[0]       # [16, 128] in +-1

    ours = [int(sum((1 << c) for c in range(x.shape[0]) if x[c, t] > 0))
            for t in range(x.shape[1])]

    n_ch, T_ = x.shape
    diff = [bin(a ^ b).count("1") for a, b in zip(ours, theirs)]
    tot = sum(diff)
    print(f"\nours vs theirs on the same clip: {tot} of {n_ch * T_} bits differ "
          f"({tot / (n_ch * T_):.1%})")
    per_ch = [sum(((ours[t] ^ theirs[t]) >> c) & 1 for t in range(T_))
              for c in range(n_ch)]
    print("  per channel: " + "  ".join(f"{c}:{d}" for c, d in
                                        enumerate(per_ch)))
    worst = max(range(n_ch), key=lambda c: per_ch[c])
    print(f"\n  worst channel {worst} ({per_ch[worst]}/{T_} frames)")
    print("  A channel much worse than the rest usually means its threshold or "
          "its centre frequency moved, not that the whole front end is wrong.")
    print("  Above roughly 10% overall, the trained weights no longer describe "
          "this front end and the network wants retraining on matched data.")


if __name__ == "__main__":
    main()
