"""Does the trigger depend on what it heard before? (wake plan step 3.5 follow-up)

eval_trigger.py scores ONE word per stream, from a fresh background. The
board showed (stream 1, firmware/stm32_wake/README.md) that after a few words
the integer background keeps a floor -- `bg_q += on - (bg_q >> 5)` stops
decaying below 32 -- and later words trigger 1..3 frames late; one of them
changed class. This tool measures that on real GSC test clips:

    streams of WORDS words, each after GAP_MIN..GAP_MAX frames of quiet,
    under several quiet conditions (zero / noise at a level relative to the
    median keyword clip; the same noise crop is added to the word itself)

and for each background-decay rule (`decay_round` in simulate_triggers:
0 floor = firmware, 16 round, 31 ceil) reports

    recall / wrong / missed   per word, scored like eval_trigger.score_case
    fresh recall              the same word with only its own gap before it
                              (what eval_trigger measures)
    shifted                   words whose capture start differs from fresh
    q.FA/min, q.duty          quiet-only streams: keyword answers per minute
                              and fraction of time capturing

Inputs are a directory of GSC wavs (<word>/<file>.wav + _background_noise_),
so a test subset copied off the training machine is enough:

    python -m experiments.eval_trigger_history --run runs/bd_base --data <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.eval_trigger import (FIRMWARE_POLICY, NATIVE, WARMUP, Capture,
                                      _padded, overlap, simulate_triggers)

RULES = {"floor": 0, "round": 16, "ceil": 31}
GAP_MIN, GAP_MAX = 100, 300     # quiet before each word, frames
TAIL = 200
QUIET_STREAM = 3000             # frames (30 s) per quiet-only stream


def load_wav(path: Path, length: int = 16000) -> np.ndarray:
    import soundfile as sf
    w, _ = sf.read(str(path), dtype="float32", always_2d=False)
    w = w.reshape(-1)[:length]
    return np.pad(w, (0, length - len(w)))


def noise_crops(noise: List[np.ndarray], n: int, rms, rng) -> np.ndarray:
    """n one-second crops scaled to `rms` (None = digital zero)."""
    out = np.zeros((n, 16000), np.float32)
    if rms is None:
        return out
    for i in range(n):
        src = noise[rng.integers(len(noise))]
        st = rng.integers(len(src) - 16000)
        c = src[st:st + 16000]
        out[i] = c * (rms / max(1e-9, float(np.sqrt((c ** 2).mean()))))
    return out


def first_hit(caps: List[Capture], span: Tuple[int, int]):
    return next((c for c in caps if overlap(c.start, c.end, *span) > 0), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/bd_base")
    ap.add_argument("--data", required=True, help="GSC dir: <word>/*.wav, _background_noise_")
    ap.add_argument("--words", type=int, default=10, help="words per stream")
    ap.add_argument("--noise", nargs="+", default=["zero", "-40", "-30", "-20"])
    ap.add_argument("--quiet-streams", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--word-active-frac", type=float, default=0.02)
    args = ap.parse_args()

    import torch
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import KEYWORDS
    from experiments.eval_streaming import _infer
    from experiments.window_offset import FRAME, word_span
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model)
    model.load_state_dict(ck["model"])
    model.eval()
    afe = AFEFrontend(cfg.afe)
    load_afe_state(afe, ck["afe"])
    afe.eval()

    def to_bits(w: np.ndarray) -> np.ndarray:                 # [N, L] -> [N, 100, C]
        with torch.no_grad():
            return np.concatenate([
                (afe(torch.as_tensor(w[i:i + 256]), target_T=NATIVE) > 0)
                .to(torch.uint8).numpy().transpose(0, 2, 1)
                for i in range(0, len(w), 256)])

    data = Path(args.data)
    files = sorted(p for kw in KEYWORDS for p in (data / kw).glob("*.wav"))
    waves = np.stack([load_wav(p) for p in files])
    labels = np.array([KEYWORDS.index(p.parent.name) for p in files])
    noise = [load_wav(p, 10 ** 9) for p in sorted((data / "_background_noise_").glob("*.wav"))]
    noise = [n[:np.flatnonzero(n).max() + 1] if n.any() else n for n in noise]
    wa, wb = word_span(torch.as_tensor(waves), active_frac=args.word_active_frac)
    spans = np.stack([wa.numpy() // FRAME, np.maximum(wb.numpy() // FRAME, wa.numpy() // FRAME + 1)], 1)
    ref_rms = float(np.median(np.sqrt((waves ** 2).mean(1))))
    n_str = len(files) // args.words
    print(f"{len(files)} keyword clips, {len(noise)} noise files, median RMS "
          f"{20 * np.log10(ref_rms):.1f} dBFS; {n_str} streams x {args.words} words; "
          f"policy {FIRMWARE_POLICY.label}")

    pol = FIRMWARE_POLICY
    every = np.ones(16, bool)
    print(f"\n{'noise':>5} {'rule':>5} | {'recall':>6} {'wrong':>6} {'missed':>6} | "
          f"{'fresh':>6} {'shifted':>8} | {'q.FA/min':>8} {'q.duty':>6} | bg_q at end (median)")
    for ci, cond in enumerate(args.noise):
        rng = np.random.default_rng(args.seed + ci)
        rms = None if cond == "zero" else ref_rms * 10 ** (float(cond) / 20)
        word_bits = to_bits(waves + noise_crops(noise, len(waves), rms, rng))
        n_quiet = 4 * n_str * args.words + args.quiet_streams * (QUIET_STREAM // NATIVE + 1)
        quiet = to_bits(noise_crops(noise, n_quiet, rms, rng)).reshape(-1, 16)
        qpos = 0

        def take(n: int) -> np.ndarray:
            nonlocal qpos
            if qpos + n > len(quiet):
                qpos = 0
            qpos += n
            return quiet[qpos - n:qpos]

        order = rng.permutation(len(files))[:n_str * args.words].reshape(n_str, args.words)
        streams, fresh = [], []          # fresh: (stream, word frame offset in it, clip)
        for row in order:
            parts, words = [take(WARMUP)], []
            pos = WARMUP
            for i in row:
                gap = int(rng.integers(GAP_MIN, GAP_MAX + 1))
                g = take(gap)
                parts += [g, word_bits[i]]
                fresh.append((np.concatenate([parts[0], g, word_bits[i], take(TAIL)]),
                              WARMUP + gap, i))
                words.append((pos + gap, i))
                pos += gap + NATIVE
            parts.append(take(TAIL))
            streams.append((np.concatenate(parts), words))
        quiets = [np.concatenate([take(WARMUP), take(QUIET_STREAM)]) for _ in range(args.quiet_streams)]

        for name, r in RULES.items():
            runs = [(s, simulate_triggers(s, pol, every, len(s), decay_round=r), w) for s, w in streams]
            fr = [(s, simulate_triggers(s, pol, every, len(s), decay_round=r), a, i) for s, a, i in fresh]
            qr = [(s, simulate_triggers(s, pol, every, len(s), decay_round=r)) for s in quiets]
            windows = ([s[c.start:c.end] for s, cs, _ in runs for c in cs]
                       + [s[c.start:c.end] for s, cs, _, _ in fr for c in cs]
                       + [s[c.start:c.end] for s, cs in qr for c in cs])
            _, pred = _infer(model, np.stack([_padded(x) for x in windows]), torch.device("cpu"), 256)
            it = iter(pred.tolist())
            got: Dict[int, int] = {}
            hit = wrong = missed = 0
            starts: Dict[int, int] = {}
            for s, cs, ws in runs:
                p = {c.start: next(it) for c in cs}
                for a, i in ws:
                    span = (a + spans[i][0], a + spans[i][1])
                    c = first_hit(cs, span)
                    if c is None:
                        missed += 1
                        continue
                    starts[i] = c.start - a
                    hit += p[c.start] == labels[i]
                    wrong += p[c.start] != labels[i] and p[c.start] < len(KEYWORDS)
            f_hit = shifted = 0
            for s, cs, a, i in fr:
                p = {c.start: next(it) for c in cs}
                c = first_hit(cs, (a + spans[i][0], a + spans[i][1]))
                if c is not None:
                    f_hit += p[c.start] == labels[i]
                    shifted += i in starts and starts[i] != c.start - a
            fa = awake = 0
            for s, cs in qr:
                fa += sum(next(it) < len(KEYWORDS) for _ in cs)
                awake += len(cs) * NATIVE
            qmin = args.quiet_streams * QUIET_STREAM / 6000
            n = len(fresh)
            bg_end = []
            for s, _ in streams:   # background left at the end of each stream
                on = s.sum(1)
                bg = int(on[:WARMUP].sum())
                for o in on[WARMUP:]:
                    if (o << 5) < bg + (pol.delta << 5):
                        bg += int(o) - ((bg + r) >> 5)
                bg_end.append(bg)
            print(f"{cond:>5} {name:>5} | {hit / n:6.3f} {wrong / n:6.3f} {missed / n:6.3f} | "
                  f"{f_hit / n:6.3f} {shifted:4d}/{len(starts):<3d} | {fa / qmin:8.2f} "
                  f"{awake / (args.quiet_streams * QUIET_STREAM):6.3f} | {np.median(bg_end) / 32:.2f} ch")


if __name__ == "__main__":
    main()
