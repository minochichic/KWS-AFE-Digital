"""Fit the threshold grid to the board itself: which software threshold makes the
software AFE fire where the real comparator fired, at each reference voltage.

    python -m experiments.thr_calib out/afe/live/*.jsonl --arb-dir <dir of the .arb files> \
        --sim analog/AFE_board/artifacts/thr_grid_sim_20261116.csv \
        --out analog/AFE_board/artifacts/thr_grid_board_<date>.csv

experiments/thr_grid.py sim builds the same table from the circuit model, and that
model is shakiest exactly where it matters (detector below 1 mV and above 4 mV,
the high bands). Here every cell is measured end to end -- filter, detector, DC,
comparator, the real timing -- because the input is a capture the board made:

  experiments/afe_web.py records every capture with the ARB that was playing and
  the V_thr typed on its page. For each capture, the software AFE runs on the
  same ARB (whole file, so the capture window can fall on its silences), at the
  level afe.input_pp gives every clip, and the capture is slid along that
  timeline to where it agrees best. Then for every (channel, voltage) the
  threshold is the one that best separates the board's 1s from its 0s over all
  aligned frames -- a 1-D exact search.

The agreement it reaches is the other result: a channel that stays well below
~0.9 even at its best threshold is not a threshold problem -- the software
filter or detector does not behave like that channel, and no table fixes it.

Collecting the data (T470, the bench as on 2026-10-30): set every channel to the
same voltage, type it into the page's V_thr box, play ~20 ARBs, repeat for each
of the 7 voltages. Mixed settings work too: type the 16 values.
"""
from __future__ import annotations

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.afe_clip import bits_of

NATIVE = 100
PAD = NATIVE                                     # frames of silence either side for sliding


def load_records(paths: List[Path]) -> List[dict]:
    """Captures that can calibrate: an ARB that had settled, and voltages that had."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ref") and r.get("vthr_mv") and r.get("clip"):
                out.append(r)
    return out


def arb_wave(path: Path, sr: int = 16000) -> np.ndarray:
    """The whole .arb (leads and tails included) at `sr`, offset removed. Scale does not
    matter: input_pp sets the level."""
    import torch
    import torchaudio
    from experiments.awg_thresholds import read_arb
    rate, mv = read_arb(path)
    mv = mv - mv[0]                              # the lead is digital silence
    x = torchaudio.functional.resample(torch.as_tensor(mv, dtype=torch.float32), int(rate), sr)
    return x.numpy()


class SoftwareAFE:
    """Raw envelopes of whole ARBs through the training front end (same bank, same
    compression, same input_pp), whatever the file length."""

    def __init__(self, config: str, overrides: dict) -> None:
        from train.config import load_config
        self.cfg = load_config(config, overrides)
        if self.cfg.afe.input_pp <= 0:
            raise SystemExit("set afe.input_pp (the bench sends every clip at one Vpp)")
        self._afe: Dict[int, object] = {}

    def raw(self, x: np.ndarray) -> np.ndarray:
        import copy
        import torch
        from data.afe import AFEFrontend
        hop = int(self.cfg.afe.sample_rate * self.cfg.afe.envelope_win_ms / 1000)
        n = -(-len(x) // hop) * hop
        if n not in self._afe:
            a = copy.deepcopy(self.cfg.afe)
            a.clip_ms = 1000.0 * n / a.sample_rate
            a.threshold_grid_path = ""
            self._afe[n] = AFEFrontend(a).eval()
        w = torch.as_tensor(np.pad(x, (0, n - len(x))), dtype=torch.float32)[None]
        with torch.no_grad():
            return self._afe[n].envelopes(w, raw=True)[0].numpy()     # [16, T]


def best_threshold(v: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Threshold t maximizing mean((v >= t) == y), and that agreement. Exact."""
    order = np.argsort(-v)
    vs, ys = v[order], y[order].astype(np.int64)
    ones_above = np.concatenate([[0], np.cumsum(ys)])            # firing the top k
    zeros_above = np.arange(len(vs) + 1) - ones_above
    correct = ones_above + (len(vs) - ys.sum() - zeros_above)
    k = int(np.argmax(correct))
    if k == 0:
        t = vs[0] * 1.001 + 1e-12                                # fires nowhere
    elif k == len(vs):
        t = vs[-1] * 0.999                                       # fires everywhere
    else:
        t = 0.5 * (vs[k - 1] + vs[k])
    return float(t), float(correct[k] / len(vs))


def window(env: np.ndarray, s: int) -> np.ndarray:
    """[16, NATIVE] of the padded timeline starting at s."""
    return env[:, s:s + NATIVE]


def align(env: np.ndarray, bits: np.ndarray, thr: np.ndarray) -> Tuple[int, float]:
    """Offset into the padded timeline where (env >= thr) agrees most with the capture."""
    best = (0, -1.0)
    b = bits.T.astype(bool)                                      # [16, 100]
    for s in range(env.shape[1] - NATIVE + 1):
        a = float(((window(env, s) >= thr[:, None]) == b).mean())
        if a > best[1]:
            best = (s, a)
    return best


def calibrate(records: List[dict], envs: Dict[str, np.ndarray], iters: int = 3):
    """-> {(ch, mV): (threshold, agreement, frames, board rate)}, offsets, per-capture agreement."""
    padded = {k: np.pad(e, ((0, 0), (PAD, PAD)), mode="edge") for k, e in envs.items()}
    bits = [bits_of(r["clip"]) for r in records]
    # start: per capture, the per-channel threshold firing on as many frames of the
    # whole timeline as the capture holds 1s -- crude, only to find the first offsets
    thr_rec = []
    for r, b in zip(records, bits):
        e = padded[r["ref"]]
        t = np.array([np.sort(e[c])[::-1][max(0, int(b[:, c].sum()) - 1)] if b[:, c].any()
                      else e[c].max() * 1.001 for c in range(16)])
        thr_rec.append(t)
    fit = {}
    for _ in range(iters):
        offs = [align(padded[r["ref"]], b, t)[0] for r, b, t in zip(records, bits, thr_rec)]
        groups = defaultdict(lambda: ([], []))
        for r, b, s in zip(records, bits, offs):
            w = window(padded[r["ref"]], s)
            for c in range(16):
                vals, ys = groups[(c, float(r["vthr_mv"][c]))]
                vals.append(w[c])
                ys.append(b[:, c])
        fit = {}
        for key, (vals, ys) in groups.items():
            v, y = np.concatenate(vals), np.concatenate(ys)
            t, acc = best_threshold(v, y)
            fit[key] = (t, acc, len(v), float(y.mean()))
        thr_rec = [np.array([fit[(c, float(r["vthr_mv"][c]))][0] for c in range(16)]) for r in records]
    agree = [align(padded[r["ref"]], b, t) for r, b, t in zip(records, bits, thr_rec)]
    return fit, [a[0] - PAD for a in agree], [a[1] for a in agree]


def to_grid(fit: dict, levels: List[float], sim: np.ndarray = None):
    """Measured cells; the rest from the sim table scaled to this channel's measured ones."""
    g = np.full((16, len(levels)), np.nan)
    src = [[""] * len(levels) for _ in range(16)]
    for (c, mv), (t, *_rest) in fit.items():
        if mv in levels:
            g[c, levels.index(mv)] = t
            src[c][levels.index(mv)] = "board"
    if sim is not None:
        for c in range(16):
            have = [j for j in range(len(levels)) if np.isfinite(g[c, j]) and np.isfinite(sim[c, j])
                    and sim[c, j] > 0]
            scale = float(np.median([g[c, j] / sim[c, j] for j in have])) if have else 1.0
            for j in range(len(levels)):
                if not np.isfinite(g[c, j]) and np.isfinite(sim[c, j]):
                    g[c, j] = sim[c, j] * scale
                    src[c][j] = "sim x%.2f" % scale if have else "sim"
    return g, src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="out/afe/live/*.jsonl from afe_web")
    ap.add_argument("--arb-dir", type=Path, required=True, help="where <ref>.arb are")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--spice-matrix", default="analog/AFE/artifacts/filterbank_matrix_board.csv")
    ap.add_argument("--input-pp", type=float, default=0.29, help="afe.input_pp the run will use")
    ap.add_argument("--sim", type=Path, default=None,
                    help="thr_grid sim table: fills voltages with no captures, scaled per channel")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from data.thr_grid import load_grid, write_grid
    recs = load_records(args.logs)
    if not recs:
        raise SystemExit("no capture carries both a settled ARB and V_thr -- type V_thr on the page")
    sw = SoftwareAFE(args.config, {"afe.spice_matrix_path": args.spice_matrix,
                                   "afe.input_pp": args.input_pp})
    envs = {}
    for ref in sorted({r["ref"] for r in recs}):
        p = args.arb_dir / f"{ref}.arb"
        if not p.is_file():
            raise SystemExit(f"{p} not found")
        envs[ref] = sw.raw(arb_wave(p))
    fit, offs, agree = calibrate(recs, envs)

    seen = sorted({mv for (_, mv) in fit})
    sim = None
    if args.sim:
        lv, sim = load_grid(args.sim)
        levels = sorted(set(lv) | set(seen))
        sim = np.stack([np.array([sim[c, lv.index(v)] if v in lv else np.nan for v in levels])
                        for c in range(16)])
    else:
        levels = seen
    g, src = to_grid(fit, levels, sim)

    print(f"{len(recs)} captures, {len(envs)} ARBs, voltages {', '.join(f'{v:g}' for v in seen)}")
    print(f"capture agreement after alignment: median {np.median(agree):.3f}, "
          f"min {min(agree):.3f}; offsets {min(offs)}..{max(offs)} frames vs the ARB start\n")
    print("per channel and voltage: agreement at the best threshold (board firing rate)")
    print(f"{'ch':>3} | " + "".join(f"{v:>14g}" for v in seen))
    for c in range(16):
        cells = "".join(f"{fit[(c, v)][1]:>8.3f} ({fit[(c, v)][3]:.2f})" if (c, v) in fit else f"{'':>14}"
                        for v in seen)
        print(f"{c:>3} | {cells}")
    weak = sorted({c for (c, v), f in fit.items() if f[1] < 0.9})
    if weak:
        print(f"\nbelow 0.9 even at the best threshold: ch {weak} -- the software front end does not "
              f"model these channels; check them on the board before trusting their cells")
    write_grid(args.out, levels, g, [
        f"experiments/thr_calib.py, {datetime.date.today()}, {len(recs)} captures from "
        + ", ".join(p.name for p in args.logs)[:300],
        f"raw envelope at input_pp={args.input_pp}, filterbank {args.spice_matrix}",
        "cells from board captures unless listed: "
        + ", ".join(f"ch{c}@{levels[j]:g}={src[c][j]}" for c in range(16) for j in range(len(levels))
                    if src[c][j] != "board")[:600],
        f"train with afe.input_pp={args.input_pp} afe.threshold_grid_path=<this file>",
    ])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
