"""Pure offset cost -- what a sliding window ACTUALLY does to accuracy.

The shift curve we have been reading is not measuring what we thought.  Both
`augment._time_shift` and the notebook's `accuracy(shift_ms=)` zero-pad: they
push part of the clip out of the 1 s buffer and drop it.  So at +-300 ms the
word is not merely misplaced, it is CUT, and 30% of the clip is replaced by
digital silence.  Two artifacts follow, neither of which exists on the board:

  * truncation -- no amount of training fixes missing samples, which is why
    +-300 ms augmentation bought +0.33pp of curve area (essentially nothing).
  * artificial silence -- a zero region collapses the cross-channel max toward
    the floor, the degenerate case in EXPERIMENTS.md 4-4.  Real rooms hiss.

A sliding window over a continuous stream does neither.  Some window always
contains the whole word; only its POSITION inside the window changes, and the
rest of the window holds room noise.  This script measures that, and only that:
find the word, move it without cutting it, fill the remainder with real
background noise at the clip's own noise floor.

The gap between this curve and the shift curve is the part of the "+-11pp at
+-300 ms" scare that was measurement artifact.  What survives here is the real
requirement the FPGA decision rule has to meet.

Usage:
    python experiments/window_offset.py [tag] [--fill noise|zero|both]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.afe import AFEFrontend                                     # noqa: E402
from data.speech_commands import build_dataloaders                   # noqa: E402
from models.binary_matchboxnet import BinaryMatchboxNet              # noqa: E402
from train.config import load_config                                 # noqa: E402

SR = 16000
FRAME = 160                     # 10 ms, matches the envelope grid
ACTIVE_FRAC = 0.10              # frame is "word" if rms > this * peak rms


# --------------------------------------------------------------------------- #
def word_span(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First and last active sample index per clip. x: [B, L] -> ([B], [B]).

    Energy gate on 10 ms frames.  Deliberately the same grid the AFE bins on,
    so a span boundary can never land mid-bin and smear one frame of the word
    into the fill.
    """
    b, length = x.shape
    n = length // FRAME
    rms = x[:, : n * FRAME].reshape(b, n, FRAME).pow(2).mean(-1).sqrt()
    gate = rms > rms.amax(1, keepdim=True) * ACTIVE_FRAC
    idx = torch.arange(n, device=x.device).expand(b, n)
    first = torch.where(gate, idx, torch.full_like(idx, n)).amin(1)
    last = torch.where(gate, idx, torch.full_like(idx, -1)).amax(1)
    return first * FRAME, (last + 1) * FRAME


def noise_floor_rms(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
                    ) -> torch.Tensor:
    """RMS of each clip OUTSIDE its word -- the room it was recorded in."""
    idx = torch.arange(x.shape[1], device=x.device).expand_as(x)
    out = (idx < a[:, None]) | (idx >= b[:, None])
    n = out.sum(1).clamp_min(1)
    return ((x * out).pow(2).sum(1) / n).sqrt()


def reposition(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               delta: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """Move each clip's word by `delta[i]` samples inside the same 1 s window.

    Nothing is dropped: callers pick `delta` so the shifted span still fits.
    The vacated region is `fill`, not zero.
    """
    length = x.shape[1]
    idx = torch.arange(length, device=x.device).expand_as(x)
    d = delta[:, None]
    word = torch.gather(x, 1, (idx - d).clamp(0, length - 1))
    inside = (idx >= a[:, None] + d) & (idx < b[:, None] + d)
    return torch.where(inside, word, fill)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tag", nargs="?", default="af_k1_ref")
    p.add_argument("--root", default=None, help="dataset root, if not the one "
                   "recorded in the run's config")
    p.add_argument("--fill", choices=["noise", "zero", "both"], default="both")
    p.add_argument("--steps", type=int, default=9, help="positions across the "
                   "window, 0 = word flush left, 1 = flush right")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # The run's OWN config, not configs/base.yaml.  base.yaml still says
    # normalize=minmax; the frozen front end lives in the notebook's BASE
    # overrides, and reading the wrong one evaluates a real checkpoint through
    # the wrong normalization without ever erroring.
    saved = f"runs/{args.tag}/config.yaml"
    if not os.path.isfile(saved):
        sys.exit(f"{saved} 없음 -- 그 태그로 학습한 적이 있는지 확인")
    cfg = load_config(saved)
    if args.root:
        cfg.data.root = args.root
    print(f"프론트엔드: {cfg.afe.filterbank_source}/{cfg.afe.compression}/"
          f"{cfg.afe.normalize} floor={cfg.afe.xmax_floor_frac} "
          f"k={cfg.afe.comparators_per_channel}")
    afe = AFEFrontend(cfg.afe).to(dev).eval()
    model = BinaryMatchboxNet(cfg.model).to(dev).eval()
    ck = torch.load(f"runs/{args.tag}/best.pt", map_location=dev,
                    weights_only=True)
    model.load_state_dict(ck["model"])
    afe.load_state_dict(ck["afe"])

    _, _, te = build_dataloaders(cfg.data, cfg.train.batch_size, SR,
                                 seed=cfg.train.seed)

    # Position is normalized, not absolute: p = 0 puts the word flush against
    # the start of the window, p = 1 flush against the end.  That is exactly
    # the trajectory a word traces as the window slides past it, and unlike an
    # absolute +-ms sweep it is feasible for EVERY clip regardless of word
    # length -- so the population never changes with position and short words
    # are not silently over-represented.
    ps = [i / (args.steps - 1) for i in range(args.steps)]
    fills = ["noise", "zero"] if args.fill == "both" else [args.fill]
    hit = {(f, i): 0 for f in fills for i in range(args.steps)}
    kept = total = 0
    travel = 0.0

    g = torch.Generator(device="cpu").manual_seed(0)
    for x, y in te:
        x, y = x.to(dev), y.to(dev)
        total += y.numel()
        a, b = word_span(x)
        ok = b > a
        if not ok.any():
            continue
        x, y, a, b = x[ok], y[ok], a[ok], b[ok]
        kept += y.numel()

        # How far this clip's word can travel at all.  Stationary silence fills
        # the window, so its range is ~0 and it stays put at every p -- which
        # is the honest answer: sliding does not change a steady noise clip.
        span = (x.shape[1] - (b - a)).clamp_min(0)
        travel += float(span.sum()) / SR * 1000.0

        floor = noise_floor_rms(x, a, b)
        for f in fills:
            if f == "zero":
                bed = torch.zeros_like(x)
            else:
                # White-ish bed at the clip's own measured floor.  The point is
                # only that the window is not digitally dead; matching spectrum
                # would need a noise bank and does not change the comparison.
                bed = torch.randn(x.shape, generator=g).to(dev)
                bed = bed * (floor / bed.std(1).clamp_min(1e-9))[:, None]
            for i, p in enumerate(ps):
                delta = (span.float() * p).long() - a
                xd = reposition(x, a, b, delta, bed)
                pred = model(afe(xd, target_T=cfg.model.T)).argmax(1)
                hit[(f, i)] += (pred == y).sum().item()

    if not kept:
        sys.exit("no clip had a detectable word span")

    print(f"{args.tag}   {kept}/{total} 클립, 평균 이동 가능 폭 "
          f"{travel / kept:.0f} ms  (창 안에서 단어가 움직일 수 있는 거리)\n")
    print(f"{'위치':>7}{''.join(f'{f:>12}' for f in fills)}   기준=중앙")
    mid = args.steps // 2
    base = {f: hit[(f, mid)] / kept for f in fills}
    for i, p in enumerate(ps):
        acc = "".join(f"{hit[(f, i)] / kept:>12.4f}" for f in fills)
        dlt = "".join(f"{(hit[(f, i)] / kept - base[f]) * 100:>+9.1f}pp"
                      for f in fills)
        print(f"{p:>7.2f}{acc}  {dlt}")
    for f in fills:
        col = [hit[(f, i)] / kept for i in range(args.steps)]
        print(f"\n{f:>6}: 평균 {sum(col) / len(col):.4f}  "
              f"최저 {min(col):.4f}  최고 {max(col):.4f}  "
              f"낙폭 {(max(col) - min(col)) * 100:.1f}pp")


if __name__ == "__main__":
    main()
