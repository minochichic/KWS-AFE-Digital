"""How many fractional bits does the tail need?

Fourteen of the twenty-one layers end in a fused threshold, so they are integer
compares and have no format to choose. Three do not:

    conv2_pw  binary MAC -> integer acc -> BN -> relu -> REAL  (conv3's input)
    conv3     int8 weights x REAL       -> BN -> relu -> REAL  (conv4's input)
    conv4     REAL weights x REAL       -> logits -> mean over time -> argmax

Those reals become fixed point in hardware, and the width is a decision with no
analytic answer -- unlike the accumulator widths, where the bound settled it
(docs/ROADMAP.md P5-1). It has to be measured.

WHAT THE FORMAT IS. A fixed-point value is an integer with an agreed binary
point: with `f` fractional bits the integer 37 means 37/2^f. The hardware only
ever sees integers; the point exists in the interpretation. So two numbers per
site:

    integer bits  -- from the measured range. Too few and values clip.
    fraction bits -- the thing to sweep. Too few and rounding costs accuracy.

Integer bits come from runs/<tag>/ranges.json, which already recorded these
three sites as `real` for exactly this purpose. Note conv4's range is
asymmetric ([-68.1, 26.1] on xl_g12): two's complement cannot be, so the wider
side sizes it and the positive side runs slack. That is a cost, not a bug.

TWO WAYS TO ANSWER, AND THEY CHECK EACH OTHER.

  sweep_tail()   quantise and measure test accuracy, per fraction width. The
                 direct answer, and the one that counts.
  logit_margin() the top-1 minus top-2 gap. argmax only cares about ordering,
                 so a quantisation step below that gap cannot change a
                 prediction. Gives the answer for conv4 without a training run
                 and predicts where the sweep should start failing.

Run from the notebook (see lab.ipynb 8c) or:
    python -m experiments.tail_fixedpoint --tag xl_g12
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# the three sites, and which module output each one is
SITES = ("conv2", "conv3", "conv4")


def quantize(x: torch.Tensor, int_bits: int, frac_bits: int) -> torch.Tensor:
    """Round to a fixed-point grid and clamp to what the format can hold.

    `int_bits` INCLUDES the sign, so the range is [-2^(i-1), 2^(i-1) - 2^-f],
    matching two's complement. Clamping rather than wrapping: hardware would
    saturate here, and a wrap would turn a large value into a wrong-signed
    small one, which is much harder to spot in an accuracy number.
    """
    scale = float(2 ** frac_bits)
    hi = float(2 ** (int_bits - 1)) - 1.0 / scale
    lo = -float(2 ** (int_bits - 1))
    return torch.clamp(torch.round(x * scale) / scale, lo, hi)


@dataclass
class TailFormat:
    """One candidate format: integer bits per site, one shared fraction width."""

    frac_bits: int
    int_bits: Dict[str, int] = field(default_factory=dict)
    quant_conv4_weight: bool = True

    def label(self) -> str:
        ib = ",".join(f"{s}:{self.int_bits.get(s, '?')}" for s in SITES)
        return f"f={self.frac_bits} [{ib}]"


def int_bits_from_ranges(path: str | Path,
                         guard: float = 1.25) -> Dict[str, int]:
    """Integer bits per site from the measured ranges, with headroom.

    `guard` widens the range before sizing, because the ranges came from the
    test set and a slightly larger activation must saturate rather than wrap.
    """
    data = json.loads(Path(path).read_text())
    out: Dict[str, int] = {}
    for s in data["sites"]:
        # ranges.json names them by module path; conv2's real output is the
        # pointwise half, so match on the site's tail
        for name in SITES:
            if s["name"].endswith(f"{name}.conv") or \
                    s["name"].endswith(f"{name}.pw"):
                need = max(abs(s["lo"]), abs(s["hi"])) * guard
                out[name] = max(2, int(math.ceil(math.log2(need + 1))) + 1)
    return out


def install(model, fmt: TailFormat) -> List:
    """Quantise each site's output; optionally conv4's weight too."""
    handles = []

    def hook(name):
        ib = fmt.int_bits.get(name)
        if ib is None:
            return None

        def fn(mod, inp, out):
            return quantize(out, ib, fmt.frac_bits)
        return fn

    for name in SITES:
        fn = hook(name)
        if fn is not None and name in model.stages:
            handles.append(model.stages[name].register_forward_hook(fn))

    if fmt.quant_conv4_weight and "conv4" in model.stages:
        conv = model.stages["conv4"].conv
        w = conv.weight.detach().clone()
        ib = max(2, int(math.ceil(math.log2(
            float(w.abs().max()) + 1))) + 1)
        with torch.no_grad():
            conv.weight.copy_(quantize(w, ib, fmt.frac_bits))
        handles.append(_WeightRestore(conv, w))
    return handles


class _WeightRestore:
    """Undo a weight edit when the sweep moves to the next format."""

    def __init__(self, conv, original):
        self.conv, self.original = conv, original

    def remove(self):
        with torch.no_grad():
            self.conv.weight.copy_(self.original)


@torch.no_grad()
def accuracy(afe, model, loader, T, device="cpu") -> float:
    ok = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += (model(afe(x, target_T=T)).argmax(1) == y).sum().item()
        n += y.numel()
    return ok / n


def sweep_tail(afe, model, loader, T, ranges_path: str | Path,
               fracs: Tuple[int, ...] = (12, 10, 8, 6, 5, 4, 3, 2),
               device: str = "cpu") -> List[Tuple[int, float]]:
    """Accuracy against fraction width, plus the float baseline."""
    ib = int_bits_from_ranges(ranges_path)
    base = accuracy(afe, model, loader, T, device)
    print(f"{'frac':>5}{'step':>10}{'test':>9}{'Δ vs float':>12}")
    print(f"{'float':>5}{'-':>10}{base:>9.4f}{'':>12}")
    rows = []
    for f in fracs:
        fmt = TailFormat(frac_bits=f, int_bits=ib)
        hs = install(model, fmt)
        try:
            a = accuracy(afe, model, loader, T, device)
        finally:
            for h in hs:
                h.remove()
        rows.append((f, a))
        print(f"{f:>5}{1.0 / 2 ** f:>10.4f}{a:>9.4f}"
              f"{(a - base) * 100:>+11.1f}pp")
    print(f"\n정수부 (실측 범위 + guard): "
          + "  ".join(f"{k}={v}b" for k, v in sorted(ib.items())))
    return rows


@torch.no_grad()
def logit_margin(afe, model, loader, T, device="cpu",
                 qs=(0.001, 0.01, 0.05, 0.25, 0.5)) -> Dict[float, float]:
    """Top-1 minus top-2 gap. argmax cannot flip while the step stays below it.

    Reported as low quantiles, since what matters is the CLOSEST calls -- the
    median gap says nothing about how many predictions are one rounding away
    from changing.
    """
    gaps = []
    for x, _ in loader:
        lg = model(afe(x.to(device), target_T=T))
        top2 = lg.topk(2, dim=1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
    g = torch.cat(gaps)
    out = {q: float(g.quantile(q)) for q in qs}
    print(f"logit 격차 (1등 − 2등), n={g.numel()}")
    print(f"{'분위':>7}{'격차':>10}{'이보다 크려면 frac':>20}")
    for q, v in out.items():
        need = max(0, math.ceil(-math.log2(v))) if v > 0 else 99
        print(f"{q:>7.3f}{v:>10.4f}{need:>18}b")
    print("\n해석: 그 분위의 표본은 양자화 간격이 격차보다 작아야 순서가 안 뒤집힌다.")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    afe = AFEFrontend(cfg.afe).to(dev).eval()
    model = BinaryMatchboxNet(cfg.model).to(dev).eval()
    ck = torch.load(run / "best.pt", map_location=dev, weights_only=True)
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])
    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]

    logit_margin(afe, model, te, cfg.model.T, dev)
    print()
    sweep_tail(afe, model, te, cfg.model.T, run / "ranges.json", device=dev)


if __name__ == "__main__":
    main()
