"""Is `weight_ste_clip` doing anything, and should it be?

afe.ste_clip turned out to be a constant inherited rather than chosen: 1.0 is
the BinaryNet convention for latent WEIGHTS, and it had been copied onto the
AFE comparator, where the quantity being measured lives in a completely
different range. Narrowing it was worth 9.4 points.

model.weight_ste_clip is the other field -- the one 1.0 was actually written
for. This checks whether it is right, and the question is NOT the same as the
AFE's:

  AFE      `sign_ste(env - thr, clip)`. The clip decides which SAMPLES vote on
           where the threshold goes. Too wide meant distant samples drowning
           out the ones near the boundary.

  weights  `sign_ste(w, clip)`. The clip decides which WEIGHTS still receive
           gradient. Its purpose (binary_ops.py) is to stop pushing a latent
           weight further out once it has saturated past +-clip -- so a weight
           at 3.0 is not driven to 4.0 when its sign cannot change again.

So the failure modes are opposite. Too NARROW freezes weights that should still
be free to cross zero. Too WIDE never engages at all, and the latent weights
drift outward with nothing to stop them -- which does not corrupt the forward
pass (only the sign is used) but does mean a weight has to travel further to
change its mind, so the network gets progressively harder to move.

The number that decides it is what fraction of |w| sits inside the clip, per
layer. At ~100% the clip is inert; well under 100% and it is actively freezing.

Run (needs the checkpoint, so on the training box):
    python -m experiments.weight_clip --tag fx_ste003
    python -m experiments.weight_clip --tag fx_d0 --tag fx_ste003
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch


def rows(model, clip: float) -> List[dict]:
    """Per binary layer: the latent weight spread and how much the clip binds."""
    from models.binary_ops import BinaryConv1d
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, BinaryConv1d):
            continue
        w = mod.weight.detach().reshape(-1).double()
        a = w.abs()
        out.append(dict(
            name=name,
            n=w.numel(),
            median=float(a.median()),
            p95=float(a.quantile(0.95)),
            mx=float(a.max()),
            inside=float((a <= clip).double().mean()),
            # alpha is mean|W| per output channel and is what the fold absorbs;
            # it says what scale the layer settled at regardless of the clip
            alpha=float(a.mean()),
            # weights this close to zero flip sign on almost any gradient, so
            # they are the ones still genuinely undecided
            near0=float((a <= 0.01 * float(a.mean())).double().mean()),
        ))
    return out


def show(tag: str, model, clip: float) -> None:
    rs = rows(model, clip)
    if not rs:
        raise SystemExit(f"{tag}: no BinaryConv1d layers found")
    print(f"=== {tag}   weight_ste_clip = {clip}")
    print(f"  {'layer':<16}{'weights':>9}{'med|w|':>10}{'p95|w|':>10}"
          f"{'max|w|':>10}{'inside':>9}{'alpha':>10}")
    for r in rs:
        mark = "" if r["inside"] > 0.999 else ("   <- clip binds"
                                               if r["inside"] < 0.99 else "")
        print(f"  {r['name']:<16}{r['n']:>9,}{r['median']:>10.4f}"
              f"{r['p95']:>10.4f}{r['mx']:>10.4f}{r['inside']:>8.1%}"
              f"{r['alpha']:>10.4f}{mark}")

    allw = torch.cat([torch.tensor([r["median"]]) for r in rs])
    inert = all(r["inside"] > 0.999 for r in rs)
    worst = min(rs, key=lambda r: r["inside"])
    print(f"\n  every layer inside the clip: {inert}")
    if inert:
        print(f"  The clip never engages -- max |w| over all layers is "
              f"{max(r['mx'] for r in rs):.3f}, well under {clip}.")
        print("  It is doing nothing, so lowering it is a real change and "
              "raising it is a no-op.")
        print("  Whether it SHOULD engage is the open question: with no ceiling "
              "the latent\n  weights are free to drift, and a weight far from "
              "zero needs a large gradient\n  to ever change sign again.")
    else:
        print(f"  {worst['name']} has {1 - worst['inside']:.1%} of its weights "
              f"outside, frozen by the clip.")

    # The scale the layers actually settled at, which is what says whether 1.0
    # is the right order of magnitude at all.
    med = sorted(r["median"] for r in rs)
    print(f"\n  median |w| across layers: {med[0]:.4f} .. {med[-1]:.4f}")
    ratio = clip / med[len(med) // 2]
    print(f"  the clip is {ratio:.0f}x the typical |w|.")
    if ratio > 20:
        print("  That is the same shape of mismatch afe.ste_clip had. It does "
              "not follow that\n  the fix is the same -- the two clips do "
              "different jobs -- but it is worth a run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    from train.config import load_config
    from models.binary_matchboxnet import BinaryMatchboxNet

    for i, tag in enumerate(args.tag):
        if i:
            print()
        run = Path(args.runs) / tag
        cfg = load_config(str(run / "config.yaml"))
        model = BinaryMatchboxNet(cfg.model).eval()
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        show(tag, model, float(getattr(cfg.model, "weight_ste_clip", 1.0)))


if __name__ == "__main__":
    main()
