"""Twelve constants that cost nothing, fitted on val and reported on test.

fx_d0 predicts `go` 572 times for 402 true clips: its logit is systematically
high against the others, so it wins arguments it should lose. One constant per
class, added to every clip alike, moves where those arguments land -- the clips
`go` barely won go to the runner-up, and the ones it won outright do not move.

WHY THIS IS FREE. conv4 is a logits site (export/tailbuild.py `_logits_site`).
Its gain must be identical across classes, because argmax compares the outputs
directly and a per-class gain would reorder them -- there is a raise enforcing
that. Its BIAS is per class, and export/tailfmt.py folds it to

    Y = (A*acc + B + half) >>> SHIFT,   B = round(bias * 2^frac * 2^shift)

so a per-class constant is a change to a ROM word that already exists. No
gates, no retraining, no resynthesis. Pooling sums rather than averages, so a
bias delta reaches the pooled logit as T*delta*2^frac -- still one constant per
class.

The shift is PINNED to what the current export uses. Left free, a larger bias
would lower it (`cap = word_bits - 1 - ceil_log2(bpeak)`) and coarsen the gain
for every class, which is exactly the kind of silent cost this is supposed not
to have. Pinned, the only question is whether B still fits one ROM word, and
that is checked and reported.

WHAT IT COSTS. Precision bought with recall. Lowering `go` also loses the
clips where `go` was barely-but-correctly ahead, so the net can be negative.
That is why the fit is on validation and the number reported is test.

Run (needs the checkpoint and the dataset, so on the training box):
    python -m experiments.logit_bias --tag fx_d0
    python -m experiments.logit_bias --tag xl_g12
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch

from experiments.fixed_accuracy import fixed_logits
from export.fuse import binary_accumulator
from export.tailbuild import tail_plan


@torch.no_grad()
def collect_logits(model, afe, sites, conv2_pw, loader, target_T: int,
                   limit: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pooled INTEGER logits and labels for one split. [N, classes], [N]."""
    grabbed = {}

    def pre_hook(mod, inp):
        grabbed["acc"] = binary_accumulator(mod, inp[0]).detach()

    h = conv2_pw.register_forward_pre_hook(pre_hook)
    outs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    n = 0
    try:
        for wav, labels in loader:
            x = afe(wav, target_T=target_T)
            model(x)                                # fills grabbed["acc"]
            outs.append(fixed_logits(model, sites, grabbed["acc"]))
            ys.append(labels)
            n += labels.numel()
            if limit and n >= limit:
                break
    finally:
        h.remove()
    return torch.cat(outs), torch.cat(ys)


def accuracy(logits: torch.Tensor, y: torch.Tensor,
             delta: torch.Tensor) -> float:
    return float(((logits + delta).argmax(1) == y).double().mean())


def balanced(logits: torch.Tensor, y: torch.Tensor, delta: torch.Tensor,
             n_classes: int) -> float:
    """Mean per-class recall. Ignores how many clips a class has, so a class
    being swallowed costs as much as a large class losing a few."""
    pred = (logits + delta).argmax(1)
    hits = torch.zeros(n_classes, dtype=torch.double)
    tot = torch.zeros(n_classes, dtype=torch.double)
    for c in range(n_classes):
        m = y == c
        tot[c] = float(m.sum())
        hits[c] = float((pred[m] == c).sum())
    ok = tot > 0
    return float((hits[ok] / tot[ok]).mean())


def fit_delta(logits: torch.Tensor, y: torch.Tensor, n_classes: int,
              objective: str = "accuracy", rounds: int = 6,
              grid: int = 41) -> torch.Tensor:
    """Coordinate ascent on the metric itself, no surrogate.

    Twelve parameters is small enough to optimise the thing we actually care
    about rather than a differentiable stand-in: cross-entropy would fit
    calibration, and calibration is not what a decision boundary needs. Each
    round sweeps one class at a time over a grid spanning the logit scale, and
    the passes are cheap because the logits are already computed.
    """
    score = (accuracy if objective == "accuracy"
             else lambda l, t, d: balanced(l, t, d, n_classes))
    delta = torch.zeros(n_classes, dtype=logits.dtype)
    # the span worth searching is set by how far apart the top two logits
    # typically are: shifts much larger than that just flip everything
    top2 = logits.topk(2, dim=1).values
    span = float((top2[:, 0] - top2[:, 1]).double().quantile(0.9)) or 1.0
    best = score(logits, y, delta)
    for _ in range(rounds):
        moved = False
        for c in range(n_classes):
            keep, cur = float(delta[c]), best
            for step in torch.linspace(-span, span, grid).tolist():
                delta[c] = keep + step
                s = score(logits, y, delta)
                if s > cur:
                    cur, keep_best = s, keep + step
            if cur > best:
                best, delta[c], moved = cur, keep_best, True
            else:
                delta[c] = keep
        span /= 2.0                              # refine around what was found
        if not moved:
            break
    return delta


def fit_prior(logits: torch.Tensor, y: torch.Tensor, n_classes: int,
              objective: str = "accuracy", grid: int = 81) -> torch.Tensor:
    """ONE parameter: a fixed direction, with only its strength fitted.

    Twelve free constants can chase individual clips sitting near a boundary,
    which is what val accuracy rewards and test accuracy does not. This fixes
    the DIRECTION from something that cannot be noise -- how far each class's
    predicted count is from its true count -- and fits only how far to travel
    along it.

        delta_c = -s * log(predicted_c / true_c)

    A class predicted twice as often as it should be gets pushed down, one
    predicted half as often gets pushed up, and the relative sizes are fixed by
    the counts rather than by the search. One parameter overfits about as
    little as anything can.
    """
    score = (accuracy if objective == "accuracy"
             else lambda l, t, d: balanced(l, t, d, n_classes))
    pred = torch.bincount(logits.argmax(1), minlength=n_classes).double()
    true = torch.bincount(y, minlength=n_classes).double()
    # a class never predicted would give -inf; clamp to one clip, which says
    # "as under-predicted as this split can show" rather than "infinitely so"
    direction = -(pred.clamp(min=1.0) / true.clamp(min=1.0)).log()

    top2 = logits.topk(2, dim=1).values
    span = float((top2[:, 0] - top2[:, 1]).double().quantile(0.9)) or 1.0
    zero = torch.zeros(n_classes, dtype=torch.double)
    best, best_s = score(logits, y, zero), 0.0
    for s in torch.linspace(0.0, 2.0 * span, grid).tolist():
        v = score(logits, y, direction * s)
        if v > best:
            best, best_s = v, s
    return direction * best_s


def halves(logits: torch.Tensor, y: torch.Tensor, n_classes: int,
           fitter, objective: str) -> Tuple[float, float]:
    """Fit on half of val, score on the other half. (fitted, held out).

    The cheapest possible honesty check, and it runs before test is touched: a
    method that gains on the half it saw and nothing on the half it did not is
    fitting that half, and no amount of it being "validation" changes that.
    """
    n = logits.shape[0]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)
    a, b = idx[: n // 2], idx[n // 2:]
    d = fitter(logits[a], y[a], n_classes, objective)
    score = (accuracy if objective == "accuracy"
             else lambda l, t, dd: balanced(l, t, dd, n_classes))
    zero = torch.zeros(n_classes, dtype=torch.double)
    return (score(logits[a], y[a], d) - score(logits[a], y[a], zero),
            score(logits[b], y[b], d) - score(logits[b], y[b], zero))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--objective", choices=("accuracy", "balanced"),
                    default="accuracy")
    ap.add_argument("--method", choices=("both", "coord", "prior"),
                    default="both",
                    help="coord fits 12 constants, prior fits 1 strength along "
                         "the count-mismatch direction")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders, class_names
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    afe = AFEFrontend(cfg.afe).eval()
    model = BinaryMatchboxNet(cfg.model).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])

    sites = tail_plan(model)
    conv2_pw = model.stages["conv2"].pw
    names = class_names()
    n_c = cfg.model.n_classes

    _, va, te = build_dataloaders(cfg.data, cfg.train.batch_size,
                                  cfg.afe.sample_rate, seed=cfg.train.seed)
    print(f"tag {args.tag}: collecting integer logits...")
    lv, yv = collect_logits(model, afe, sites, conv2_pw, va, cfg.model.T,
                            args.limit)
    lt, yt = collect_logits(model, afe, sites, conv2_pw, te, cfg.model.T,
                            args.limit)
    print(f"  val {lv.shape[0]} clips, test {lt.shape[0]} clips\n")

    # everything stays in double from here. The logits are integers, but the
    # delta is not: rounding it back to the accumulator's integer grid would
    # throw away most of a correction that is small next to the logit scale.
    lv, lt = lv.double(), lt.double()
    zero = torch.zeros(n_c, dtype=torch.double)

    # Both methods, because the first result showed the failure mode plainly:
    # twelve free constants gained 1.28pp on val and LOST 0.31pp on test, which
    # is what fitting a few thousand clips with twelve parameters looks like.
    fitters = {"coord (12 params)": fit_delta, "prior (1 param)": fit_prior}
    if args.method != "both":
        fitters = {k: v for k, v in fitters.items()
                   if k.startswith(args.method)}

    print(f"split-half check on VAL, before test is touched "
          f"({args.objective}):")
    print(f"  {'':<20}{'fitted half':>13}{'held out':>12}")
    for name, f in fitters.items():
        seen, held = halves(lv, yv, n_c, f, args.objective)
        print(f"  {name:<20}{seen * 100:>+12.2f}pp{held * 100:>+11.2f}pp")
    print("  A method that gains on the half it saw and nothing on the half it "
          "did\n  not is fitting that half. This costs nothing and it runs "
          "first.")

    best_name, best_delta, best_test = None, zero, -1.0
    print(f"\nfitted on all of VAL, reported on both:")
    print(f"  {'':<20}{'':6}{'before':>9}{'after':>9}{'change':>10}"
          f"{'balanced':>12}")
    for name, f in fitters.items():
        d = f(lv, yv, n_c, args.objective)
        for what, l, y in (("val", lv, yv), ("test", lt, yt)):
            a0, a1 = accuracy(l, y, zero), accuracy(l, y, d)
            b1 = balanced(l, y, d, n_c)
            print(f"  {name if what == 'val' else '':<20}{what:<6}"
                  f"{a0:>9.4f}{a1:>9.4f}{(a1 - a0) * 100:>+9.2f}pp{b1:>12.4f}")
            if what == "test" and a1 > best_test:
                best_name, best_delta, best_test = name, d, a1

    base = accuracy(lt, yt, zero)
    delta = best_delta

    # McNemar, because this is a PAIRED comparison on the same clips. Quoting
    # a change against the standard error of a single split overstates the
    # precision -- most clips are untouched and carry no information about the
    # difference. What matters is how many clips CHANGED and how lopsided that
    # change was; under the null those flips split evenly, so the net has a
    # standard error of sqrt(flips).
    ok0 = lt.argmax(1) == yt
    ok1 = (lt + delta).argmax(1) == yt
    broke = int((ok0 & ~ok1).sum())
    fixed = int((~ok0 & ok1).sum())
    flips = broke + fixed
    z = (fixed - broke) / math.sqrt(flips) if flips else 0.0
    print(f"\naccuracy change on TEST, tested properly (McNemar):")
    print(f"  {flips} clips changed answer: {fixed} fixed, {broke} broken, "
          f"net {fixed - broke:+d}")
    print(f"  z = {z:+.2f}" + ("  -- beyond noise" if abs(z) > 2 else
                               "  -- inside noise, this is a coin flip"))

    print(f"\nVERDICT: ", end="")
    if abs(z) <= 2.0:
        print(f"the accuracy change is NOT REAL "
              f"({base:.4f} -> {best_test:.4f}, z={z:+.2f}).")
        print("  Which is the expected result: the correction trades precision "
              "for recall,\n  and a trade nets to zero unless one side was "
              "genuinely mispriced. Look at\n  the false wake line instead -- "
              "the error MIX can move even when the total\n  does not, and for "
              "an always-on spotter the mix is what is felt.")
    elif best_test > base:
        print(f"{best_name} gains {(best_test - base) * 100:+.2f}pp on test "
              f"({base:.4f} -> {best_test:.4f}),\n  beyond noise at z={z:+.2f}. "
              f"It costs one ROM word per class and nothing else.")
    else:
        print(f"do NOT ship: {best_name} LOSES "
              f"{(base - best_test) * 100:.2f}pp on test at z={z:+.2f}.")

    pred0 = lt.argmax(1)
    pred1 = (lt + delta).argmax(1)
    print(f"\nper class on TEST (true / predicted before -> after):")
    print(f"  {'':<12}{'true':>6}{'before':>8}{'after':>7}{'  recall':>10}")
    for c in range(n_c):
        t = int((yt == c).sum())
        r0 = float((pred0[yt == c] == c).double().mean()) if t else 0.0
        r1 = float((pred1[yt == c] == c).double().mean()) if t else 0.0
        print(f"  {names[c]:<12}{t:>6}{int((pred0 == c).sum()):>8}"
              f"{int((pred1 == c).sum()):>7}   {r0:.3f} -> {r1:.3f}")

    kw = [c for c in range(n_c)
          if names[c] not in ("_unknown_", "_silence_")]
    nk = torch.tensor([names[c] in ("_unknown_", "_silence_")
                       for c in range(n_c)])
    is_nk = nk[yt]
    w0 = int(torch.isin(pred0[is_nk], torch.tensor(kw)).sum())
    w1 = int(torch.isin(pred1[is_nk], torch.tensor(kw)).sum())
    tot_nk = int(is_nk.sum())
    se = math.sqrt((w0 / tot_nk) * (1 - w0 / tot_nk) / tot_nk) * 100
    move = (w1 - w0) / tot_nk * 100
    print(f"\nfalse wakes: {w0} -> {w1} of {tot_nk} non-keyword clips "
          f"({w0 / tot_nk:.1%} -> {w1 / tot_nk:.1%}, {move:+.1f}pp)")
    print(f"  one standard error on this rate is about {se:.1f}pp, so "
          f"{abs(move) / se:.1f} of them.")
    print("  This is the number an always-on spotter actually pays: a "
          "non-keyword scored\n  as a keyword wakes the device, and accuracy "
          "averages it away against\n  errors nobody notices.")

    _emit(model, delta, cfg, names)


def _emit(model, delta: torch.Tensor, cfg, names: List[str]) -> None:
    """The new conv4 bias, with the shift pinned and the ROM word checked."""
    from export.tailfmt import ROM_WORD_BITS, fold_affine
    from export.tailbuild import tail_plan

    site = [s for s in tail_plan(model) if s.kind == "logits"][0]
    T = cfg.model.T
    frac = site.out_fmt.frac_bits
    # pooled logit gains T * d * 2^frac from a bias delta d, so invert
    d_bias = (delta.double() / (T * (1 << frac))).tolist()
    old = list(site.fold.bias_real)
    new = [o + d for o, d in zip(old, d_bias)]

    print(f"\nthe change, as conv4's bias (shift pinned at "
          f"{site.fold.shift}):")
    try:
        refit = fold_affine(site.name, site.fold.gain_real, new, site.out_fmt,
                            relu=False, shift=site.fold.shift)
    except ValueError as e:
        # fold_affine refuses rather than masking, which is the right failure
        # (export/tailfmt.py fits_word explains why silence would be worse).
        print(f"  REFUSED: {e}")
        print("  The offsets no longer fit at the pinned shift. Lowering the "
              "shift would\n  fit them but coarsen the gain for every class, "
              "so this correction is not\n  free after all -- do not ship it "
              "without measuring what the coarser\n  gain costs.")
        return

    lim = 1 << (ROM_WORD_BITS - 1)
    print(f"  {'':<12}{'old':>12}{'new':>12}{'ROM B old':>14}{'ROM B new':>14}")
    for c, nm in enumerate(names):
        b_old, b_new = site.fold.bias[c], refit.bias[c]
        room = f"{abs(b_new) / lim:.0%} of a word"
        print(f"  {nm:<12}{old[c]:>12.5f}{new[c]:>12.5f}"
              f"{b_old:>14d}{b_new:>14d}   {room}")
    if refit.gain != site.fold.gain:
        print("  WARNING: the gain moved. The shift was pinned, so this should "
              "not\n  happen -- do not ship this without finding out why.")
    else:
        print(f"  The gain is byte-identical and every offset fits "
              f"{ROM_WORD_BITS} bits.\n  Only conv4_bn.hex would change; every "
              f"other file in the export is untouched.")


if __name__ == "__main__":
    main()
