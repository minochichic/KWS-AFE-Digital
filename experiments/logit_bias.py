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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--objective", choices=("accuracy", "balanced"),
                    default="accuracy")
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
    delta = fit_delta(lv, yv, n_c, args.objective)

    rows = [("val", lv, yv), ("test", lt, yt)]
    print(f"fitted on VAL by {args.objective}, reported on both:")
    print(f"  {'':6}{'before':>9}{'after':>9}{'change':>9}"
          f"{'   balanced before/after':>26}")
    for what, l, y in rows:
        a0, a1 = accuracy(l, y, zero), accuracy(l, y, delta)
        b0 = balanced(l, y, zero, n_c)
        b1 = balanced(l, y, delta, n_c)
        print(f"  {what:<6}{a0:>9.4f}{a1:>9.4f}{(a1 - a0) * 100:>+8.2f}pp"
              f"{b0:>13.4f}{b1:>13.4f}")
    print("  A test gain well under the val gain means twelve parameters "
          "found\n  val's noise. The fit never sees test, so this comparison "
          "is honest.")

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
    print(f"\nfalse wakes: {w0} -> {w1} of {tot_nk} non-keyword clips "
          f"({w0 / tot_nk:.1%} -> {w1 / tot_nk:.1%})")

    _emit(model, delta, cfg, names)


def _emit(model, delta: torch.Tensor, cfg, names: List[str]) -> None:
    """The new conv4 bias, with the shift pinned and the ROM word checked."""
    from export.tailfmt import ROM_WORD_BITS, fits_word, fold_affine
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
    print(f"  {'':<12}{'old':>12}{'new':>12}{'ROM B old':>14}{'ROM B new':>14}")
    refit = fold_affine(site.name, site.fold.gain_real, new, site.out_fmt,
                        relu=False, shift=site.fold.shift)
    ok = True
    for c, nm in enumerate(names):
        b_old, b_new = site.fold.bias[c], refit.bias[c]
        fit = fits_word(b_new, ROM_WORD_BITS)
        ok &= fit
        print(f"  {nm:<12}{old[c]:>12.5f}{new[c]:>12.5f}"
              f"{b_old:>14d}{b_new:>14d}{'' if fit else '  <- OVERFLOWS'}")
    if refit.gain != site.fold.gain:
        print("  WARNING: the gain moved. The shift was pinned, so this should "
              "not\n  happen -- do not ship this without finding out why.")
    elif ok:
        print(f"  The gain is byte-identical and every offset fits "
              f"{ROM_WORD_BITS} bits.\n  Only conv4_bn.hex changes; every other "
              f"file in the export is untouched.")
    else:
        print("  An offset does not fit a ROM word. Lower the shift for this "
              "layer\n  (export/tailfmt.py picks min(gain want, offset cap)) "
              "and re-check the gain.")


if __name__ == "__main__":
    main()
