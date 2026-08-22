"""Where the sixteen thresholds actually sit, and what moving them would buy.

experiments/fixed_accuracy.py measures how often each channel fires and turns
that into bits. This asks the next two questions, both of which need the
envelope distribution rather than just the binary output:

  1. Did TRAINING put the threshold there, or did the initialiser?
     data/afe.py:730 sets it to the channel mean. The mean of a right-skewed
     distribution sits well above the median, and speech energy is about as
     right-skewed as it gets -- most frames are quiet. If the trained value has
     barely moved from the initial one, the fix is the initialiser, and that is
     a one-line change rather than a research problem.

  2. What would a different placement give?
     Firing rate and total bits at each quantile, measured on real envelopes.
     A retraining run costs hours; this costs a forward pass, and it says
     whether the run is worth starting.

WHAT THIS DOES NOT SAY. Bits are a CEILING, not a score. A channel firing 50%
of the time on noise carries a full bit of noise, and the network would be
better off without it. So a placement that raises the total is worth TRYING,
not worth assuming -- the only number that settles it is accuracy after a
retrain. What the ceiling does say for certain is the other direction: a
channel at 3% is nearly constant, and no training recovers a constant input.

Run (needs the checkpoint and the dataset, so on the training box):
    python -m experiments.threshold_placement --tag fx_d0
    python -m experiments.threshold_placement --tag xl_g12 --clips 2048
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def entropy_bits(p: float) -> float:
    """How much a one-bit channel firing at rate p actually carries."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def quantile_of(env_c: torch.Tensor, value: float) -> float:
    """Where `value` sits in this channel's own distribution, in [0, 1]."""
    return float((env_c <= value).double().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--clips", type=int, default=1024,
                    help="clips to measure the distribution on")
    ap.add_argument("--split", choices=("training", "testing"),
                    default="training",
                    help="training (default) answers 'where did the fit leave "
                         "them'; testing matches fixed_accuracy's firing rates")
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    afe = AFEFrontend(cfg.afe).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    load_afe_state(afe, ck["afe"])

    if afe.n_comparators != 1:
        raise SystemExit(f"k={afe.n_comparators} comparators per channel; this "
                         f"reads the k=1 layout only")

    # The TRAIN split by default, because that is what the thresholds were
    # fitted to: the question is where the initialiser and the optimiser left
    # them relative to the data they SAW. That split is augmented, so its
    # firing rates will not match fixed_accuracy's, which reads the clean test
    # split -- --split testing lines the two up when that is what is wanted.
    loaders = build_dataloaders(cfg.data, cfg.train.batch_size,
                                cfg.afe.sample_rate, seed=cfg.train.seed)
    src = loaders[0] if args.split == "training" else loaders[2]
    waves = []
    got = 0
    for wav, _ in src:
        waves.append(wav)
        got += wav.shape[0]
        if got >= args.clips:
            break
    waves = torch.cat(waves)[:args.clips]

    with torch.no_grad():
        # raw=False: the same scaling init_thresholds() sees, so the numbers are
        # in threshold coordinates and directly comparable
        env = afe._envelopes_chunked(waves, raw=False)      # [N, C, T]
    n_ch = cfg.afe.n_channels
    flat = env.permute(1, 0, 2).reshape(n_ch, -1)

    trained = afe.threshold.detach().reshape(-1)
    init = flat.mean(dim=1)                     # what data/afe.py:730 would set
    median = flat.median(dim=1).values

    aug = getattr(cfg.afe, "aug_gain_db", None)
    print(f"tag {args.tag}, {waves.shape[0]} {args.split} clips, "
          f"{flat.shape[1]:,} envelope values per channel")
    if args.split == "training" and aug:
        print(f"  (this split is augmented, aug_gain_db={aug}; firing rates "
              f"here will NOT\n   match experiments/fixed_accuracy, which "
              f"reads the clean test split)")
    print()

    print("did training move the threshold, or is this where init left it?")
    print(f"  {'ch':>2} {'init(mean)':>11} {'trained':>9} {'moved':>8} "
          f"{'median':>8} {'trained sits at':>16} {'fires':>7} {'bits':>6}")
    total_bits = 0.0
    for c in range(n_ch):
        t, i0 = float(trained[c]), float(init[c])
        q = quantile_of(flat[c], t)
        fires = 1.0 - q
        b = entropy_bits(fires)
        total_bits += b
        rel = (t - i0) / abs(i0) * 100 if i0 else float("nan")
        print(f"  {c:>2} {i0:>11.4f} {t:>9.4f} {rel:>+7.1f}% "
              f"{float(median[c]):>8.4f} {q:>15.1%} {fires:>6.1%} {b:>6.2f}")
    print(f"  total {total_bits:.2f} of {n_ch} bits per frame "
          f"({total_bits / n_ch:.0%} of capacity)")

    moved = ((trained - init).abs() / init.abs().clamp(min=1e-9)).mean() * 100
    print(f"\n  mean |move| from the initial value: {float(moved):.1f}%")
    if float(moved) < 10.0:
        print("  Training barely moved them, so the placement is the "
              "INITIALISER's.\n  data/afe.py:730 uses the channel mean; the "
              "table above shows where\n  that lands relative to the median. "
              "Changing it is a one-line fix.")
    else:
        print("  Training moved them a long way, so this placement is one the "
              "optimiser\n  worked for rather than one it was handed.")
        print("  That does NOT make it init-independent. A large move from a "
              "bad start can\n  still end in a basin the start chose; the only "
              "way to know is to begin\n  somewhere else and see where it "
              "lands. Moving is evidence about the\n  journey, not about the "
              "destination.")

    # Separating the two ways this loses information, because they have
    # different fixes. Being dark is about where the thresholds sit on average;
    # being uneven is about them disagreeing with each other. Entropy is
    # concave, so at a fixed mean firing rate the even allocation carries the
    # most -- which makes the gap a price the uneven one is paying.
    mean_rate = sum(1.0 - quantile_of(flat[c], float(trained[c]))
                    for c in range(n_ch)) / n_ch
    even = n_ch * entropy_bits(mean_rate)
    full = n_ch * entropy_bits(0.5)
    print(f"\n  where the {n_ch - total_bits:.2f} missing bits went:")
    print(f"    {full - even:>5.2f} to being dark    (mean firing "
          f"{mean_rate:.1%}, not 50%)")
    print(f"    {even - total_bits:>5.2f} to being uneven  (the same mean, "
          f"spread across channels)")
    print("  The second number is what the optimiser chose to pay: it spent "
          "sixteen free\n  parameters making channels disagree, and that costs "
          "information by itself.\n  It was optimising classification, not "
          "information, so this is a description\n  of the trade rather than a "
          "verdict on it.")

    # What another placement would give, measured rather than assumed. Each
    # channel's threshold is put at the same quantile OF ITS OWN distribution,
    # which is what a quantile initialiser would do.
    print(f"\nif every threshold were placed at quantile q of its own channel:")
    print(f"  {'q':>6} {'fires':>7} {'bits':>7}  {'vs now':>8}")
    for q in (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
        b = n_ch * entropy_bits(1.0 - q)
        print(f"  {q:>6.2f} {1 - q:>6.0%} {b:>7.2f}  {b - total_bits:>+8.2f}")
    print("  Bits are a ceiling, not a score: a channel firing on noise still "
          "counts\n  here. Treat a higher row as worth a retrain, not as a "
          "predicted gain.")

    # The channels that are provably wasted, which is the one claim the ceiling
    # supports outright.
    dark = [(c, 1.0 - quantile_of(flat[c], float(trained[c])))
            for c in range(n_ch)]
    worst = sorted(dark, key=lambda t: t[1])[:4]
    print(f"\nquietest channels: " +
          ", ".join(f"ch{c} {r:.1%}" for c, r in worst))
    print("  A channel this close to constant cannot be trained back into "
          "usefulness;\n  it is the threshold that has to move.")


if __name__ == "__main__":
    main()
