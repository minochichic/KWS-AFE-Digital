"""What the HARDWARE scores on the full test set, not what the float model does.

Every accuracy figure in this project so far -- 0.8445 for xl_g12 -- comes from
the float model. The thing that will run on the board is not that model: its
tail is fixed point, and export/golden.py has only ever checked the two clips it
dumps. Two clips is not an accuracy measurement.

WHAT IS AND IS NOT APPROXIMATED. The binary layers are exact. `sign(BN(acc))`
becomes an integer compare with no rounding at all (export/fuse.py), so those
fifteen layers give bit-identical answers. Only the tail quantises: conv2_pw,
conv3 and conv4 land on a 1/64 grid. So the gap this measures is entirely the
tail's, which is why it should be small -- and "should be" is the reason to
measure it.

Reports three numbers:

  float     the model as trained
  fixed     the integer path, which is what the RTL computes
  agree     how often the two pick the same class

`agree` matters separately from accuracy: the two can differ on a clip and both
be right or both be wrong, so a drop in accuracy and a drop in agreement are
different findings.

Run (needs the checkpoint and the dataset, so on the training box):
    python -m experiments.fixed_accuracy --tag xl_g12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export.tailbuild import apply_site, int8_weights, tail_plan
from export.fuse import binary_accumulator
from models.binary_matchboxnet import BinaryMatchboxNet
from models.quant_ops import QuantConv1d


@torch.no_grad()
def fixed_logits(model: BinaryMatchboxNet, sites, acc2: torch.Tensor
                 ) -> torch.Tensor:
    """Chain the tail in integers from conv2_pw's accumulator. [N, classes].

    The same walk as export/golden.py: each dense conv is fed the QUANTISED
    output of the layer before it, not the float model's, because that is what
    the register holds. Feeding it the float activations produces something that
    looks right to within a percent and is not what the hardware does.
    """
    cur = acc2.round().to(torch.int64)
    for s in sites:
        if s.in_fmt is not None:
            w = (int8_weights(s.conv)[0] if isinstance(s.conv, QuantConv1d)
                 else s.weights)
            cur = torch.round(torch.nn.functional.conv1d(
                cur.double(), w.double(), None, s.conv.stride, s.conv.padding,
                s.conv.dilation, s.conv.groups)).to(torch.int64)
        cur = apply_site(s, cur)
    # the pool never divides: the same positive factor on every class, and
    # argmax ignores it
    return cur.sum(dim=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--vos", type=float, default=None,
                    help="평가 시 비교기 오프셋(정규화 단위). 체크포인트가 학습된 "
                         "값과 무관하게 덮어쓴다. 3 mV / K=100 mV -> 0.03")
    ap.add_argument("--vos-fixed", action="store_true",
                    help="오프셋을 채널당 하나로 고정(실물에 충실). 기본값은 "
                         "data/afe.py 처럼 클립마다 새로 뽑는 것 = 훨씬 가혹하다")
    ap.add_argument("--vos-seed", type=int, default=0,
                    help="--vos-fixed 의 draw. 보드 한 장 = seed 하나")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many clips (0 = the whole split)")
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    if args.vos is not None:
        cfg.afe.comparator_vos = args.vos
    afe = AFEFrontend(cfg.afe).eval()
    if args.vos is not None and args.vos_fixed and args.vos > 0.0:
        # 실물 비교기의 Vos 는 소자별 고정 DC 오차다. data/afe.py 는 클립마다
        # 다시 뽑는데(학습 강건성용) 그건 실제보다 가혹하다 -- 캘리브레이션으로
        # 없앨 수 있는 것과 없는 것이 그 차이에 걸려 있다. 여기서는 채널당 하나를
        # 뽑아 임계값에 그냥 더해두고, forward 의 랜덤 주입은 끈다.
        g = torch.Generator().manual_seed(args.vos_seed)
        draw = args.vos * torch.randn(afe.threshold.numel(), generator=g)
        with torch.no_grad():
            afe.threshold += draw.to(afe.threshold.dtype)
        afe.cfg.comparator_vos = 0.0
        print(f"오프셋 고정 draw (seed {args.vos_seed}), 정규화 단위:")
        print("  " + " ".join(f"{v:+.4f}" for v in draw.tolist()))
    model = BinaryMatchboxNet(cfg.model).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])

    sites = tail_plan(model)
    conv2_pw = model.stages["conv2"].pw

    grabbed = {}

    def pre_hook(mod, inp):
        grabbed["acc"] = binary_accumulator(mod, inp[0]).detach()

    h = conv2_pw.register_forward_pre_hook(pre_hook)

    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]

    n = fl_ok = fx_ok = same = 0
    confusion = torch.zeros(cfg.model.n_classes, cfg.model.n_classes,
                            dtype=torch.int64)
    # How often each channel is +1 across the whole split. A channel stuck at
    # one value carries no information no matter how well the network is
    # trained, and an absolute threshold has no way to notice: speech energy
    # falls with frequency, so one fixed threshold per channel sits at a
    # different place on each channel's dynamic range.
    fire = torch.zeros(cfg.afe.n_channels, dtype=torch.int64)
    fire_n = 0
    try:
        for wav, labels in te:
            x = afe(wav, target_T=cfg.model.T)
            fire += (x > 0).sum(dim=(0, 2))
            fire_n += x.shape[0] * x.shape[2]
            float_logits = model(x)                 # the hook fires in here
            fx = fixed_logits(model, sites, grabbed["acc"])

            fl_pred = float_logits.argmax(1)
            fx_pred = fx.argmax(1)
            n += labels.numel()
            fl_ok += int((fl_pred == labels).sum())
            fx_ok += int((fx_pred == labels).sum())
            same += int((fl_pred == fx_pred).sum())
            for t, p in zip(labels.tolist(), fx_pred.tolist()):
                confusion[t, p] += 1
            if args.limit and n >= args.limit:
                break
    finally:
        h.remove()

    print(f"tag {args.tag}, {n} clips\n")
    print(f"{'':10}{'accuracy':>10}{'':4}{'note'}")
    print(f"{'float':<10}{fl_ok / n:>10.4f}    모델 그대로")
    print(f"{'fixed':<10}{fx_ok / n:>10.4f}    RTL 이 계산하는 것")
    print(f"{'agree':<10}{same / n:>10.4f}    두 경로가 같은 클래스를 고른 비율")
    print(f"\ngap {(fl_ok - fx_ok) / n * +100:+.2f} pp "
          f"({fl_ok - fx_ok:+d} clips)")
    print("\nThe binary layers are exact, so this gap is entirely the tail's "
          "fixed point.")

    # The reading lives in experiments/confusion.py on plain lists, so it can be
    # unit tested without a checkpoint (tests/test_confusion.py). A diagnostic
    # that is quietly wrong points the next month of work at the wrong problem.
    from data.speech_commands import class_names
    from experiments.confusion import class_stats, report

    names = class_names()
    print()
    print(report(confusion.tolist(), names))

    # WHAT THE NETWORK SAYS WHEN THE INPUT SAYS NOTHING.
    #
    # An all -1 image is not a hypothetical. An absolute-threshold front end
    # produces exactly this for a clip too quiet to light any channel, which
    # docs/ROADMAP.md names as track 2's failure mode -- at -18 dB fx_* falls to
    # 0.248 because the image goes uniformly -1.
    #
    # The answer is a fixed property of the weights, not of any clip: with no
    # information to go on the network still has to pick, and whichever class it
    # picks collects every uninformative clip. That is what turns one class into
    # a sink, so the class named here should be the same one the block above
    # flags as absorbing. If they match, the fix is where the thresholds sit --
    # not the architecture, and not the tail.
    with torch.no_grad():
        quiet = -torch.ones(1, cfg.afe.n_channels, cfg.model.T)
        h2 = conv2_pw.register_forward_pre_hook(pre_hook)
        try:
            fl_q = int(model(quiet).argmax(1)[0])
            fx_q = int(fixed_logits(model, sites, grabbed["acc"]).argmax(1)[0])
        finally:
            h2.remove()

    # WHAT THE INPUT ACTUALLY CARRIES, BEFORE ASKING ANYTHING OF THE NETWORK.
    #
    # Each channel is one bit per frame, so a channel that is +1 (or -1) almost
    # always spends its bit saying nothing. Entropy puts a number on it: a
    # channel firing half the time carries a full bit, one firing 1% of the
    # time carries 0.08. Summed over sixteen channels this is the ceiling on
    # what any network downstream could possibly separate.
    #
    # This is where an absolute threshold is structurally exposed. Speech energy
    # falls with frequency, so a per-channel fixed threshold lands at a
    # different point on each channel's range -- the loud low bands saturate and
    # the quiet high bands go dark, both at the same time. A relative threshold
    # divides that tilt out.
    import math
    rate = (fire.double() / max(1, fire_n)).tolist()
    bits = [0.0 if r <= 0 or r >= 1 else
            -(r * math.log2(r) + (1 - r) * math.log2(1 - r)) for r in rate]
    print(f"\nwhat each channel actually carries (over all "
          f"{fire_n // cfg.model.T} clips; ch0 is the lowest band):")
    print(f"  {'ch':>2} {'fires':>7} {'bits':>6}")
    for c, (r, b) in enumerate(zip(rate, bits)):
        mark = ("  <- saturated" if r > 0.9 else
                "  <- dark" if r < 0.05 else "")
        bar = "#" * int(round(b * 20))
        print(f"  {c:>2} {r:>6.1%} {b:>6.2f}  {bar}{mark}")
    print(f"  total {sum(bits):.2f} of {cfg.afe.n_channels} bits per frame "
          f"({sum(bits) / cfg.afe.n_channels:.0%} of the input's capacity)")
    stuck = [c for c, r in enumerate(rate) if r > 0.9 or r < 0.05]
    if stuck:
        print(f"  channels {stuck} spend their bit saying nothing. No amount of "
              f"training\n  recovers a channel that is constant -- the "
              f"threshold is in the wrong place.")

    print(f"\nwith nothing in the input at all (every channel -1, every frame):")
    print(f"  float says {names[fl_q]}, the integer path says {names[fx_q]}")
    sinks = [s for s in class_stats(confusion.tolist(), names) if s.absorbing]
    if not sinks:
        print("  No class is absorbing, so this fallback is not costing "
              "anything yet.")
    else:
        worst = min(sinks, key=lambda s: s.precision)
        if worst.idx == fx_q:
            print(f"  {worst.name} is also the class absorbing its neighbours "
                  f"(recall {worst.recall:.3f},\n  precision "
                  f"{worst.precision:.3f}). Same class, so the quiet clips are "
                  f"landing there:\n  a threshold placement problem, not an "
                  f"architecture one.")
        else:
            print(f"  The absorbing class is {worst.name}, not {names[fx_q]}, "
                  f"so the sink is not\n  simply where silence goes -- look at "
                  f"the pairs above instead.")


if __name__ == "__main__":
    main()
