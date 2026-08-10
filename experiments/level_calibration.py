"""Does Speech Commands span the loudness range the board will actually meet?

`xlse` has one number in it, lse_temp_frac, and it is not a design choice: the
diode's blur is n*V_T, pinned near 26-39 mV by physics, so

    frac = n*V_T / envelope swing = 1 / loudness

The pre-amp is already at its x10 ceiling (ANALOG.md 6-3, headroom and GBW give
the same limit), so nothing we build can narrow the range -- 60-85 dB SPL is a
25 dB span, an 18x swing in envelope, and frac travels 0.55 down to 0.03 with it.

Training therefore has to see that range, and the question is whether it already
does. It might: Speech Commands was recorded on whatever phone each volunteer
had, at whatever distance, so its clips are NOT level-normalised (verified --
the loader pads and stacks raw waveforms, nothing scales them). Whether that
accidental spread matches 25 dB is what this measures.

Two distributions, because they answer different halves:

  * per FRAME -- what the circuit literally experiences. lse_temp is frozen, so
    every frame gets its own effective frac, including quiet frames inside a
    loud clip.
  * per CLIP -- what an SPL change, or a gain augmentation, would move. This is
    the one that compares to the 25 dB deployment span.

Usage:
    python experiments/level_calibration.py [tag] [--spl 74] [--split test]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.afe import AFEFrontend                                     # noqa: E402
from data.speech_commands import (SILENCE_INDEX,                     # noqa: E402
                                  build_dataloaders)
from train.config import load_config                                 # noqa: E402

SR = 16000
VT = 25.852                       # kT/q at 300 K, mV

# ANALOG.md 6-2, pre-amp x10. "optimistic" assumes the whole band lands in one
# channel; the note there says a formant channel really gets about a third.
SPL_MV = {60: 47.0, 65: 84.0, 70: 150.0, 74: 237.0, 85: 840.0}
FORMANT_SHARE = 1 / 3


def pct(v: torch.Tensor, q: float) -> float:
    return float(v.quantile(q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tag", nargs="?", default="af_k1_ref")
    p.add_argument("--root", default=None)
    p.add_argument("--spl", type=float, default=74.0,
                   help="SPL the MEDIAN Speech Commands clip is assumed to be")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-batches", type=int, default=0, help="0 = all")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    saved = f"runs/{args.tag}/config.yaml"
    if not os.path.isfile(saved):
        sys.exit(f"{saved} 없음")
    cfg = load_config(saved)
    if args.root:
        cfg.data.root = args.root
    afe = AFEFrontend(cfg.afe).to(dev).eval()
    ck = torch.load(f"runs/{args.tag}/best.pt", map_location=dev,
                    weights_only=True)
    afe.load_state_dict(ck["afe"])

    tr, _, te = build_dataloaders(cfg.data, cfg.train.batch_size, SR,
                                  seed=cfg.train.seed)
    loader = tr if args.split == "train" else te

    frame_pk, clip_pk, lab = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            env = afe.envelopes(x.to(dev), raw=True)      # [B, C, T], pre-norm
            fp = env.amax(dim=1)                          # [B, T] the OR node
            frame_pk.append(fp.flatten().cpu())
            # A clip's loudness is what it looks like WHILE the word is being
            # said, not averaged with its own silence -- half of a 1 s clip is
            # typically not speech, so a median would measure the room.
            clip_pk.append(fp.quantile(0.90, dim=1).cpu())
            lab.append(y.cpu())
    frame_pk = torch.cat(frame_pk)
    clip_pk = torch.cat(clip_pk)
    lab = torch.cat(lab)
    # Drop the silence class before measuring spread. Those clips are background
    # crops from full-scale recordings and run ~3x LOUDER than the speech
    # (EXPERIMENTS.md 6), so leaving them in would widen the range with the one
    # thing the comparison is not about -- how loud people speak.
    speech = lab != SILENCE_INDEX
    clip_pk = clip_pk[speech]

    # sqrt(mel + 1e-6) bottoms out at exactly 1e-3, so silence and zero padding
    # do not reach 0 -- they pile up ON the guard. Left in, they would dominate
    # the low quantiles and report a spread that is numerical, not acoustic.
    # This is the same guard that broke xmax_floor (EXPERIMENTS.md 4-4).
    guard = 1e-3 if cfg.afe.compression == "sqrt" else 0.0
    active = frame_pk > guard * 2.0
    quiet_share = 1.0 - float(active.float().mean())
    frame_act = frame_pk[active]

    # init_fixed_scale() pools EVERY frame -- silence included -- and takes the
    # median, so the anchor the frozen temperature is built on is itself pulled
    # down by however much of the dataset is not speech.
    TYP = float(frame_pk.median())
    TYP_ACT = float(frame_act.median())
    nominal = float(cfg.afe.lse_temp_frac)

    print(f"{args.tag}  [{args.split}]  클립 {clip_pk.numel()}개, "
          f"프레임 {frame_pk.numel()}개")
    print(f"프론트엔드 {cfg.afe.filterbank_source}/{cfg.afe.compression}, "
          f"lse_temp_frac(설정) = {nominal}\n")

    print(f"0) 앵커 점검 -- init_fixed_scale는 무음 프레임까지 섞어 중앙값을 낸다")
    print(f"   sqrt 가드(1e-3) 근처 프레임 : {quiet_share:>6.1%}")
    print(f"   TYP (전체 프레임 중앙값)     : {TYP:.5f}   ← 실제 쓰이는 앵커")
    print(f"   TYP (말하는 프레임만)        : {TYP_ACT:.5f}   "
          f"({TYP_ACT / max(TYP, 1e-12):.2f}x)")
    if TYP_ACT > 1.5 * TYP:
        print("   ⚠️ 앵커가 무음 쪽으로 끌려갔다 → lse_temp가 실제보다 작다"
              " (= 실제보다 딱딱한 max로 학습)")

    # --- 1. spread ---------------------------------------------------------- #
    print("\n1) GSC가 실제로 얼마나 넓은가  (앵커 없이도 답이 나오는 부분)")
    print(f"{'':>16}{'p5':>10}{'p25':>10}{'중앙':>10}{'p75':>10}{'p95':>10}"
          f"{'p95/p5':>9}{'dB폭':>8}")
    for name, v in (("프레임(말)", frame_act), ("클립 레벨", clip_pk)):
        qs = [pct(v, q) for q in (.05, .25, .5, .75, .95)]
        ratio = qs[4] / max(qs[0], 1e-12)
        print(f"{name:>14}" + "".join(f"{x / TYP:>10.2f}" for x in qs)
              + f"{ratio:>9.1f}x{20 * math.log10(ratio):>7.1f}")
    print("   (값은 앵커 TYP 대비 배율. 배율이 곧 frac 배율의 역수다.)")

    span_clip = pct(clip_pk, .95) / max(pct(clip_pk, .05), 1e-12)
    need = SPL_MV[85] / SPL_MV[60]
    print(f"\n   GSC 클립 레벨 폭 : {20 * math.log10(span_clip):>5.1f} dB "
          f"({span_clip:.1f}x)")
    print(f"   배치 SPL 폭      : {20 * math.log10(need):>5.1f} dB "
          f"({need:.1f}x)   60~85 dB SPL")
    verdict = ("충분 -- 학습이 이미 배치 범위를 덮는다"
               if span_clip >= need else
               "부족 -- 게인 증강으로 메워야 하는 폭이 있다")
    print(f"   → {verdict}")
    if span_clip < need:
        extra = need / span_clip
        print(f"     모자란 폭 {20 * math.log10(extra):.1f} dB "
              f"→ aug_gain_db 약 [-{10 * math.log10(extra):.0f}, "
              f"+{10 * math.log10(extra):.0f}]")

    # --- 2. the frac the circuit actually sees ------------------------------ #
    print("\n2) 프레임마다 회로가 실제로 겪는 frac  "
          f"(lse_temp는 동결이므로 frac = {nominal} x TYP/프레임피크)")
    print(f"{'':>16}{'p5':>10}{'p25':>10}{'중앙':>10}{'p75':>10}{'p95':>10}")
    qs = [nominal * TYP / max(pct(frame_act, q), 1e-12)
          for q in (.95, .75, .5, .25, .05)]      # reversed: big peak = small frac
    print(f"{'frac':>14}" + "".join(f"{x:>10.2f}" for x in qs))

    # --- 3. the anchor ------------------------------------------------------ #
    print(f"\n3) 앵커 -- \"중앙값 GSC 클립 = {args.spl:.0f} dB SPL\"이라고 두면")
    mv = SPL_MV.get(args.spl)
    if mv is None:
        print(f"   {args.spl} dB는 표에 없음 (ANALOG.md 6-2: "
              f"{sorted(SPL_MV)})")
        return
    for label, swing in (("낙관 (전대역이 한 채널)", mv),
                         ("현실 (포먼트 채널 1/3)", mv * FORMANT_SHARE)):
        for n_id, nlab in ((1.0, "n=1.0"), (1.5, "n=1.5")):
            f0 = n_id * VT / swing
            print(f"   {label:<22} {nlab}  스윙 {swing:>5.0f} mV  "
                  f"→ frac(중앙) {f0:>5.2f}   "
                  f"설정값 {nominal} 대비 {f0 / nominal:>4.1f}x")
    print("\n   설정값이 이 표보다 작으면, 실제보다 딱딱한 max로 학습한 것이다"
          "\n   (= 회로가 더 소프트해서 분모가 더 부푼다).")


if __name__ == "__main__":
    main()
