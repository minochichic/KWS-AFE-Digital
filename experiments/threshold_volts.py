"""Per-channel thresholds from a checkpoint, and what they would be in volts.

Two things live under the word "threshold" and they are not the same number.

  LEARNED    afe.threshold, sixteen values in the normalised envelope domain:
             (log_mel - lo) / (hi - lo), with lo and hi dataset-level constants
             SHARED across channels. This script reads them exactly.

  SHIPPED    analog/AFE_tuning/artifacts/r7r8_real.md, sixteen R7/R8 dividers
             built from Vthr = Venv_DC + f * swing, where swing is measured per
             channel and f is a SINGLE hand-picked 0.380 (swing_real.py:43).

They are currently disconnected. The resistor values on the board do not come
from any trained model, and the trained thresholds have never been converted to
volts. Joining them is ROADMAP P4-b, still open, and this script exists to make
the gap visible rather than to close it.

WHY IT IS NOT A SUBSTITUTION. Setting f := thr looks obvious and is not
justified, for two reasons that both have to be resolved by measurement:

  compression   ours is log_mel; the detector's V+ is an amplitude. A fraction
                of a log range is not a fraction of a linear swing.
  scope         our lo/hi are one pair shared by all sixteen channels; the
                swings are per channel and span 28-65 mV.

So the volts column below is printed under a NAMED assumption and labelled as
such. Read it for shape -- which channels sit high, which sit low, how much the
two runs disagree -- not as a number to give the analog side.

Run (needs the checkpoints, so on the training box):
    python -m experiments.threshold_volts --tag fx_ste001 --tag fx_ste0003
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import torch

SUPPLY_V = 1.8          # swing_per_channel.py SUPPLY
TOTAL_K = 1000.0        # nominal R7+R8, kOhm
SHIPPED_F = 0.380       # swing_real.py F_TRIP -- one constant for all channels


def measured_swing(root: Path) -> List[dict]:
    """Parse the measured per-channel table out of r7r8_real.md.

    Parsed rather than copied so the numbers cannot drift from the analog
    side's own record. Columns: ch, f_c, opamp, swing, margin, Vthr, R7, R8.
    """
    p = root / "analog/AFE_tuning/artifacts/r7r8_real.md"
    if not p.is_file():
        raise SystemExit(f"{p} 가 없다 -- 아날로그 실측 표를 찾을 수 없다.")
    rows = []
    for line in p.read_text().splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) != 9 or not re.fullmatch(r"\d+", cells[0]):
            continue
        rows.append(dict(ch=int(cells[0]), f_c=float(cells[1]), opamp=cells[2],
                         swing_mv=float(cells[3]), vthr_mv=float(cells[5]),
                         r7=float(cells[6]), r8=float(cells[7])))
    if len(rows) != 16:
        raise SystemExit(f"{p} 에서 16행이 아니라 {len(rows)}행을 읽었다.")
    return sorted(rows, key=lambda r: r["ch"])


def thresholds(runs: str, tag: str) -> torch.Tensor:
    ck = torch.load(Path(runs) / tag / "best.pt", map_location="cpu",
                    weights_only=True)
    thr = ck["afe"]["threshold"].reshape(-1)
    if thr.numel() != 16:
        raise SystemExit(f"{tag}: 임계값이 16개가 아니라 {thr.numel()}개다 "
                         f"(comparators_per_channel != 1?)")
    return thr.double()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True,
                    help="repeat to compare runs")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    swing = measured_swing(root)
    thrs: Dict[str, torch.Tensor] = {t: thresholds(args.runs, t) for t in args.tag}

    # --- what the checkpoints actually hold --------------------------------
    print("학습된 임계값 (정규화 도메인, 체크포인트 그대로)\n")
    head = f"  {'ch':>2}" + "".join(f"{t:>13}" for t in args.tag)
    if len(args.tag) == 2:
        head += f"{'배율':>9}"
    print(head)
    a, b = args.tag[0], args.tag[-1]
    for c in range(16):
        line = f"  {c:>2}" + "".join(f"{float(thrs[t][c]):>13.4f}" for t in args.tag)
        if len(args.tag) == 2:
            x, y = float(thrs[a][c]), float(thrs[b][c])
            line += f"{(y / x if x else float('nan')):>8.2f}x"
        print(line)
    for t in args.tag:
        v = thrs[t]
        print(f"\n  {t}: 범위 {float(v.min()):.4f} ~ {float(v.max()):.4f} "
              f"= {float(v.max() / v.min()):.0f}배,  중앙값 {float(v.median()):.4f}")
    if len(args.tag) == 2:
        r = (thrs[b] / thrs[a])
        moved = int(((r > 2) | (r < 0.5)).sum())
        print(f"  두 런 사이에서 2배 이상 움직인 채널: {moved} / 16")

    # --- the analog side's own measured numbers ----------------------------
    print("\n\n아날로그 실측 (analog/AFE_tuning/artifacts/r7r8_real.md)\n")
    print(f"  {'ch':>2}{'f_c[Hz]':>9}{'opamp':>10}{'swing[mV]':>11}"
          f"{'Venv_DC':>9}{'현재 Vthr':>11}{'R7[kΩ]':>9}{'R8[kΩ]':>9}")
    for r in swing:
        dc = r["vthr_mv"] - SHIPPED_F * r["swing_mv"]
        print(f"  {r['ch']:>2}{r['f_c']:>9.0f}{r['opamp']:>10}"
              f"{r['swing_mv']:>11.2f}{dc:>9.1f}{r['vthr_mv']:>11.2f}"
              f"{r['r7']:>9.2f}{r['r8']:>9.2f}")
    print(f"\n  현재 값은 전부 f = {SHIPPED_F} 라는 단일 상수에서 나왔다 "
          f"(swing_real.py:43).")
    print("  학습된 임계값이 아니다. 지금은 두 숫자가 연결되어 있지 않다.")

    # --- the naive map, clearly labelled -----------------------------------
    print("\n\n가정: f := 학습된 임계값 (근거 없음, 모양만 보라)\n")
    print("  Vthr = Venv_DC + thr x swing,   R8 = Vthr/1.8 x 1000kΩ,  R7 = 1000 - R8")
    print(f"\n  {'ch':>2}{'swing':>8}" +
          "".join(f"{t[:11]+' Vthr':>17}" for t in args.tag) +
          f"{'현재 Vthr':>11}")
    for c, r in enumerate(swing):
        dc = r["vthr_mv"] - SHIPPED_F * r["swing_mv"]
        cells = ""
        for t in args.tag:
            v = dc + float(thrs[t][c]) * r["swing_mv"]
            over = "!" if v > dc + r["swing_mv"] else " "
            cells += f"{v:>16.2f}{over}"
        print(f"  {c:>2}{r['swing_mv']:>8.2f}{cells}{r['vthr_mv']:>11.2f}")

    print("\n  ! = 그 채널의 측정 스윙 꼭대기보다 높다 -> 절대 안 켜진다.")
    print("      그런 표시가 있으면 그 자체가 이 가정이 틀렸다는 증거다:")
    print("      학습된 프론트엔드에서 그 채널은 분명히 발화하고 있기 때문이다.")
    print("\n  두 도메인이 다르다 (log_mel 분수 vs 선형 스윙 분수, 전역 lo/hi vs")
    print("  채널별 스윙). 회로에 넘길 숫자를 만들려면 그 사상을 실측으로")
    print("  정해야 하고, 그게 ROADMAP P4-b 다. 이 표는 그 자리가 비어 있다는")
    print("  것을 보이기 위한 것이지 그 자리를 채운 게 아니다.")


if __name__ == "__main__":
    main()
