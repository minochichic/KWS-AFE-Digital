"""실물 보드의 트립 포인트를 흉내낸 CSV 를 만든다 (보드 없이 전략을 미리 재려고).

시나리오: 저항은 어떤 값으로든 박히고, 비교기마다 고정 오프셋이 더해진다. 보드가
생기면 그 **합**을 채널별로 한 번 재서 학습에 넣으면 된다 (`threshold_init=measured`
+ `threshold_trainable=false`). 오프셋을 상쇄할 필요도, 오프셋에 강건해질 필요도
없다 -- 변하지 않는 값이므로 알기만 하면 된다.

여기서 미리 답하는 질문: **임계값이 최적이 아니라 하드웨어가 정한 자리에 있으면
정확도를 얼마나 잃는가.** 보드가 없어도 잴 수 있다.

오프셋 draw 는 `fixed_accuracy --vos-fixed --vos-seed S` 와 **같은 난수**를 쓴다.
그래야 다음 두 숫자가 같은 축 위에 놓인다:

  (a) 이 오프셋을 그냥 맞은 `bd_base`          -> fixed_accuracy 로 이미 쟀다
  (b) 이 오프셋을 알고 재학습한 모델           -> 이 CSV 로 학습해서 잰다

  python -m experiments.make_trip_points --tag bd_base --vos 0.03 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def rel(p: Path) -> str:
    """리포 기준 상대경로, 밖이면 절대경로 그대로 (--out 이 어디든 죽지 않게)."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base",
                    help="저항 공칭값을 가져올 런 (그 학습된 임계값을 씀)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--vos", type=float, default=0.03,
                    help="비교기 오프셋 표준편차, 정규화 단위 (3 mV / K 100 mV)")
    ap.add_argument("--seed", type=int, default=0, help="보드 한 장 = seed 하나")
    ap.add_argument("--nominal", type=float, default=None,
                    help="체크포인트 대신 전 채널 공통 공칭 임계값을 쓴다. "
                         "동료가 저항을 대충 한 값으로 박는 경우")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.nominal is not None:
        base = torch.full((16,), float(args.nominal), dtype=torch.float64)
        src = f"공칭 {args.nominal} (전 채널 동일)"
    else:
        ck = torch.load(Path(args.runs) / args.tag / "best.pt",
                        map_location="cpu", weights_only=True)
        base = ck["afe"]["threshold"].reshape(-1).double()
        if base.numel() != 16:
            raise SystemExit(f"{args.tag}: 임계값이 {base.numel()}개다")
        src = f"{args.runs}/{args.tag}/best.pt"

    # fixed_accuracy --vos-fixed 와 동일한 draw 여야 비교가 성립한다.
    g = torch.Generator().manual_seed(args.seed)
    draw = args.vos * torch.randn(16, generator=g).double()
    trip = base + draw

    out = Path(args.out or ROOT / "analog/AFE_board/artifacts"
               / f"trip_points_{args.tag}_vos{args.vos:g}_s{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, trip.numpy(), fmt="%.6f")

    print(f"공칭 출처: {src}")
    print(f"오프셋:    std {args.vos:g} (정규화), seed {args.seed}\n")
    print(f"{'ch':>3}{'공칭':>9}{'오프셋':>10}{'트립':>10}   비고")
    dead = 0
    for c in range(16):
        note = ""
        if trip[c] <= 0.0:
            note = "<- 항상 켜짐 (트립 <= 0)"
            dead += 1
        elif abs(draw[c]) > base[c]:
            note = "<- 오프셋이 공칭보다 큼"
        print(f"{c:>3}{base[c]:>9.4f}{draw[c]:>+10.4f}{trip[c]:>10.4f}   {note}")

    print(f"\n트립 범위 {trip.min():.4f} ~ {trip.max():.4f}")
    if dead:
        print(f"⚠️  트립 <= 0 인 채널 {dead}개: 실물이라면 항상 +1 이다. 학습이 "
              f"이걸 '알고' 나머지로 보상할 수 있는지가 이 실험의 절반이다.")
    print(f"\n저장: {rel(out)}")
    print("\n재학습:")
    print(f"  python -m train.train --config configs/base.yaml \\\n"
          f"    --tag {args.tag}_trip_s{args.seed} \\\n"
          f"    afe.spice_matrix_path=analog/AFE/artifacts/filterbank_matrix_board.csv \\\n"
          f"    afe.threshold_init=measured \\\n"
          f"    afe.threshold_measured_path={rel(out)} \\\n"
          f"    afe.threshold_trainable=false")
    print("\n비교 대상 (같은 오프셋을 '모르고' 맞은 경우):")
    print(f"  python -m experiments.fixed_accuracy --tag {args.tag} "
          f"--vos {args.vos:g} --vos-fixed --vos-seed {args.seed}")


if __name__ == "__main__":
    main()
