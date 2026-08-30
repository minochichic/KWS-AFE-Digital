"""물리적 오차 1 mV 가 채널마다 다르게 아프다 — 얼마나 다른가.

우리 시뮬은 필터뱅크를 **채널별 피크 정규화**해서 쓴다. 그래서 모델 안에서는 16채널이
모두 같은 동적 범위를 갖는다. 실물은 아니다: 0.4 mV 마이크 입력에서 ch2 는 90.13 mV,
ch15 는 15.99 mV 를 흔든다 (`analog/AFE_board/artifacts/swing_board.csv`). 검출기의
루프여유가 고주파에서 줄기 때문이고, **그 손실은 필터 행렬에 들어 있지 않다.**

결과: 비교기 오프셋이든 저항 공차든 잡음이든, **같은 mV 가 채널마다 다른 크기의
정규화 임계값 이동**이 된다. ch15 는 ch2 의 5.6 배다.

`fixed_accuracy --vos` 는 이걸 못 본다 -- 모든 채널에 같은 정규화 단위를 뿌린다.
`train/config.py:105` 가 xmax/xmix 에 대해 경고한 것과 같은 함정이고, 여기서는
채널마다 스케일이 다르다는 형태로 나타난다.

두 가지를 낸다:

  1. **민감도**  채널별로 임계값을 물리 1 mV 만큼 밀었을 때 뒤집히는 프레임 수.
     스윙(mV→정규화 환산)과 임계값 근처 분포 밀도가 곱해진 값이다. 발화율이 50%
     에 가까울수록 밀도가 높아 더 많이 뒤집힌다.

  2. **정확도**  물리적으로 맞춰 뿌린 오차에서의 test 정확도. 채널마다 다른
     정규화 크기로 흔든다 -- 실물이 그렇게 아프다.

    python -m experiments.swing_sensitivity --tag bd_base --mv 0.5 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SWING_CSV = ROOT / "analog/AFE_board/artifacts/swing_board.csv"


def swings_mv():
    """채널별 스윙 [mV]. board_swing.py 가 쓴 CSV 를 읽는다 (복사하지 않는다)."""
    if not SWING_CSV.is_file():
        raise SystemExit(f"{SWING_CSV} 가 없다. board_swing.py --amp 0.4m 를 먼저.")
    d = np.loadtxt(SWING_CSV, delimiter=",", skiprows=1)
    return d[:, 0].astype(int), d[:, 3] * 1e3          # ch, swing_V -> mV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--mv", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                    help="물리 오차 크기 [mV] (채널별 스윙으로 환산해 뿌린다)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)

    ch, sw = swings_mv()
    C = cfg.afe.n_channels
    if len(sw) != C:
        raise SystemExit(f"스윙 {len(sw)}개, 채널 {C}개 -- 안 맞는다")

    # 정규화 1.0 이 몇 mV 인가. 전역 lo/hi 라 스케일은 하나여야 하지만, 채널마다
    # 실제 스윙이 다르므로 "그 채널에서 정규화 1.0 에 해당하는 mV" 는 스윙 자체다.
    # 따라서 물리 e mV 는 그 채널에서 e/swing_c 만큼의 정규화 이동이다.
    per_mv = 1.0 / sw                                   # 정규화 단위 / mV

    afe = AFEFrontend(cfg.afe).eval()
    model = BinaryMatchboxNet(cfg.model).eval()
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])
    thr0 = afe.threshold.detach().clone()

    print(f"태그 {args.tag}\n")
    print("채널별 환산 — 물리 1 mV 가 정규화로 얼마인가")
    print(f"{'ch':>3}{'스윙[mV]':>10}{'thr':>8}{'1mV = 정규화':>14}{'ch2 대비':>10}")
    print("-" * 46)
    base = per_mv[2]
    for c in range(C):
        print(f"{c:>3}{sw[c]:>10.2f}{float(thr0[c]):>8.3f}"
              f"{per_mv[c]:>14.4f}{per_mv[c]/base:>9.1f}x")

    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]

    def score(delta):
        """delta [C] 를 임계값에 더하고 정확도와 채널별 뒤집힌 비트 수를 낸다."""
        with torch.no_grad():
            afe.threshold.copy_(thr0 + delta)
        ok = n = 0
        flips = torch.zeros(C, dtype=torch.int64)
        with torch.no_grad():
            for wav, lab in te:
                with torch.no_grad():
                    afe.threshold.copy_(thr0)
                    ref = afe(wav, target_T=cfg.model.T) > 0
                    afe.threshold.copy_(thr0 + delta)
                    x = afe(wav, target_T=cfg.model.T)
                flips += (( x > 0) != ref).sum(dim=(0, 2))
                ok += (model(x).argmax(1) == lab).sum().item()
                n += lab.numel()
        return ok / n, flips, n

    with torch.no_grad():
        afe.threshold.copy_(thr0)
    a0, _, n = score(torch.zeros(C))
    print(f"\n기준 정확도 {a0:.4f} ({n} 클립)\n")

    print("물리 오차를 채널별로 환산해 뿌렸을 때")
    print(f"{'오차[mV]':>10}{'정확도':>10}{'낙폭':>9}{'뒤집힌 비트':>13}")
    print("-" * 44)
    g = torch.Generator().manual_seed(args.seed)
    sign = torch.randn(C, generator=g).sign()          # 채널마다 방향이 다르다
    per = torch.tensor(per_mv, dtype=torch.float32)
    rows = []
    for e in args.mv:
        d = sign * per * e
        acc, flips, _ = score(d)
        tot = int(flips.sum())
        print(f"{e:>10.2f}{acc:>10.4f}{(acc-a0)*100:>+8.2f}p{tot:>13,}")
        rows.append((e, flips))

    e, flips = rows[-1]
    print(f"\n채널별 뒤집힌 비트 — 오차 {e:g} mV 에서")
    print(f"{'ch':>3}{'스윙':>9}{'뒤집힘':>11}{'비율':>8}")
    print("-" * 32)
    tot = int(flips.sum())
    order = sorted(range(C), key=lambda c: -int(flips[c]))
    for c in order:
        print(f"{c:>3}{sw[c]:>9.1f}{int(flips[c]):>11,}"
              f"{100*int(flips[c])/max(1,tot):>7.1f}%")
    print("\n  스윙이 작은 채널이 위쪽에 몰리면 노파심이 맞았던 것이다.")
    print("  대응은 아날로그 쪽이다: 검출기 이득을 낮춰 고역 루프여유를 되찾거나,")
    print("  최상단 f_c 를 내리거나. 학습으로 고칠 수 있는 종류가 아니다.")


if __name__ == "__main__":
    main()
