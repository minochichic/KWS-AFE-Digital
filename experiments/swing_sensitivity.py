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
    # 학습에 이미 스윙 배율이 들어갔으면 환산이 달라진다. 그때는 정규화 도메인이
    # 전압에 비례하므로 물리 오차가 전 채널에 **균일**하게 작용한다 -- 여기서
    # 다시 1/swing_c 를 곱하면 이중 적용이라 없는 격차를 만들어낸다.
    scaled = bool(getattr(cfg.afe, "spice_swing_path", "") or "")
    if scaled:
        per_mv = np.full_like(per_mv, 1.0 / sw.max())
        print("⚠️ 이 런은 spice_swing_path 로 학습됐다 -> 환산이 전 채널 균일하다"
              f" ({per_mv[0]:.4f}/mV). 격차 열은 전부 1.0x 가 정상이다.\n")

    afe = AFEFrontend(cfg.afe).eval()
    model = BinaryMatchboxNet(cfg.model).eval()
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])
    # ⚠️ 잠재값이 아니라 **유효 임계값** 에 섭동을 준다.
    # threshold_min 은 straight-through clamp 라 잠재값이 자유롭게 흘러간다
    # (실제로 -6.269 까지 간 채널이 있다). 섭동을 clamp 앞에 더하면 clamp 가
    # 그걸 흡수해서 그 채널이 "오차에 완전 면역" 인 것처럼 보이는데, **실물 저항에는
    # clamp 가 없다.** 회로는 유효값으로 만들어지고 오차는 그 위에 얹힌다.
    tmin = float(getattr(cfg.afe, "threshold_min", 0.0) or 0.0)
    thr0 = afe.threshold.detach().clone()
    if tmin > 0.0:
        n_pinned = int((thr0 < tmin).sum())
        thr0 = thr0.clamp(min=tmin)
        afe.cfg.threshold_min = 0.0          # 모델이 다시 clamp 하지 않도록
        print(f"threshold_min={tmin:g}: 유효 임계값으로 환산 "
              f"(하한에 눌린 채널 {n_pinned}/{C}개). 섭동은 그 위에 얹는다.\n")

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
    # 이 스크립트를 쓸 때는 스윙이 순위를 정할 것이라 예상했다. bd_base 에서는
    # 아니었다: 순위상관이 임계값과 +0.98, 스윙과는 +0.49 다. 가장 분명한 반례가
    # ch2 -- 스윙 90.1 mV 로 두 번째로 큰데 뒤집힘 2 위다. 임계값이 0.008 이라서다.
    #
    # 이유: 임계값이 낮으면 엔벨로프 분포의 **밀집 구간**에 앉는다. 거기서는 조금만
    # 밀어도 많은 프레임이 경계를 넘는다. 임계값이 높으면 꼬리에 앉아 주변이 성기다.
    # firing_stability 가 같은 것을 다른 각도로 보여준다: 임계값 낮은 채널은 발화율
    # 30~36% (분포 한가운데), 높은 채널은 8~10% (꼬리) 였다.
    tr = [float(thr0[c]) for c in range(C)]
    def _rank(x):
        import statistics as st
        r = sorted(range(len(x)), key=lambda i: x[i])
        out = [0.0] * len(x)
        for pos, i in enumerate(r):
            out[i] = pos
        m = st.mean(out); sd = st.pstdev(out) or 1.0
        return [(v - m) / sd for v in out]
    fr = _rank([-int(flips[c]) for c in range(C)])
    print(f"\n  순위상관 — 뒤집힘 vs 임계값 "
          f"{sum(a*b for a, b in zip(_rank(tr), fr))/C:+.2f}, "
          f"vs 스윙 {sum(a*b for a, b in zip(_rank(list(sw)), fr))/C:+.2f}")
    print("  임계값 쪽이 크면 이건 아날로그가 아니라 **우리** 문제다: 임계값이")
    print("  분포 한가운데 앉아 있어서다. 모델 선택이나 임계값 하한으로 움직인다.")
    print("  스윙 쪽이 크면 아날로그다 -- 검출기 이득이나 최상단 f_c.")
    print("\n  주의: 위 오차는 전 채널이 공차 한계에 있는 최악 가정이다(크기 고정,")
    print("  부호만 무작위). 실제 분포라면 낙폭은 이보다 작다.")


if __name__ == "__main__":
    main()
