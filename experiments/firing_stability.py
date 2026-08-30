"""채널별 발화율이 정말 "고정"인가 — 그리고 하드웨어가 치르는 값은 무엇인가.

`fixed_accuracy` 는 전 클립·전 프레임을 합산한 **평균 하나**를 채널마다 낸다
(ch15 35.3% 처럼). 그 숫자로는 두 질문에 답할 수 없다:

  1. 클립마다도 그런가?  `normalize="fixed"` 는 절대 임계다 -- AGC 가 없으므로
     큰 소리엔 많이, 작은 소리엔 적게 켜진다. 평균 35% 가 "늘 35%" 일 수도,
     "대부분 0% 인데 몇 클립에서 90%" 일 수도 있고 **하드웨어 얘기가 전혀 다르다.**

  2. 전이가 몇 번인가?  전력을 정하는 건 발화율이 아니라 **비교기가 뒤집히는 횟수**
     다 (CLAUDE.md 3.5: 임계값은 정확도 파라미터이자 이벤트 발생률 파라미터).
     35% 가 한 덩어리면 전이 2 번, 번갈아 켜지면 프레임마다 전이다. 같은 발화율에
     이벤트 수가 수십 배 차이 난다.

두 태그를 주면 **런 간 안정성**도 같이 본다. 저항비를 얼리려면 그 값이 재현돼야
하는데, 임계값 자체는 순위상관 +0.53 으로 불안정하다는 것이 이미 측정돼 있다
(`docs/ABLATIONS.md`). 발화율이 그보다 안정적인지는 따로 볼 문제다.

    python -m experiments.firing_stability --tag bd_base
    python -m experiments.firing_stability --tag bd_base --tag bd_base_trip_s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def measure(tag: str, runs: str, limit: int):
    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import build_dataloaders

    run = Path(runs) / tag
    cfg = load_config(str(run / "config.yaml"))
    afe = AFEFrontend(cfg.afe).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    load_afe_state(afe, ck["afe"])

    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]

    C = cfg.afe.n_channels
    rates, trans, loud = [], [], []
    n = 0
    with torch.no_grad():
        for wav, _ in te:
            x = afe(wav, target_T=cfg.model.T)          # [B, C, T] in {-1,+1}
            on = (x > 0)
            rates.append(on.float().mean(dim=2))        # [B, C] 클립별 발화율
            # 전이 = 인접 프레임에서 값이 바뀐 횟수. 비교기가 실제로 뒤집히는
            # 횟수의 하한이다 (프레임 안의 펄스는 sticky OR 로 이미 뭉쳐졌다).
            trans.append((on[:, :, 1:] != on[:, :, :-1]).float().sum(dim=2))
            loud.append(wav.abs().amax(dim=1) if wav.dim() == 2
                        else wav.abs().amax(dim=(1, 2)))
            n += x.shape[0]
            if limit and n >= limit:
                break
    return (torch.cat(rates), torch.cat(trans), torch.cat(loud),
            cfg.afe.envelope_win_ms, cfg.model.T, tag)


def report(rates, trans, loud, win_ms, T, tag):
    C = rates.shape[1]
    print(f"\n=== {tag} — {rates.shape[0]} 클립, {C} 채널, "
          f"프레임 {T} × {win_ms:g} ms ===\n")

    print("클립별 발화율 분포 (평균 하나로는 안 보이는 것)")
    print(f"{'ch':>3}{'평균':>8}{'p10':>7}{'p50':>7}{'p90':>7}{'폭':>7}"
          f"{'무음%':>7}{'포화%':>7}")
    print("-" * 60)
    for c in range(C):
        r = rates[:, c]
        p10, p50, p90 = [float(torch.quantile(r, q)) for q in (.1, .5, .9)]
        silent = float((r < 0.01).float().mean())       # 그 클립에서 거의 안 켜짐
        sat = float((r > 0.90).float().mean())
        print(f"{c:>3}{float(r.mean())*100:>7.1f}%{p10*100:>6.1f}%"
              f"{p50*100:>6.1f}%{p90*100:>6.1f}%{(p90-p10)*100:>6.1f}%"
              f"{silent*100:>6.1f}%{sat*100:>6.1f}%")

    print("\n  폭 = p90 − p10. 크면 '고정'이 아니라 소리 크기를 따라간다는 뜻이다.")
    print("  무음% = 그 채널이 사실상 안 켜진 클립의 비율. 그 클립에서는 비트가 없다.")

    # ---- 전이: 전력을 정하는 것 -------------------------------------------- #
    sec = T * win_ms / 1000.0
    print(f"\n프레임 전이 — 이쪽이 이벤트·인터럽트·전력이다 (클립 {sec:g} 초)")
    print(f"{'ch':>3}{'전이/클립':>11}{'전이/초':>10}{'발화율':>8}{'전이/발화':>11}")
    print("-" * 45)
    for c in range(C):
        t = float(trans[:, c].mean())
        r = float(rates[:, c].mean())
        on_frames = r * T
        print(f"{c:>3}{t:>11.1f}{t/sec:>10.1f}{r*100:>7.1f}%"
              f"{(t/on_frames if on_frames > 0 else 0):>11.2f}")
    tot = float(trans.sum(dim=1).mean())
    print(f"\n  16채널 합계: 클립당 {tot:.0f} 전이 = 초당 {tot/sec:.0f}")
    print("  전이/발화 가 1 에 가까우면 켜짐이 흩어져 있다(전이가 많다).")
    print("  0 에 가까우면 한 덩어리로 켜진다(전이가 적다) -- 같은 발화율이라도")
    print("  하드웨어 비용이 수십 배 다르다.")

    # ---- 음량 의존성 ------------------------------------------------------- #
    print("\n음량과의 상관 (절대 임계라 있어야 정상이다)")
    lo = loud.double()
    lo = (lo - lo.mean()) / (lo.std() + 1e-12)
    cs = []
    for c in range(C):
        r = rates[:, c].double()
        r = (r - r.mean()) / (r.std() + 1e-12)
        cs.append(float((r * lo).mean()))
    print("  " + " ".join(f"{v:+.2f}" for v in cs))
    print(f"  평균 {sum(cs)/len(cs):+.3f}  — 0 에 가까우면 AGC 처럼 굴고 있다는 뜻이라")
    print("  오히려 이상하다. fixed 는 음량을 따라가는 것이 정상이다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", action="append", required=True,
                    help="반복하면 런 간 비교")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--limit", type=int, default=0, help="클립 수 상한(빠른 확인용)")
    args = ap.parse_args()

    got = [measure(t, args.runs, args.limit) for t in args.tag]
    for g in got:
        report(*g)

    if len(got) >= 2:
        print("\n=== 런 간 발화율 안정성 ===")
        a, b = got[0][0].mean(dim=0).double(), got[1][0].mean(dim=0).double()
        d = (a - b).abs()
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        ra = (ra - ra.mean()) / ra.std()
        rb = (rb - rb.mean()) / rb.std()
        print(f"  {got[0][5]} vs {got[1][5]}")
        print(f"  채널별 발화율 차이: 평균 {float(d.mean())*100:.1f}pp, "
              f"최대 {float(d.max())*100:.1f}pp")
        print(f"  순위상관: {float((ra * rb).mean()):+.2f}")
        print("  낮으면 채널별 값을 저항으로 얼릴 근거가 약하다는 뜻이다.")


if __name__ == "__main__":
    main()
