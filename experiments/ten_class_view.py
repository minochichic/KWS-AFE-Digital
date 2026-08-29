"""12클래스 런을 10클래스 기준으로 재집계한다 -- 아날로그 런과 같은 축에 놓으려고.

`an_*` 런은 키워드 10개뿐인 전송본으로 학습해서 10클래스다. `bd_base` 는 12클래스라
0.825 를 그대로 비교할 수 없다. 그런데 어느 쪽으로 치우치는지는 산수로 안 나온다:

  `_unknown_` 를 빼면  -> 우리 최악(0.644)이 빠지니 **올라간다**
  `_silence_` 를 빼면  -> 우리 최고(0.993)가 빠지니 **내려간다**
  오답 선택지 2개가 사라지면 -> **올라간다**

그래서 잰다. 같은 체크포인트를 두 가지로 본다:

  하한  키워드 클립만, argmax 는 12개 그대로.
        `_silence_`/`_unknown_` 로 찍으면 오답. 10클래스 모델이라면 애초에 고를 수
        없는 답이므로 **실제보다 불리하다.**

  상한  키워드 클립만, argmax 를 **키워드 10개 로짓으로 제한**.
        10클래스 헤드가 할 법한 것에 가깝지만, 진짜 10클래스로 학습하면 그 두
        클래스에 쓰이던 용량이 풀리므로 **정확히 같지는 않다.**

진짜 10클래스 값은 이 둘 사이에 있다. `an_bn` 0.730 이 그 구간 어디에 놓이는지가
"전송본 데이터가 우리 시뮬보다 얼마나 약한가"의 첫 정직한 답이다.

    python -m experiments.ten_class_view --tag bd_base
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--compare", type=float, default=None,
                    help="나란히 놓을 10클래스 정확도 (예: an_bn 의 0.730)")
    args = ap.parse_args()

    from train.config import load_config
    from data.afe import AFEFrontend, load_afe_state
    from data.speech_commands import (KEYWORDS, build_dataloaders, class_names,
                                      SILENCE_INDEX, UNKNOWN_INDEX)
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    afe = AFEFrontend(cfg.afe).to(device).eval()
    model = BinaryMatchboxNet(cfg.model).to(device).eval()
    ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    load_afe_state(afe, ck["afe"])

    te = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.afe.sample_rate,
                           seed=cfg.train.seed)[2]

    kw = torch.tensor(list(range(len(KEYWORDS))), device=device)
    n = n12 = lo = hi = 0
    to_extra = 0
    per = torch.zeros(len(KEYWORDS), 2, dtype=torch.long)   # [정답(상한), 전체]
    with torch.no_grad():
        for wav, y in te:
            wav, y = wav.to(device), y.to(device)
            logits = model(afe(wav, target_T=cfg.model.T))
            p12 = logits.argmax(1)
            n12 += (p12 == y).sum().item()
            n += y.numel()

            m = y < len(KEYWORDS)                    # 키워드 클립만
            if not m.any():
                continue
            yk, p_all = y[m], p12[m]
            lo += (p_all == yk).sum().item()
            to_extra += ((p_all == SILENCE_INDEX) |
                         (p_all == UNKNOWN_INDEX)).sum().item()
            p10 = logits[m][:, kw].argmax(1)         # 키워드 로짓으로만 argmax
            hi += (p10 == yk).sum().item()
            for c in range(len(KEYWORDS)):
                s = yk == c
                per[c, 0] += (p10[s] == c).sum().item()
                per[c, 1] += s.sum().item()

    nk = int(per[:, 1].sum())
    print(f"\n체크포인트: {run}/best.pt   테스트 {n:,} 클립 "
          f"(키워드 {nk:,}, 나머지 {n-nk:,})\n")
    print(f"12클래스 그대로            {n12/n:.4f}   <- 기록된 값")
    print(f"10클래스 하한 (12-way)     {lo/nk:.4f}   "
          f"키워드 클립 중 {to_extra}개가 silence/unknown 으로 샜다")
    print(f"10클래스 상한 (10-way)     {hi/nk:.4f}   argmax 를 키워드로 제한")
    if args.compare is not None:
        print(f"\n전송본 10클래스            {args.compare:.4f}")
        d_lo, d_hi = args.compare - lo / nk, args.compare - hi / nk
        print(f"  하한 대비 {d_lo*100:+.1f}pp,  상한 대비 {d_hi*100:+.1f}pp")
        if d_hi < -0.02:
            print("  -> 클래스 수로는 설명 안 되는 실제 격차다. "
                  "남은 변수는 시간 정렬과 회로다.")
        elif d_lo > 0.02:
            print("  -> 전송본이 오히려 낫다. 예상 밖이니 조건을 다시 볼 것.")
        else:
            print("  -> 두 값이 겹친다. 클래스 수 차이로 대부분 설명된다.")

    print(f"\n{'클래스':>8}{'10-way recall':>15}{'clips':>8}")
    for c, w in enumerate(KEYWORDS):
        print(f"{w:>8}{per[c,0].item()/max(int(per[c,1]),1):>15.3f}"
              f"{int(per[c,1]):>8}")


if __name__ == "__main__":
    main()
