"""val_acc 진동이 BatchNorm running 통계 때문인지 판별한다.

`an_*` 런에서 train acc 는 0.77 로 평탄한데 val 은 0.22~0.73 을 오간다. lr=1e-5
구간에서도 에폭 사이 20pp 가 튀는데, 그 학습률에서 가중치는 거의 안 움직이므로
가중치 변화로는 설명이 안 된다. 남는 후보는 **train 과 eval 이 서로 다른 BN 통계를
쓴다**는 것이다:

  train  배치 통계  -> 활성 분포가 밀려도 그 자리에서 다시 정규화된다
  eval   running 통계 -> 밀린 분포를 낡은 통계로 정규화한다

이진 가중치는 잠재값이 0 을 지날 때 부호가 뒤집히므로 활성 분포가 계속 밀린다.
train 은 그걸 흡수하고 eval 은 못 흡수한다면, running 통계를 **다시 재기만 해도**
정확도가 올라야 한다. 세 가지를 같은 체크포인트로 잰다:

  eval(그대로)     저장된 running 통계
  eval(재보정)     학습셋을 forward 해서 running 통계를 다시 채운 뒤
  train-BN         배치 통계 그대로 (상한; 실제 배포에는 못 쓴다)

재보정이 크게 올리면 원인은 BN 이고, 고치는 방법도 그것이다. 셋이 비슷하면 BN 이
아니므로 다른 데를 봐야 한다.

    python -m experiments.bn_gap --tag an_v3b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def evaluate(model, loader, device) -> float:
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ok += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    return ok / max(n, 1)


def evaluate_batch_stats(model, loader, device) -> float:
    """BN 을 train 모드로 두고 평가 = 배치 통계. 상한이지 배포 가능한 값이 아니다
    (추론 때 배치가 없다)."""
    model.train()
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ok += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    return ok / max(n, 1)


def recalibrate_bn(model, loader, device, n_batches: int = 200) -> None:
    """running 통계를 학습셋으로 다시 채운다. 가중치는 건드리지 않는다."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None          # None = 누적 평균 (지수 EMA 가 아니라)
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(device))
            if i + 1 >= n_batches:
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--batches", type=int, default=200,
                    help="재보정에 쓸 학습 배치 수")
    args = ap.parse_args()

    from train.config import load_config
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not getattr(cfg.data, "analog_csv_root", ""):
        raise SystemExit(f"{args.tag} 는 아날로그 CSV 런이 아니다. "
                         f"AFE 런은 experiments/fixed_accuracy.py 를 쓴다.")
    from data.analog_spectrogram import build_analog_dataloaders
    train_loader, val_loader, test_loader = build_analog_dataloaders(
        cfg.data, cfg.train.batch_size, target_T=cfg.model.T)

    model = BinaryMatchboxNet(cfg.model).to(device)
    ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])

    print(f"\n체크포인트: {run}/best.pt (epoch {ck.get('epoch','?')})\n")
    print(f"{'':22s}{'val':>9}{'test':>9}")
    a_v = evaluate(model, val_loader, device)
    a_t = evaluate(model, test_loader, device)
    print(f"{'eval (그대로)':22s}{a_v:>9.4f}{a_t:>9.4f}")

    b_v = evaluate_batch_stats(model, val_loader, device)
    b_t = evaluate_batch_stats(model, test_loader, device)
    print(f"{'train-BN (배치통계)':22s}{b_v:>9.4f}{b_t:>9.4f}   <- 상한")

    recalibrate_bn(model, train_loader, device, args.batches)
    c_v = evaluate(model, val_loader, device)
    c_t = evaluate(model, test_loader, device)
    print(f"{'eval (재보정)':22s}{c_v:>9.4f}{c_t:>9.4f}")

    print()
    gain = c_t - a_t
    if gain > 0.02:
        print(f"→ 재보정만으로 test +{gain*100:.1f}pp. **원인은 BN running 통계다.**")
        print("   고치는 법: bn_momentum 을 낮추거나(창을 늘리거나), 학습 끝에")
        print("   running 통계를 다시 재고 저장한다. 가중치는 그대로여도 된다.")
    elif b_t - a_t > 0.02:
        print(f"→ 재보정은 안 듣는데 배치통계는 +{(b_t-a_t)*100:.1f}pp 다. running "
              f"통계를 '언제' 쟀느냐가 아니라 **하나의 통계로는 부족**하다는 뜻 "
              f"(입력이 시간축으로 심하게 비정상적이다).")
    else:
        print("→ 셋이 비슷하다. **BN 은 원인이 아니다.** 진동은 다른 데서 온다 "
              "-- 데이터 자체의 천장을 의심할 차례다.")


if __name__ == "__main__":
    main()
