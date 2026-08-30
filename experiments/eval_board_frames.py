"""학습된 네트워크를 동료의 회로 출력 프레임에 그대로 태운다 (재학습 없음).

`swing_sensitivity` 는 **우리** 파이프라인이 만든 프레임의 임계값을 흔든다. 여기서는
프레임 자체가 남의 것이다: 동료가 SPICE 로 필터+검출기를 돌리고 자기 Vthr 로
이진화해 보낸 CSV 다. 그래서 이 스크립트가 재는 것은 정확히 하나 --

    **우리 네트워크가 자기 임계값이 아닌 임계값으로 만든 이미지를 얼마나 견디는가.**

이건 이미 두 번 잰 적이 있다 (`bd_base_trip_s0` 0.734, `bd_nom0.1_trip_s0` 0.669).
둘 다 우리 엔벨로프 파이프라인은 공유한 채 임계값만 바꾼 것이었는데도 9~15pp 를
잃었다. 여기서는 체인 전체가 다르므로 그보다 나쁠 것을 예상하고 본다. **낮은
숫자는 회로가 나쁘다는 뜻이 아니다** -- 이식(transplant)의 대가다. 회로의 실력을
알고 싶으면 재학습해야 한다:

    python -m train.train --overrides data.analog_csv_root=vthr_762b7b0c1885

어느 체크포인트를 태울지가 중요하다. `docs/request_v4/target_firing_rates.csv` 의
목표 발화율은 **`bd_base`** 에서 뽑은 것이므로, 회로가 그 목표를 맞췄다면 `bd_base`
가 가장 잘 맞는 짝이다. 얼마나 맞췄는지도 같이 찍는다 -- 안 맞았다면 낮은 정확도의
원인이 거기 있고, 그건 우리가 아니라 아날로그 쪽에서 움직여야 하는 값이다.

    python -m experiments.eval_board_frames --tag bd_base
    python -m experiments.eval_board_frames --tag bd_base --csv-root vthr_762b7b0c1885
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "docs/request_v4/target_firing_rates.csv"


def firing_report(bits: np.ndarray) -> None:
    """전달받은 프레임의 채널별 발화율 vs 우리가 요청한 목표."""
    if not TARGETS.is_file():
        return
    rows = list(csv.DictReader(TARGETS.open()))
    got = bits.mean(axis=(0, 2)) * 100.0
    print("\n채널별 발화율 — 우리가 요청한 목표 대비")
    print(f"{'ch':>3}{'f_c[Hz]':>9}{'목표':>8}{'받은':>8}{'배수':>9}")
    print("-" * 37)
    off = []
    for r in rows:
        c = int(r["ch"])
        tgt, cur = float(r["target_fire_pct"]), float(got[c])
        ratio = tgt / cur if cur > 0 else float("inf")
        off.append(abs(np.log(max(ratio, 1e-9))))
        print(f"{c:>3}{float(r['f_c_hz']):>9.0f}{tgt:>7.1f}%{cur:>7.1f}%{ratio:>8.2f}x")
    near = sum(1 for r, g in zip(rows, got)
               if g > 0 and 0.7 <= float(r["target_fire_pct"]) / g <= 1.43)
    print(f"\n  목표 ±43% 안: {near}/{len(rows)}채널, "
          f"로그평균 이탈 {np.exp(np.mean(off)):.2f}x")
    if near < len(rows) // 2:
        print("  목표를 못 맞췄다 -> 아래 정확도는 '이 네트워크가 다른 임계값을")
        print("  얼마나 견디는가'이지 회로의 실력이 아니다. 재학습해야 알 수 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base",
                    help="체크포인트. 목표 발화율을 뽑은 런과 같아야 의미가 있다.")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv-root", default="",
                    help="비우면 런의 config 에 적힌 analog_csv_root 를 쓴다")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    from train.config import load_config
    from data.analog_spectrogram import build_analog_dataloaders, load_all
    from data.speech_commands import KEYWORDS
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    if args.csv_root:
        cfg.data.analog_csv_root = args.csv_root
    if not cfg.data.analog_csv_root:
        raise SystemExit("--csv-root 를 주거나 config 의 data.analog_csv_root 를 채울 것")

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = BinaryMatchboxNet(cfg.model).eval()
    model.load_state_dict(ck["model"])
    print(f"태그 {args.tag} (epoch {ck.get('epoch')}), "
          f"프레임 {cfg.data.analog_csv_root}")
    print("AFE 없음 -- 입력이 이미 이진이다. 이 런이 학습한 임계값은 쓰이지 않는다.\n")

    bits, _, _ = load_all(cfg.data.analog_csv_root)
    firing_report(bits)

    loaders = build_analog_dataloaders(cfg.data, cfg.train.batch_size,
                                       target_T=cfg.model.T, seed=cfg.train.seed)
    loader = loaders[{"train": 0, "val": 1, "test": 2}[args.split]]

    names = KEYWORDS + ["_silence_", "_unknown_"]
    K = cfg.model.n_classes
    hit = np.zeros(K, dtype=np.int64)
    tot = np.zeros(K, dtype=np.int64)
    with torch.no_grad():
        for x, lab in loader:
            pred = model(x).argmax(1)
            for t, p in zip(lab.tolist(), pred.tolist()):
                tot[t] += 1
                hit[t] += (t == p)

    n = int(tot.sum())
    print(f"\n{args.split} 정확도 {hit.sum()/n:.4f} ({n} 클립)\n")
    print(f"{'class':<12}{'clips':>7}{'정확도':>9}")
    print("-" * 28)
    for i in np.argsort(-(hit / np.maximum(tot, 1))):
        if tot[i]:
            print(f"{names[i] if i < len(names) else i:<12}{tot[i]:>7}"
                  f"{hit[i]/tot[i]:>9.3f}")
    # unknown 은 우리 baseline 에서 최악(0.644)이었다. 여기서 크게 높으면 회로가
    # 좋아서가 아니라 unknown/ 파일명이 재명명돼 화자 분리가 깨진 것을 먼저 의심할 것
    # (data/analog_spectrogram.py 의 ⚠️ 참조).
    print("\n비교축: bd_base 12클래스 0.8249 (우리 시뮬), 그중 _unknown_ 0.644 / "
          "_silence_ 0.993")


if __name__ == "__main__":
    main()
