"""우리 시뮬로 학습한 모델에 동료 전송본을 그대로 넣어본다 (재학습 없음).

두 이진 이미지가 **같은 표현**이라면 가중치가 그대로 통해야 한다. 안 통하면
표현이 다른 것이고, 얼마나 다른지가 곧 "우리 시뮬 vs 실제 회로"의 거리다.
재학습이 필요 없으므로 몇 초에 끝나고, 학습 하이퍼파라미터가 섞이지 않는다.

`bd_base` 는 12클래스인데 전송본은 키워드 10개뿐이므로 **argmax 를 키워드 로짓
10개로 제한**한다 (`ten_class_view.py` 의 상한과 같은 방식). 그래야 같은 축이다.

읽는 법:

  0.80 근처   두 전단이 사실상 호환된다. 남은 격차는 학습 조건 쪽이다.
  0.40~0.6    표현이 겹치긴 하는데 어긋난다. 무엇이 어긋나는지는
              --shift / --flip 로 좁힐 수 있다.
  0.10 근처   전혀 안 통한다 = 완전히 다른 표현.

`--shift` 는 시간축을 밀어보고, `--flip` 은 채널 순서를 뒤집어 본다. 둘 중
하나로 크게 회복되면 그 축이 어긋나 있었다는 뜻이다 (row_order 는 이미 맞췄지만,
맞췄다는 것과 우리 모델이 기대하는 순서인 것은 별개다).

    python -m experiments.cross_apply --tag bd_base \\
        --csv analog/AFE_board/transfer_v3_native/spectrogram-native.tar.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base", help="12클래스 학습 런")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", required=True, help="전송본 (디렉터리 또는 tar.gz)")
    ap.add_argument("--shift", type=int, nargs="*", default=[0],
                    help="시간축 이동 프레임 (예: -20 -10 0 10 20)")
    ap.add_argument("--flip", action="store_true", help="채널 순서를 뒤집어서도 본다")
    args = ap.parse_args()

    from train.config import load_config
    from data.analog_spectrogram import load_all, _split_lists
    from data.afe import pad_or_crop
    from data.speech_commands import KEYWORDS
    from models.binary_matchboxnet import BinaryMatchboxNet

    run = Path(args.runs) / args.tag
    cfg = load_config(str(run / "config.yaml"))
    if cfg.model.n_classes < len(KEYWORDS):
        raise SystemExit(f"{args.tag} 는 {cfg.model.n_classes}클래스다; "
                         f"12클래스 런이어야 한다.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BinaryMatchboxNet(cfg.model).to(device).eval()
    ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])

    bits, labels, keys = load_all(args.csv)
    where = _split_lists(cfg.data.root)
    m = np.array([where.get(k, "train") for k in keys]) == "test"
    bits, labels = bits[m], labels[m]
    print(f"\n{args.tag} 가중치 그대로, 전송본 test {len(labels):,} 클립\n")

    kw = torch.arange(len(KEYWORDS), device=device)

    def score(shift: int, flip: bool) -> float:
        ok = 0
        for i in range(0, len(labels), 512):
            b = bits[i:i + 512]
            if flip:
                b = b[:, ::-1]
            x = torch.from_numpy(np.ascontiguousarray(b)).float() * 2 - 1
            if shift:
                x = torch.roll(x, shift, dims=2)
                if shift > 0:
                    x[:, :, :shift] = -1.0
                else:
                    x[:, :, shift:] = -1.0
            x = pad_or_crop(x, cfg.model.T, pad_value=-1.0).to(device)
            y = torch.from_numpy(labels[i:i + 512]).to(device)
            with torch.no_grad():
                ok += (model(x)[:, kw].argmax(1) == y).sum().item()
        return ok / len(labels)

    print(f"{'시프트':>8}{'그대로':>10}" + (f"{'채널뒤집기':>12}" if args.flip else ""))
    best = (None, -1.0)
    for s in args.shift:
        a = score(s, False)
        row = f"{s:>8d}{a:>10.4f}"
        if a > best[1]:
            best = ((s, False), a)
        if args.flip:
            b = score(s, True)
            row += f"{b:>12.4f}"
            if b > best[1]:
                best = ((s, True), b)
        print(row)

    (s, f), acc = best
    print(f"\n최고 {acc:.4f}  (시프트 {s}, 채널뒤집기 {f})   랜덤은 0.100")
    if acc > 0.6:
        print("→ 두 전단이 대체로 호환된다. 남은 격차는 학습 조건 쪽을 볼 차례다.")
    elif acc > 0.25:
        print("→ 표현이 겹치지만 어긋난다. 시프트/뒤집기로 회복된 만큼이 그 축의 몫이다.")
    else:
        print("→ 사실상 안 통한다. 두 전단이 만드는 이진 이미지가 서로 다른 표현이다.\n"
              "   가중치를 옮길 수 없다는 뜻이고, 전송본은 처음부터 학습해야 한다.")


if __name__ == "__main__":
    main()
