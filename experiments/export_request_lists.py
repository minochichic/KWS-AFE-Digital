"""동료에게 넘길 `_unknown_` / `_silence_` 목록을 우리 파이프라인 그대로 뽑는다.

v3 전송본에는 키워드 10개뿐이라 12클래스 비교가 안 된다. 나머지 둘을 받아야 하는데,
**"아무거나 3,800개"로는 안 된다** -- 우리가 어떤 클립을 고르는지는 시드가 정하고,
동료가 다른 걸 고르면 `_unknown_` 의 구성이 달라져 그 클래스 정확도를 비교할 수
없다. 그래서 우리가 실제로 쓰는 목록을 그대로 뽑아 넘긴다.

두 클래스는 성격이 다르다:

  `_unknown_`  키워드 10개를 뺀 25개 단어에서 **골라낸 것**이므로 파일명 목록이면 된다.
  `_silence_`  `_background_noise_` 에서 **잘라 만든 것**이라 파일명만으로는 부족하다.
               (파일, 시작 샘플, 길이) 를 줘야 같은 오디오가 재현된다.

우리 파이프라인(`data/speech_commands.py`)의 선택 로직을 다시 구현하지 않고
**그대로 불러서** 뽑는다 -- 다시 구현하면 언젠가 어긋난다.

데이터셋이 있는 학습 박스에서 돌린다:
    python -m experiments.export_request_lists --out docs/request_v4
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="docs/request_v4")
    args = ap.parse_args()

    from train.config import load_config
    from data.speech_commands import (KEYWORDS, SpeechCommands12,
                                      _load_noise_waves, subsample_indices)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    unk_rows, sil_rows = [], []
    for split in ("train", "val", "test"):
        ds = SpeechCommands12.from_torchaudio(cfg.data, split, seed=cfg.train.seed)
        base = ds._items._base                      # torchaudio SPEECHCOMMANDS
        # _real = 키워드 전부 + 골라낸 unknown. 앞쪽 키워드 개수를 세어 뒤를 자른다.
        n_kw = sum(1 for i in ds._real
                   if Path(base._walker[i]).parent.name in set(KEYWORDS))
        for i in ds._real[n_kw:]:
            p = Path(base._walker[i])
            unk_rows.append([split, p.parent.name, p.stem])

        # silence 는 __getitem__ 이 만드는 것과 같은 시드로 크롭 위치를 재현한다.
        noise = _load_noise_waves(base)
        names = sorted(p.name for p in
                       (Path(base._path) / "_background_noise_").glob("*.wav"))
        for s in range(ds._n_silence):
            g = torch.Generator().manual_seed(ds._silence_seed + s)
            j = int(torch.randint(len(noise), (1,), generator=g))
            src = noise[j].reshape(-1)
            L = ds.sample_rate
            start = (0 if src.numel() <= L else
                     int(torch.randint(src.numel() - L + 1, (1,), generator=g)))
            sil_rows.append([split, names[j], start, L])

        print(f"{split:5s}  키워드 {n_kw:6,}  unknown {len(ds._real)-n_kw:5,}  "
              f"silence {ds._n_silence:5,}")

    up = out / "unknown_list.csv"
    with up.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "word", "clip"])
        w.writerows(unk_rows)

    sp = out / "silence_crops.csv"
    with sp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "noise_file", "start_sample", "length_samples"])
        w.writerows(sil_rows)

    print(f"\n저장: {up}  ({len(unk_rows):,} 클립)")
    print(f"      {sp}  ({len(sil_rows):,} 크롭)")
    print("\n동료는 이 두 파일대로 돌리면 우리와 **정확히 같은 클립**이 된다.")
    print("silence 는 (파일, 시작샘플) 로 잘라야 재현된다 -- 파일명만으로는 부족하다.")


if __name__ == "__main__":
    main()
