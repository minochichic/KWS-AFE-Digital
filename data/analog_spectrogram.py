"""동료 아날로그 모델이 뽑아준 이진 스펙트로그램을 그대로 읽는다.

`data/afe.py` 는 회로를 **소프트웨어로 흉내내서** 이진 이미지를 만든다. 여기서는
그럴 필요가 없다 -- 동료가 회로를 전달함수로 만들어 클립마다 `[16, 100]` 이진
이미지를 이미 뽑아놨다. 그래서 이 경로에는 AFE 가 없다:

    CSV [16, 100] {0,1}  ->  {-1,+1}  ->  T=128 로 패딩  ->  BinaryMatchboxNet

우리 시뮬이 근사하던 것들이 여기서는 **원본**이다: 검출기 비선형(PCHIP), C3 평활,
비교기 임계값(0.97~0.99 V, 5 mV 양자화)이 전부 아날로그 쪽에서 이미 적용됐다.
따라서 `normalize` / `compression` / `comparator_vos` / `threshold_*` 는
이 경로에서 **읽히지 않는다**.

## ⚠️ 10 클래스다, 12 가 아니다

패키지에 키워드 10개만 있다 -- `_silence_` 도 `_unknown_` 도 없다. 그러므로
여기서 나온 정확도는 **`bd_base` 의 0.825 와 직접 비교할 수 없다.** 12클래스에서
`_unknown_` 은 우리 최악(0.644)이고 `_silence_` 는 최고(0.993)라, 둘을 빼면
숫자가 어느 쪽으로 움직일지 계산으로 정해지지 않는다. 비교하려면 동료에게
그 두 종류도 같은 체인으로 뽑아달라고 해야 한다.

분할은 Speech Commands 공식 목록(`validation_list.txt` / `testing_list.txt`)을
그대로 쓴다. 파일명이 원본 wav 와 같으므로 정확히 대응되고, 그래야 기존 기록과
같은 축에 놓인다.
"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.speech_commands import KEYWORDS

N_CH, N_FRAMES = 16, 100


def _split_lists(gsc_root: str) -> Dict[str, str]:
    """"<word>/<clip>" -> "val" | "test". 목록에 없으면 train 이다."""
    leaf = Path(os.path.expanduser(gsc_root)) / "SpeechCommands" / "speech_commands_v0.02"
    out: Dict[str, str] = {}
    for name, split in (("validation_list.txt", "val"), ("testing_list.txt", "test")):
        p = leaf / name
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} 가 없다. 아날로그 CSV 는 파일명으로 공식 분할에 대응시키므로 "
                f"원본 Speech Commands 트리가 필요하다 (data.root 확인).")
        for line in p.read_text().split():
            out[line.strip().removesuffix(".wav")] = split
    return out


def _iter_source(root: Path):
    """(키, 단어, [16,100] uint8) 을 순서대로. 디렉터리도 tar.gz 도 받는다.

    CSV 38,546개를 낱개로 커밋하면 161 MB / 38k 오브젝트라 clone 과 status 가
    영구히 느려진다. tar.gz 는 5.8 MB 한 덩어리이고 순차 읽기가 오히려 빠르다
    (3초 vs 6초). 그래서 리포에는 tar 를 넣고 여기서 바로 읽는다.
    """
    if root.is_dir():
        for f in sorted(root.glob("*/*.csv")):
            yield f"{f.parent.name}/{f.stem}", f.parent.name, np.loadtxt(
                f, delimiter=",", dtype=np.uint8, ndmin=2)
        return
    with tarfile.open(root, "r:*") as tf:
        for m in tf:
            # macOS 가 만드는 AppleDouble(._foo) 은 CSV 가 아니다.
            name = Path(m.name)
            if not m.isfile() or name.suffix != ".csv" or name.name.startswith("._"):
                continue
            buf = tf.extractfile(m)
            if buf is None:
                continue
            yield (f"{name.parent.name}/{name.stem}", name.parent.name,
                   np.loadtxt(io.BytesIO(buf.read()), delimiter=",",
                              dtype=np.uint8, ndmin=2))


def _scan(csv_root: str) -> List[Tuple[Path, int, str]]:
    """(csv 경로, 라벨, "<word>/<clip>") 목록. 라벨은 KEYWORDS 순서를 따른다."""
    root = Path(os.path.expanduser(csv_root))
    if not root.exists():
        raise FileNotFoundError(f"analog_csv_root={root} 가 없다.")
    idx = {w: i for i, w in enumerate(KEYWORDS)}
    items: List[Tuple[Path, int, str]] = []
    if root.is_dir():
        names = [(f, f"{f.parent.name}/{f.stem}") for f in sorted(root.glob("*/*.csv"))]
    else:
        with tarfile.open(root, "r:*") as tf:
            names = []
            for m in tf.getmembers():
                n = Path(m.name)
                if (m.isfile() and n.suffix == ".csv"
                        and not n.name.startswith("._")):
                    names.append((n, f"{n.parent.name}/{n.stem}"))
        names.sort(key=lambda t: t[1])
    for f, key in names:
        word = key.split("/", 1)[0]
        if word not in idx:
            raise ValueError(
                f"{root} 안의 '{word}' 는 KEYWORDS 에 없다. 이 로더는 키워드 10개만 "
                f"다룬다 -- _silence_/_unknown_ 이 들어오면 클래스 정의를 먼저 정해야 "
                f"한다.")
        items.append((f, idx[word], key))
    if not items:
        raise ValueError(f"{root} 에서 CSV 를 하나도 못 찾았다.")
    return items


def load_all(csv_root: str, cache: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """모든 클립을 [N, 16, 100] uint8 로. 38k 개를 매 에폭 파싱할 수는 없다."""
    root = Path(os.path.expanduser(csv_root))
    cache_path = (root / "_cache_bits.npz" if root.is_dir()
                  else root.parent / f"_cache_{root.name.split('.')[0]}.npz")
    items = _scan(csv_root)
    keys = [k for _, _, k in items]
    if cache and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=False)
        if list(z["keys"]) == keys:
            return z["bits"], z["labels"], keys
        print(f"[analog] 캐시가 현재 파일 목록과 다르다 -> 다시 읽는다 ({cache_path})")

    print(f"[analog] CSV {len(items)}개 읽는 중 ({'디렉터리' if root.is_dir() else 'tar'}) ...")
    idx = {w: i for i, w in enumerate(KEYWORDS)}
    order = {k: i for i, (_, _, k) in enumerate(items)}
    bits = np.zeros((len(items), N_CH, N_FRAMES), dtype=np.uint8)
    labels = np.zeros(len(items), dtype=np.int64)
    seen = 0
    for key, word, a in _iter_source(root):
        if a.shape != (N_CH, N_FRAMES):
            raise ValueError(f"{key} 가 {a.shape} 이다; ({N_CH}, {N_FRAMES}) 여야 한다.")
        i = order[key]                    # tar 순서와 정렬 순서가 달라도 맞춘다
        bits[i], labels[i] = a, idx[word]
        seen += 1
    if seen != len(items):
        raise ValueError(f"{seen}개만 읽혔다 (목록은 {len(items)}개).")
    if cache:
        try:
            np.savez_compressed(cache_path, bits=bits, labels=labels,
                                keys=np.array(keys))
            print(f"[analog] 캐시 저장: {cache_path}")
        except OSError as e:                       # 읽기 전용 매체 등
            print(f"[analog] 캐시 저장 실패(무시): {e}")
    return bits, labels, keys


class AnalogSpectrograms(Dataset):
    """[16, T] {-1,+1} 텐서와 라벨. AFE 없음 -- 이미 이진이다."""

    def __init__(self, bits: np.ndarray, labels: np.ndarray, target_T: int = 128):
        self.bits, self.labels, self.target_T = bits, labels, target_T

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.bits[i].astype(np.float32)) * 2.0 - 1.0
        if self.target_T != x.shape[-1]:
            from data.afe import pad_or_crop        # 같은 규칙(-1 패딩)을 쓴다
            x = pad_or_crop(x.unsqueeze(0), self.target_T, pad_value=-1.0)[0]
        return x, int(self.labels[i])


def build_analog_dataloaders(cfg_data, batch_size: int, target_T: int = 128,
                             num_workers: Optional[int] = None):
    """공식 분할대로 (train, val, test) DataLoader."""
    bits, labels, keys = load_all(cfg_data.analog_csv_root)
    where = _split_lists(cfg_data.root)
    split = np.array([where.get(k, "train") for k in keys])

    loaders = []
    nw = cfg_data.num_workers if num_workers is None else num_workers
    for name in ("train", "val", "test"):
        m = split == name
        ds = AnalogSpectrograms(bits[m], labels[m], target_T)
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=(name == "train"), num_workers=nw,
                                  drop_last=False))
        print(f"[analog] {name:5s} {int(m.sum()):6d} clips")
    return tuple(loaders)
