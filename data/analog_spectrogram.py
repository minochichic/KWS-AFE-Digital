"""동료 아날로그 모델이 뽑아준 이진 스펙트로그램을 그대로 읽는다.

`data/afe.py` 는 회로를 **소프트웨어로 흉내내서** 이진 이미지를 만든다. 여기서는
그럴 필요가 없다 -- 동료가 회로를 전달함수로 만들어 클립마다 `[16, 100]` 이진
이미지를 이미 뽑아놨다. 그래서 이 경로에는 AFE 가 없다:

    CSV [16, 100] {0,1}  ->  {-1,+1}  ->  T=128 로 패딩  ->  BinaryMatchboxNet

우리 시뮬이 근사하던 것들이 여기서는 **원본**이다: 검출기 비선형(PCHIP), C3 평활,
비교기 임계값(0.97~0.99 V, 5 mV 양자화)이 전부 아날로그 쪽에서 이미 적용됐다.
따라서 `normalize` / `compression` / `comparator_vos` / `threshold_*` 는
이 경로에서 **읽히지 않는다**.

## 클래스 수는 전송본마다 다르다

첫 전송본에는 키워드 10개뿐이라 `bd_base` 의 0.825 와 직접 비교할 수 없었다.
`vthr_762b7b0c1885` (2026-08-31) 부터 `silence/` 와 `unknown/` 이 같은 체인으로
들어와서 12클래스가 됐다. 둘 다 있으면 12클래스로, 없으면 10클래스로 읽는다.

## ⚠️ silence/unknown 은 공식 분할을 못 쓴다

키워드 10개는 파일명이 원본 wav 와 같아서 공식 목록
(`validation_list.txt` / `testing_list.txt`)에 정확히 대응된다. 나머지 둘은 아니다:

  unknown/  `backward_0000.csv` 처럼 **재명명**돼 화자 해시가 사라졌다. 공식 분할은
            화자 단위로 나누는데 그 근거가 없으니 파일명 해시로 80:10:10 한다.
            **같은 화자가 train 과 test 에 동시에 들어갈 수 있다** -- unknown 정확도가
            우리 baseline(0.644, 최악 클래스)보다 크게 높게 나오면 회로가 좋아서가
            아니라 이 누수일 가능성을 먼저 의심할 것.
  silence/  `doing_the_dishes_0000.csv` -- 배경소음 6개 파일을 자른 것이다. 우리
            baseline 도 같은 6개에서 매 split 을 잘라 쓰므로(speech_commands.py 의
            synthesize_silence) 이 점은 새 문제가 아니다.

두 클래스는 split 마다 `unknown_fraction` / `silence_fraction` 만큼으로 다시
줄인다. 동료는 각각 4001개를 줬는데 그대로 쓰면 키워드 한 개의 클래스보다 커져서
baseline 과 클래스 균형이 달라지고, 0.825 와 같은 축에 놓을 수 없게 된다.
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

from data.speech_commands import (KEYWORDS, SILENCE_INDEX, UNKNOWN_INDEX,
                                  subsample_indices)

N_CH, N_FRAMES = 16, 100

# 동료 전송본의 디렉터리 이름 -> 우리 클래스 번호. 우리 쪽 라벨은 `_silence_` /
# `_unknown_` 이지만 전송본은 밑줄 없이 쓴다.
_EXTRA_DIRS = {"silence": SILENCE_INDEX, "unknown": UNKNOWN_INDEX}


def row_order(root: Path) -> Optional[List[int]]:
    """export_settings.json 의 row_order -> CSV 행 -> 채널번호 매핑.

    v3 전송본은 행을 **역순**으로 쓴다 (행0 = ch15). 그대로 읽으면 우리 모델의
    ch0(최저역) 자리에 최고역이 들어간다. 하드코딩하지 않고 매니페스트에서 읽는다
    -- 다음 전송본이 순서를 바꿔도 조용히 틀리면 안 된다.
    """
    import json
    text = None
    if root.is_dir():
        f = root / "export_settings.json"
        if f.is_file():
            text = f.read_text()
    else:
        with tarfile.open(root, "r:*") as tf:
            for m in tf:
                if m.name.endswith("export_settings.json") and m.isfile():
                    buf = tf.extractfile(m)
                    if buf is not None:
                        text = buf.read().decode("utf-8")
                    break
    if text is None:
        return None
    names = (json.loads(text).get("matrix") or {}).get("row_order")
    if not names:
        return None
    if len(names) != N_CH:
        raise ValueError(f"row_order 가 {len(names)}개다; {N_CH}개여야 한다.")
    # "ch07" -> 7. perm[i] = i번째 CSV 행이 담고 있는 채널 번호.
    return [int(str(n).lower().lstrip("ch")) for n in names]


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
        if word in idx:
            label = idx[word]
        elif word in _EXTRA_DIRS:
            label = _EXTRA_DIRS[word]
        else:
            raise ValueError(
                f"{root} 안의 '{word}' 는 KEYWORDS 에도 {sorted(_EXTRA_DIRS)} 에도 "
                f"없다. 디렉터리 이름이 클래스와 어떻게 대응되는지 먼저 정해야 한다.")
        items.append((f, label, key))
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
        want = row_order(root) or list(range(N_CH))
        if list(z["keys"]) == keys and list(z.get("perm", want)) == want:
            return z["bits"], z["labels"], keys
        print(f"[analog] 캐시가 현재 파일 목록과 다르다 -> 다시 읽는다 ({cache_path})")

    perm = row_order(root)
    if perm is not None and perm != list(range(N_CH)):
        print(f"[analog] row_order 가 표준이 아니다 -> 재배열한다 (행0 = ch{perm[0]:02d})")
    print(f"[analog] CSV {len(items)}개 읽는 중 ({'디렉터리' if root.is_dir() else 'tar'}) ...")
    # 라벨은 _scan 이 이미 정했다 (silence/unknown 포함). 여기서 다시 매기면
    # 두 곳이 어긋날 수 있다.
    order = {k: (i, lab) for i, (_, lab, k) in enumerate(items)}
    bits = np.zeros((len(items), N_CH, N_FRAMES), dtype=np.uint8)
    labels = np.zeros(len(items), dtype=np.int64)
    seen = 0
    short = 0
    for key, word, a in _iter_source(root):
        if a.shape[0] != N_CH:
            raise ValueError(f"{key} 의 행이 {a.shape[0]}개다; {N_CH}개여야 한다.")
        if a.shape[1] > N_FRAMES:
            raise ValueError(
                f"{key} 가 {a.shape[1]}열이다; {N_FRAMES}열을 넘을 수 없다 "
                f"(1초 = 100 bin).")
        if a.shape[1] < N_FRAMES:
            # 클립 원래 타이밍 export 는 열 수가 소스 길이를 따른다
            # ("columns": "source_duration_10ms_bins"). 짧은 wav 는 **뒤가** 비므로
            # 뒤로 채운다 -- data/speech_commands.py 의 _fix 가 파형에 하는 것과
            # 같다 (F.pad(wave, (0, length - n))). 가운데 정렬로 채우면 시간축이
            # 원본과 어긋난다.
            a = np.pad(a, ((0, 0), (0, N_FRAMES - a.shape[1])))
            short += 1
        if perm is not None:
            out = np.empty_like(a)
            out[perm] = a                 # CSV 행 j -> 채널 perm[j]
            a = out
        i, lab = order[key]               # tar 순서와 정렬 순서가 달라도 맞춘다
        bits[i], labels[i] = a, lab
        seen += 1
    if seen != len(items):
        raise ValueError(f"{seen}개만 읽혔다 (목록은 {len(items)}개).")
    if short:
        print(f"[analog] {short:,}개가 {N_FRAMES}열 미만이라 뒤를 0 으로 채웠다 "
              f"({short/len(items)*100:.1f}%)")
    if cache:
        try:
            np.savez_compressed(cache_path, bits=bits, labels=labels,
                                keys=np.array(keys),
                                perm=np.array(perm if perm else list(range(N_CH))))
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


def _hash_split(key: str) -> str:
    """공식 목록에 없는 클립을 80:10:10 으로. 파일명만 보고 정한다.

    `hash()` 를 쓰면 PYTHONHASHSEED 때문에 세션마다 분할이 달라진다
    (speech_commands.py 의 _SPLIT_OFFSET 주석과 같은 함정). md5 는 고정이다.
    """
    import hashlib
    v = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100
    return "val" if v < 10 else ("test" if v < 20 else "train")


def build_analog_dataloaders(cfg_data, batch_size: int, target_T: int = 128,
                             num_workers: Optional[int] = None, seed: int = 0):
    """공식 분할대로 (train, val, test) DataLoader.

    키워드는 공식 목록을 그대로 쓴다. silence/unknown 은 목록에 없어서 그냥
    두면 **전부 train 으로 떨어져 val/test 에 두 클래스가 0개**가 된다 -- 조용히
    틀리는 쪽이라 해시로 나눈다. 그 뒤 split 마다 `silence_fraction` /
    `unknown_fraction` 만큼으로 줄여 baseline 과 클래스 균형을 맞춘다.
    """
    from data.speech_commands import _SPLIT_OFFSET

    bits, labels, keys = load_all(cfg_data.analog_csv_root)
    where = _split_lists(cfg_data.root)
    words = [k.split("/", 1)[0] for k in keys]
    split = np.array([where[k] if k in where else
                      (_hash_split(k) if w in _EXTRA_DIRS else "train")
                      for k, w in zip(keys, words)])
    extra = np.array([w in _EXTRA_DIRS for w in words])

    keep = np.ones(len(keys), dtype=bool)
    frac = {SILENCE_INDEX: cfg_data.silence_fraction,
            UNKNOWN_INDEX: cfg_data.unknown_fraction}
    for name in ("train", "val", "test"):
        in_split = split == name
        n_keyword = int((in_split & ~extra).sum())
        # speech_commands.py 와 같은 규칙: split 마다 다른, 그러나 프로세스 사이에서
        # 재현되는 시드.
        split_seed = seed + 100003 * _SPLIT_OFFSET.get(name, 0)
        for label, f in frac.items():
            idx = np.flatnonzero(in_split & (labels == label))
            if idx.size == 0:
                continue
            n_keep = round(n_keyword * f)
            pick = subsample_indices(idx.size, n_keep, split_seed + label)
            drop = np.setdiff1d(idx, idx[pick])
            keep[drop] = False

    loaders = []
    nw = cfg_data.num_workers if num_workers is None else num_workers
    names12 = KEYWORDS + ["_silence_", "_unknown_"]
    for name in ("train", "val", "test"):
        m = (split == name) & keep
        ds = AnalogSpectrograms(bits[m], labels[m], target_T)
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=(name == "train"), num_workers=nw,
                                  drop_last=False))
        cnt = np.bincount(labels[m], minlength=len(names12))
        print(f"[analog] {name:5s} {int(m.sum()):6d} clips  "
              + " ".join(f"{names12[i][:4]}={c}" for i, c in enumerate(cnt) if c))
    return tuple(loaders)
