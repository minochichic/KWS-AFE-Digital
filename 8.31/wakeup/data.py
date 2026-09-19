"""Google Speech Commands v2 — 대상 단어 대 나머지(이진).

12-class 분류가 아니라 검출 과제이므로 라벨은 두 개뿐이다.
  1 = 대상 단어
  0 = 그 밖의 모든 것 (다른 34개 단어 + 무음/소음)

음성(비대상)이 압도적으로 많다. 그대로 쓰면 "항상 0"이 좋은 해가 되므로
비율을 config 로 맞추고, 손실에도 양성 가중을 준다.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SR = 16000
_SPLIT = {"train": "training", "val": "validation", "test": "testing"}
# 분할마다 다른 부분표본을 뽑되 세션 간에는 재현되도록
_SPLIT_OFFSET = {"train": 1, "val": 2, "test": 3}


def _fix_length(w: torch.Tensor, n: int) -> torch.Tensor:
    w = w.reshape(-1)
    if w.numel() > n:
        return w[:n]
    if w.numel() < n:
        return F.pad(w, (0, n - w.numel()))
    return w


class TargetVsRest(Dataset):
    """(wave [L], y) — y 는 0/1 float."""

    def __init__(self, items: Sequence[Tuple], noise: Sequence[torch.Tensor],
                 target: str, split: str, *, neg_per_pos: float = 4.0,
                 silence_fraction: float = 0.15, seed: int = 0,
                 clip_ms: float = 1000.0, augment: bool = False,
                 noise_snr_db: Tuple[float, float] = (5.0, 20.0),
                 shift_ms: float = 100.0) -> None:
        self.items = items
        self.noise = list(noise)
        self.n = int(round(SR * clip_ms / 1000.0))
        self.augment = augment
        self.snr = noise_snr_db
        self.shift = int(round(SR * shift_ms / 1000.0))

        pos, neg = [], []
        for i, word in enumerate(self._words(items)):
            (pos if word == target else neg).append(i)
        if not pos:
            raise ValueError(
                f"대상 단어 {target!r} 가 {split} 분할에 없다. "
                "Speech Commands 의 35개 단어 중 하나여야 한다.")

        rng = random.Random(seed + 100003 * _SPLIT_OFFSET.get(split, 0))
        rng.shuffle(neg)
        n_neg = min(len(neg), int(round(len(pos) * neg_per_pos)))
        self.pos, self.neg = pos, neg[:n_neg]
        self.n_silence = int(round(len(pos) * silence_fraction))
        self.sil_seed = seed + 777

    @staticmethod
    def _words(items) -> List[str]:
        return [it[1] for it in items]

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg) + self.n_silence

    def _aug(self, w: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        # 시간 이동 — START 가 어디서 뜨든 견디게 한다
        if self.shift > 0:
            s = int(torch.randint(-self.shift, self.shift + 1, (1,),
                                  generator=g).item())
            w = torch.roll(w, s)
        # 잡음 혼합 — 실제 환경의 SNR 범위
        if self.noise and torch.rand(1, generator=g).item() < 0.5:
            nz = self.noise[int(torch.randint(len(self.noise), (1,),
                                              generator=g).item())]
            if nz.numel() > self.n:
                o = int(torch.randint(nz.numel() - self.n, (1,),
                                      generator=g).item())
                nz = nz[o:o + self.n]
            nz = _fix_length(nz, self.n)
            snr = self.snr[0] + torch.rand(1, generator=g).item() * (
                self.snr[1] - self.snr[0])
            ps, pn = w.pow(2).mean(), nz.pow(2).mean().clamp(min=1e-12)
            w = w + nz * torch.sqrt(ps / (pn * 10 ** (snr / 10)))
        return w

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_real = len(self.pos) + len(self.neg)
        if i < n_real:
            j = self.pos[i] if i < len(self.pos) else self.neg[i - len(self.pos)]
            w = _fix_length(self.items[j][0], self.n)
            y = 1.0 if i < len(self.pos) else 0.0
            if self.augment:
                w = self._aug(w, torch.Generator().manual_seed(self.sil_seed + i))
        else:                                   # 무음/소음 = 음성
            g = torch.Generator().manual_seed(self.sil_seed + i)
            if self.noise:
                nz = self.noise[int(torch.randint(len(self.noise), (1,),
                                                  generator=g).item())]
                o = int(torch.randint(max(1, nz.numel() - self.n), (1,),
                                      generator=g).item())
                w = _fix_length(nz[o:o + self.n], self.n)
                w = w * torch.rand(1, generator=g).item()
            else:
                w = torch.zeros(self.n)
            y = 0.0
        return w, torch.tensor(y)


# --------------------------------------------------------------------- 적재
class _Lazy(Sequence):
    """torchaudio SPEECHCOMMANDS -> (wave, word) 시퀀스. 클립은 필요할 때 읽는다."""

    def __init__(self, base) -> None:
        self._b = base

    def __len__(self) -> int:
        return len(self._b)

    def __getitem__(self, i):
        it = self._b[i]
        return it[0], it[2]      # (wave, sr, label, speaker, utt)


def _noise_waves(base) -> List[torch.Tensor]:
    import torchaudio
    d = Path(base._path) / "_background_noise_"
    out = []
    if d.is_dir():
        for p in sorted(d.glob("*.wav")):
            w, sr = torchaudio.load(str(p))
            out.append(w.reshape(-1))
    return out


def build(root: str, target: str, split: str, *, seed: int = 0,
          augment: bool = False, **kw) -> TargetVsRest:
    import torchaudio
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS(
        root=root, url="speech_commands_v0.02", download=True,
        subset=_SPLIT[split])
    return TargetVsRest(_Lazy(base), _noise_waves(base), target, split,
                        seed=seed, augment=augment, **kw)


def loaders(root: str, target: str, batch_size: int, *, seed: int = 0,
            num_workers: int = 4, **kw):
    out = {}
    for sp in ("train", "val", "test"):
        ds = build(root, target, sp, seed=seed, augment=(sp == "train"), **kw)
        out[sp] = DataLoader(ds, batch_size=batch_size, shuffle=(sp == "train"),
                             num_workers=num_workers, drop_last=(sp == "train"),
                             pin_memory=True)
    return out


def raw_batch(root: str, split: str, n: int, *, seed: int = 0,
              clip_ms: float = 1000.0):
    """(wave [n, L], words [n]) — 단어 라벨을 그대로 돌려준다.

    여러 대상 단어를 비교할 때 쓴다. 단어마다 파일을 다시 읽지 않고 같은
    클립 묶음에 라벨만 바꿔 달면 되므로 훨씬 빠르고, 비교도 공정해진다.
    """
    import random as _r
    import torchaudio
    root = os.path.expanduser(root)
    base = torchaudio.datasets.SPEECHCOMMANDS(
        root=root, url="speech_commands_v0.02", download=True,
        subset=_SPLIT[split])
    idx = list(range(len(base)))
    _r.Random(seed).shuffle(idx)
    idx = idx[:n]
    L = int(round(SR * clip_ms / 1000.0))
    waves, words = [], []
    for i in idx:
        it = base[i]
        waves.append(_fix_length(it[0], L))
        words.append(it[2])
    return torch.stack(waves), words
