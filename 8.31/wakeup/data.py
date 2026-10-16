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
                 shift_ms: float = 100.0,
                 extra_pos: Optional[Sequence[torch.Tensor]] = None,
                 enroll_frac: float = 0.3,
                 enroll_only: bool = False,
                 extra_neg: Optional[Sequence[torch.Tensor]] = None,
                 neg_frac: float = 0.5,
                 burst_fraction: float = 0.0,
                 burst_ms: Tuple[float, float] = (8.0, 100.0)) -> None:
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

        # 시연자 녹음(enrollment). 데이터셋 양성이 수천 개인데 녹음은 수십 개
        # 뿐이라 그대로 섞으면 묻힌다. 양성의 enroll_frac 만큼을 차지하도록
        # 반복해 넣는다. 증강(시간 이동·잡음)은 train 분할에서 그대로 걸린다.
        self.extra = list(extra_pos or [])
        self.n_extra = 0
        if self.extra and enroll_only:
            # 화자 종속 모드: 데이터셋 양성을 **버리고** 시연자 녹음만 양성으로
            # 쓴다. 섞어 쓰면(enroll_frac) 40개뿐인 파라미터가 "데이터셋 일반화"
            # 와 "한 사람"을 동시에 못 맞추고 둘 다 놓친다(실측: 시험 TPR
            # 0.575 -> 0.373, s2 가 음성의 58% 를 통과시키며 판별을 포기).
            # 상용 기기의 enrollment 도 결국 사용자에게 맞추는 것이다.
            # 양성 개수는 그대로 두어 배치 구성이 종전과 같게 한다.
            self.n_extra = max(len(self.extra), len(pos))
            self.pos = []
        elif self.extra and enroll_frac > 0:
            want = int(round(len(pos) * enroll_frac / (1.0 - enroll_frac)))
            self.n_extra = max(len(self.extra), want)

        # 시연자가 직접 녹음한 **음성**. 이게 왜 중요한가: 양성이 시연자의
        # 마이크·방에서 나오고 음성이 Speech Commands 에서만 나오면, 단어를
        # 배우는 것보다 "이 녹음 채널"을 배우는 게 더 쉬워서 학습이 그쪽으로
        # 샌다(실측: 형판이 전 채널 ON 으로 수렴, 스트림 FA/h 6000).
        # 같은 마이크·같은 방의 음성을 섞으면 그 지름길이 막힌다.
        self.extra_neg = list(extra_neg or [])
        self.n_extra_neg = 0
        if self.extra_neg and neg_frac > 0:
            want = int(round(len(self.neg) * neg_frac / (1.0 - neg_frac)))
            self.n_extra_neg = max(len(self.extra_neg), want)

        self.n_silence = int(round(len(pos) * silence_fraction))
        self.sil_seed = seed + 777
        # 짧은 소리 + 무음 (음성). TrainConfig.burst_neg 주석 참고.
        self.n_burst = int(round(len(pos) * burst_fraction))
        self.burst_len = (int(round(SR * burst_ms[0] / 1000.0)),
                          int(round(SR * burst_ms[1] / 1000.0)))

    def _burst(self, i: int) -> torch.Tensor:
        """다른 단어(3/4)나 잡음(1/4)을 8~100 ms 잘라 약한 배경 위에 놓는다.

        단어에서는 에너지가 가장 큰 구간을 자른다 -- 아무 데나 자르면 무음
        조각이 나와 START 조차 안 서서 아무것도 가르치지 못한다. 잡음 조각은
        같은 단어 조각의 RMS 에 맞춘다. 대상 단어는 쓰지 않는다(/ʃ/ 조각이
        음성이 되면 양성과 정면으로 부딪힌다).
        """
        g = torch.Generator().manual_seed(self.sil_seed + 31337 + i)
        n = self.n

        def rnd() -> float:
            return float(torch.rand(1, generator=g))

        w = _fix_length(self.items[self.neg[int(torch.randint(
            len(self.neg), (1,), generator=g))]][0], n)
        L = self.burst_len[0] + int(rnd() * (self.burst_len[1] - self.burst_len[0]))
        L = max(1, min(L, n // 2))
        e = torch.nn.functional.avg_pool1d(w.pow(2).view(1, 1, -1), L, 1).view(-1)
        c = int(e.argmax())
        frag = w[c:c + L].clone()
        if self.noise and rnd() < 0.25:
            nz = self.noise[int(torch.randint(len(self.noise), (1,), generator=g))]
            o = int(torch.randint(max(1, nz.numel() - L), (1,), generator=g))
            nf = _fix_length(nz[o:o + L], L)
            frag = nf * frag.pow(2).mean().sqrt() / nf.pow(2).mean().sqrt().clamp(min=1e-9)
        frag = frag * (0.5 + 1.5 * rnd())               # 0.5~2 배
        base = torch.zeros(n)
        if self.noise and rnd() < 0.5:                   # 무음 클래스와 같은 약한 배경
            nz = self.noise[int(torch.randint(len(self.noise), (1,), generator=g))]
            o = int(torch.randint(max(1, nz.numel() - n), (1,), generator=g))
            base = _fix_length(nz[o:o + n], n) * rnd() * 0.1
        p = int(n * (0.1 + 0.6 * rnd()))
        base[p:p + L] += frag[:max(0, min(L, n - p))]
        return base

    @staticmethod
    def _words(items) -> List[str]:
        return [it[1] for it in items]

    def __len__(self) -> int:
        return (len(self.pos) + self.n_extra + len(self.neg)
                + self.n_extra_neg + self.n_burst + self.n_silence)

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
        n_pos, n_ex = len(self.pos), self.n_extra
        n_neg, n_exn = len(self.neg), self.n_extra_neg
        n_real = n_pos + n_ex + n_neg + n_exn
        if i < n_real:
            if i < n_pos:                      # 데이터셋 양성
                w = _fix_length(self.items[self.pos[i]][0], self.n)
                y = 1.0
            elif i < n_pos + n_ex:             # 시연자 녹음 양성 (반복)
                w = _fix_length(self.extra[(i - n_pos) % len(self.extra)], self.n)
                y = 1.0
            elif i < n_pos + n_ex + n_neg:     # 데이터셋 음성
                w = _fix_length(self.items[self.neg[i - n_pos - n_ex]][0], self.n)
                y = 0.0
            else:                              # 시연자 녹음 음성 (반복)
                j = (i - n_pos - n_ex - n_neg) % len(self.extra_neg)
                w = _fix_length(self.extra_neg[j], self.n)
                y = 0.0
            if self.augment:
                w = self._aug(w, torch.Generator().manual_seed(self.sil_seed + i))
        elif i < n_real + self.n_burst:         # 짧은 소리 + 무음 = 음성
            w = self._burst(i)
            y = 0.0
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


def positive_clip_rms(root: str, target: str, split: str = "train",
                      n: int = 400) -> float:
    """데이터셋 대상 클립의 RMS 중앙값. enrollment 레벨을 맞출 기준이다."""
    ds = build(root, target, split)
    vals = []
    for i in range(min(n, len(ds.pos))):
        w = _fix_length(ds.items[ds.pos[i]][0], ds.n)
        vals.append(float(w.pow(2).mean().sqrt()))
    vals.sort()
    return vals[len(vals) // 2] if vals else 0.05


def load_enroll(path: str, *, clip_ms: float = 1000.0,
                target_rms: Optional[float] = None,
                verbose: bool = True) -> List[torch.Tensor]:
    """녹음 폴더(또는 파일)를 발화 단위 1초 클립으로 만든다.

    target_rms 가 주어지면 녹음 전체에 **하나의 이득**을 걸어 발화 RMS 의
    중앙값을 그 값에 맞춘다. 발화마다 따로 정규화하지 않는 이유: 실제 회로는
    PreAMP 이득이 고정이라 발화 간 크기 차이를 그대로 받는다. 그 변동은
    남겨 두고 체계적인 레벨 차이만 없앤다.

    이걸 안 하면 enrollment 클립이 데이터셋보다 조용해 START 를 못 만들고,
    학습에서 양성의 30% 가 통째로 버려진다(실측: START 검출률 94% -> 73.8%).
    """
    from pathlib import Path as _P
    from .continuous import split_utterances
    p = _P(os.path.expanduser(path))
    files = sorted(p.glob("*.wav")) if p.is_dir() else [p]
    out: List[torch.Tensor] = []
    for f in files:
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(f))
            wav = wav.mean(0)
        except Exception:
            import wave as _w
            import numpy as _np
            with _w.open(str(f), "rb") as h:
                sr = h.getframerate()
                a = _np.frombuffer(h.readframes(h.getnframes()),
                                   dtype=_np.int16).astype(_np.float32) / 32768
                if h.getnchannels() > 1:
                    a = a.reshape(-1, h.getnchannels()).mean(1)
            wav = torch.from_numpy(a.copy())
        if sr != SR:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav / wav.abs().max().clamp(min=1e-9) * 0.25
        clips = split_utterances(wav, clip_s=clip_ms / 1000.0)
        if target_rms and clips:
            r = sorted(float(c.pow(2).mean().sqrt()) for c in clips)
            med = r[len(r) // 2]
            g = target_rms / max(med, 1e-9)
            clips = [c * g for c in clips]
            if verbose:
                print(f"    {f.name}: 발화 {len(clips)}개, RMS 중앙 "
                      f"{med:.4f} -> {target_rms:.4f} (이득 {g:.2f}배)")
        out += clips
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


def split_extra(extra_pos, enroll_only: bool, *, verbose: bool = True) -> dict:
    """시연자 녹음을 어느 분할에 넣을지 정한다. loaders 에서 떼어낸 이유는
    데이터셋 없이도 이 규칙만 따로 시험하기 위해서다."""
    if not extra_pos:
        return {"train": None, "val": None, "test": None}
    if not enroll_only:
        return {"train": list(extra_pos), "val": None, "test": None}
    n = len(extra_pos)
    if n < 6:                                  # 셋으로 못 나눈다
        if verbose:
            print(f"  경고: 녹음이 {n}발화뿐이라 train/val/test 를 나누지 못했다. "
                  "val/test TPR 은 학습 정확도이지 일반화가 아니다.")
        return {"train": list(extra_pos), "val": list(extra_pos),
                "test": list(extra_pos)}
    a, b = int(round(n * 0.6)), int(round(n * 0.8))
    tr, va, te = (list(extra_pos[:a]), list(extra_pos[a:b]),
                  list(extra_pos[b:]))
    if verbose:
        print(f"  화자 종속: 학습 {len(tr)} / 검증 {len(va)} / 시험 {len(te)}발화, "
              f"셋 다 겹치지 않는다. 데이터셋 양성은 쓰지 않는다.")
        print("  * 이 세 분할은 모두 **한 녹음**에서 나왔다. 마이크·방·날이 같아서"
              " 시험 TPR 은 낙관적이다.")
        print("  * 진짜 시험은 따로 녹음한 파일을 --eval-wav 로 재는 것뿐이다.")
    return {"train": tr, "val": va, "test": te}


def loaders(root: str, target: str, batch_size: int, *, seed: int = 0,
            num_workers: int = 4, extra_pos=None, enroll_only: bool = False,
            extra_neg=None, **kw):
    """extra_pos(시연자 녹음)를 어디에 넣을지는 모드에 따라 다르다.

    섞어 쓰는 모드(enroll_frac)에서는 **train 분할에만** 넣는다. val/test 에
    섞으면 자기 목소리로 자기를 평가하게 되어 숫자가 거짓이 된다.

    화자 종속 모드(enroll_only)에서는 데이터셋 양성이 아예 빠지므로 val 에도
    양성이 있어야 한다 -- 없으면 TPR 이 정의되지 않아 체크포인트를 고를 수가
    없다. 그래서 녹음을 train 3 : val 1 로 **겹치지 않게** 쪼갠다. 진짜
    시험은 따로 녹음한 voice/test 를 --eval-wav 로 재는 것이고, 그 파일은
    여기 어디에도 들어가지 않는다.
    """
    ex_by_split = split_extra(extra_pos, enroll_only)
    # 시연자 음성은 모드와 무관하게 셋으로 나눈다. val/test 에도 있어야
    # "내 목소리로 다른 말을 할 때 안 켜지는가" 를 잴 수 있다.
    neg_by_split = split_extra(extra_neg, True, verbose=False)
    out = {}
    for sp in ("train", "val", "test"):
        ex = ex_by_split[sp]
        ds = build(root, target, sp, seed=seed, augment=(sp == "train"),
                   extra_pos=ex, enroll_only=enroll_only,
                   extra_neg=neg_by_split[sp], **kw)
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


def speaker_counts(root: str, splits: Sequence[str]) -> dict:
    """{화자: {단어: 개수}} — **파형을 읽지 않는다.**

    word_clips 로 전체를 세려다 105k 클립의 파형을 전부 메모리에 올려
    OOM 으로 죽었다(약 7 GB). 라벨만 필요하면 파일 경로만 보면 된다.
    Speech Commands 의 경로는 '<단어>/<화자>_nohash_<n>.wav' 형식이다.
    """
    import torchaudio
    from collections import defaultdict
    root = os.path.expanduser(root)
    out: dict = defaultdict(lambda: defaultdict(int))
    for sp in splits:
        base = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, url="speech_commands_v0.02", download=False,
            subset=_SPLIT[sp])
        for rel in getattr(base, "_walker", []):
            parts = str(rel).replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            word, fn = parts[-2], parts[-1]
            spk = fn.split("_nohash_")[0]
            out[spk][word] += 1
    return {k: dict(v) for k, v in out.items()}


def word_clips(root: str, splits: Sequence[str], word: Optional[str] = None,
               *, clip_ms: float = 1000.0, limit: int = 0):
    """(wave [n, L], speakers [n], words [n]) — 화자 ID 를 같이 돌려준다.

    화자별 검출률을 재려면 누가 말했는지 알아야 한다. torchaudio 의
    SPEECHCOMMANDS 는 (wave, sr, label, speaker_id, utterance_number) 를 준다.
    word 를 주면 그 단어만, 없으면 전부.
    """
    import torchaudio
    root = os.path.expanduser(root)
    L = int(round(SR * clip_ms / 1000.0))
    waves, spk, wrd = [], [], []
    for sp in splits:
        base = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, url="speech_commands_v0.02", download=False,
            subset=_SPLIT[sp])
        for i in range(len(base)):
            it = base[i]
            if word is not None and it[2] != word:
                continue
            waves.append(_fix_length(it[0], L))
            spk.append(it[3]); wrd.append(it[2])
            if limit and len(waves) >= limit:
                break
        if limit and len(waves) >= limit:
            break
    if not waves:
        raise ValueError(f"클립이 없다: {word!r} in {splits}")
    return torch.stack(waves), spk, wrd
