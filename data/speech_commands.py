"""Google Speech Commands v2 -> 12-class pipeline (CLAUDE.md 3).

12 classes = 10 keywords (yes no up down left right on off stop go)
             + _silence_ + _unknown_.

Split strategy: torchaudio's SPEECHCOMMANDS already implements the dataset's
official 80:10:10 split (validation_list.txt / testing_list.txt, training =
the rest, with _background_noise_ excluded). We wrap it and add only the
12-class relabeling that the raw dataset does not do:

* the 10 keywords keep their identity (indices 0..9),
* every other spoken word becomes _unknown_ (index 11), subsampled to roughly
  one keyword-class worth so it does not dominate (Cerutti IV-A: unknown is a
  *subset* of the remaining 25 words),
* _silence_ (index 10) is synthesized from the _background_noise_ clips, with
  a fraction of pure-zero clips (Warden's split convention).

Class order is fixed and load-bearing: the exported hardware indexes outputs
by it, so KEYWORDS + [silence, unknown] must never be reordered.

Only `build_dataloaders` / `SpeechCommands12.from_torchaudio` touch torchaudio
or the disk; everything else is pure and unit-tested with in-memory items.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train.config import DataConfig

KEYWORDS: List[str] = ["yes", "no", "up", "down", "left", "right",
                       "on", "off", "stop", "go"]
SILENCE_INDEX = 10
UNKNOWN_INDEX = 11
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"

_KW_TO_IDX = {kw: i for i, kw in enumerate(KEYWORDS)}

# Stable per-split seed offsets. NOT hash(split): Python string hashing is
# salted per process (PYTHONHASHSEED), which would make the unknown subsample
# and silence clips differ between Colab sessions -- breaking reproducibility
# (CLAUDE.md 5).
_SPLIT_OFFSET = {"training": 1, "validation": 2, "testing": 3,
                 "train": 1, "val": 2, "test": 3}


def class_names() -> List[str]:
    return KEYWORDS + [SILENCE_LABEL, UNKNOWN_LABEL]


def label_to_index(word: str) -> int:
    """Map a raw Speech Commands word to its 12-class index."""
    if word in _KW_TO_IDX:
        return _KW_TO_IDX[word]
    if word == SILENCE_LABEL:
        return SILENCE_INDEX
    return UNKNOWN_INDEX


def subsample_indices(n_total: int, n_keep: int, seed: int) -> List[int]:
    """Pick min(n_keep, n_total) distinct indices from range(n_total)."""
    if n_keep >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_keep))


def synthesize_silence(noise_waves: Sequence[torch.Tensor], n: int, length: int,
                       seed: int, zero_fraction: float = 0.1) -> torch.Tensor:
    """n 1-second silence clips: random crops of background noise, plus a
    fraction of pure-zero clips. Returns [n, length]."""
    g = torch.Generator().manual_seed(seed)
    clips = torch.zeros(n, length)
    n_zero = int(round(n * zero_fraction))
    for i in range(n):
        if i < n_zero or not noise_waves:
            continue                                  # leave as zeros
        src = noise_waves[int(torch.randint(len(noise_waves), (1,), generator=g))]
        src = src.reshape(-1)
        if src.numel() <= length:
            clips[i, :src.numel()] = src
        else:
            start = int(torch.randint(src.numel() - length + 1, (1,), generator=g))
            clips[i] = src[start:start + length]
    return clips


class FixedLengthCollate:
    """Pad/crop every waveform to `length` and stack into a batch.

    Speech Commands clips are ~1 s but not exactly; the AFE needs a rectangular
    [B, L] batch. Padding is zero (silence), matching the AFE's own handling.
    """

    def __init__(self, length: int = 16000) -> None:
        self.length = length

    def _fix(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.reshape(-1)
        n = wave.numel()
        if n < self.length:
            return F.pad(wave, (0, self.length - n))
        return wave[:self.length]

    def __call__(self, batch: Sequence[Tuple[torch.Tensor, int]]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        waves = torch.stack([self._fix(w) for w, _ in batch])
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return waves, labels


class SpeechCommands12(Dataset):
    """12-class view over Speech Commands items.

    `items` is any sequence of (waveform, word); on Colab it comes from
    torchaudio (see from_torchaudio), in tests it is in-memory. Silence clips
    are appended and generated lazily in __getitem__.
    """

    def __init__(self, items: Sequence[Tuple[torch.Tensor, str]],
                 noise_waves: Sequence[torch.Tensor], cfg: DataConfig,
                 split: str, seed: int = 0, sample_rate: int = 16000,
                 augment: bool = False) -> None:
        self.cfg = cfg
        self.split = split
        self.sample_rate = sample_rate
        self._noise = list(noise_waves)

        # Waveform augment, TRAIN split only. Built from cfg + this split's
        # noise pool; bypassed entirely when all knobs are off so the no-aug
        # baseline is preserved exactly.
        self._augment = None
        if augment:
            from data.augment import WaveformAugment
            aug = WaveformAugment(
                sample_rate, cfg.aug_time_shift_ms, cfg.aug_noise_prob,
                tuple(cfg.aug_noise_snr_db), self._noise,
                tuple(getattr(cfg, "aug_gain_db", (0.0, 0.0))))
            if not aug.is_noop():
                self._augment = aug

        keyword: List[int] = []
        unknown: List[int] = []
        for i, (_w, word) in enumerate(items):
            (keyword if word in _KW_TO_IDX else unknown).append(i)
        self._items = items

        # unknown is subsampled to ~keyword-class size so it cannot dominate.
        n_keyword = len(keyword)
        # split-dependent, process-stable seed so train/val/test never draw
        # identical subsets yet reproduce across sessions
        split_seed = seed + 100003 * _SPLIT_OFFSET.get(split, 0)
        n_unknown = round(n_keyword * cfg.unknown_fraction)
        kept_unknown = [unknown[j]
                        for j in subsample_indices(len(unknown), n_unknown,
                                                   split_seed)]

        self._real = keyword + kept_unknown          # indices into items
        self._n_silence = round(n_keyword * cfg.silence_fraction)
        self._silence_seed = split_seed + 777

    def __len__(self) -> int:
        return len(self._real) + self._n_silence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < len(self._real):
            wave, word = self._items[self._real[idx]]
            wave = wave.reshape(-1)
            if self._augment is not None:        # real utterances only
                wave = self._augment(wave)
            return wave, label_to_index(word)
        # silence: generated on demand, deterministic per (split, position)
        s = idx - len(self._real)
        clip = synthesize_silence(self._noise, 1, self.sample_rate,
                                  seed=self._silence_seed + s)[0]
        return clip, SILENCE_INDEX

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torchaudio(cls, cfg: DataConfig, split: str, seed: int = 0,
                        augment: bool = False) -> "SpeechCommands12":
        """Build from the downloaded torchaudio dataset (Colab path).

        split in {"training","validation","testing"}. Downloads on first call.
        `augment` should be True only for the training split.
        """
        import os

        import torchaudio  # local import: not needed for the pure-logic path

        # torchaudio downloads the tarball straight into `root` and assumes it
        # already exists -- create it, otherwise the .partial temp file fails
        # with FileNotFoundError.
        os.makedirs(cfg.root, exist_ok=True)

        subset = {"train": "training", "val": "validation",
                  "test": "testing"}.get(split, split)
        base = torchaudio.datasets.SPEECHCOMMANDS(
            root=cfg.root, url=f"speech_commands_{cfg.version}",
            download=True, subset=subset)

        # (waveform, word) lazily; torchaudio yields
        # (wave, sr, label, speaker_id, utterance_number).
        items = _LazyTorchaudioItems(base)
        noise = _load_noise_waves(base)
        return cls(items, noise, cfg, split=split, seed=seed, augment=augment)


class _LazyTorchaudioItems(Sequence):
    """Adapt torchaudio SPEECHCOMMANDS to a sequence of (waveform, word)
    without loading every clip up front."""

    def __init__(self, base) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, str]:
        wave, _sr, label, *_ = self._base[i]
        return wave, label


def _load_noise_waves(base) -> List[torch.Tensor]:
    """Load the _background_noise_ clips that back the silence class."""
    import os
    import torchaudio

    root = os.path.join(base._path, "_background_noise_")
    if not os.path.isdir(root):
        return []
    waves = []
    for name in sorted(os.listdir(root)):
        if name.endswith(".wav"):
            w, _sr = torchaudio.load(os.path.join(root, name))
            waves.append(w.reshape(-1))
    return waves


def ensure_dataset(root: str, cache_tar: Optional[str] = None) -> str:
    """Make the extracted Speech Commands tree available at `root`.

    Order of preference:
      1. `root` already extracted -> use it.
      2. `cache_tar` exists (e.g. on Drive) -> extract it onto local disk.
      3. otherwise download via torchaudio, then (if cache_tar given) pack the
         extracted tree into cache_tar for next session.

    The cache is a SINGLE tar, extracted onto local `root`, not read in place:
    reading 105k tiny wavs from Drive during training is slow, but streaming
    one big tar is fine. Note Colab's download is ~25 s, so the main win is
    avoiding torchaudio's re-extraction and network dependence, not raw speed.
    """
    import os
    import shutil
    import subprocess
    import tarfile

    sc = os.path.join(root, "SpeechCommands")
    leaf = os.path.join(sc, "speech_commands_v0.02")

    def _ready() -> bool:
        # A key list file the loader needs; presence of the SpeechCommands dir
        # alone is NOT enough (a partial/interrupted extract has the dir but
        # not this file, which then fails deep inside torchaudio).
        return os.path.isfile(os.path.join(leaf, "testing_list.txt"))

    def _clean_partial() -> None:
        if os.path.isdir(sc):
            print(f"[dataset] 불완전한 추출 감지 -> 삭제: {sc}")
            shutil.rmtree(sc, ignore_errors=True)

    if _ready():
        return root
    _clean_partial()
    os.makedirs(root, exist_ok=True)

    def _untar(src: str, dst: str) -> None:
        if shutil.which("tar"):
            subprocess.run(["tar", "xf", src, "-C", dst], check=True)
        else:
            with tarfile.open(src) as t:
                t.extractall(dst)

    def _mktar(src_dir: str, arcname: str, dst: str) -> None:
        if shutil.which("tar"):
            subprocess.run(["tar", "cf", dst, "-C", os.path.dirname(src_dir),
                            arcname], check=True)
        else:
            with tarfile.open(dst, "w") as t:
                t.add(src_dir, arcname=arcname)

    if cache_tar and os.path.exists(cache_tar):
        print(f"[dataset] Drive 캐시에서 복원: {cache_tar}")
        _untar(cache_tar, root)
        if _ready():
            return root
        print("[dataset] 캐시가 불완전 -> 삭제하고 재다운로드")
        _clean_partial()

    import torchaudio
    print("[dataset] Speech Commands v2 다운로드")
    torchaudio.datasets.SPEECHCOMMANDS(
        root=root, url="speech_commands_v0.02", download=True)
    if not _ready():
        raise RuntimeError(
            f"dataset extract incomplete: {leaf}/testing_list.txt missing")
    if cache_tar:
        os.makedirs(os.path.dirname(cache_tar), exist_ok=True)
        print(f"[dataset] 다음 세션을 위해 Drive에 캐시 저장: {cache_tar}")
        _mktar(sc, "SpeechCommands", cache_tar)
    return root


def build_dataloaders(cfg: DataConfig, batch_size: int, sample_rate: int = 16000,
                      num_workers: Optional[int] = None, seed: int = 0
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """(train, val, test) loaders. Colab entry point; downloads on first call."""
    nw = cfg.num_workers if num_workers is None else num_workers
    collate = FixedLengthCollate(sample_rate)

    def loader(split: str, shuffle: bool) -> DataLoader:
        # augment the training split only (val/test stay clean for fair eval)
        ds = SpeechCommands12.from_torchaudio(
            cfg, split, seed=seed, augment=(split == "training"))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=nw, collate_fn=collate,
                          pin_memory=True, drop_last=shuffle)

    return (loader("training", True),
            loader("validation", False),
            loader("testing", False))
