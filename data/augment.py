"""Waveform augmentation for training (CLAUDE.md 4).

Applied on the raw 1-D waveform BEFORE the AFE, on the training split only.
Transforms:

* time shift: roll the signal by a random offset in +/- `time_shift_ms`, with
  zero fill (NOT circular -- wrapping speech around is unphysical).
* background noise: with probability `noise_prob`, add a random crop of a
  `_background_noise_` clip scaled to a random SNR. This is the augmentation
  that matters most for an AFE (Cerutti's whole point is noise robustness --
  the binarization thresholds learn to be robust).
* partial keyword window: keep a controlled fraction of the detected keyword
  at either edge of a one-second window.  Only keyword-labelled examples may
  use it; unknown and silence are never relabelled as keywords.

Design: when all knobs are off (`is_noop()`), the augment is bypassed entirely
so the no-aug baseline is preserved byte-for-byte. Randomness draws from
torch's global RNG, so a fixed train seed keeps runs reproducible.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch


class WaveformAugment:
    def __init__(self, sample_rate: int, time_shift_ms: float = 0.0,
                 noise_prob: float = 0.0,
                 noise_snr_db: Tuple[float, float] = (5.0, 30.0),
                 noise_waves: Optional[Sequence[torch.Tensor]] = None,
                 gain_db: Tuple[float, float] = (0.0, 0.0),
                 keyword_partial_prob: float = 0.0,
                 keyword_partial_min_coverage: float = 0.75,
                 keyword_partial_max_coverage: float = 1.0,
                 keyword_partial_active_frac: float = 0.02,
                 keyword_partial_fill: str = "zero") -> None:
        self.sr = sample_rate
        self.max_shift = int(round(time_shift_ms * sample_rate / 1000.0))
        self.noise_prob = float(noise_prob)
        self.snr_lo, self.snr_hi = float(noise_snr_db[0]), float(noise_snr_db[1])
        self.noise: List[torch.Tensor] = [n.reshape(-1)
                                          for n in (noise_waves or [])]
        # Random loudness. NOT in MatchboxNet (its MFCC + normalization already
        # absorbs level); this exists because a FIXED comparator threshold does
        # not -- with normalize="fixed" a quiet clip fires almost nothing and a
        # loud one saturates, which the 2x2 measured as the dominant loss
        # (~-8pp, twice the filter-shape cost). Scaling the waveform is exactly
        # the nuisance the hardware cannot remove, so training over it is the
        # one purely-software lever against it.
        self.gain_lo, self.gain_hi = float(gain_db[0]), float(gain_db[1])
        self.partial_prob = float(keyword_partial_prob)
        self.partial_min = float(keyword_partial_min_coverage)
        self.partial_max = float(keyword_partial_max_coverage)
        self.partial_active_frac = float(keyword_partial_active_frac)
        self.partial_fill = keyword_partial_fill
        if not 0.0 <= self.partial_prob <= 1.0:
            raise ValueError("keyword_partial_prob must be in [0, 1]")
        if not 0.0 < self.partial_min <= self.partial_max <= 1.0:
            raise ValueError(
                "keyword partial coverage must satisfy 0 < min <= max <= 1"
            )
        if not 0.0 < self.partial_active_frac < 1.0:
            raise ValueError("keyword_partial_active_frac must be in (0, 1)")
        if self.partial_fill not in ("zero", "noise"):
            raise ValueError("keyword_partial_fill must be 'zero' or 'noise'")

    def is_noop(self) -> bool:
        """True if this augment would leave every waveform unchanged."""
        no_shift = self.max_shift <= 0
        no_noise = self.noise_prob <= 0.0 or not self.noise
        no_gain = self.gain_lo == 0.0 and self.gain_hi == 0.0
        no_partial = self.partial_prob <= 0.0
        return no_shift and no_noise and no_gain and no_partial

    def __call__(self, wave: torch.Tensor, *, keyword: bool = False
                 ) -> torch.Tensor:
        wave = wave.reshape(-1)
        if keyword and self.partial_prob > 0.0 and \
                float(torch.rand(())) < self.partial_prob:
            wave = self._partial_keyword_window(wave)
        if self.max_shift > 0:
            wave = self._time_shift(wave)
        if self.noise and self.noise_prob > 0.0 and \
                float(torch.rand(())) < self.noise_prob:
            wave = self._add_noise(wave)
        if self.gain_lo != 0.0 or self.gain_hi != 0.0:
            wave = self._gain(wave)
        return wave

    # ------------------------------------------------------------------ #
    def _time_shift(self, wave: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(-self.max_shift, self.max_shift + 1, ()))
        if k == 0:
            return wave
        out = torch.zeros_like(wave)
        if k > 0:                       # shift right (delay), zero the front
            out[k:] = wave[:-k]
        else:                           # shift left (advance), zero the tail
            out[:k] = wave[-k:]
        return out

    def _gain(self, wave: torch.Tensor) -> torch.Tensor:
        """Scale the whole clip by a random gain drawn uniformly in dB.

        Applied LAST so it also scales the added noise -- a real level change
        moves signal and background together; scaling only the speech would
        secretly change the SNR too and confound the experiment.
        """
        db = self.gain_lo + (self.gain_hi - self.gain_lo) * float(torch.rand(()))
        return wave * (10.0 ** (db / 20.0))

    def _partial_keyword_window(self, wave: torch.Tensor) -> torch.Tensor:
        """Put 75--100% (configurable) of a keyword at a window edge.

        The crop is aligned to the 10 ms AFE frame grid.  Keeping the tail at
        the left edge models a word that began before the current snapshot;
        keeping the head at the right edge models one that ends afterwards.
        The original pre/post-word recording context stays attached to the
        retained word, and only the newly exposed side uses the configured bed.
        """
        length = self.sr
        if wave.numel() < length:
            fixed = torch.zeros(length, dtype=wave.dtype, device=wave.device)
            fixed[:wave.numel()] = wave
            wave = fixed
        elif wave.numel() > length:
            wave = wave[:length]

        frame = max(1, int(round(self.sr * 0.010)))
        n_frames = length // frame
        if n_frames == 0:
            return wave
        rms = wave[:n_frames * frame].reshape(n_frames, frame).pow(2).mean(1).sqrt()
        peak = rms.max()
        if float(peak) <= 0.0:
            return wave
        active = torch.nonzero(
            rms > peak * self.partial_active_frac, as_tuple=False
        ).flatten()
        if active.numel() == 0:
            return wave

        first = int(active[0])
        last = int(active[-1]) + 1
        word_frames = last - first
        min_keep = max(1, math.ceil(word_frames * self.partial_min))
        max_keep = max(min_keep, int(word_frames * self.partial_max))
        keep_frames = int(torch.randint(min_keep, max_keep + 1, ()))

        out = self._partial_bed(wave)
        if bool(torch.randint(0, 2, ())):
            # Retain the word head and align the crop to the snapshot's end.
            cut = min(length, (first + keep_frames) * frame)
            source = wave[:cut]
            out[-source.numel():] = source
        else:
            # Retain the word tail and align the crop to the snapshot's start.
            cut = max(0, (last - keep_frames) * frame)
            source = wave[cut:]
            out[:source.numel()] = source
        return out

    def _partial_bed(self, wave: torch.Tensor) -> torch.Tensor:
        if self.partial_fill == "zero" or not self.noise:
            return torch.zeros_like(wave)
        src = self.noise[int(torch.randint(len(self.noise), ()))]
        src = src.to(dtype=wave.dtype, device=wave.device)
        if src.numel() < wave.numel():
            src = src.repeat(wave.numel() // src.numel() + 1)
        start = int(torch.randint(src.numel() - wave.numel() + 1, ()))
        return src[start:start + wave.numel()].clone()

    def _add_noise(self, wave: torch.Tensor) -> torch.Tensor:
        n = wave.numel()
        src = self.noise[int(torch.randint(len(self.noise), ()))]
        if src.numel() < n:             # tile short noise clips
            src = src.repeat(n // src.numel() + 1)
        start = int(torch.randint(src.numel() - n + 1, ()))
        noise = src[start:start + n]

        snr = self.snr_lo + (self.snr_hi - self.snr_lo) * float(torch.rand(()))
        ps = wave.pow(2).mean().clamp_min(1e-12)
        pn = noise.pow(2).mean().clamp_min(1e-12)
        # scale noise so 10*log10(ps / P(scaled_noise)) == snr
        scale = torch.sqrt(ps / pn / (10.0 ** (snr / 10.0)))
        return wave + scale * noise
