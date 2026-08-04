"""Waveform augmentation for training (CLAUDE.md 4).

Applied on the raw 1-D waveform BEFORE the AFE, on the training split only.
Two transforms:

* time shift: roll the signal by a random offset in +/- `time_shift_ms`, with
  zero fill (NOT circular -- wrapping speech around is unphysical).
* background noise: with probability `noise_prob`, add a random crop of a
  `_background_noise_` clip scaled to a random SNR. This is the augmentation
  that matters most for an AFE (Cerutti's whole point is noise robustness --
  the binarization thresholds learn to be robust).

Design: when all knobs are off (`is_noop()`), the augment is bypassed entirely
so the no-aug baseline is preserved byte-for-byte. Randomness draws from
torch's global RNG, so a fixed train seed keeps runs reproducible.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


class WaveformAugment:
    def __init__(self, sample_rate: int, time_shift_ms: float = 0.0,
                 noise_prob: float = 0.0,
                 noise_snr_db: Tuple[float, float] = (5.0, 30.0),
                 noise_waves: Optional[Sequence[torch.Tensor]] = None,
                 gain_db: Tuple[float, float] = (0.0, 0.0)) -> None:
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

    def is_noop(self) -> bool:
        """True if this augment would leave every waveform unchanged."""
        no_shift = self.max_shift <= 0
        no_noise = self.noise_prob <= 0.0 or not self.noise
        no_gain = self.gain_lo == 0.0 and self.gain_hi == 0.0
        return no_shift and no_noise and no_gain

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.reshape(-1)
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
