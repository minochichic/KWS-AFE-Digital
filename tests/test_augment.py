"""Waveform augmentation tests.

Guards two things above all: (1) when off, the augment is an EXACT no-op so the
no-aug baseline is preserved; (2) noise is added at the requested SNR.
"""

from __future__ import annotations

import math

import torch

from data.augment import WaveformAugment


def test_off_is_exact_noop() -> None:
    aug = WaveformAugment(16000, time_shift_ms=0.0, noise_prob=0.0,
                          noise_waves=[torch.randn(20000)])
    assert aug.is_noop()
    w = torch.randn(16000)
    assert torch.equal(aug(w), w)                    # byte-for-byte identical


def test_noise_prob_without_noise_pool_is_noop() -> None:
    aug = WaveformAugment(16000, noise_prob=1.0, noise_waves=[])
    assert aug.is_noop()


# --------------------------------------------------------------------------- #
# time shift
# --------------------------------------------------------------------------- #
def test_time_shift_preserves_length() -> None:
    aug = WaveformAugment(16000, time_shift_ms=5.0)
    w = torch.randn(16000)
    out = aug(w)
    assert out.shape == w.shape


def test_time_shift_moves_an_impulse_with_zero_fill() -> None:
    torch.manual_seed(0)
    aug = WaveformAugment(1000, time_shift_ms=100.0)   # up to +/-100 samples
    w = torch.zeros(1000); w[500] = 1.0
    out = aug(w)
    # exactly one nonzero, still magnitude 1, and not circular (no wrap)
    nz = torch.nonzero(out).flatten()
    assert nz.numel() == 1 and out[nz].item() == 1.0
    assert 400 <= nz.item() <= 600                     # within +/-100 of 500


def test_time_shift_is_not_circular() -> None:
    aug = WaveformAugment(1000, time_shift_ms=50.0)
    w = torch.full((1000,), 5.0)                       # all nonzero
    for _ in range(20):                                # until a nonzero shift
        out = aug(w)
        if not torch.equal(out, w):
            # a circular roll of all-5s stays all-5s; zero-fill introduces 0s
            assert (out == 0).any()
            return
    raise AssertionError("no shift occurred in 20 tries")


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def test_noise_changes_signal_and_keeps_length() -> None:
    aug = WaveformAugment(16000, noise_prob=1.0, noise_snr_db=(10.0, 10.0),
                          noise_waves=[torch.randn(40000)])
    w = torch.randn(16000)
    out = aug(w)
    assert out.shape == w.shape
    assert not torch.equal(out, w)


def test_noise_hits_target_snr() -> None:
    torch.manual_seed(0)
    target = 15.0
    aug = WaveformAugment(16000, noise_prob=1.0, noise_snr_db=(target, target),
                          noise_waves=[torch.randn(40000)])
    w = torch.randn(16000)
    added = aug(w) - w                                 # the scaled noise
    snr = 10.0 * math.log10(w.pow(2).mean() / added.pow(2).mean())
    assert abs(snr - target) < 0.5


def test_short_noise_clip_is_tiled() -> None:
    aug = WaveformAugment(16000, noise_prob=1.0, noise_snr_db=(20.0, 20.0),
                          noise_waves=[torch.randn(4000)])   # shorter than clip
    out = aug(torch.randn(16000))
    assert out.shape == (16000,)                       # no crash, right length
