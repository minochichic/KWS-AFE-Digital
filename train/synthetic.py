"""Synthetic tone dataset for the overfit sanity check (FIRST_TASK.md step 5).

Each class k gets its own carrier frequency, mel-spaced across the AFE band so
adjacent classes land in different filterbank channels. Samples within a class
differ by phase, small frequency jitter, amplitude and noise -- enough
variation that hitting 100% train accuracy requires the whole chain
(AFE thresholds -> binary stages -> epilogue) to actually learn, not just a
lucky constant.

This file is NOT the real data pipeline (that is step 6); it exists so the
training loop can be proven before any download.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def class_frequencies(n_classes: int, f_lo: float = 150.0,
                      f_hi: float = 6500.0) -> torch.Tensor:
    """One carrier per class, equally spaced on the mel scale."""
    def mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def inv_mel(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    return inv_mel(torch.linspace(mel(f_lo), mel(f_hi), n_classes))


def make_tone_dataset(n_classes: int = 12, per_class: int = 4,
                      sample_rate: int = 16000, seed: int = 0,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (waves [N, sample_rate], labels [N]), N = n_classes*per_class."""
    g = torch.Generator().manual_seed(seed)
    freqs = class_frequencies(n_classes)
    t = torch.arange(sample_rate) / sample_rate

    waves, labels = [], []
    for k in range(n_classes):
        for _ in range(per_class):
            f = freqs[k] * (1.0 + 0.02 * torch.randn((), generator=g))
            phase = 2 * math.pi * torch.rand((), generator=g)
            amp = 0.4 + 0.2 * torch.rand((), generator=g)
            noise = 0.02 * torch.randn(sample_rate, generator=g)
            waves.append(amp * torch.sin(2 * math.pi * f * t + phase) + noise)
            labels.append(k)
    return torch.stack(waves), torch.tensor(labels)
