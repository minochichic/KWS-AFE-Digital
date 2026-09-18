"""합성 과제. 실제 음성을 쓰기 전에 구조가 학습되는지부터 확인한다.

양성은 발성 시작으로부터 정해진 네 시각에 정해진 주파수가 오는 파형이고,
음성(비대상)은 같은 구조에 주파수만 다르다. 발성 시작 시각 t0 는 매번 다르게
두어 START 검출과 정렬까지 함께 시험한다.

이 과제를 못 풀면 실제 데이터로 갈 이유가 없다.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

# 양성이 각 상태에서 내는 주파수 [Hz]
TARGET_HZ: List[float] = [400.0, 1200.0, 700.0, 2500.0]
# 음성이 뽑아 쓰는 후보
DISTRACTOR_HZ: List[float] = [300.0, 600.0, 900.0, 1400.0, 2000.0, 3000.0, 4000.0]


def _tone(f: float, n: int, sr: int, gen: torch.Generator) -> torch.Tensor:
    phase = torch.rand(1, generator=gen).item() * 6.283185
    t = torch.arange(n, dtype=torch.float32) / sr
    return torch.sin(2 * 3.14159265 * f * t + phase)


def make_batch(n: int, tau_frames: List[int], *, sr: int = 16000,
               clip_ms: float = 1000.0, frame_ms: float = 10.0,
               burst_ms: float = 60.0, onset_ms: float = 30.0,
               noise: float = 0.02, seed: int = 0,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(wave [n, L], y [n], onset [n]) — y=1 이 대상, onset 은 진짜 t0 프레임.

    onset 을 돌려주는 이유: START 검출의 지터를 재려면 정답이 있어야 한다.
    난수 흐름을 밖에서 재현하려 하면 양성/음성의 소비 횟수가 달라 어긋난다.
    """
    gen = torch.Generator().manual_seed(seed)
    L = int(round(sr * clip_ms / 1000.0))
    nb = int(round(sr * burst_ms / 1000.0))
    no = int(round(sr * onset_ms / 1000.0))
    tau_ms = [f * frame_ms for f in tau_frames]

    waves = torch.randn(n, L, generator=gen) * noise
    y = (torch.rand(n, generator=gen) < 0.5).float()
    onset = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        t0 = 50.0 + torch.rand(1, generator=gen).item() * 200.0     # 50~250 ms
        s0 = int(round(sr * t0 / 1000.0))
        onset[i] = int(round(t0 / frame_ms))
        # 발성 시작 표지 — 광대역 버스트. START 는 여기서 뜬다.
        waves[i, s0:s0 + no] += torch.randn(no, generator=gen) * 0.5

        if y[i] > 0:
            freqs = list(TARGET_HZ[:len(tau_ms)])
        else:
            freqs = []
            for j in range(len(tau_ms)):
                k = torch.randint(len(DISTRACTOR_HZ), (1,), generator=gen).item()
                freqs.append(DISTRACTOR_HZ[k])
            # 우연히 양성과 같아지면 한 곳을 확실히 어긋나게 둔다
            if all(abs(a - b) < 1.0 for a, b in zip(freqs, TARGET_HZ[:len(tau_ms)])):
                freqs[0] = DISTRACTOR_HZ[0]

        for tm, f in zip(tau_ms, freqs):
            a = s0 + int(round(sr * tm / 1000.0))
            b = min(a + nb, L)
            if a >= L:
                continue
            waves[i, a:b] += _tone(f, b - a, sr, gen) * 0.35

    return waves, y, onset
