"""연속 오디오 평가 — 검출률과 시간당 오검출(FA/h)을 같은 스트림에서 잰다.

클립 단위 FPR 은 "단어 하나당 오검출 확률"이라 시간당 횟수가 아니다. 상시 대기
회로에서 사용성을 좌우하는 것은 후자이고, 계획서 3.4 가 약속한 것도 그쪽이다.

비대상 단어를 무작위 간격으로 이어 붙이고 사이사이 대상 단어를 심은 뒤,
게이트 수준 상태기계를 처음부터 끝까지 한 번에 돌린다. 클립 경계에서 끊지
않으므로 START 가 실제 회로처럼 계속 뜨고 timeout 으로 닫힌다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SR = 16000


# --------------------------------------------------------------- 스트림 생성
def build_stream(waves: torch.Tensor, words: Sequence[str], target: str, *,
                 minutes: float = 10.0, targets_per_min: float = 6.0,
                 gap_s: Tuple[float, float] = (0.3, 1.5),
                 noise: Optional[Sequence[torch.Tensor]] = None,
                 snr_db: Tuple[float, float] = (10.0, 30.0),
                 seed: int = 0) -> Tuple[torch.Tensor, List[int]]:
    """(stream [L], target_onsets [samples]) — 대상 단어의 시작 표본 위치."""
    g = torch.Generator().manual_seed(seed)
    tgt = [i for i, w in enumerate(words) if w == target]
    oth = [i for i, w in enumerate(words) if w != target]
    if not tgt or not oth:
        raise ValueError(f"대상 {target!r} 또는 비대상 클립이 없다.")

    total = int(minutes * 60 * SR)
    # 대상 하나당 비대상 몇 개를 끼울지
    per_target = max(1, int(round(60.0 / targets_per_min /
                                  (1.0 + sum(gap_s) / 2))))
    out, onsets, n = [], [], 0
    while n < total:
        for _ in range(per_target):
            i = oth[int(torch.randint(len(oth), (1,), generator=g))]
            out.append(waves[i]); n += waves[i].numel()
            gap = gap_s[0] + torch.rand(1, generator=g).item() * (gap_s[1] - gap_s[0])
            ng = int(gap * SR)
            out.append(torch.zeros(ng)); n += ng
        i = tgt[int(torch.randint(len(tgt), (1,), generator=g))]
        onsets.append(n)
        out.append(waves[i]); n += waves[i].numel()
        gap = gap_s[0] + torch.rand(1, generator=g).item() * (gap_s[1] - gap_s[0])
        ng = int(gap * SR)
        out.append(torch.zeros(ng)); n += ng

    stream = torch.cat(out)
    if noise:
        # 소음과 SNR 을 스트림당 한 번만 뽑으면 그 스트림이 사실상 한 가지
        # 조건만 시험하게 되어, 스트림 간 FA/h 편차가 3배씩 벌어진다(실측:
        # 191 / 198 / 65). 30초 블록마다 다시 뽑아 한 스트림이 범위를 덮게 한다.
        blk = int(30.0 * SR)
        ps_all = stream.pow(2).mean().clamp(min=1e-12)
        for a in range(0, stream.numel(), blk):
            b = min(a + blk, stream.numel())
            nz = noise[int(torch.randint(len(noise), (1,), generator=g))]
            rep = int((b - a) // nz.numel()) + 1
            off = int(torch.randint(max(1, nz.numel() - 1), (1,), generator=g))
            nz = nz.roll(off).repeat(rep)[:b - a]
            snr = snr_db[0] + torch.rand(1, generator=g).item() * (snr_db[1] - snr_db[0])
            pn = nz.pow(2).mean().clamp(min=1e-12)
            stream[a:b] = stream[a:b] + nz * torch.sqrt(
                ps_all / (pn * 10 ** (snr / 10)))
    return stream, onsets


# --------------------------------------------------------------- 연속 특징
@torch.no_grad()
def stream_features(model, stream: torch.Tensor, chunk_s: float = 60.0
                    ) -> torch.Tensor:
    """연속 파형 -> [1, C, T] 이진 특징.

    클립용 envelopes() 는 1초를 native_T 칸으로 adaptive pooling 하므로 쓸 수
    없다. 여기서는 STFT 프레임을 그대로 10 ms 격자로 쓴다(hop 이 이미 10 ms).
    정규화 상수 fixed_lo/hi 와 문턱 theta 는 학습된 값을 그대로 쓴다.
    """
    fe = model.frontend
    cfg = fe.cfg
    ratio = int(round(cfg.envelope_win_ms / cfg.stft_hop_ms))
    dev = fe.threshold.device
    parts = []
    n = int(chunk_s * SR)
    for i in range(0, stream.numel(), n):
        seg = stream[i:i + n].to(dev)
        if seg.numel() < int(0.05 * SR):
            break
        band = fe._bands(seg.unsqueeze(0))
        if cfg.compression == "log":
            band = torch.log(band + 1e-10)
        else:
            band = torch.sqrt(band + 1e-10)
        if ratio > 1:
            band = F.max_pool1d(band, ratio, ratio)
        env = (band - fe.fixed_lo) / (fe.fixed_hi - fe.fixed_lo + 1e-10)
        parts.append((env >= fe.threshold.view(1, -1, 1)).float().cpu())
    return torch.cat(parts, dim=2)


# --------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate_stream(board, feats: torch.Tensor, onsets: Sequence[int], *,
                    frame_ms: float = 10.0,
                    credit_window_s: Tuple[float, float] = (-0.1, 0.9)
                    ) -> Dict[str, float]:
    """WAKE 상승 에지를 사건으로 세고, 대상 발화에 귀속시킨다.

    대상 시작 전후 credit_window 안에서 뜬 WAKE 는 그 발화의 검출로 본다.
    나머지는 전부 오검출이다.
    """
    T = feats.shape[2]
    wake = board.run(feats)["wake_seq"][0]
    edges = ((wake > 0.5) & (torch.cat([torch.zeros(1), wake[:-1]]) <= 0.5))
    ev = edges.nonzero().flatten().tolist()            # 검출 사건의 프레임

    fps = 1000.0 / frame_ms
    lo, hi = credit_window_s
    windows = [(o / SR + lo, o / SR + hi) for o in onsets]
    used, hit = set(), 0
    for a, b in windows:
        for j, f in enumerate(ev):
            if j in used:
                continue
            t = f / fps
            if a <= t <= b:
                used.add(j); hit += 1
                break
    fa = len(ev) - len(used)
    hours = T / fps / 3600.0
    return {
        "tpr": hit / max(len(onsets), 1),
        "n_targets": len(onsets),
        "n_wakes": len(ev),
        "n_false": fa,
        "fa_per_hour": fa / max(hours, 1e-9),
        "hours": hours,
        "frames": T,
    }
