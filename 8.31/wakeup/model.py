"""전체 모델: 파형 -> 이진 특징 -> START 정렬 -> 형판 정합 -> WAKE.

회로와 같은 순서로 계산한다. 학습이 끝나면 각 부분의 파라미터가 그대로
회로 상수가 되고, hard() 는 반올림한 정수 상수로 회로와 동일하게 판정한다.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .frontend import Frontend
from .head import TemplateHead


def detect_start(x: torch.Tensor, k: int,
                 min_frames: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """sum_c x[c,t] >= k 가 min_frames 연속으로 성립하는 첫 프레임.

    x: [B, C, T] in {0,1}
    반환 (start [B] long, found [B] bool) -- start 는 그 구간의 첫 프레임

    회로에서는 채널 버스를 저항으로 합산해 비교기에 넣고(k), 그 출력을 RC 나
    시프트레지스터로 지속 확인하는 것(min_frames)에 해당한다. 미분 대상이
    아니므로 인덱스로만 쓴다.
    """
    active = (x.sum(dim=1) >= k)                    # [B, T] bool
    if min_frames > 1:
        # 연속 구간의 첫 프레임을 찾는다
        w = torch.ones(1, 1, min_frames, device=x.device)
        run = F.conv1d(active.float().unsqueeze(1), w).squeeze(1)   # [B, T-m+1]
        active = run >= min_frames
    found = active.any(dim=1)
    # argmax 는 최초 최댓값 위치를 준다 = 첫 True
    start = active.float().argmax(dim=1)
    return start.long(), found


def gather_states(x: torch.Tensor, start: torch.Tensor, tau: torch.Tensor,
                  window: int = 0) -> torch.Tensor:
    """각 상태의 채점 창에서 채널 줄을 뽑는다.

    x: [B, C, T], start: [B], tau: [S]  ->  [B, S, W, C],  W = 2*window+1
    window=0 이면 W=1 이고 예전과 같다.
    """
    B, C, T = x.shape
    S = tau.shape[0]
    off = torch.arange(-window, window + 1, device=x.device)        # [W]
    W = off.shape[0]
    idx = (start.view(-1, 1, 1) + tau.view(1, -1, 1) + off.view(1, 1, -1))
    idx = idx.clamp(0, T - 1).reshape(B, S * W)                     # [B, S*W]
    g = x.gather(2, idx.unsqueeze(1).expand(B, C, S * W))           # [B, C, S*W]
    return g.transpose(1, 2).reshape(B, S, W, C).contiguous()


class WakeupModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.frontend = Frontend(cfg.frontend)
        self.head = TemplateHead(cfg.head, cfg.frontend.n_channels)
        self.register_buffer("tau", torch.tensor(cfg.head.tau, dtype=torch.long))

    # ------------------------------------------------------------------ 전방
    def features(self, wave: torch.Tensor) -> torch.Tensor:
        return self.frontend(wave)

    def align(self, x: torch.Tensor, jitter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        start, found = detect_start(x, self.cfg.start.k,
                                    self.cfg.start.min_frames)
        if jitter and self.cfg.start.jitter > 0:
            j = self.cfg.start.jitter
            noise = torch.randint(-j, j + 1, start.shape, device=start.device)
            start = (start + noise).clamp(min=0)
        return start, found

    def forward(self, wave: torch.Tensor, jitter: bool = False) -> Dict[str, torch.Tensor]:
        x = self.features(wave)                              # [B, C, T]
        start, found = self.align(x, jitter)
        xs = gather_states(x, start, self.tau,
                           self.cfg.head.match_window)     # [B, S, W, C]
        out = self.head(xs)
        # START 를 못 잡으면 그 발화는 판정 자체가 불가능하다. 회로도 같다.
        out["wake_logit"] = torch.where(
            found, out["wake_logit"], torch.full_like(out["wake_logit"], -20.0))
        out["found"] = found
        out["start"] = start
        return out

    # --------------------------------------------------------------- 초기화
    @torch.no_grad()
    def init_from_data(self, wave: torch.Tensor, y: torch.Tensor) -> None:
        """스케일 -> 문턱 -> 형판 -> k 순서로 데이터에서 초기화한다.

        순서가 중요하다. 문턱이 정해져야 이진 특징이 생기고, 이진 특징이 있어야
        형판 원형을 뽑을 수 있고, 형판이 있어야 count 분포가 생겨 k 를 맞출 수 있다.
        """
        was = self.training
        self.eval()
        self.frontend.init_fixed_scale(wave)
        self.frontend.init_thresholds(wave)   # cfg.init_on_rate 를 따른다
        x = self.features(wave)
        start, found = self.align(x, jitter=False)
        xs = gather_states(x, start, self.tau, self.cfg.head.match_window)
        self.head.init_templates(xs[(y > 0.5) & found])
        self.refit_k(wave, y)
        self.train(was)

    # ------------------------------------------------------------- k 재적합
    @torch.no_grad()
    def refit_k(self, wave: torch.Tensor, y: torch.Tensor,
                max_fpr: Optional[float] = None) -> torch.Tensor:
        if max_fpr is None:
            max_fpr = self.cfg.head.k_max_fpr
        was = self.training
        self.eval()
        x = self.features(wave)
        start, found = self.align(x, jitter=False)
        xs = gather_states(x, start, self.tau, self.cfg.head.match_window)
        count = self.head(xs)["count"]
        # START 를 못 잡은 클립은 어차피 탈락이므로 양성에서 빼고 맞춘다
        keep = found | (y <= 0.5)
        k = self.head.fit_k(count[keep], y[keep], max_fpr=max_fpr)
        self.train(was)
        return k

    @torch.no_grad()
    def hard(self, wave: torch.Tensor) -> Dict[str, torch.Tensor]:
        """반올림 상수로 회로와 동일하게 판정한다."""
        x = self.features(wave)
        start, found = self.align(x, jitter=False)
        xs = gather_states(x, start, self.tau, self.cfg.head.match_window)
        out = self.head.hard(xs)
        out["wake"] = out["wake"] * found.float()
        out["found"] = found
        return out

    # ------------------------------------------------------------------ 손실
    def loss(self, wave: torch.Tensor, y: torch.Tensor,
             pos_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        out = self(wave, jitter=self.training)
        bce = F.binary_cross_entropy_with_logits(
            out["wake_logit"], y,
            pos_weight=torch.as_tensor(pos_weight, device=y.device,
                                       dtype=y.dtype))
        pen = self.head.penalties()
        total = bce + pen["l1"] + pen["floor"]
        return {"loss": total, "bce": bce, "l1": pen["l1"], "floor": pen["floor"],
                "wake_logit": out["wake_logit"], "found": out["found"]}

    # ------------------------------------------------------------------ 내보내기
    @torch.no_grad()
    def export(self, vdd: float = 1.8) -> Dict[str, object]:
        d = self.head.export(vdd)
        d["tau"] = list(self.cfg.head.tau)
        d["timeout"] = self.cfg.head.timeout
        d["start_k"] = self.cfg.start.k
        d["match_window"] = self.cfg.head.match_window
        d["theta"] = self.frontend.threshold.detach().cpu()
        d["wake_width_ms"] = (self.cfg.head.timeout - self.cfg.head.tau[-1]) * \
            self.cfg.frontend.envelope_win_ms
        return d
