"""형판 정합부.

상태 s 마다 시각 tau(s) 의 채널 한 줄을 뽑아 형판 M(s,c) 와 대조하고, 맞은
개수가 허용 오차 k(s) 이상이면 통과로 본다. 네 상태를 전부 통과해야 WAKE.

여기서 나오는 값이 그대로 회로 상수가 된다.
  M(s,c)  -> 저항의 연결 여부와 극성
  k(s)    -> 비교기 기준전압 분압비
  m(s)    -> 그 상태가 실제로 보는 채널 수 (= 저항 개수)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import HeadConfig
from .ste import sign_ste, gate_ste


class TemplateHead(nn.Module):
    def __init__(self, cfg: HeadConfig, n_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.C = n_channels
        S = cfg.n_states

        # 형판은 M = g * sign(w) 로 분해한다. 세 값 {1, 0, X} 를 미분 가능한
        # 두 부분으로 나눈 것: g 가 "보는가", sign(w) 가 "무엇이어야 하는가".
        self.weight = nn.Parameter(torch.randn(S, n_channels) * 0.1)
        self.gate_logit = nn.Parameter(torch.full((S, n_channels), cfg.gate_init))
        # 허용 오차. 실수로 두고 내보낼 때 반올림한다.
        self.k = nn.Parameter(torch.full((S,), n_channels * 0.7))

    # ------------------------------------------------------------------ 형판
    def ternary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(g, s). M = g * s, g in {0,1}, s in {-1,+1}."""
        g = gate_ste(self.gate_logit, self.cfg.gate_ste_clip)
        s = sign_ste(self.weight, self.cfg.weight_ste_clip)
        return g, s

    def used_channels(self) -> torch.Tensor:
        """m(s) — 상태별로 실제 보는 채널 수."""
        g, _ = self.ternary()
        return g.sum(dim=1)

    # ------------------------------------------------------------------ 정합
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, S, C] in {0,1} — 각 상태의 tau 에서 뽑힌 채널 한 줄.

        반환
          wake_logit [B]     WAKE 로짓 — 상태별 여유의 soft-min
          count      [B, S]  상태별 일치 개수
          m          [S]     상태별 사용 채널 수
        """
        if x.dim() != 3 or x.shape[1] != self.cfg.n_states or x.shape[2] != self.C:
            raise ValueError(
                f"x 는 [B, {self.cfg.n_states}, {self.C}] 여야 하는데 "
                f"{tuple(x.shape)} 다.")
        g, s = self.ternary()                                   # [S, C]

        # 채널별 일치도. s=+1 이면 x, s=-1 이면 1-x. g=0 이면 세지 않는다.
        #   0.5 + s*(x - 0.5)  ==  x        (s=+1)
        #                      ==  1 - x    (s=-1)
        agree = g.unsqueeze(0) * (0.5 + s.unsqueeze(0) * (x - 0.5))
        count = agree.sum(dim=2)                                # [B, S]
        m = g.sum(dim=1)                                        # [S]

        # count >= k 를 부드럽게. k-0.5 는 회로의 V_TH 와 같은 위치다.
        z = (count - (self.k.unsqueeze(0) - 0.5)) / self.cfg.temperature
        # AND 는 "가장 약한 상태가 정한다" 이므로 min 의 완화형을 쓴다.
        # 시그모이드의 곱을 쓰면 P->1 부근에서 log(1-P) 가 수치적으로 무너지고,
        # 정작 가장 고쳐야 할 확신에 찬 오검출에서 기울기가 사라진다.
        t = self.cfg.and_temperature
        wake_logit = -t * torch.logsumexp(-z / t, dim=1)
        return {"wake_logit": wake_logit, "count": count, "m": m}

    # -------------------------------------------------------------- 추론(경성)
    @torch.no_grad()
    def hard(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """반올림한 정수 상수로 회로와 똑같이 판정한다."""
        g, s = self.ternary()
        agree = g.unsqueeze(0) * (0.5 + s.unsqueeze(0) * (x - 0.5))
        count = agree.sum(dim=2)
        k = self.k.round().clamp(min=0)
        pass_s = (count >= k.unsqueeze(0)).float()              # [B, S]
        return {"wake": pass_s.prod(dim=1), "pass_s": pass_s, "count": count}

    # ------------------------------------------------------------- 초기화
    @torch.no_grad()
    def init_templates(self, xs_pos: torch.Tensor, scale: float = 1.0) -> None:
        """양성의 다수결 패턴으로 형판을 초기화한다.

        형판은 결국 "그 시각에 이 채널이 켜져 있는가"의 원형(prototype)이므로,
        무작위가 아니라 데이터에서 시작하는 것이 맞다. 문턱을 채널 평균으로
        초기화하는 것과 같은 이치다.

        무작위 초기화는 시드에 따라 학습이 무너진다(실측: acc 0.795±0.187,
        최저 0.574). 원형 초기화는 그 분산을 없앤다.

        xs_pos: [N, S, C] — 양성 클립만 모은 상태별 채널 패턴
        """
        if xs_pos.numel() == 0:
            return
        proto = xs_pos.mean(dim=0)                  # [S, C] in [0,1]
        self.weight.copy_((proto - 0.5) * 2.0 * scale)
        self.gate_logit.fill_(self.cfg.gate_init)

    # ------------------------------------------------------------- k 재적합
    @torch.no_grad()
    def fit_k(self, count: torch.Tensor, y: torch.Tensor,
              max_fpr: float = 0.05, sweeps: int = 4) -> torch.Tensor:
        """정수 k(s) 를 좌표상승으로 직접 고른다.

        k 는 정수 판정 문턱이라 경사하강에 맡기면 안 된다. 학습 초기에는 형판이
        무작위라 양성이 전부 탈락하고, 그 압력으로 k 가 바닥까지 내려간 뒤에는
        양성이 여유롭게 통과해 버려 다시 올릴 기울기가 사라진다(실측 확인).
        회로에서도 k 는 결국 반올림된 정수이므로 탐색이 정직하다.

        count [N, S], y [N] -> 갱신된 k [S]
        """
        m = self.used_channels()
        S = count.shape[1]
        k = self.k.detach().clone().round().clamp(min=0)
        pos, neg = (y > 0.5), (y <= 0.5)
        if pos.sum() == 0 or neg.sum() == 0:
            return k

        def score(kv: torch.Tensor) -> float:
            """FPR 을 상한 아래로 묶고 그 안에서 TPR 을 최대화한다.

            TPR - w*FPR 같은 가중합을 쓰면 w >= 2 에서 "전부 거부"(TPR 0, FPR 0)가
            최적이 되어 버린다(실측: w=3 에서 TPR 0.000). 상시 대기 회로의 사양은
            어차피 "FA/h 를 얼마 이하로 두고 검출률을 올린다" 이므로 그대로 쓴다.
            """
            ok = (count >= kv.unsqueeze(0)).all(dim=1)
            tpr = ok[pos].float().mean().item()
            fpr = ok[neg].float().mean().item()
            if fpr <= max_fpr:
                return 1.0 + tpr            # 가능 영역: TPR 최대화
            return -fpr                     # 불가능 영역: 일단 FPR 을 낮추는 쪽으로

        best = score(k)
        for _ in range(sweeps):
            moved = False
            for s in range(S):
                hi = int(m[s].item())
                for cand in range(0, hi + 1):
                    if cand == int(k[s].item()):
                        continue
                    trial = k.clone()
                    trial[s] = cand
                    sc = score(trial)
                    if sc > best + 1e-9:
                        best, k, moved = sc, trial, True
            if not moved:
                break
        self.k.copy_(k.to(self.k.dtype))
        return k

    # ------------------------------------------------------------------ 벌점
    def penalties(self) -> Dict[str, torch.Tensor]:
        """L1 은 X 를 늘려 저항을 줄이고, 하한은 너무 줄어드는 것을 막는다."""
        g, _ = self.ternary()
        m = g.sum(dim=1)
        l1 = self.cfg.l1_gate * g.sum() / (self.cfg.n_states * self.C)
        floor = self.cfg.min_channels_weight * torch.relu(
            self.cfg.min_channels - m).mean()
        return {"l1": l1, "floor": floor}

    # ------------------------------------------------------------------ 내보내기
    @torch.no_grad()
    def export(self, vdd: float = 1.8) -> Dict[str, object]:
        """학습 결과 -> 회로 상수."""
        g, s = self.ternary()
        M = (g * s).round().to(torch.int8)          # +1 / -1 / 0(=X)
        m = g.sum(dim=1)
        k = self.k.round().clamp(min=0)
        # V_TH(s) = VDD * (k - 0.5) / m(s). m 이 상태마다 달라 공유할 수 없다.
        vth = torch.where(m > 0, vdd * (k - 0.5) / m.clamp(min=1),
                          torch.zeros_like(m))
        # 판정 여유 = 한 채널이 바뀔 때의 전압 변화의 절반
        margin = torch.where(m > 0, vdd / (2 * m.clamp(min=1)),
                             torch.zeros_like(m))
        # k <= 0 이면 "0개 이상 일치"라 무조건 통과하고, m == 0 이면 볼 채널이
        # 없다. 둘 다 그 상태를 없는 것으로 만든다 -- 그런데 회로에서는 비교기와
        # 플립플롭을 그대로 먹으므로, 보드 크기를 정하기 전에 반드시 드러나야 한다.
        active = ((k > 0) & (m > 0)).cpu()
        return {
            "M": M.cpu(),                     # [S, C], 0 은 미연결(X)
            "k": k.cpu().to(torch.int32),
            "m": m.cpu().to(torch.int32),
            "V_TH": vth.cpu(),
            "margin_V": margin.cpu(),
            "n_resistors": int(m.sum().item()),
            "active": active,                 # [S] bool
            "n_active_states": int(active.sum().item()),
        }
