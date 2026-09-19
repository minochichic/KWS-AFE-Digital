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
    @staticmethod
    def _as_window(x: torch.Tensor) -> torch.Tensor:
        """[B,S,C] 를 [B,S,1,C] 로 올려 창 없는 호출도 같은 경로를 타게 한다."""
        return x.unsqueeze(2) if x.dim() == 3 else x

    def _counts(self, x: torch.Tensor):
        """x: [B, S, W, C] -> (count [B,S,W], m [S])."""
        g, sg = self.ternary()                                  # [S, C]
        # 채널별 일치도. sg=+1 이면 x, sg=-1 이면 1-x. g=0 이면 세지 않는다.
        #   0.5 + sg*(x - 0.5)  ==  x        (sg=+1)
        #                       ==  1 - x    (sg=-1)
        gv = g.view(1, -1, 1, self.C)
        sv = sg.view(1, -1, 1, self.C)
        agree = gv * (0.5 + sv * (x - 0.5))
        return agree.sum(dim=3), g.sum(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, S, C] 또는 [B, S, W, C] in {0,1}.

        W > 1 이면 창 안에서 **한 번이라도** 맞으면 통과다(OR). 회로에서는
        PASS 를 셋 우세 래치로 두고 디코더 출력 여러 개를 OR 해 만든다.

        반환
          wake_logit [B]     WAKE 로짓 — 상태별 여유의 soft-min
          count      [B, S]  상태별 일치 개수 (창 안 최대)
          m          [S]     상태별 사용 채널 수
        """
        x = self._as_window(x)
        if x.dim() != 4 or x.shape[1] != self.cfg.n_states or x.shape[3] != self.C:
            raise ValueError(
                f"x 는 [B, {self.cfg.n_states}, W, {self.C}] 여야 하는데 "
                f"{tuple(x.shape)} 다.")
        cw, m = self._counts(x)                                 # [B,S,W], [S]
        count = cw.amax(dim=2)                                  # OR = 창 안 최대

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
        cw, _ = self._counts(self._as_window(x))
        count = cw.amax(dim=2)
        k = self.k.round().clamp(min=1)
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

        xs_pos: [N, S, C] 또는 [N, S, W, C] — 양성 클립만 모은 패턴
        """
        if xs_pos.numel() == 0:
            return
        if xs_pos.dim() == 4:                       # 창의 중앙을 원형으로 쓴다
            xs_pos = xs_pos[:, :, xs_pos.shape[2] // 2, :]
        proto = xs_pos.mean(dim=0)                  # [S, C] in [0,1]
        self.weight.copy_((proto - 0.5) * 2.0 * scale)
        self.gate_logit.fill_(self.cfg.gate_init)

    @torch.no_grad()
    def balance_k(self, count: torch.Tensor, y: torch.Tensor,
                  max_fpr: float = 0.05) -> torch.Tensor:
        """학습용 k — 상태들이 고르게 거르도록 맞춘다.

        fit_k 를 학습 중에 쓰면 되먹임이 생긴다. fit_k 가 어떤 상태를 k=1 로
        두면 그 상태의 여유 z 가 크게 양수가 되고, soft-min AND 는 가장 약한
        상태에만 기울기를 보내므로 그 상태는 영영 기울기를 못 받는다. 못 배우니
        계속 쓸모없고, 쓸모없으니 계속 k=1 이다. 실측에서 s3/s4 의 형판이 끝까지
        전부 0(아무것도 못 배움)이었다.

        그래서 학습 중에는 "상태들이 독립이라면 각자 max_fpr^(1/S)" 지점에
        묶어 둔다. 동작점으로 좋아서가 아니라, 모든 상태가 기울기를 받게 하기
        위해서다. 진짜 동작점은 학습이 끝난 뒤 fit_k 가 고른다.
        """
        neg = y <= 0.5
        if neg.sum() < 8:
            return self.k.detach().clone()
        m = self.used_channels()
        S = count.shape[1]
        per = max(1e-6, max_fpr) ** (1.0 / max(S, 1))
        cn = count[neg].float()
        bal = torch.stack([
            cn[:, s].quantile(1.0 - per).ceil().clamp(1, max(1, int(m[s].item())))
            for s in range(S)]).to(self.k.dtype)
        self.k.copy_(bal)
        return bal

    # ------------------------------------------------------------- k 재적합
    @torch.no_grad()
    def fit_k(self, count: torch.Tensor, y: torch.Tensor,
              max_fpr: float = 0.05, sweeps: int = 4) -> torch.Tensor:
        """정수 k(s) 를 좌표상승으로 직접 고른다.

        k 는 정수 판정 문턱이라 경사하강에 맡기면 안 된다. 학습 초기에는 형판이
        무작위라 양성이 전부 탈락하고, 그 압력으로 k 가 바닥까지 내려간 뒤에는
        양성이 여유롭게 통과해 버려 다시 올릴 기울기가 사라진다(실측 확인).
        회로에서도 k 는 결국 반올림된 정수이므로 탐색이 정직하다.

        좌표상승은 한 번에 한 상태만 움직이므로 시작점에서 못 빠져나온다.
        "강한 상태 하나 + 나머지 무료 통과"와 "네 상태가 고르게 거름"은 둘 다
        FPR 상한을 만족하는데, 전자에서 후자로 가려면 여러 상태를 **동시에**
        움직여야 한다. 실측에서 계속 전자에 갇혔다(상태별 음성 통과율
        1.000 / 0.044 / 0.998 / 0.998).

        그래서 균형 잡힌 시드에서도 출발해 보고 더 나은 쪽을 택한다.

        count [N, S], y [N] -> 갱신된 k [S]
        """
        m = self.used_channels()
        S = count.shape[1]
        pos, neg = (y > 0.5), (y <= 0.5)
        cur = self.k.detach().clone().round().clamp(min=1)
        if pos.sum() == 0 or neg.sum() == 0:
            return cur

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

        def climb(seed: torch.Tensor):
            kk = seed.clone()
            bb = score(kk)
            for _ in range(sweeps):
                moved = False
                for s in range(S):
                    hi = int(m[s].item())
                    # k=0 은 "0개 이상 일치"라 무조건 통과다. 회로에서는 비교기와
                    # 플립플롭을 그대로 먹으면서 판정에 기여하지 않고, V_TH 가
                    # 음수(= 만들 수 없는 전압)가 된다. 후보에서 뺀다.
                    for cand in range(1, hi + 1):
                        if cand == int(kk[s].item()):
                            continue
                        trial = kk.clone()
                        trial[s] = cand
                        sc = score(trial)
                        if sc > bb + 1e-9:
                            bb, kk, moved = sc, trial, True
                if not moved:
                    break
            return bb, kk

        # 균형 시드. 상태들이 독립이라면 각자 max_fpr^(1/S) 를 내면 곱해서
        # 상한에 맞는다. 그 지점의 k 를 음성 count 분포에서 바로 읽는다.
        # 이게 없으면 "강한 상태 하나 + 나머지 무료 통과"에서 못 빠져나온다.
        per = max(1e-6, max_fpr) ** (1.0 / max(S, 1))
        cn = count[neg].float()
        bal = torch.stack([
            cn[:, s].quantile(1.0 - per).ceil().clamp(1, max(1, int(m[s].item())))
            for s in range(S)]).to(cur.dtype)

        # 양성을 거의 다 통과시키는 느슨한 시드도 하나
        cp = count[pos].float()
        loose = torch.stack([cp[:, s].quantile(0.10).floor()
                             for s in range(S)]).clamp(min=1).to(cur.dtype)

        best, k = max((climb(sd) for sd in (cur, bal, loose)),
                      key=lambda t: t[0])
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
        k = self.k.round().clamp(min=1)
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
