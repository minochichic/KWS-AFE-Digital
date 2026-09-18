"""게이트 수준 시뮬레이터 — 디지털 보드를 실제 부품 단위로 돌린다.

학습 모델(`model.hard()`)은 "start+tau 프레임을 뽑아 형판과 대조"라고 한 줄로
계산하지만, 실제 회로는 발진기·카운터·디코더·게이트·플립플롭이 시간을 따라
돌아가며 같은 결과를 내야 한다. 둘이 비트 단위로 일치하는지 확인하기 전에
스키매틱을 그리면 안 된다 (CLAUDE.md 5절: 골든 벡터 없이 RTL 쓰지 말 것 —
스키매틱도 같다).

여기서 일부러 모사하는 것들:
  · 카운터가 CLK 상승에서 증가하고, SAMPLE 은 DECODE·/CLK 로 후반부에 뜬다
  · 디코더가 덮는 비트 수 (2개면 6비트뿐 -> 64 카운트마다 별칭)
  · CLR 이 비동기라 RUN 과 PASS 를 즉시 되돌린다
  · IDLE 에서 카운터가 0 에 묶여 있다

모사하지 않는 것: 전파 지연, 준비/유지 시간, 소자 공차. 그건 별도 단계다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

# 프레임 하나를 두 위상으로 나눈다. 카운터는 CLK 상승(위상 0 시작)에서 바뀌고,
# SAMPLE 은 /CLK 가 1 인 위상 1 에서만 뜬다. 이것이 헛펄스를 피하는 방법이다.
PHASE_HIGH, PHASE_LOW = 0, 1


@dataclass
class DigitalBoard:
    """학습이 확정한 상수로 구성된 판별부 회로."""

    template: torch.Tensor          # [S, C] in {-1, 0, +1}; 0 = X(미연결)
    k: Sequence[int]                # [S] 상태별 허용 오차
    tau: Sequence[int]              # [S] 채점 시각 (프레임)
    timeout: int
    start_k: int
    counter_bits: int = 8
    # 3-to-8 디코더 개수. 2 면 하위 6비트만 디코드되어 64 카운트마다 별칭이 생긴다.
    n_decoders: int = 3

    def __post_init__(self) -> None:
        self.template = torch.as_tensor(self.template)
        if self.template.dim() != 2:
            raise ValueError(f"template 은 [S, C] 여야 한다: {tuple(self.template.shape)}")
        if len(self.k) != self.template.shape[0] or len(self.tau) != self.template.shape[0]:
            raise ValueError("k, tau 의 길이가 상태 수와 다르다.")

    # ------------------------------------------------------------------ 상수
    @property
    def n_states(self) -> int:
        return self.template.shape[0]

    @property
    def n_channels(self) -> int:
        return self.template.shape[1]

    @property
    def decode_mask(self) -> int:
        """디코더가 실제로 보는 비트. 나머지 상위 비트는 판정에 안 들어간다."""
        return (1 << min(3 * self.n_decoders, self.counter_bits)) - 1

    def decode_hits(self, count: int, value: int) -> bool:
        """카운터 == value 인가. 덮이지 않은 상위 비트는 무시된다."""
        m = self.decode_mask
        return (count & m) == (value & m)

    def aliases(self) -> Dict[int, List[int]]:
        """디코드 값마다, 같은 신호를 또 내는 카운트들. 비어 있어야 정상이다."""
        top = 1 << self.counter_bits
        out: Dict[int, List[int]] = {}
        for v in list(self.tau) + [self.timeout]:
            dup = [c for c in range(top)
                   if c != v and self.decode_hits(c, v) and c <= self.timeout]
            if dup:
                out[v] = dup
        return out

    # ---------------------------------------------------------------- 조합논리
    def _match(self, col: torch.Tensor) -> torch.Tensor:
        """MATCH(s) — 저항 평균 + 비교기. col: [B, C] in {0,1} -> [B, S]."""
        t = self.template.to(col.device)
        g = (t != 0).float()                       # 연결된 채널
        sgn = torch.sign(t).float()                # +1 이면 ON 요구, -1 이면 OFF 요구
        agree = g.unsqueeze(0) * (0.5 + sgn.unsqueeze(0) * (col.unsqueeze(1) - 0.5))
        cnt = agree.sum(dim=2)                     # [B, S]
        kk = torch.tensor(list(self.k), dtype=cnt.dtype, device=cnt.device)
        return (cnt >= kk.unsqueeze(0)).float()

    def _start(self, col: torch.Tensor) -> torch.Tensor:
        """START detect — 채널 합산 + 비교기. col: [B, C] -> [B]."""
        return (col.sum(dim=1) >= self.start_k).float()

    # ------------------------------------------------------------------ 실행
    @torch.no_grad()
    def run(self, x: torch.Tensor, trace: bool = False) -> Dict[str, object]:
        """이진 특징을 시간순으로 흘려보낸다.

        x: [B, C, T] in {0,1} — 비교기 출력 (프레임당 한 값, 회로에서는 10 ms)
        반환 wake [B], wake_frame [B] (WAKE 가 처음 뜬 프레임, 없으면 -1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, T = x.shape
        if C != self.n_channels:
            raise ValueError(f"채널이 {C}개인데 형판은 {self.n_channels}채널이다.")
        dev = x.device
        S = self.n_states

        run_ff = torch.zeros(B, device=dev)             # RUN latch
        cnt = torch.zeros(B, dtype=torch.long, device=dev)
        pass_ff = torch.zeros(B, S, device=dev)         # PASS F/F
        wake = torch.zeros(B, device=dev)
        wake_frame = torch.full((B,), -1, dtype=torch.long, device=dev)
        started = torch.zeros(B, device=dev)            # START 를 한 번이라도 봤나

        tau_t = torch.tensor(list(self.tau), device=dev)
        rows: List[dict] = []

        for f in range(T):
            col = x[:, :, f]                            # 이 프레임의 채널 값

            # ── 위상 0 (CLK 상승) : RUN=1 이면 카운터 증가, RUN=0 이면 MR 로 0 유지
            cnt = torch.where(run_ff > 0.5, cnt + 1, torch.zeros_like(cnt))
            # START 는 이 프레임 값으로 판정되고, RUN latch 를 세운다.
            # 카운터가 0 인 프레임이 START 프레임이 되도록 증가 뒤에 세운다.
            st = self._start(col)
            newly = (st > 0.5) & (run_ff < 0.5)
            run_ff = torch.where(newly, torch.ones_like(run_ff), run_ff)
            cnt = torch.where(newly, torch.zeros_like(cnt), cnt)
            started = torch.maximum(started, st)

            # ── 위상 1 (/CLK = 1) : 카운터가 안정된 뒤에만 채점한다
            active = run_ff > 0.5
            dec = torch.stack(
                [((cnt & self.decode_mask) == (int(t) & self.decode_mask)) & active
                 for t in tau_t], dim=1).float()        # [B, S]
            dec_to = (((cnt & self.decode_mask) ==
                       (self.timeout & self.decode_mask)) & active).float()

            match = self._match(col)                    # [B, S] 레벨, 상시 유효
            # SAMPLE 상승 에지에서 D 를 붙잡는다
            pass_ff = torch.where(dec > 0.5, match, pass_ff)

            w = pass_ff.prod(dim=1) * active.float()
            first = (w > 0.5) & (wake_frame < 0)
            wake_frame = torch.where(first, torch.full_like(wake_frame, f), wake_frame)
            wake = torch.maximum(wake, w)

            # ── timeout : 비동기 CLR 로 RUN 과 PASS 를 동시에 되돌린다
            clr = dec_to > 0.5
            run_ff = torch.where(clr, torch.zeros_like(run_ff), run_ff)
            pass_ff = torch.where(clr.unsqueeze(1), torch.zeros_like(pass_ff), pass_ff)
            cnt = torch.where(clr, torch.zeros_like(cnt), cnt)

            if trace:
                rows.append({
                    "f": f, "cnt": cnt.clone(), "run": run_ff.clone(),
                    "dec": dec.clone(), "match": match.clone(),
                    "pass": pass_ff.clone(), "wake": w.clone(),
                })

        out: Dict[str, object] = {"wake": wake, "wake_frame": wake_frame,
                                  "found": started}
        if trace:
            out["trace"] = rows
        return out


def board_from_export(rep: Dict, n_decoders: int = 3) -> DigitalBoard:
    """`export.build_report()` 결과에서 보드를 만든다."""
    tmpl = torch.tensor([s["template"] for s in rep["states"]], dtype=torch.int8)
    return DigitalBoard(
        template=tmpl,
        k=[s["k"] for s in rep["states"]],
        tau=[s["tau_frames"] for s in rep["states"]],
        timeout=rep["timeout_frames"],
        start_k=rep["start_k"],
        n_decoders=n_decoders,
    )


def board_from_model(model, n_decoders: int = 3) -> DigitalBoard:
    e = model.export()
    return DigitalBoard(
        template=e["M"],
        k=[int(v) for v in e["k"]],
        tau=list(e["tau"]),
        timeout=int(e["timeout"]),
        start_k=int(e["start_k"]),
        n_decoders=n_decoders,
    )
