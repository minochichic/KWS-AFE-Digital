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


def ripple_transients(count: int) -> List[int]:
    """리플 카운터가 count -> count+1 로 갈 때 잠깐 거치는 값들.

    LSB 부터 차례로 뒤집히므로, 끝자리 1 들이 하나씩 0 이 되는 동안의 값이
    출력에 나타난다. 11(1011) -> 10(1010) -> 8(1000) -> 12(1100).
    """
    out, c, bit = [], count, 1
    while c & bit:
        c &= ~bit
        out.append(c)
        bit <<= 1
    return out


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
    # 채점 창. tau(s) ± match_window 안에서 한 번이라도 맞으면 통과.
    # 회로에서는 PASS 를 에지 트리거 D 플립플롭이 아니라 **셋 우세 래치**로 두고,
    # 디코더 출력 2w+1 개를 OR 해 창을 만든다. 래치는 한 번 서면 CLR 까지 유지된다.
    match_window: int = 0
    # START 조건이 몇 프레임 연속이어야 하나 (시프트 레지스터 깊이). 1 이면 순간 조건.
    start_frames: int = 1
    # start_frames = N > 1 일 때 시각 원점을 맞추는 방법. tau·timeout 은 학습
    # 모델과 같이 "연속 구간의 첫 프레임" 기준으로 받고, 회로 카운트로는 여기서 바꾼다.
    #  "causal": RUN 은 N 번째 프레임에 선다(회로는 그때 처음 안다). 그 프레임이
    #            카운트 0 이고 채널은 지연 없이 본다 -> 디코드 값이 tau-(N-1) 로
    #            당겨지고, 음수가 되는 창 칸은 잘린다. 부품 추가 없음.
    #  "delay" : 채널을 N-1 프레임 지연(레지스터 8*(N-1)비트)해 Template 에 넣는다
    #            -> 학습 모델과 같은 함수. 지연 레지스터는 프레임 에지에서 표본화한다.
    start_mode: str = "causal"
    # 위험 재현용 스위치. 실제 회로는 둘 다 True 여야 한다.
    #  gate_run: S 에 RUN 을 넣는다. 빼면 IDLE 동안 MR 로 묶인 카운트 0 이 창으로 읽힌다.
    #  gate_clk: S 를 카운트가 안정된 반주기에만 연다. 빼면 리플 카운터가 올라가는
    #            순간 거치는 중간값(예: 11->12 는 10, 8 을 지난다)이 창으로 읽힌다.
    gate_run: bool = True
    gate_clk: bool = True

    def __post_init__(self) -> None:
        self.template = torch.as_tensor(self.template)
        if self.template.dim() != 2:
            raise ValueError(f"template 은 [S, C] 여야 한다: {tuple(self.template.shape)}")
        if len(self.k) != self.template.shape[0] or len(self.tau) != self.template.shape[0]:
            raise ValueError("k, tau 의 길이가 상태 수와 다르다.")
        if self.start_frames < 1:
            raise ValueError(f"start_frames 는 1 이상이어야 한다: {self.start_frames}")
        if self.start_mode not in ("causal", "delay"):
            raise ValueError(f"start_mode 는 causal/delay 중 하나: {self.start_mode}")
        for s in range(self.template.shape[0]):
            if not self.window_counts(s):
                raise ValueError(
                    f"상태 {s+1} 의 창이 전부 START 확인 이전이다 (tau={self.tau[s]}, "
                    f"창 ±{self.match_window}, 당김 {self.count_shift}). causal 로는 못 본다.")
        if self.timeout_count <= max(max(self.window_counts(s))
                                     for s in range(self.template.shape[0])):
            raise ValueError("timeout 이 마지막 창보다 앞이다.")

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

    # ------------------------------------------------- 학습 시각 -> 회로 카운트
    @property
    def data_delay(self) -> int:
        """Template 에 들어가는 채널의 지연 (프레임)."""
        return self.start_frames - 1 if self.start_mode == "delay" else 0

    @property
    def count_shift(self) -> int:
        """디코드 값을 학습 시각보다 몇 프레임 당기나."""
        return self.start_frames - 1 if self.start_mode == "causal" else 0

    def window_counts(self, s: int) -> List[int]:
        """상태 s 의 창을 이루는 카운터 값들 = 디코더 출력 중 OR 로 묶을 것."""
        w, sh = self.match_window, self.count_shift
        return [int(self.tau[s]) + o - sh for o in range(-w, w + 1)
                if int(self.tau[s]) + o - sh >= 0]

    @property
    def timeout_count(self) -> int:
        """CLR 을 거는 카운터 값. causal 이면 끝나는 절대 시각을 지키려고 같이 당긴다."""
        return int(self.timeout) - self.count_shift

    @property
    def wake_count(self) -> int:
        """마지막 상태의 창 중심 카운트 (연속 평가에서 사건 시각을 잡는 데 쓴다)."""
        return int(self.tau[-1]) - self.count_shift

    def aliases(self) -> Dict[int, List[int]]:
        """디코드 값마다, 같은 신호를 또 내는 카운트들. 비어 있어야 정상이다."""
        top = 1 << self.counter_bits
        out: Dict[int, List[int]] = {}
        vals = [c for s in range(self.n_states) for c in self.window_counts(s)]
        for v in vals + [self.timeout_count]:
            dup = [c for c in range(top)
                   if c != v and self.decode_hits(c, v) and c <= self.timeout_count]
            if dup:
                out[v] = dup
        return out

    # ---------------------------------------------------------------- 조합논리
    def _match_counts(self, col: torch.Tensor) -> torch.Tensor:
        """일치 개수 [B, S] — 비교기에 들어가기 전의 저항 평균 전압에 해당."""
        t = self.template.to(col.device)
        g = (t != 0).float()
        sgn = torch.sign(t).float()
        agree = g.unsqueeze(0) * (0.5 + sgn.unsqueeze(0) * (col.unsqueeze(1) - 0.5))
        return agree.sum(dim=2)

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
        """START 조건 한 프레임분 — 채널 합산 + 비교기. col: [B, C] -> [B].
        start_frames 연속 확인은 run/scan_segments 의 시프트 레지스터가 한다."""
        return (col.sum(dim=1) >= self.start_k).float()

    def _hits(self, count: int, s: int) -> bool:
        return any(self.decode_hits(count, v) for v in self.window_counts(s))

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

        if not self.gate_clk and self.match_window == 0:
            raise ValueError("gate_clk=False 재현은 창(latch) 구성에서만 한다.")

        run_ff = torch.zeros(B, device=dev)             # RUN latch
        cnt = torch.zeros(B, dtype=torch.long, device=dev)
        pass_ff = torch.zeros(B, S, device=dev)         # PASS F/F
        wake = torch.zeros(B, device=dev)
        wake_frame = torch.full((B,), -1, dtype=torch.long, device=dev)
        wake_seq = torch.zeros(B, T, device=dev)   # 연속 평가용 WAKE 시계열
        started = torch.zeros(B, device=dev)            # START 를 한 번이라도 봤나
        # START 시프트 레지스터 — 최근 start_frames 프레임의 START 조건
        hist = torch.zeros(B, self.start_frames, device=dev)
        d = self.data_delay
        to_c = self.timeout_count
        rows: List[dict] = []

        for f in range(T):
            col = x[:, :, f]                            # 이 프레임의 채널 값
            # Template 이 보는 값. delay 구성이면 지연 레지스터를 거친 값이다.
            col_t = x[:, :, f - d] if f >= d else torch.zeros_like(col)

            # ── 위상 0 (CLK 상승) : RUN=1 이면 카운터 증가, RUN=0 이면 MR 로 0 유지
            prev = cnt.clone()
            stepped = run_ff > 0.5
            cnt = torch.where(stepped, cnt + 1, torch.zeros_like(cnt))
            # START 조건을 시프트 레지스터에 넣고, 마지막 N 프레임이 모두 참이면
            # RUN latch 를 세운다. 카운터가 0 인 프레임 = START 를 확인한 프레임.
            hist = torch.cat([hist[:, 1:], self._start(col).unsqueeze(1)], dim=1)
            st = (hist > 0.5).all(dim=1)
            newly = st & (run_ff < 0.5)
            run_ff = torch.where(newly, torch.ones_like(run_ff), run_ff)
            cnt = torch.where(newly, torch.zeros_like(cnt), cnt)
            started = torch.maximum(started, st.float())

            # ── 위상 1 (/CLK = 1) : 카운터가 안정된 뒤에만 채점한다
            active = run_ff > 0.5
            gate = active if self.gate_run else torch.ones_like(active)
            m = self.decode_mask
            dec = torch.stack(
                [torch.stack([(cnt & m) == (v & m) for v in self.window_counts(s)],
                             0).any(0) & gate
                 for s in range(S)], dim=1).float()     # [B, S] -- 창 OR
            dec_to = (((cnt & m) == (to_c & m)) & active).float()

            match = self._match(col_t)                  # [B, S] 레벨, 상시 유효
            if self.match_window == 0:
                # SAMPLE 상승 에지에서 D 를 붙잡는다 (에지 트리거 D 플립플롭)
                pass_ff = torch.where(dec > 0.5, match, pass_ff)
            else:
                # 창 안에서 한 번이라도 맞으면 선다 (S = MATCH·WIN·CLK·RUN).
                # 한 번 선 PASS 는 CLR 까지 내려오지 않는다.
                pass_ff = torch.maximum(pass_ff, dec * match)
                if not self.gate_clk:
                    # CLK 로 막지 않으면 카운터가 올라가는 순간의 중간값도 S 를 연다
                    for b in stepped.nonzero().flatten().tolist():
                        for tv in ripple_transients(int(prev[b])):
                            for s in range(S):
                                if self._hits(tv, s) and match[b, s] > 0.5:
                                    pass_ff[b, s] = 1.0

            w = pass_ff.prod(dim=1) * active.float()
            first = (w > 0.5) & (wake_frame < 0)
            wake_frame = torch.where(first, torch.full_like(wake_frame, f), wake_frame)
            wake = torch.maximum(wake, w)
            wake_seq[:, f] = w

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
                                  "found": started, "wake_seq": wake_seq}
        if trace:
            out["trace"] = rows
        return out


@torch.no_grad()
def scan_segments(board: "DigitalBoard", x: torch.Tensor):
    """상태기계를 한 번 돌려 계측 구간마다 상태별 최대 일치 개수를 기록한다.

    CLR 은 timeout 만 보고 k 와 무관하므로, 구간 분할과 채점 시각은 k 가 바뀌어도
    똑같다. 따라서 구간마다 "각 상태가 채점 시각들에서 본 최대 count" 만 알면
    어떤 k 에 대해서도 PASS = (maxcount >= k) 로 답이 나온다. 창을 쓰는 설정에서
    PASS 는 셋 우세 래치이므로 창 안 최대가 곧 통과 여부다.

    반환 (starts [n_seg], maxcount [n_seg, S]) -- starts 는 각 구간에서 RUN 이 선
    프레임(START 를 확인한 프레임). 사건 시각은 starts + board.wake_count 다.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] != 1:
        raise ValueError("연속 스캔은 스트림 하나만 받는다.")
    if not (board.gate_run and board.gate_clk):
        raise ValueError("scan_segments 는 올바른 회로(gate_run, gate_clk)만 돈다.")
    B, C, T = x.shape
    S = board.n_states
    d, to_c = board.data_delay, board.timeout_count
    hist = [0.0] * board.start_frames                      # START 시프트 레지스터

    run_ff, cnt = 0.0, 0
    seg_start, seg_max = None, None
    starts, maxes = [], []
    for f in range(T):
        col = x[:, :, f]
        col_t = x[:, :, f - d] if f >= d else torch.zeros_like(col)
        cnt = cnt + 1 if run_ff > 0.5 else 0
        hist = hist[1:] + [float(board._start(col)[0])]
        if all(h > 0.5 for h in hist) and run_ff < 0.5:
            run_ff, cnt = 1.0, 0
            seg_start = f
            seg_max = torch.zeros(S)
        if run_ff > 0.5:
            m = board._match_counts(col_t)[0]               # [S]
            for si in range(S):
                if board._hits(cnt, si):
                    seg_max[si] = max(float(seg_max[si]), float(m[si]))
            if board.decode_hits(cnt, to_c):
                starts.append(seg_start); maxes.append(seg_max)
                run_ff, cnt, seg_start, seg_max = 0.0, 0, None, None
    if seg_start is not None:
        starts.append(seg_start); maxes.append(seg_max)
    if not starts:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, S)
    return torch.tensor(starts), torch.stack(maxes)


START_CHOICES = ("single", "causal", "delay")


def _start_args(start: str, min_frames: int) -> Dict[str, object]:
    """START 구현 선택 -> DigitalBoard 인자.

    single : 1프레임 순간 조건. 2026-10 까지의 연속 스트림 수치(FA/h, 검출률)가
             전부 이 회로로 잰 것이다 -- 학습 설정의 min_frames 를 무시했었다.
    causal : min_frames 연속, 원점은 확인한 프레임. 부품 추가 없음.
    delay  : min_frames 연속, 채널 지연 레지스터로 학습 모델과 같은 원점.
    """
    if start not in START_CHOICES:
        raise ValueError(f"start 는 {START_CHOICES} 중 하나: {start}")
    if start == "single":
        return {"start_frames": 1, "start_mode": "causal"}
    return {"start_frames": int(min_frames), "start_mode": start}


def board_from_export(rep: Dict, *, start: str, n_decoders: int = 3) -> DigitalBoard:
    """`export.build_report()` 결과에서 보드를 만든다."""
    tmpl = torch.tensor([s["template"] for s in rep["states"]], dtype=torch.int8)
    return DigitalBoard(
        template=tmpl,
        k=[s["k"] for s in rep["states"]],
        tau=[s["tau_frames"] for s in rep["states"]],
        timeout=rep["timeout_frames"],
        start_k=rep["start_k"],
        n_decoders=n_decoders,
        match_window=int(rep.get("match_window", 0)),
        **_start_args(start, int(rep.get("start_min_frames", 1))),
    )


def board_from_model(model, *, start: str, n_decoders: int = 3) -> DigitalBoard:
    """start 에 기본값을 두지 않는다. 기본값이 있으면 어느 START 회로로 잰
    수치인지가 호출부에서 안 보인다 -- 실제로 그렇게 min_frames 가 빠졌다."""
    e = model.export()
    return DigitalBoard(
        template=e["M"],
        k=[int(v) for v in e["k"]],
        tau=list(e["tau"]),
        timeout=int(e["timeout"]),
        start_k=int(e["start_k"]),
        n_decoders=n_decoders,
        match_window=int(model.cfg.head.match_window),
        **_start_args(start, model.cfg.start.min_frames),
    )
