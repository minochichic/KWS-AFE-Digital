"""디지털 판별부 넷리스트 — 부품·핀·신호 단위. 그리고 그 넷리스트 자체를 돌리는 논리 시뮬.

gatelevel.DigitalBoard 는 "카운터가 몇이면 창이 열린다" 수준의 블록 모델이다.
여기서는 같은 회로를 **실제 칩의 핀**으로 펼치고(어느 138 출력이 어느 NAND 입력에
가는지, 어느 저항이 CH 에 붙고 어느 것이 /CH 에 붙는지), 그 배선 그대로 논리를
돌려 DigitalBoard 와 프레임 단위로 대조한다. 배선 실수는 여기서 잡혀야 한다 --
KiCad 로 옮긴 뒤에는 눈으로만 확인할 수 있다.

  python -m wakeup.netlist --params hardware/f256_ok_s3.json --out hardware   (8.31/ 에서)

전원 1.8 V, 74LVC/74LV 계열. 74HC 는 권장 VCC 하한이 2.0 V 라 쓸 수 없다.

시뮬에서 일부러 모사하는 것
  · 74LV393 이 CP 하강 에지에서 세고, MR 이 비동기로 0 을 잡는다 (두 단 리플 연결)
  · 74LVC74A 의 /SD·/RD 비동기 동작. 둘 다 Low 면 Q=/Q=High
  · CLR 이 카운터를 되돌려 스스로 꺼지는 펄스라는 것, 그 순간의 START re-arm
  · 판정 노드 전압을 저항값으로 계산 (E96 값 그대로)
모사하지 않는 것
  · 32.768 kHz 발진과 분주 -- CLK 를 직접 넣는다. 발진은 시제품에서 확인한다
  · 전파 지연과 리플 중간값 -- S 를 CLK 로 막는 근거는 gatelevel 의 위험 재현
    테스트(gate_clk=False)에 있다
  · 채널 입력은 프레임 경계(CLK 하강)에서만 바뀐다고 본다
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .gatelevel import DigitalBoard

VDD, GND, NC = "VDD", "GND", "NC"
# 외부에서 값을 넣는 넷. 시뮬에서 칩 모델이 아니라 시험 벡터가 정한다.
CLK, POR_RC = "CLK", "POR_RC"


# ===================================================================== 부품 라이브러리
@dataclass(frozen=True)
class PartType:
    name: str
    package: str
    pins: Dict[str, int]
    source: str                      # 핀 배치의 근거
    outputs: Tuple[str, ...] = ()    # 출력 핀 (넷 구동 검사용)
    power: Tuple[str, str] = ("VCC", "GND")


def _gate14(name: str, src: str) -> PartType:
    """7400/7408 배치 -- 2입력 게이트 4개."""
    pins = {"1A": 1, "1B": 2, "1Y": 3, "2A": 4, "2B": 5, "2Y": 6, "GND": 7,
            "3Y": 8, "3A": 9, "3B": 10, "4Y": 11, "4A": 12, "4B": 13, "VCC": 14}
    return PartType(name, "SO14", pins, src, ("1Y", "2Y", "3Y", "4Y"))


LIB: Dict[str, PartType] = {
    "74LVC14A": PartType(
        "74LVC14A", "SO14",
        {"1A": 1, "1Y": 2, "2A": 3, "2Y": 4, "3A": 5, "3Y": 6, "GND": 7,
         "4Y": 8, "4A": 9, "5Y": 10, "5A": 11, "6Y": 12, "6A": 13, "VCC": 14},
        "7414 표준 배치 (데이터시트 대조 필요)", ("1Y", "2Y", "3Y", "4Y", "5Y", "6Y")),
    "74LVC08A": _gate14("74LVC08A", "7408 표준 배치 (데이터시트 대조 필요)"),
    "74LVC00A": _gate14("74LVC00A", "7400 표준 배치 (데이터시트 대조 필요)"),
    "74LVC30A": PartType(
        "74LVC30A", "SO14",
        {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "GND": 7, "Y": 8,
         "NC9": 9, "NC10": 10, "G": 11, "H": 12, "NC13": 13, "VCC": 14},
        "7430 표준 배치 (Nexperia 74LVC30A 존재·1.8 V 확인, 핀 대조 필요)", ("Y",)),
    "74LVC138A": PartType(
        "74LVC138A", "SO16",
        {"A0": 1, "A1": 2, "A2": 3, "/E1": 4, "/E2": 5, "E3": 6, "/Y7": 7, "GND": 8,
         "/Y6": 9, "/Y5": 10, "/Y4": 11, "/Y3": 12, "/Y2": 13, "/Y1": 14, "/Y0": 15,
         "VCC": 16},
        "74138 표준 배치 (데이터시트 대조 필요)",
        tuple(f"/Y{i}" for i in range(8))),
    "74LVC74A": PartType(
        "74LVC74A", "SO14",
        {"1/RD": 1, "1D": 2, "1CP": 3, "1/SD": 4, "1Q": 5, "1/Q": 6, "GND": 7,
         "2/Q": 8, "2Q": 9, "2/SD": 10, "2CP": 11, "2D": 12, "2/RD": 13, "VCC": 14},
        "7474 표준 배치 (데이터시트 대조 필요)", ("1Q", "1/Q", "2Q", "2/Q")),
    "74LV393": PartType(
        "74LV393", "SO14",
        {"1CP": 1, "1MR": 2, "1Q0": 3, "1Q1": 4, "1Q2": 5, "1Q3": 6, "GND": 7,
         "2Q3": 8, "2Q2": 9, "2Q1": 10, "2Q0": 11, "2MR": 12, "2CP": 13, "VCC": 14},
        "74393 표준 배치 (데이터시트 대조 필요)",
        ("1Q0", "1Q1", "1Q2", "1Q3", "2Q0", "2Q1", "2Q2", "2Q3")),
    "74LV4060": PartType(
        "74LV4060", "SO16",
        {"Q11": 1, "Q12": 2, "Q13": 3, "Q5": 4, "Q4": 5, "Q6": 6, "Q3": 7, "GND": 8,
         "CTC": 9, "RTC": 10, "RS": 11, "MR": 12, "Q8": 13, "Q7": 14, "Q9": 15,
         "VCC": 16},
        "Nexperia 74LV4060 Rev.5 표 2 (확인). Q7 = ÷256",
        ("Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q11", "Q12", "Q13", "RTC", "CTC")),
    "LPV7215": PartType(
        "LPV7215", "SOT-23-5",
        {"OUT": 1, "V-": 2, "IN+": 3, "IN-": 4, "V+": 5},
        "TI LPV7215 데이터시트 5절 Pin Functions, DBV (확인)", ("OUT",),
        power=("V+", "V-")),
    "MMBT3904": PartType(
        "MMBT3904", "SOT-23", {"B": 1, "E": 2, "C": 3},
        "SOT-23 일반 배치 (제조사 데이터시트 대조 필요)", (), power=()),
}

# 게이트 종류 -> (부품, 단위별 (입력 핀들, 출력 핀))
GATE_UNITS: Dict[str, Tuple[str, List[Tuple[Tuple[str, ...], str]]]] = {
    "inv": ("74LVC14A", [((f"{i}A",), f"{i}Y") for i in range(1, 7)]),
    "and2": ("74LVC08A", [((f"{i}A", f"{i}B"), f"{i}Y") for i in range(1, 5)]),
    "nand2": ("74LVC00A", [((f"{i}A", f"{i}B"), f"{i}Y") for i in range(1, 5)]),
    "nand8": ("74LVC30A", [(("A", "B", "C", "D", "E", "F", "G", "H"), "Y")]),
}
FUNC = {"74LVC14A": "inv", "74LVC08A": "and2", "74LVC00A": "nand2", "74LVC30A": "nand8"}

E96 = (100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140,
       143, 147, 150, 154, 158, 162, 165, 169, 174, 178, 182, 187, 191, 196, 200,
       205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267, 274, 280, 287,
       294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
       422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549, 562, 576, 590,
       604, 619, 634, 649, 665, 681, 698, 715, 732, 750, 768, 787, 806, 825, 845,
       866, 887, 909, 931, 953, 976)
LPV7215_VOS_MAX = 8e-3          # V, 1.8 V 온도 전 범위 (데이터시트 6.5)


def e96_divider(ratio: float, total=(0.9e6, 1.1e6)) -> Tuple[float, float]:
    """R8/(R7+R8) = ratio 에 가장 가까운 E96 쌍 (합계 약 1 MΩ)."""
    vals = [m * 1e2 for m in E96] + [m * 1e3 for m in E96] + [m * 1e4 for m in E96 if m < 110]
    best = None
    for r7 in vals:
        for r8 in vals:
            if not total[0] <= r7 + r8 <= total[1]:
                continue
            err = abs(r8 / (r7 + r8) - ratio)
            if best is None or err < best[0]:
                best = (err, r7, r8)
    return best[1], best[2]


def fmt_r(r: float) -> str:
    if r >= 1e6:
        return f"{r/1e6:g}M"
    if r >= 1e3:
        return f"{r/1e3:g}k"
    return f"{r:g}"


# ======================================================================= 넷리스트
@dataclass
class Passive:
    ref: str
    kind: str          # R C Y D
    value: str
    a: str             # 넷 (D 는 a=애노드, b=캐소드)
    b: str
    ohms: float = 0.0
    note: str = ""


@dataclass
class Netlist:
    parts: Dict[str, Tuple[str, Dict[str, str]]] = field(default_factory=dict)
    passives: List[Passive] = field(default_factory=list)
    connector: Dict[str, str] = field(default_factory=dict)   # 신호 -> 50핀 번호(미정)
    notes: List[str] = field(default_factory=list)
    open_items: List[str] = field(default_factory=list)
    params: Dict = field(default_factory=dict)
    analog: Dict[str, Dict] = field(default_factory=dict)     # 판정 노드 요약

    # ---------------------------------------------------------------- 조회
    def pin_rows(self) -> List[Tuple[str, str, int, str, str]]:
        rows = []
        for ref in sorted(self.parts, key=_refkey):
            pt, conn = self.parts[ref]
            t = LIB[pt]
            for name, num in sorted(t.pins.items(), key=lambda kv: kv[1]):
                rows.append((ref, pt, num, name, conn.get(name, NC)))
        return rows

    def net_map(self) -> Dict[str, List[str]]:
        nets: Dict[str, List[str]] = defaultdict(list)
        for ref, pt, num, name, net in self.pin_rows():
            if net != NC:
                nets[net].append(f"{ref}.{num}({name})")
        for p in self.passives:
            nets[p.a].append(f"{p.ref}.1")
            nets[p.b].append(f"{p.ref}.2")
        for sig in self.connector:
            nets[sig].append(f"J1.{self.connector[sig]}({sig})")
        return dict(nets)

    def drivers(self) -> Dict[str, List[str]]:
        d: Dict[str, List[str]] = defaultdict(list)
        for ref, (pt, conn) in self.parts.items():
            for o in LIB[pt].outputs:
                if conn.get(o, NC) != NC:
                    d[conn[o]].append(f"{ref}.{o}")
        for sig in self.connector:
            if sig not in (VDD, GND):
                d[sig].append(f"J1({sig})")
        return dict(d)

    # ---------------------------------------------------------------- 검사
    def check(self) -> List[str]:
        """배선 규칙 위반을 문자열로 돌려준다. 비어 있어야 정상이다."""
        errs = []
        drv = self.drivers()
        for net, ds in drv.items():
            if len(ds) > 1 and net not in (VDD, GND):
                errs.append(f"넷 {net} 을 둘 이상이 구동한다: {ds}")
            if net in (VDD, GND):
                errs.append(f"출력 {ds} 가 전원 {net} 에 물려 있다")
        resistive = {p.a for p in self.passives} | {p.b for p in self.passives}
        for ref, pt, num, name, net in self.pin_rows():
            t = LIB[pt]
            if name.startswith("NC"):
                continue
            if name in t.power:
                want = VDD if name == t.power[0] else GND
                if net != want:
                    errs.append(f"{ref}.{num}({name}) 전원이 {net}")
                continue
            if name in t.outputs:
                continue
            if net == NC:
                errs.append(f"{ref}.{num}({name}) 입력이 떠 있다")
            elif net not in (VDD, GND) and net not in drv and net not in resistive:
                errs.append(f"{ref}.{num}({name}) 의 넷 {net} 을 아무도 구동하지 않는다")
        nets = self.net_map()
        for net, users in nets.items():
            if len(users) < 2 and net not in (VDD, GND):
                errs.append(f"넷 {net} 에 연결이 하나뿐이다: {users}")
        return errs


def _refkey(ref: str) -> Tuple[str, int]:
    head = ref.rstrip("0123456789")
    return head, int(ref[len(head):] or 0)


class _Builder:
    def __init__(self) -> None:
        self.nl = Netlist()
        self.n = defaultdict(int)
        self.free: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.purpose: Dict[str, List[str]] = defaultdict(list)

    def ref(self, prefix: str) -> str:
        self.n[prefix] += 1
        return f"{prefix}{self.n[prefix]}"

    def part(self, pt: str, conn: Dict[str, str], why: str = "") -> str:
        r = self.ref("U" if pt not in ("MMBT3904",) else "Q")
        self.nl.parts[r] = (pt, dict(conn))
        if why:
            self.purpose[r].append(why)
        return r

    def gate(self, kind: str, inputs: Sequence[str], out: str, why: str = "") -> str:
        pt, units = GATE_UNITS[kind]
        if not self.free[kind]:
            r = self.part(pt, {})
            self.free[kind] = [(r, i) for i in range(len(units))]
        r, u = self.free[kind].pop(0)
        ins, o = units[u]
        if len(inputs) > len(ins):
            raise ValueError(f"{kind} 입력이 {len(ins)}개뿐인데 {len(inputs)}개를 달았다")
        conn = self.nl.parts[r][1]
        pad = VDD if kind.startswith("nand") or kind.startswith("and") else GND
        for i, pin in enumerate(ins):
            conn[pin] = inputs[i] if i < len(inputs) else pad
        conn[o] = out
        if why:
            self.purpose[r].append(why)
        return r

    def res(self, a: str, b: str, ohms: float, note: str = "", value: str = "") -> None:
        self.nl.passives.append(Passive(self.ref("R"), "R", value or fmt_r(ohms),
                                        a, b, ohms, note))

    def cap(self, a: str, b: str, value: str, note: str = "") -> None:
        self.nl.passives.append(Passive(self.ref("C"), "C", value, a, b, 0.0, note))

    def finish(self) -> Netlist:
        # 남는 게이트: 입력은 GND 에 묶고 출력은 띄운다 (CMOS 입력을 띄우면 안 된다)
        for kind, left in self.free.items():
            pt, units = GATE_UNITS[kind]
            for r, u in left:
                ins, o = units[u]
                conn = self.nl.parts[r][1]
                for pin in ins:
                    conn[pin] = GND
                conn[o] = NC
                self.purpose[r].append("(남는 단위: 입력 GND)")
        for r, (pt, conn) in self.nl.parts.items():
            t = LIB[pt]
            if t.power:
                conn.setdefault(t.power[0], VDD)
                conn.setdefault(t.power[1], GND)
            for name in t.pins:
                conn.setdefault(name, NC)
        return self.nl


# ======================================================================= 생성
def build_netlist(board: DigitalBoard, *, vdd: float = 1.8, r_template: float = 1e6,
                  name: str = "") -> Netlist:
    """확정 상수(DigitalBoard)를 부품·핀 단위 넷리스트로 펼친다."""
    if board.start_frames != 1 or board.start_k != 1:
        raise ValueError("이 생성기는 1프레임 START, 채널 1개 이상(OR) 만 만든다.")
    if not (board.gate_run and board.gate_clk) or board.match_window == 0:
        raise ValueError("S 게이팅이 있는 latch 구성(match_window > 0)만 만든다.")
    S, C = board.n_states, board.n_channels
    to = board.timeout_count
    if not 0 < to <= 63:
        raise ValueError(f"timeout {to} 은 6비트(<=63)여야 한다.")
    b = _Builder()
    nl = b.nl
    nl.params = {"name": name, "vdd": vdd, "template": board.template.tolist(),
                 "k": list(board.k), "tau": list(board.tau),
                 "match_window": board.match_window, "timeout": to,
                 "windows": [board.window_counts(s) for s in range(S)]}
    ch = [f"CH{c}" for c in range(C)]
    nch = [f"/CH{c}" for c in range(C)]
    for c in range(C):
        nl.connector[ch[c]] = "ICD"
    nl.connector[VDD] = "ICD"
    nl.connector[GND] = "ICD"

    # ── 클럭: 32.768 kHz crystal + 74LV4060, Q7(14번 핀) = ÷256 = 128 Hz
    b.part("74LV4060", {"RS": "OSC_IN", "RTC": "OSC_OUT", "MR": GND, "Q7": CLK},
           "프레임 클럭 128 Hz")
    nl.passives.append(Passive(b.ref("Y"), "Y", "32.768kHz 12.5pF", "OSC_IN", "OSC_X",
                               note="AB26TRB-32.768KHZ-T 또는 FC-135"))
    b.res("OSC_IN", "OSC_OUT", 10e6, "Rbias (시제품 조정)")
    b.res("OSC_OUT", "OSC_X", 330e3, "drive 제한 (시제품 조정)")
    b.cap("OSC_IN", GND, "22pF", "부하 C (시제품 조정)")
    b.cap("OSC_X", GND, "22pF", "부하 C (시제품 조정)")

    # ── 채널 반전 (Template 의 0 자리와 START OR 에 쓴다)
    for c in range(C):
        b.gate("inv", [ch[c]], nch[c], f"/CH{c}")

    # ── POR: VDD-C-노드-R-GND. 켜는 순간 노드가 High -> 반전 -> POR_N Low
    b.cap(VDD, POR_RC, "1uF", "POR")
    b.res(POR_RC, GND, 1e6, "POR (τ ≈ 1 s)")
    b.gate("inv", [POR_RC], "POR_N", "POR (Schmitt)")

    # ── START: OR(CH) = NAND(/CH). RUN latch 의 /SD 는 POR 동안 막는다
    b.gate("nand8", nch, "START", "START = OR(CH0..CH7)")
    b.gate("nand2", ["START", "POR_N"], "/SD_RUN", "RUN set")

    # ── 프레임 카운터: 74LV393 두 단 리플. RUN=0 이면 MR 로 0 에 묶인다
    b.part("74LV393", {"1CP": CLK, "1MR": "RUN_N", "1Q0": "C0", "1Q1": "C1",
                       "1Q2": "C2", "1Q3": "C3", "2CP": "C3", "2MR": "RUN_N",
                       "2Q0": "C4", "2Q1": "C5"}, "프레임 카운터 6비트")
    inv_bits: Dict[int, str] = {}

    def nbit(i: int) -> str:
        if i not in inv_bits:
            inv_bits[i] = f"/C{i}"
            b.gate("inv", [f"C{i}"], f"/C{i}", f"/C{i}")
        return inv_bits[i]

    # ── 시각 디코더: 그룹(상위 3비트)마다 74LVC138A 하나, 하위 3비트가 주소
    used = {v for s in range(S) for v in board.window_counts(s)}
    groups = sorted({v >> 3 for v in used})
    yout: Dict[int, str] = {}
    for g in groups:
        bits = {i: (g >> (i - 3)) & 1 for i in (3, 4, 5)}
        ones = [i for i in (3, 4, 5) if bits[i]]
        zeros = [i for i in (3, 4, 5) if not bits[i]]
        low: List[str] = []     # /E1, /E2 -- 0 일 때 통과
        if ones:
            high = f"C{ones[0]}"
            low += [nbit(i) for i in ones[1:]]
        else:
            high = nbit(zeros[-1])
            zeros = zeros[:-1]
        low += [f"C{i}" for i in zeros]
        assert len(low) == 2
        conn = {"A0": "C0", "A1": "C1", "A2": "C2", "/E1": low[0], "/E2": low[1],
                "E3": high}
        for a in range(8):
            v = 8 * g + a
            if v in used:
                yout[v] = f"/D{v}"
            conn[f"/Y{a}"] = yout.get(v, NC)
        b.part("74LVC138A", conn, f"디코드 {8*g}..{8*g+7}")

    # ── 창: 디코더 출력(active low) 들의 NAND = OR
    for s in range(S):
        wc = board.window_counts(s)
        if len(wc) > 8:
            raise ValueError(f"상태 {s+1} 창이 {len(wc)}칸 -- 74LVC30A 입력 8개를 넘는다")
        b.gate("nand8", [yout[v] for v in wc], f"WIN{s+1}",
               f"WIN{s+1} = 카운트 {wc[0]}..{wc[-1]}")

    # ── timeout: NAND(카운터 비트 패턴, CLK) = CLR_N. CLK 로 막아 리플 중간값을 피한다
    pat = [f"C{i}" if (to >> i) & 1 else nbit(i) for i in range(6)]
    b.gate("nand8", pat + [CLK], "CLR_N", f"CLR = 카운트 {to} · CLK")
    b.gate("and2", ["CLR_N", "POR_N"], "/RD", "/RD = CLR_N · POR_N")

    # ── RUN latch (74LVC74A 비동기만 사용: D, CP 는 GND)
    ffs: List[Tuple[str, str, str]] = [("/SD_RUN", "RUN", "RUN_N")]
    # ── Template 정합: 같은 값 저항 m 개 -> 비교기 IN+, 분압 -> IN-
    for s in range(S):
        row = board.template[s].tolist()
        m = sum(1 for t in row if t != 0)
        k = int(board.k[s])
        for c, t in enumerate(row):
            if t == 0:
                continue
            b.res(ch[c] if t > 0 else nch[c], f"SC{s+1}", r_template,
                  f"s{s+1} ch{c} ({'1' if t > 0 else '0'})")
        ratio = (k - 0.5) / m
        r7, r8 = e96_divider(ratio)
        b.res(VDD, f"VTH{s+1}", r7, f"s{s+1} 분압 상단 R7")
        b.res(f"VTH{s+1}", GND, r8, f"s{s+1} 분압 하단 R8")
        vth = vdd * r8 / (r7 + r8)
        step = vdd / m
        margin = min(vth - (k - 1) * step, k * step - vth)
        nl.analog[f"s{s+1}"] = {
            "m": m, "k": k, "R7": r7, "R8": r8, "vth": vth,
            "vth_ideal": vdd * ratio, "step": step, "margin": margin,
            "margin_ideal": vdd / (2 * m),
            # n 개 High 일 때 VDD 에서 흘러드는 전류 = VDD·n(m-n)/(m·R), n=m/2 에서 최대
            "worst_template_uA": vdd * (m // 2) * (m - m // 2) / (m * r_template) * 1e6,
        }
        b.part("LPV7215", {"IN+": f"SC{s+1}", "IN-": f"VTH{s+1}",
                           "OUT": f"MATCH{s+1}", "V+": VDD, "V-": GND},
               f"s{s+1} 정합 비교기")
        # S = MATCH · WIN · CLK · RUN  ->  /S = NAND(MATCH·GATE, WIN)
        if s == 0:
            b.gate("and2", [CLK, "RUN"], "GATE", "GATE = CLK · RUN")
        b.gate("and2", [f"MATCH{s+1}", "GATE"], f"X{s+1}", f"X{s+1} = MATCH{s+1}·GATE")
        b.gate("nand2", [f"X{s+1}", f"WIN{s+1}"], f"/S{s+1}", f"/S{s+1}")
        ffs.append((f"/S{s+1}", f"PASS{s+1}", NC))

    for i in range(0, len(ffs), 2):
        conn: Dict[str, str] = {}
        for j, (sd, q, qn) in enumerate(ffs[i:i + 2], start=1):
            conn.update({f"{j}/SD": sd, f"{j}/RD": "/RD", f"{j}D": GND,
                         f"{j}CP": GND, f"{j}Q": q, f"{j}/Q": qn})
        if len(ffs[i:i + 2]) == 1:          # 남는 FF: 비동기 입력 High 로 묶는다
            conn.update({"2/SD": VDD, "2/RD": VDD, "2D": GND, "2CP": GND})
        b.part("74LVC74A", conn, " / ".join(q for _, q, _ in ffs[i:i + 2]))

    # ── WAKE = 모든 PASS 의 AND
    acc = "PASS1"
    for s in range(1, S):
        out = "WAKE" if s == S - 1 else f"PASS1_{s+1}"
        b.gate("and2", [acc, f"PASS{s+1}"], out, out)
        acc = out
    if S == 1:
        raise ValueError("상태 1개는 이 생성기에서 다루지 않는다.")

    # ── 출력단: WAKE -> NPN -> LED
    b.res("WAKE", "Q_B", 10e3, "베이스 저항")
    q = b.part("MMBT3904", {"B": "Q_B", "E": GND, "C": "LED_K"}, "LED 드라이버")
    nl.passives.append(Passive(b.ref("D"), "D", "LED", "LED_A", "LED_K"))
    b.res("VLED", "LED_A", 680, "LED 전류 (VLED 3.3 V 가정, 약 2 mA)")
    nl.connector["VLED"] = "미정"

    nl = b.finish()
    # 디커플링: IC 마다 100 nF
    for r, (pt, conn) in sorted(nl.parts.items(), key=lambda kv: _refkey(kv[0])):
        if pt != "MMBT3904":
            nl.passives.append(Passive(b.ref("C"), "C", "100nF", VDD, GND,
                                       note=f"{r} 디커플링"))
    nl.passives.append(Passive(b.ref("C"), "C", "10uF", VDD, GND, note="벌크"))
    nl.notes = [f"{r}: {', '.join(p)}" for r, p in
                sorted(b.purpose.items(), key=lambda kv: _refkey(kv[0]))]
    nl.open_items = [
        "50핀 커넥터 핀 번호 -- ICD 확정 후 (CH0..CH7, VDD, GND)",
        "VLED: 1.8 V 로는 LED 순방향 전압을 못 넘는다. 3.3 V 등 별도 레일 필요",
        "32.768 kHz 발진 상수(Rbias, drive 저항, 부하 C) -- 1.8 V 에서 시제품 확인",
        "CLR 은 자기 종료 펄스다 (CLR -> RUN 리셋 -> 카운터 0 -> 디코드 해제). "
        "74LVC74A /RD 최소 펄스 폭과 대조",
        "표준 배치로 적은 핀 번호 -- 실제 구매 부품 데이터시트와 대조",
    ]
    return nl


# =================================================================== 넷리스트 시뮬
class HazardError(RuntimeError):
    pass


class NetSim:
    """넷리스트 그대로 논리를 돌린다. 칩 모델 + 저항 판정 노드."""

    def __init__(self, nl: Netlist, vdd: float = 1.8, *, init_ff: int = 1,
                 init_count: int = 5) -> None:
        self.nl, self.vdd = nl, vdd
        self.v: Dict[str, int] = defaultdict(int)
        self.v[VDD], self.v[GND] = 1, 0
        self.ext = {VDD, GND, CLK, POR_RC} | {s for s in nl.connector if s.startswith("CH")}
        self.ff: Dict[Tuple[str, int], Dict] = {}
        self.cnt: Dict[Tuple[str, int], Dict] = {}
        # 판정 노드: 저항으로만 연결된 넷
        self.res_of: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for p in nl.passives:
            if p.kind == "R":
                self.res_of[p.a].append((p.b, p.ohms))
                self.res_of[p.b].append((p.a, p.ohms))
        for r, (pt, conn) in nl.parts.items():
            if pt == "74LVC74A":
                for j in (1, 2):
                    self.ff[(r, j)] = {"q": init_ff, "both": False}
            if pt == "74LV393":
                for j in (1, 2):
                    self.cnt[(r, j)] = {"n": init_count & 15, "cp": 1}
        self.v[CLK] = 1
        self.order = sorted(nl.parts, key=_refkey)

    # ---------------------------------------------------------------- 평가
    def _set(self, net: str, val: int) -> bool:
        if net == NC:
            return False
        if net in self.ext:
            raise RuntimeError(f"외부 넷 {net} 을 칩이 구동하려 한다")
        if self.v[net] != val:
            self.v[net] = val
            return True
        return False

    def _volt(self, net: str) -> float:
        if net in self.res_of and net not in (VDD, GND):
            g = sum(1.0 / r for _, r in self.res_of[net])
            return sum(self.vdd * self.v[o] / r for o, r in self.res_of[net]) / g
        return self.vdd * self.v[net]

    def _eval(self) -> bool:
        ch = False
        for r in self.order:
            pt, cn = self.nl.parts[r]
            g = cn.get
            if pt in FUNC:
                kind = FUNC[pt]
                for ins, o in GATE_UNITS[kind][1]:
                    if cn[o] == NC:
                        continue
                    x = [self.v[cn[i]] for i in ins]
                    y = {"inv": lambda: 1 - x[0], "and2": lambda: int(all(x)),
                         "nand2": lambda: 1 - int(all(x)),
                         "nand8": lambda: 1 - int(all(x))}[kind]()
                    ch |= self._set(cn[o], y)
            elif pt == "74LVC138A":
                en = (self.v[g("/E1")] == 0 and self.v[g("/E2")] == 0
                      and self.v[g("E3")] == 1)
                a = self.v[g("A0")] + 2 * self.v[g("A1")] + 4 * self.v[g("A2")]
                for i in range(8):
                    ch |= self._set(cn[f"/Y{i}"], 0 if (en and i == a) else 1)
            elif pt == "74LVC74A":
                for j in (1, 2):
                    if cn[f"{j}CP"] != GND:
                        raise RuntimeError(f"{r} FF{j} 의 CP 가 GND 가 아니다 -- 비동기 전용")
                    st = self.ff[(r, j)]
                    sd, rd = self.v[cn[f"{j}/SD"]], self.v[cn[f"{j}/RD"]]
                    if sd == 0 and rd == 0:
                        st["both"] = True
                        q, qn = 1, 1
                    else:
                        if st["both"] and sd == 1 and rd == 1:
                            raise HazardError(f"{r} FF{j}: /SD 와 /RD 가 동시에 풀렸다")
                        st["both"] = False
                        if sd == 0:
                            st["q"] = 1
                        elif rd == 0:
                            st["q"] = 0
                        q, qn = st["q"], 1 - st["q"]
                    ch |= self._set(cn[f"{j}Q"], q)
                    ch |= self._set(cn[f"{j}/Q"], qn)
            elif pt == "74LV393":
                for j in (1, 2):
                    st = self.cnt[(r, j)]
                    cp, mr = self.v[cn[f"{j}CP"]], self.v[cn[f"{j}MR"]]
                    if mr:
                        st["n"] = 0
                    elif st["cp"] == 1 and cp == 0:
                        st["n"] = (st["n"] + 1) & 15
                    st["cp"] = cp
                    for bit in range(4):
                        ch |= self._set(cn[f"{j}Q{bit}"], (st["n"] >> bit) & 1)
            elif pt == "LPV7215":
                vp, vm = self._volt(cn["IN+"]), self._volt(cn["IN-"])
                if abs(vp - vm) < 1e-9:
                    raise HazardError(f"{r}: 입력 두 개가 같은 전압 ({vp:.4f} V)")
                ch |= self._set(cn["OUT"], int(vp > vm))
            # 74LV4060 은 CLK 로 대신한다. MMBT3904 는 논리에 안 들어간다.
        return ch

    def settle(self, limit: int = 64) -> None:
        for _ in range(limit):
            if not self._eval():
                return
        raise HazardError("논리가 안정되지 않는다 (발진)")

    # ---------------------------------------------------------------- 실행
    @torch.no_grad()
    def run(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [C, T] in {0,1}. 프레임마다 CLK 하강(채널 갱신) -> CLK 상승."""
        C, T = x.shape
        chans = [f"CH{c}" for c in range(C)]
        # 전원 투입: POR 이 latch 를 지운 뒤 풀린다
        for c in chans:
            self.v[c] = 0
        self.v[POR_RC] = 1
        self.settle()
        self.v[POR_RC] = 0
        self.settle()
        wake = torch.zeros(T)
        run = torch.zeros(T)
        S = len(self.nl.params["k"])
        pas = torch.zeros(S, T)
        for f in range(T):
            self.v[CLK] = 0
            for c in range(C):
                self.v[chans[c]] = int(x[c, f] > 0.5)
            self.settle()
            lo = self.v["WAKE"]
            self.v[CLK] = 1
            self.settle()
            wake[f] = float(lo or self.v["WAKE"])
            run[f] = float(self.v["RUN"])
            for s in range(S):
                pas[s, f] = float(self.v[f"PASS{s+1}"])
        return {"wake_seq": wake, "run": run, "pass": pas}


# ===================================================================== 문서
def write_doc(nl: Netlist, out: Path, sim: Optional[Dict] = None) -> Path:
    p = nl.params
    L: List[str] = []
    a = L.append
    a(f"# 디지털 판별부 넷리스트 — {p.get('name', '')}\n")
    a("`python -m wakeup.netlist` 로 생성. 손으로 고치지 말 것 — 생성기를 고친다.\n")
    a(f"- VDD {p['vdd']} V, 74LVC/74LV 계열 · START 1프레임(채널 1개 이상) · "
      f"창 ±{p['match_window']} · timeout 카운트 {p['timeout']}")
    for s, w in enumerate(p["windows"]):
        a(f"- s{s+1}: Template `{_tmpl(p['template'][s])}`, k={p['k'][s]}, "
          f"창 카운트 {w[0]}..{w[-1]}")
    a("")
    errs = nl.check()
    a("## 배선 검사\n")
    a("통과 — 넷마다 구동원 하나, 떠 있는 입력 없음, 전원 핀 정상." if not errs
      else "**실패**\n\n" + "\n".join(f"- {e}" for e in errs))
    if sim:
        a(f"\n## 넷리스트 시뮬 대조\n\n{sim['frames']}프레임, 시나리오 {sim['cases']}개에서 "
          f"gatelevel 모델과 **불일치 {sim['mismatch']}** (WAKE {sim['wake_mismatch']}, "
          f"내부 PASS·RUN {sim['inner_mismatch']}). WAKE 가 선 프레임 {sim['wake_frames']}개.")
    a("\n## 열린 항목\n")
    L += [f"- {o}" for o in nl.open_items]

    a("\n## 판정 노드\n")
    a("| 상태 | m | k | R7 / R8 (E96) | V_TH | 이상값 | margin | 이상 margin | "
      "오프셋·공차 예산 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s, d in nl.analog.items():
        budget = LPV7215_VOS_MAX + 0.02 * d["vth"] * (1 - d["vth"] / p["vdd"]) \
            + 0.01 * d["step"]
        a(f"| {s} | {d['m']} | {d['k']} | {fmt_r(d['R7'])} / {fmt_r(d['R8'])} | "
          f"{d['vth']*1e3:.1f} mV | {d['vth_ideal']*1e3:.1f} mV | "
          f"{d['margin']*1e3:.1f} mV | {d['margin_ideal']*1e3:.1f} mV | "
          f"{budget*1e3:.1f} mV |")
    a("\n예산 = 비교기 오프셋 최대 8 mV + 분압 1% 저항 최악 + Template 1% 저항 근사. "
      "margin 보다 작으면 판정이 바뀌지 않는다.\n")

    a("## 부품 (BOM)\n")
    a("| 참조 | 부품 | 패키지 | 쓰임 | 핀 배치 근거 |")
    a("|---|---|---|---|---|")
    purpose = dict(n.split(": ", 1) for n in nl.notes)
    for r in sorted(nl.parts, key=_refkey):
        pt = nl.parts[r][0]
        a(f"| {r} | {pt} | {LIB[pt].package} | {purpose.get(r, '')} | {LIB[pt].source} |")
    a("\n| 참조 | 종류 | 값 | 1번 | 2번 | 비고 |")
    a("|---|---|---|---|---|---|")
    for q in nl.passives:
        a(f"| {q.ref} | {q.kind} | {q.value} | {q.a} | {q.b} | {q.note} |")

    a("\n## 부품별 핀\n")
    cur = None
    for ref, pt, num, name, net in nl.pin_rows():
        if ref != cur:
            a(f"\n**{ref} {pt}**\n\n| 핀 | 이름 | 넷 |\n|---|---|---|")
            cur = ref
        a(f"| {num} | {name} | {net} |")

    a("\n## 넷별 연결\n\n| 넷 | 연결 |\n|---|---|")
    for net, users in sorted(nl.net_map().items()):
        if net in (VDD, GND):
            users = [u for u in users if not u.startswith("C")] + [
                f"(+ 디커플링 C {sum(1 for u in users if u.startswith('C'))}개)"]
        a(f"| {net} | {', '.join(users)} |")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"netlist_{p.get('name') or 'board'}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def _tmpl(row: Sequence[int]) -> str:
    return " ".join({1: "1", -1: "0", 0: "X"}[int(t)] for t in row)


def params_from_circuit(rep: Dict, name: str, source: str = "") -> Dict:
    """export.build_report() 결과(circuit.json) -> hardware/*.json 형식."""
    if int(rep.get("start_min_frames", 1)) != 1:
        print(f"  주의: 학습 START 가 x{rep['start_min_frames']}프레임이다. 보드는 1프레임"
              " -- --board-start-only 로 학습한 모델이어야 tau 원점이 맞는다.")
    if int(rep["start_k"]) != 1:
        raise SystemExit(f"start_k={rep['start_k']} -- 보드 START 는 채널 1개 이상(OR)뿐이다.")
    st = rep["states"]
    return {"name": name, "source": source or "circuit.json",
            "template": [s["template"] for s in st], "k": [s["k"] for s in st],
            "tau": [s["tau_frames"] for s in st],
            "match_window": int(rep.get("match_window", 0)),
            "timeout": int(rep["timeout_frames"]), "start_k": 1,
            "frame_ms": rep.get("frame_ms")}


def board_from_params(d: Dict) -> DigitalBoard:
    """hardware/*.json -> DigitalBoard (single START)."""
    return DigitalBoard(template=torch.tensor(d["template"]), k=d["k"], tau=d["tau"],
                        timeout=d["timeout"], start_k=d.get("start_k", 1),
                        n_decoders=3, match_window=d["match_window"],
                        start_frames=1, start_mode="causal")


def compare_streams(nl: Netlist, board: DigitalBoard, xs: Sequence[torch.Tensor],
                    vdd: float) -> Dict:
    """여러 시험 스트림에서 넷리스트 시뮬과 gatelevel 을 프레임 단위로 대조한다."""
    mism, frames, wf, inner = 0, 0, 0, 0
    for x in xs:
        ref = board.run(x.unsqueeze(0), trace=True)
        want = ref["wake_seq"][0]
        got = NetSim(nl, vdd).run(x)
        mism += int((want != got["wake_seq"]).sum())
        # WAKE 는 AND 라 한 상태의 오배선을 가릴 수 있다 -- 내부 상태도 대조한다
        rp = torch.stack([r["pass"][0] for r in ref["trace"]], dim=1)     # [S, T]
        rr = torch.stack([r["run"][0] for r in ref["trace"]])
        inner += int((rp != got["pass"]).any(dim=0).sum()) + int((rr != got["run"]).sum())
        frames += x.shape[1]
        wf += int(want.sum())
    return {"mismatch": mism + inner, "wake_mismatch": mism, "inner_mismatch": inner,
            "frames": frames, "cases": len(xs), "wake_frames": wf}


def make_streams(board: DigitalBoard, n: int = 12, T: int = 900, seed: int = 0
                 ) -> List[torch.Tensor]:
    """대조용 스트림. 밀도가 제각각인 소리 덩어리와 무음을 섞는다.

    잡음을 고르게 깔면 안 된다 -- 확정 Template 은 무음에서도 통과하므로
    (s1 7/8, s2 5/6) 거의 모든 구간이 WAKE 가 되고, 그러면 배선을 틀려도
    대조가 못 잡는다(실제로 그랬다). 일치 개수가 k 언저리에 오도록 밀도를
    0~0.6 으로 흔들고, Template 을 창 안팎에 심고, 연속 켜짐(re-arm)을 넣는다.
    """
    g = torch.Generator().manual_seed(seed)
    C, S, w = board.n_channels, board.n_states, board.match_window
    tm = board.template

    def ri(lo: int, hi: int) -> int:
        return int(torch.randint(lo, hi, (1,), generator=g))

    xs = []
    for i in range(n):
        x = torch.zeros(C, T)
        f = ri(0, 40)
        while f < T - 10:
            L = ri(3, 90)
            p = float(torch.rand(1, generator=g)) * 0.6
            x[:, f:f + L] = (torch.rand(C, min(L, T - f), generator=g) < p).float()
            if ri(0, 2):                                   # Template 을 심는다
                for s in range(S):
                    t = f + int(board.tau[s]) + ri(-w - 1, w + 2)   # 창 밖도 섞는다
                    if t < T:
                        x[:, t] = torch.where(tm[s] > 0, 1.0,
                                              torch.where(tm[s] < 0, 0.0, x[:, t]))
            f += L + ri(0, 90)
        r = ri(0, T - 200)
        x[3, r:r + 3 * board.timeout_count] = 1.0           # 연속 켜짐 -> re-arm
        xs.append(x)
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(description="디지털 판별부 넷리스트 생성·검사·대조")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", help="hardware/*.json")
    src.add_argument("--circuit", help="--emit-circuit 이 쓴 runs/.../circuit.json. "
                                       "읽은 상수를 hardware/<name>.json 으로도 남긴다")
    ap.add_argument("--name", default="", help="--circuit 일 때 이름 (기본: 폴더 이름)")
    ap.add_argument("--out", default="hardware")
    ap.add_argument("--vdd", type=float, default=1.8)
    a = ap.parse_args()
    if a.circuit:
        d = params_from_circuit(json.loads(Path(a.circuit).read_text(encoding="utf-8")),
                                a.name or Path(a.circuit).parent.name, a.circuit)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        pj = Path(a.out) / f"{d['name']}.json"
        pj.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"상수: {pj}")
    else:
        d = json.loads(Path(a.params).read_text(encoding="utf-8"))
    board = board_from_params(d)
    nl = build_netlist(board, vdd=a.vdd, name=d.get("name", Path(a.params).stem))
    errs = nl.check()
    sim = compare_streams(nl, board, make_streams(board), a.vdd)
    path = write_doc(nl, Path(a.out), sim)
    print(f"넷리스트: {path}")
    n_ic = sum(1 for pt, _ in nl.parts.values() if pt != "MMBT3904")
    print(f"  IC {n_ic}개 + 트랜지스터 {len(nl.parts) - n_ic}개, "
          f"수동소자 {len(nl.passives)}개")
    print(f"  배선 검사: {'통과' if not errs else f'{len(errs)}건 실패'}")
    for e in errs:
        print("   -", e)
    print(f"  시뮬 대조: {sim['frames']}프레임 중 불일치 {sim['mismatch']} "
          f"(WAKE 프레임 {sim['wake_frames']})")


if __name__ == "__main__":
    main()
