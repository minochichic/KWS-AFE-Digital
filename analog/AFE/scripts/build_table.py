"""Emit analog/AFE/artifacts/BUILD_TABLE.md -- values to draw the schematic from.

What the analog side needs to draw the board is not what the existing
channel_table.md has. That one carries alpha as the pre-training initialiser
under xmax/xmix; the board needs the trained xlse alphas turned into resistors
you can actually buy.

So this picks E96 pairs for Ra/Rb, and answers the question that picking them
raises: does the rounding move anything that matters. It does not, and the
table shows by how much rather than asserting it.

Alphas come from gen_afe16, so the schematic and the netlist cannot disagree.

    python3 analog/AFE/scripts/build_table.py
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_afe16 import ALPHA, DESIGN, ISINK, RTOT, VQUIESCENT  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts/BUILD_TABLE.md"

# Measured on the 16-channel netlist, .op with delta = 0. margin_c = alpha_c * this.
LSE_FLOOR = 0.0700

E96 = [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137,
       140, 143, 147, 150, 154, 158, 162, 165, 169, 174, 178, 182, 187, 191,
       196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267,
       274, 280, 287, 294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374,
       383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523,
       536, 549, 562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
       750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976]


def e96_values(lo: float, hi: float) -> list[float]:
    out = []
    for dec in range(2, 8):                       # 100 ohm .. 97.6 Mohm
        for m in E96:
            v = m * 10 ** (dec - 2)
            if lo <= v <= hi:
                out.append(float(v))
    return out


def ohm(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g} M"
    if v >= 1e3:
        return f"{v / 1e3:g} k"
    return f"{v:g} "


def pick(alpha: float, vals: list[float]) -> tuple[float, float, float]:
    """Closest E96 Ra/Rb with the sum kept near RTOT. alpha = Rb/(Ra+Rb).

    The sum term is a tie-break, not a constraint. alpha error is harmless at
    this scale (see the table's own delta column) while a sum that wanders
    changes that channel's divider current, so it is worth trading a little
    alpha accuracy to keep the 16 dividers drawing the same.
    """
    best = None
    for rb in vals:
        ra_ideal = rb * (1.0 - alpha) / alpha if alpha > 0 else float("inf")
        for ra in vals:
            if not 0.5 * ra_ideal <= ra <= 2.0 * ra_ideal:
                continue
            tot = ra + rb
            if not 0.5 * RTOT <= tot <= 2.0 * RTOT:
                continue
            score = abs(rb / tot - alpha) + 1e-3 * abs(math.log(tot / RTOT))
            if best is None or score < best[0]:
                best = (score, ra, rb)
    assert best is not None, f"no E96 pair for alpha={alpha}"
    return best[1], best[2], best[2] / (best[1] + best[2])


def main() -> None:
    with DESIGN.open(newline="", encoding="utf-8") as fh:
        d = list(csv.DictReader(fh))
    vals = e96_values(1e4, 5e6)

    L = []
    A = L.append
    A("# 제작용 값 표 — 스키매틱을 이 표에서 그린다")
    A("")
    A("`analog/AFE/scripts/build_table.py`가 생성한다. α는 `gen_afe16.py`와 같은")
    A("출처(runs/xl_g12)라 **넷리스트와 어긋날 수 없다**.")
    A("")
    A("기존 [`channel_table.md`](../../../proposal/artifacts/channel_table.md)의 α는")
    A("**학습 전 초기값(xmax/xmix)**이다. 제작에 쓸 것은 아래 표다.")
    A("")
    A("---")
    A("")
    A("## 1. 채널별 — 필터 + 검출기 + 임계 분압")
    A("")
    A("`RA` `C` `R1`은 기존 설계 그대로다 (**건드리면 mel 매칭이 깨진다**).")
    A("`Ra` `Rb`가 새로 바뀌는 부분 — 기존 `R7`/`R8`를 대체한다.")
    A("")
    A("| ch | f_c (Hz) | RA (kΩ) | C (nF) | R1 (kΩ) | α (학습) | **Ra (E96)** | **Rb (E96)** | α 실제 | Δα | 무음 마진 |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    worst = None
    rows = []
    for c in range(16):
        r = d[c]
        ra, rb, act = pick(ALPHA[c], vals)
        da = act - ALPHA[c]
        margin = act * LSE_FLOOR * 1e3
        rows.append((ra, rb, act, da, margin, ra + rb))
        A(f"| {c} | {float(r['fc_sim']):.0f} | {float(r['RA'])/1e3:.2f} | "
          f"{float(r['C'])*1e9:.2f} | {float(r['R1'])/1e3:.1f} | {ALPHA[c]:.4f} | "
          f"**{ohm(ra)}Ω** | **{ohm(rb)}Ω** | {act:.4f} | {da:+.4f} | "
          f"{margin:.1f} mV |")
        if worst is None or margin < worst[1]:
            worst = (c, margin)
    A("")
    max_da = max(abs(r[3]) for r in rows)
    tots = [r[5] for r in rows]
    itot = sum(LSE_FLOOR / t for t in tots) * 1e6
    A(f"$R_a+R_b$는 {ohm(min(tots))}Ω ~ {ohm(max(tots))}Ω로 흩어진다 "
      f"(목표 {ohm(RTOT)}Ω). **맞출 필요 없다** — 합은 자유도이고, α만 맞으면")
    A(f"된다. 16개 분압 전류 총합이 무음에서 {itot:.1f} µA라 채널 간 편차는 전력에도")
    A("버퍼 구동에도 영향이 없다.")
    A("")
    A(f"**E96 반올림은 무해하다.** α 오차 최대 {max_da:.4f}, 무음 마진으로 환산하면")
    A(f"{max_da * LSE_FLOOR * 1e3:.2f} mV — 최악 채널 마진 {worst[1]:.1f} mV에 비해 무시할 수준이다.")
    A("저항 1% 공차도 α를 1% 안쪽으로만 흔들어 같은 결론이다.")
    A("(CLAUDE.md §5대로 **재학습은 하지 않았다**. 반올림 영향을 계산했을 뿐이다.)")
    A("")
    A("공통 값 (16채널 동일): `RA1 = RA2 = RA`, `C1 = C2 = C`, `R2 = R3 = 100 kΩ`,")
    A("`R4 = 10 kΩ`, `R5 = 47 kΩ`, `R6 = 8.25 kΩ`, `C3 = 100 nF`.")
    A("")
    A("---")
    A("")
    A("## 2. 임계 생성 블록 — 뱅크 전체에 1개")
    A("")
    A("기존 보드에서 **없어지는 것**: 채널별 `R7`/`R8` (1.8 V 레일 전체에 걸리던 분압).")
    A("**새로 생기는 것**은 아래가 전부다.")
    A("")
    A("| 지정 | 부품 | 값 | 연결 | 기능 |")
    A("|---|---|---|---|---|")
    A("| `DOR0..15` | BAT54 ×16 | — | 애노드 = 각 `v_env`, 캐소드 = `v_or` 공통 | 다이오드-OR |")
    A(f"| `Isink` | 정전류 싱크 ×1 | **{ISINK*1e6:.0f} µA** | `v_or` → GND | OR 노드 풀다운 |")
    A("| `XUBUF` | OPA379 ×1 | — | +입력 = `v_or`, 출력 = `v_max` | 버퍼 + 16분압 구동 |")
    A("| `Dcomp` | BAT54 ×1 | — | `v_max` → `XUBUF` −입력 (피드백) | $V_d$ 강하 상쇄 |")
    A(f"| `Iref` | 정전류 싱크 ×1 | **{ISINK*1e6:.0f} µA** | `XUBUF` −입력 → GND | 정합 다이오드에 같은 전류 |")
    A(f"| `Vref` | 분압 + 버퍼 op-amp ×1 | **{VQUIESCENT*1e3:.1f} mV** | → 16개 `Rb`의 공통단 | 무음 바닥 (δ = 0) |")
    A("")
    A("**꼭 지켜야 하는 것 세 가지:**")
    A("")
    A(f"1. `Isink`/`Iref`는 **정전류**여야 한다. 저항으로 대신하면 전류합이 `v_or`를")
    A("   따라가 log-sum-exp가 성립하지 않는다. 두 전류는 **같아야** 하며, 그래야")
    A("   `Dcomp`가 OR가 잃은 $V_d$를 정확히 되돌린다.")
    A("2. `Dcomp`는 `DOR*`와 **같은 패키지에서 뽑는다** (BAT54WT1 듀얼). 온도 드리프트가")
    A("   상쇄되는 근거가 같은 다이 위에 있다는 것이다.")
    A("3. `Vref` **버퍼는 생략 불가.** 정지점에서 D2가 역바이어스라 엔벨로프단이")
    A("   고임피던스다 — 버퍼 없이 16개 분압을 물리면 기준이 40 mV 밀린다 (시뮬 확인).")
    A("")
    A("---")
    A("")
    A("## 3. 왜 이 값들인가")
    A("")
    A(f"- **$R_a + R_b \\approx$ {ohm(RTOT)}Ω**: 자유도다. 크게 잡을수록 전력이 준다.")
    A(f"  걸리는 전압이 1.8 V 전체가 아니라 $V_{{max}}-V_{{ref}}$(~0.1 V)라 기존")
    A("  `R7`/`R8` 대비 채널당 18배 적게 먹는다.")
    A(f"- **`Isink` = {ISINK*1e6:.0f} µA**: 100 nA는 다이오드 $I_s$(2.2 nA)에 가까워")
    A("  LSE 오차가 10배가 되고, 10 µA는 나아지는 것 없이 전력만 10배다.")
    A(f"- **`Vref` = {VQUIESCENT*1e3:.1f} mV (δ = 0)**: 무음 마진은 δ가 아니라 LSE 바닥")
    A(f"  $T\\ln16$ = {LSE_FLOOR*1e3:.0f} mV에서 나온다. 마진 = α × {LSE_FLOOR*1e3:.0f} mV,")
    A(f"  최악 채널(ch{worst[0]})이 {worst[1]:.1f} mV. 근거는")
    A("  [`../SPICE_FINDINGS.md`](../SPICE_FINDINGS.md).")
    A("")
    A("**브링업에서 먼저 잴 것**: 채널별 `v_env` 정지점. 설계값은 16채널 공통")
    A(f"{VQUIESCENT*1e3:.1f} mV이지만 실제로는 `da + (R5/R4)·(vf − da)`, 즉 그 채널")
    A(f"op-amp 오프셋의 4.7배라 소자마다 다르다. 산포가 최악 채널 마진")
    A(f"{worst[1]:.1f} mV를 먹기 시작하면 `Vref`를 그만큼 올린다.")
    A("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  max |delta alpha| = {max_da:.5f}  ->  {max_da*LSE_FLOOR*1e3:.3f} mV")
    print(f"  worst margin: ch{worst[0]} at {worst[1]:.2f} mV")


if __name__ == "__main__":
    main()
