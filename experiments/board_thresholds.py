"""학습된 임계값 <-> 동료 보드의 R7/R8, 양방향.

`threshold_volts.py` 는 **옛 회로**(50-8000, 프리앰프 없음, 검출기 이득 4.7)를 상대로
같은 일을 한다. 이건 동료 확정 보드용이고, 두 방향을 다 한다:

  정방향  학습된 thr -> Vthr -> R7/R8      (동료에게 넘길 저항값)
  역방향  동료 R7/R8 -> Vthr -> thr        (그 값으로 재학습할 때 넣을 값)

## K -- 이 변환의 유일한 미지수

우리 엔벨로프는 `normalize="fixed"` 로 [0,1] 근처에 정규화돼 있고 검출기의 V+ 는
전압이다. `compression="sqrt"` 라 우리 엔벨로프도 **진폭**이므로 둘의 관계는 선형이다:

    V+  =  Venv_DC  +  env_normalised * K

K 는 "정규화 1.0 이 몇 mV 인가"이고, 마이크 감도 x 프리앰프 x 실제 음압이 정한다.
**우리는 K 를 모른다.** 16개 임계값에 미지수는 이것 하나뿐이므로, 답은 K 하나짜리
곡선이다. 그래서 이 스크립트는 K 를 인자로 받고 여러 값에 대해 표를 낸다.

옛 `threshold_volts.py` 는 이 변환을 "정당화되지 않는다"고 적었는데, 이유 두 개 중
하나는 이제 없어졌다: 그때는 `compression="log"` 라 로그 범위의 분수가 선형 스윙의
분수가 아니었다. 지금은 sqrt 라 양쪽 다 진폭이다. 남은 것은 **절대 스케일 K** 뿐이다.

## 이 스크립트가 답하지 못하는 것

비교기가 안 켜지는 방식은 두 가지이고, **하나만** 여기 보인다:

  (A) 임계값이 너무 높다     -> 엔벨로프가 그 위로 안 올라간다.
                                thr > 1 로 보이고, 재학습에도 그대로 반영된다.
  (B) 스윙이 오프셋보다 작다 -> ch14(2.75 mV)/ch15(1.65 mV) 가 LPV7215 의 ~3 mV
                                아래다. 임계값과 무관하게 안 뒤집힌다.
                                **우리 모델에는 이게 없다** (`comparator_vos: 0.0`,
                                데드존 없음). 재학습에 반영되지 않는다.

(B) 를 반영하려면 그 채널을 죽이거나 vos 를 켜야 한다. 별도 실험이다.

체크포인트가 있는 학습 박스에서 돌린다:
    python -m experiments.board_thresholds --tag bd_base --k-mv 20 40 80 100
    python -m experiments.board_thresholds --tag bd_base --mic-mv 0.1   # 옛 0.1 mV 열
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

SUPPLY_V = 1.8
TOTAL_K = 1000.0        # 명목 R7+R8 [kOhm]; 동료 BOM 도 합이 1000k 다
VOS_MV = 3.0            # LPV7215 오프셋 -- 이 아래 스윙은 안 뒤집힌다
ROOT = Path(__file__).resolve().parents[1]
SWING_MD = ROOT / "analog/AFE_board/artifacts/swing_board.md"


def board_rows():
    """동료 BOM 에서 ch -> (f_c, R7, R8, Vthr) 와, 측정된 Venv_DC/스윙."""
    import csv
    art = ROOT / "analog/AFE_board/artifacts"
    bom = sorted(art.glob("channel_components_*.csv"))[-1]
    rows = {}
    with bom.open() as f:
        for r in csv.DictReader(f):
            ch = int(float(r["ch"]))
            r7, r8 = float(r["R7_kohm"]), float(r["R8_kohm"])
            rows[ch] = dict(fc=float(r["f_c_hz"]), r7=r7, r8=r8,
                            vthr=SUPPLY_V * r8 / (r7 + r8))
    return bom.name, rows


MIC_MV = (0.1, 0.4)     # swing_board.md 의 두 열. 0.4 mV 가 설계점 (CLAUDE.md 2.2)


def measured(mic_mv: float = 0.4):
    """swing_board.md 에서 Venv_DC 와 채널별 스윙 [mV], 그리고 `≤` 채널.

    복사가 아니라 파싱이다 -- 표가 갱신되면 여기 숫자도 같이 움직여야 한다.
    실제로 갱신됐고(4f35691: "0.1 mV 에서 ch14·15 사망" 을 정정하며 표를
    마이크 진폭 0.1 / 0.4 mV 두 열로 바꿨다) 옛 절 제목을 찾던 이 함수가 멈췄다.

    `≤` 는 ngspice 가 그 진폭에서 수렴하지 못해 낮은 진폭에서 재고 선형 환산한
    값이다. 회로가 100 mV 근처에서 압축되므로 **상한**이다 -- 그 채널 집합을
    같이 돌려준다.
    """
    if mic_mv not in MIC_MV:
        raise SystemExit(f"--mic-mv 는 {MIC_MV} 중 하나 (swing_board.md 의 열)")
    if not SWING_MD.is_file():
        raise SystemExit(f"{SWING_MD} 가 없다. board_swing.py 를 먼저 돌린다.")
    txt = SWING_MD.read_text()
    m = re.search(r"Venv_DC\s*=\s*([\d.]+)\s*mV", txt)
    if not m:
        raise SystemExit("swing_board.md 에서 Venv_DC 를 못 찾았다.")
    dc = float(m.group(1))

    head = "## 스윙 (마이크 진폭별)"
    if head not in txt:
        raise SystemExit(f"swing_board.md 에 '{head}' 절이 없다.")
    body = txt.split(head, 1)[1].split("\n## ", 1)[0]
    col = MIC_MV.index(mic_mv)                  # 0 -> @0.1 mV, 1 -> @0.4 mV
    sw, upper = {}, set()
    # 좌우 2단 표: | ch | f_c | @0.1 | @0.4 | | ch | f_c | @0.1 | @0.4 |
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        for i in range(0, len(cells) - 3, 4):
            ch, _fc, *vals = cells[i:i + 4]
            if not ch.isdigit():
                continue                        # 머리 줄, 구분 줄
            v = vals[col]
            if v.startswith("≤"):
                upper.add(int(ch))
            sw[int(ch)] = float(v.lstrip("≤"))
    if len(sw) != 16:
        raise SystemExit(f"스윙을 16개가 아니라 {len(sw)}개 파싱했다: {sorted(sw)}")
    return dc, sw, upper


DC_CSV = ROOT / "analog/AFE_board/artifacts/dc_measured_20261029.csv"
POT_MV = (900.0, 990.0)  # 동료가 R7/R8 을 가변저항으로 바꾼 뒤의 조정 범위 (2026-09)


def measured_dc(path: Path = DC_CSV) -> dict:
    """ch -> (V_Filt, V_Env) [mV], 무신호 실측. `#` 줄은 주석."""
    import csv
    lines = [l for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    rows = {int(r["ch"]): (float(r["V_Filt_mV"]), float(r["V_Env_mV"]))
            for r in csv.DictReader(lines)}
    if sorted(rows) != list(range(16)):
        raise SystemExit(f"{path}: 채널이 0..15 가 아니다: {sorted(rows)}")
    return rows


def per_channel_targets(thr, env_dc: dict, swing: dict) -> list:
    """K 를 하나로 두지 않고 채널마다 그 채널의 스윙으로 둔다.

    학습은 필터뱅크를 채널마다 최대 이득 1 로 맞춘다(board_matrix.py 의
    peak-norm) -- 모든 채널을 같은 이득으로 본다. 실물은 고역이 약하다(같은
    마이크 진폭에서 ch15 16 mV, 저역 80~108 mV). 그래서 "정규화 1.0 = 설계점
    마이크 진폭에서의 그 채널 스윙" 으로 두면 K_ch = swing[ch] 이고, 기준점도
    시뮬의 공통 Venv_DC 가 아니라 채널마다 잰 V_Env 다."""
    out = []
    for c in range(16):
        t = float(thr[c])
        above = t * swing[c]
        out.append(dict(ch=c, thr=t, dc=env_dc[c], above=above,
                        target=env_dc[c] + above, peak=env_dc[c] + swing[c]))
    return out


def learned(runs: str, tag: str) -> torch.Tensor:
    ck = torch.load(Path(runs) / tag / "best.pt", map_location="cpu",
                    weights_only=True)
    thr = ck["afe"]["threshold"].reshape(-1)
    if thr.numel() != 16:
        raise SystemExit(f"{tag}: 임계값이 {thr.numel()}개다 "
                         f"(comparators_per_channel != 1?)")
    return thr.double()


def divider(vthr_mv: float):
    """Vthr -> (R7, R8) [kOhm], 합 = TOTAL_K. Vthr = 1.8 * R8/(R7+R8)."""
    frac = vthr_mv / 1000.0 / SUPPLY_V
    r8 = TOTAL_K * frac
    return TOTAL_K - r8, r8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="bd_base")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--mic-mv", type=float, default=0.4,
                    help="스윙 표의 마이크 진폭 열 [mV]: 0.1 또는 0.4 (설계점)")
    ap.add_argument("--k-mv", nargs="+", type=float, default=[20.0, 40.0, 80.0],
                    help="정규화 1.0 에 해당하는 전압 [mV]. 미지수다 -- 여러 개 준다")
    ap.add_argument("--dc-csv", type=Path, nargs="?", const=DC_CSV, default=None,
                    help="채널별 무신호 DC 실측 CSV. 주면 K_ch = 그 채널 스윙, 기준점 = "
                         "그 채널 V_Env 로 채널별 목표를 낸다 (값 없이 주면 최신 실측)")
    args = ap.parse_args()

    if args.dc_csv is not None:
        report_measured_dc(args)
        return

    bom_name, board = board_rows()
    dc, swing, upper = measured(args.mic_mv)
    thr = learned(args.runs, args.tag)

    print(f"체크포인트: {args.runs}/{args.tag}/best.pt")
    print(f"BOM:        {bom_name}")
    print(f"Venv_DC:    {dc:.2f} mV   (비교기 오프셋 가정 {VOS_MV:g} mV)")
    print(f"스윙:       마이크 {args.mic_mv:g} mV 열. ≤ = 상한 (ngspice 수렴 실패, "
          f"선형 환산): " + (", ".join(f"ch{c}" for c in sorted(upper)) or "없음") + "\n")

    dead = [c for c in range(16) if swing[c] < VOS_MV]
    if dead:
        print(f"⚠️  스윙 < {VOS_MV:g} mV 라 임계값과 무관하게 안 뒤집히는 채널: "
              f"{dead}  (" + ", ".join(f"ch{c} {swing[c]:.2f}mV" for c in dead) + ")")
        print("    아래 표의 이 행들은 저항을 어떻게 놓든 의미가 없다.\n")

    # ---------- 정방향: 학습된 thr -> Vthr -> R7/R8 ----------
    for k in args.k_mv:
        print(f"── 정방향  K = {k:g} mV  (정규화 1.0 = {k:g} mV) "
              f"{'─'*28}")
        print(f"{'ch':>3}{'f_c':>7}{'thr':>8}{'Vthr':>10}{'R7[k]':>9}"
              f"{'R8[k]':>9}{'스윙':>9}{'':>7}{'DC 위':>9}")
        over = []
        for c in range(16):
            t = float(thr[c])
            vthr = dc + t * k
            r7, r8 = divider(vthr)
            flag = ""
            if t * k > swing[c]:                 # 엔벨로프가 거기까지 못 간다
                flag = "안닿음"
                over.append(c)
            if swing[c] < VOS_MV:
                flag = "사망"
            le = "≤" if c in upper else " "
            print(f"{c:>3}{board[c]['fc']:>7.0f}{t:>8.3f}{vthr:>9.1f}m"
                  f"{r7:>9.1f}{r8:>9.1f}{le}{swing[c]:>7.2f}m{flag:>7}"
                  f"{t * k:>9.2f}m")
        if over:
            print(f"   ⚠️ K={k:g} 에서 임계값이 그 채널 스윙 위에 있는 채널: {over}")
        print()

    # ---------- 역방향: 동료 R7/R8 -> thr ----------
    print(f"── 역방향  동료 저항 -> 우리 정규화 임계값 {'─'*30}")
    hdr = f"{'ch':>3}{'f_c':>7}{'R7[k]':>9}{'R8[k]':>9}{'Vthr':>10}{'위':>8}"
    hdr += "".join(f"{'thr@K=' + f'{k:g}':>12}" for k in args.k_mv)
    print(hdr)
    for c in range(16):
        b = board[c]
        above = b["vthr"] * 1000 - dc                  # Venv_DC 위 몇 mV
        line = (f"{c:>3}{b['fc']:>7.0f}{b['r7']:>9.2f}{b['r8']:>9.2f}"
                f"{b['vthr']*1000:>9.1f}m{above:>7.1f}m")
        for k in args.k_mv:
            t = above / k
            line += f"{t:>11.3f}{'!' if t > 1.0 else ' '}"
        print(line)
    print("\n  ! = 정규화 1.0 을 넘는다 = 그 K 가 맞다면 이 채널은 절대 안 켜진다.")
    print("  재학습에 넣을 값은 위 thr 열이다:")
    print(f"    afe.threshold_trainable=false  +  체크포인트에 이 16개를 심는다.")
    print("\n  주의: K 를 모르므로 이 표는 하나의 답이 아니라 K 의 함수다.")
    print("  K 를 확정하려면 알려진 음압에서 v_env 스윙을 재야 한다 -- 동료 몫이다.")


def report_measured_dc(args) -> None:
    bom_name, board = board_rows()
    _, swing, upper = measured(args.mic_mv)
    dc = measured_dc(args.dc_csv)
    thr = learned(args.runs, args.tag)
    rows = per_channel_targets(thr, {c: dc[c][1] for c in dc}, swing)

    env = [dc[c][1] for c in range(16)]
    print(f"체크포인트: {args.runs}/{args.tag}/best.pt")
    print(f"실측 DC:    {args.dc_csv}  (V_Env {min(env):.1f}..{max(env):.1f} mV)")
    print(f"가정:       정규화 1.0 = 마이크 {args.mic_mv:g} mV 에서 그 채널의 스윙 (시뮬), "
          f"비교기 오프셋 0")
    print(f"BOM Vthr:   {bom_name} (가변저항 이전 고정 R7/R8), 가변 범위 "
          f"{POT_MV[0]:g}..{POT_MV[1]:g} mV\n")
    print(f"{'ch':>3}{'f_c':>7}{'thr':>7}{'V_Env':>8}{'DC위':>7}{'목표':>8}"
          f"{'최고점':>9}{'BOM':>6}  BOM 으로 켜지나   가변범위  오프셋")
    never, out_of_range, buried = [], [], []
    for r in rows:
        c = r["ch"]
        bom = board[c]["vthr"] * 1000
        reach = r["peak"] - bom
        never += [c] if reach < 0 else []
        in_pot = POT_MV[0] <= r["target"] <= POT_MV[1]
        out_of_range += [] if in_pot else [c]
        buried += [c] if r["above"] < VOS_MV else []
        print(f"{c:>3}{board[c]['fc']:>7.0f}{r['thr']:>7.3f}{r['dc']:>8.1f}{r['above']:>7.2f}"
              f"{r['target']:>8.1f}{r['peak']:>8.1f}{'≤' if c in upper else ' '}{bom:>6.0f}  "
              f"{'안 닿음' if reach < 0 else f'+{reach:.1f} mV':>12}   "
              f"{'O' if in_pot else '범위 밖':>6}  {'<3mV' if r['above'] < VOS_MV else ''}")
    print(f"\n  BOM 임계값으로 마이크 {args.mic_mv:g} mV 에서 안 켜지는 채널: {never}")
    print(f"  목표가 가변 범위 {POT_MV[0]:g}..{POT_MV[1]:g} mV 밖인 채널: {out_of_range}")
    print(f"  목표가 자기 DC 위 {VOS_MV:g} mV 이내 (비교기 오프셋에 묻힘): {buried}")


if __name__ == "__main__":
    main()

