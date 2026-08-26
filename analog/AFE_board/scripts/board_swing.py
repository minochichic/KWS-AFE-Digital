"""검출기 동작점과 스윙 — 동료 보드 실측 (ngspice transient).

임계값을 저항비로 바꾸려면 두 가지가 필요하다:

  Venv_DC   무신호일 때 v_env 가 앉는 전압. R7/R8 분압의 기준점.
  swing     신호가 들어왔을 때 v_env 가 그 위로 올라가는 높이.

`analog/AFE_tuning/scripts/swing_real.py` 가 같은 일을 하지만 **옛 회로**를 잰다
(50-8000, 혼합 op-amp, 검출기 이득 4.7, 프리앰프 없음). 이 보드는 ×10 프리앰프가
붙고 검출기 이득이 10 이라 마이크 입력 기준 감도가 21배 다르다. 그 표를 이 보드에
쓰면 안 된다.

입력은 **마이크 단(/v_mic)** 에서 준다. 프리앰프가 체인 안에 있으므로 그래야
"마이크에 이만큼 들어오면 v_env 가 이만큼"이라는, 교정에 쓸 수 있는 형태가 된다.

  .venv/bin/python analog/AFE_board/scripts/board_swing.py --amp 0.1m 0.3m 1m
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parents[1]
N_CH = 16


def bom():
    art = BOARD / "artifacts"
    csv = sorted(art.glob("channel_components_*.csv"))[-1]
    ch = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    co = np.genfromtxt(art / "common_components.csv", delimiter=",", names=True,
                       encoding="utf-8")
    return csv, ch, co


def netlist(row, co, drive: str, ctrl: str) -> str:
    """드라이브 문자열과 .control 을 끼운 넷리스트. 슬래시는 사본에서만 벗긴다."""
    tmpl = (BOARD / "netlists" / "netlist_preamp.cir").read_text()
    n = re.sub(r"(?<![\w.\"'])/(?=[A-Za-z_])", "", tmpl)
    n = re.sub(r"^\.param .*$", "", n, flags=re.M)
    p = {"RA": row["RA_kohm"] * 1e3, "R1": row["R1_kohm"] * 1e3,
         "R2": row["R2_kohm"] * 1e3, "R4": row["R4_kohm"] * 1e3,
         "R5": row["R5_kohm"] * 1e3, "R6": row["R6_kohm"] * 1e3,
         "R7": row["R7_kohm"] * 1e3, "R8": row["R8_kohm"] * 1e3,
         "C1": row["C_nF"] * 1e-9, "C3": row["C3_nF"] * 1e-9,
         "VMIC": float(co["VMIC_V"]), "Rmic": float(co["Rmic_kohm"]) * 1e3,
         "Cc": float(co["Cc_nF"]) * 1e-9,
         "Rpre1": float(co["Rpre1_kohm"]) * 1e3,
         "Rpre2": float(co["Rpre2_kohm"]) * 1e3,
         "Rpre3": float(co["Rpre3_kohm"]) * 1e3}
    decl = "\n".join(f".param {k} = {v:.10g}" for k, v in p.items())
    n = re.sub(r"^V4 v_mic .*$", f"V4 v_mic Net-_V3-Pad1_ {drive}", n, flags=re.M)
    return n.replace(".end", decl + "\n.control\n" + ctrl + "\n.endc\n.end")


def run(net: str, keys, strict: bool = True):
    """strict=False 면 수렴 실패 시 예외 대신 None."""
    tmp = BOARD / "netlists" / "tmp_swing.cir"
    tmp.write_text(net)
    r = subprocess.run(["ngspice", "-b", tmp.name], cwd=BOARD / "netlists",
                       capture_output=True, text=True, timeout=600)
    out = {}
    # ngspice 는 tran 이 중간에 죽어도 exit 0 이고 meas 는 0 을 돌려준다.
    # 그 0 이 "스윙 없음"으로 조용히 표에 들어가면 안 된다.
    aborted = "aborted" in r.stdout or "Timestep too small" in r.stdout
    for k in keys:
        m = re.search(re.escape(k) + r"\s*=\s*([-\d.eE+]+)", r.stdout)
        if not m or aborted:
            if not strict:
                return None
            print(r.stdout[-900:], file=sys.stderr)
            raise SystemExit(f"'{k}' 를 못 읽었다 (aborted={aborted})")
        out[k] = float(m.group(1))
    return out


def swing_at(row, co, dc: float, amp: float, fc: float, tries: int = 4):
    """진폭 amp 에서의 스윙. 수렴 실패하면 절반씩 낮춰 재시도하고, 선형 구간
    가정으로 amp 기준으로 되돌려 보고한다 (되돌렸으면 used != amp)."""
    a = amp
    for _ in range(tries):
        got = run(netlist(row, co, f"DC 0 SIN(0 {a:g} {fc:g} 0 0 0)",
                          "  tran 2u 60m\n"
                          "  meas tran vmax MAX v(v_env) from=40m to=60m\n"
                          "  print vmax"), ["vmax"], strict=False)
        # 정류기의 스윙은 0 이하일 수 없다. abort 없이 meas 가 0 을 돌려주는
        # 경우가 있어서(ngspice 가 조용히 실패한다) 값 자체로도 걸러낸다.
        if got is not None and got["vmax"] - dc > 1e-6:
            return (got["vmax"] - dc) * (amp / a), a
        a /= 2.0
    return float("nan"), a


def parse_amp(s: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*([munp]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"진폭 형식: 0.3m / 1m / 100u  (got {s})")
    return float(m.group(1)) * {"": 1, "m": 1e-3, "u": 1e-6, "n": 1e-9,
                                "p": 1e-12}[m.group(2)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amp", nargs="+", type=parse_amp, default=[3e-4],
                    help="마이크 단 정현파 진폭 (예: 0.1m 0.3m 1m)")
    ap.add_argument("--channels", type=int, nargs="*", default=None)
    args = ap.parse_args()

    csv, ch, co = bom()
    which = args.channels if args.channels is not None else range(N_CH)
    print(f"BOM: {csv.name}   구동: 마이크 단(v_mic), 각 채널 f_c\n")

    # Venv_DC 는 신호와 무관하고 채널 간에도 거의 같지만, C3/R 이 채널마다 달라
    # 완전히 같다는 보장이 없으므로 채널별로 잰다.
    hdr = f"{'ch':>3}{'f_c':>8}{'Venv_DC':>10}" + "".join(
        f"{'swing@' + f'{a*1e3:g}mV':>14}" for a in args.amp)
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for k in which:
        row = ch[k]
        fc = float(row["f_c_hz"])
        dc = run(netlist(row, co, "DC 0", "  op\n  print v(v_env)"),
                 ["v(v_env)"])["v(v_env)"]
        swings, marks = [], []
        for a in args.amp:
            sw, used = swing_at(row, co, dc, a, fc)
            swings.append(sw)
            marks.append("" if used == a else f"*{used*1e3:g}mV")
        print(f"{k:>3}{fc:>8.0f}{dc*1e3:>9.2f}m" +
              "".join(f"{s*1e3:>11.2f}m{m:>2}" for s, m in zip(swings, marks)))
        rows.append([k, fc, dc] + swings)

    a = np.array(rows)
    # 마크다운을 파싱하는 것보다 CSV 를 읽는 게 낫다 -- 표 서식이 바뀌면
    # 파서가 조용히 다른 열을 집는다 (실제로 여유 표의 f_c 를 채널 번호로 주웠다).
    out = BOARD / "artifacts" / "swing_board.csv"
    hdr2 = "ch,f_c_hz,Venv_DC_V," + ",".join(f"swing_V_at_{x:g}Vamp" for x in args.amp)
    np.savetxt(out, a, delimiter=",", header=hdr2, comments="", fmt="%.9g")
    print(f"\n저장: {out.relative_to(BOARD.parent.parent)}")
    print("* = 그 진폭에서 수렴 실패해 더 낮은 진폭으로 재서 선형 환산한 값")
    print(f"Venv_DC {a[:, 2].min()*1e3:.2f}~{a[:, 2].max()*1e3:.2f} mV")
    for i, amp in enumerate(args.amp):
        s = a[:, 3 + i]
        print(f"swing @ {amp*1e3:g} mV: {s.min()*1e3:.2f}~{s.max()*1e3:.2f} mV "
              f"(비 {s.max()/max(s.min(), 1e-12):.1f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
