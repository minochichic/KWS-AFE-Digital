"""Mic pre-amp AC design: gain vs bandwidth with the OPA379 (GBW 90 kHz).

Sweeps the target gain G and measures the actual mid-band gain and -3 dB
bandwidth. The point: OPA379's 90 kHz GBW means closed-loop BW ~ 90kHz/G, so a
high gain rolls the 8 kHz band edge off. Finds the largest G that keeps the gain
roughly flat to 8 kHz (HF is exactly what we need to amplify).

Output: artifacts/preamp_ac.png, artifacts/preamp_ac.md
Run from repo root: .venv/bin/python AFE_micamp/scripts/design_preamp.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MA = Path(__file__).resolve().parents[1]
NET = (MA / "netlists" / "preamp.cir").read_text()
GAINS = [5, 10, 20, 50]
F_EDGE = 8000.0


def run(g):
    net = re.sub(r"\.param G\s*=\s*\S+", f".param G = {g:g}", NET)
    (MA / "sim").mkdir(exist_ok=True)
    (MA / "sim" / "tmp.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(MA / "sim" / "tmp.cir")],
                   cwd=MA, capture_output=True, text=True, timeout=120)
    d = np.loadtxt(MA / "sim" / "preamp.csv")
    return d[:, 0], d[:, 1]                     # f, gain[dB]


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for g in GAINS:
        f, gdb = run(g)
        mid = gdb[(f >= 100) & (f <= 500)].mean()      # low-band flat gain
        g8 = np.interp(F_EDGE, f, gdb)                 # gain at 8 kHz
        droop = mid - g8                               # dB drop at band edge
        # -3 dB bandwidth
        below = np.where(gdb <= mid - 3)[0]
        bw = f[below[0]] if len(below) else f[-1]
        rows.append((g, 20*np.log10(g), mid, g8, droop, bw))
        ax.semilogx(f, gdb, lw=1.8, label=f"G={g} ({20*np.log10(g):.0f}dB target)")
        print(f"G={g:2d}: mid={mid:5.1f}dB  8kHz={g8:5.1f}dB  droop={droop:4.1f}dB  "
              f"-3dB BW={bw/1e3:5.1f}kHz")
    ax.axvline(F_EDGE, color="tab:red", ls="--", lw=1, label="8 kHz band edge")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("gain [dB]")
    ax.set_title("Mic pre-amp: gain vs bandwidth (OPA379, GBW 90 kHz)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(MA / "artifacts" / "preamp_ac.png", dpi=130)

    md = ["# 마이크 프리앰프 AC 설계 (OPA379, GBW 90 kHz)\n",
          "단일공급 비반전 이득단. 이득 G에서 닫힘루프 대역 ~ 90kHz/G → 고이득이면 8 kHz가 처짐.\n",
          "| 목표 G | 목표[dB] | 중역이득[dB] | 8kHz이득[dB] | 8kHz 처짐[dB] | -3dB 대역[kHz] |",
          "|---:|---:|---:|---:|---:|---:|"]
    for g, gd, mid, g8, dr, bw in rows:
        md.append(f"| {g} | {gd:.0f} | {mid:.1f} | {g8:.1f} | {dr:.1f} | {bw/1e3:.1f} |")
    md += ["",
           "- **OPA379 GBW 한계**: 이득을 올릴수록 8 kHz 이득이 처진다(HF가 정작 이득이 필요한데).",
           "- **평탄(8kHz 처짐 ≲1dB) 유지 최대 이득**이 이 op-amp의 실용 상한. 더 큰 이득이 필요하면",
           "  (a) 더 빠른 저전력 op-amp, (b) 2단 저이득 캐스케이드, (c) HF 프리엠퍼시스.",
           "",
           "→ 다음: 선택 이득으로 full-chain 통과, 채널별 V+ 스윙이 오프셋을 넘는지 검증 + R7/R8 재도출."]
    (MA / "artifacts" / "preamp_ac.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/preamp_ac.png, preamp_ac.md")


if __name__ == "__main__":
    main()
