"""Per-channel comparator divider (R7/R8) table.

The comparator reference is V- = Vpos * R8/(R7+R8) with Vpos = 1.8 V. Each
channel's V- must sit at the *middle of that channel's envelope (V_+) swing*,
so we run the full-chain transient once per channel (pass 1 only -- we just
need the V_+ min/max), take the midpoint as the target VREF, and back out an
R7/R8 divider.

IMPORTANT: VREF is signal-dependent -- it is measured here on the demo signal
(synthetic, or the wav in AFE/audio/). It is a *starting point* for the R7/R8
divider; the true operating threshold is the ML-learnable one (CLAUDE.md 3.5).
Different input levels shift the envelope and hence the ideal divider.

Divider design: fix R7+R8 = RTOTAL (default 1 MOhm -> ~1.8 uA bleed, low power),
then R8 = RTOTAL * VREF/1.8, R7 = RTOTAL - R8.

Run from repo root:  .venv/bin/python AFE/scripts/threshold_table.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

import run_transient as rt   # reuse channel_params / apply_channel / pwl / signal

AFE = Path(__file__).resolve().parents[1]
VPOS = 1.8
RTOTAL = 1e6                 # R7 + R8 (low-power divider)


def envelope_range(net_base, pwl):
    """Run pass-1 transient, return (vdmin, vdmax) of the envelope V_+."""
    out = rt.run(0.9004, pwl, net_base)
    m = {k: float(v) for k, v in re.findall(r"(vdmin|vdmax)\s*=\s*(\S+)", out)}
    if "vdmin" not in m:
        print(out[-1200:]); raise SystemExit("transient failed")
    return m["vdmin"], m["vdmax"]


def main():
    sigs = rt.calib_signals()                        # [(name, signal), ...]
    pwls = [rt.pwl_block(s) for _, s in sigs]

    rows = []
    for ch in range(16):
        RA, C, R1, fc = rt.channel_params(ch)
        net_base = rt.apply_channel(rt.NET, RA, C, R1)
        vmins, vmaxs = [], []
        for pwl in pwls:
            a, b = envelope_range(net_base, pwl)
            vmins.append(a); vmaxs.append(b)
        vmin, vmax = min(vmins), max(vmaxs)
        vref = 0.5 * (vmin + vmax)
        r8 = RTOTAL * vref / VPOS
        r7 = RTOTAL - r8
        rows.append((ch, fc, vmin, vmax, vref, r7, r8))
        print(f"ch{ch:2d}  f_c={fc:6.0f}  V+ {vmin:.4f}..{vmax:.4f}  "
              f"VREF={vref:.4f}  R7={r7/1e3:6.1f}k  R8={r8/1e3:6.1f}k")

    lines = ["# AFE 채널별 비교기 분압 R7/R8 (SPICE)\n"]
    lines.append(f"V- = 1.8 * R8/(R7+R8), R7+R8 = {RTOTAL/1e3:.0f} kΩ 고정(저전력).")
    lines.append("VREF = 해당 채널 엔벨로프(V+) 범위의 중앙값. **신호 레벨 의존** —")
    lines.append("데모 신호 기준 시작점이며, 실제 동작 임계는 ML 학습 threshold.\n")
    lines.append("| ch | f_c [Hz] | V+ min [V] | V+ max [V] | VREF (V-) [V] | R7 [kΩ] | R8 [kΩ] |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for ch, fc, vmin, vmax, vref, r7, r8 in rows:
        lines.append(f"| {ch} | {fc:.0f} | {vmin:.4f} | {vmax:.4f} | "
                     f"{vref:.4f} | {r7/1e3:.1f} | {r8/1e3:.1f} |")
    txt = "\n".join(lines) + "\n"
    (AFE / "artifacts" / "threshold_table.md").write_text(txt)
    print("\n저장: artifacts/threshold_table.md")


if __name__ == "__main__":
    main()
