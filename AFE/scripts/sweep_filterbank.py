#!/usr/bin/env python3
"""Sweep the GIC channel's RA and characterize each resulting bandpass.

For a geometric range of RA (the corner-setting resistor), run ngspice .ac on
the GIC-only netlist and extract: actual peak frequency f_c, peak gain, and Q
(from the -3 dB bandwidth). This tells us (a) whether the AFE can cover
50-8000 Hz, (b) how far the real f_c drifts from the ideal 1/(2*pi*RA*C), and
(c) where the OPA379 finite GBW starts degrading the high-frequency channels.

Run from the AFE/ directory:  python scripts/sweep_filterbank.py
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np

AFE = Path(__file__).resolve().parents[1]
BASE = (AFE / "netlists" / "gic_channel.cir").read_text()
CVAL = 10e-9


def run_one(ra: float):
    """Return (f_c, peak_gain_dB, Q, f_lo, f_hi) for a given RA, via ngspice."""
    net = re.sub(r"\.param RA\s*=\s*\S+", f".param RA = {ra:.6g}", BASE)
    tmp = AFE / "sim" / "tmp_channel.cir"
    tmp.write_text(net)
    subprocess.run(["ngspice", "-b", str(tmp)], cwd=AFE,
                   capture_output=True, text=True, timeout=60)

    csv = AFE / "sim" / "gic_ac.csv"
    d = np.loadtxt(csv)
    f, g = d[:, 0], d[:, 1]                     # freq (Hz), gain (dB)
    i = int(np.argmax(g))
    fc, gpk = f[i], g[i]
    # -3 dB bandwidth around the peak
    half = gpk - 3.0
    lo = _cross(f[:i + 1], g[:i + 1], half, rising=True)
    hi = _cross(f[i:], g[i:], half, rising=False)
    q = fc / (hi - lo) if (lo and hi and hi > lo) else float("nan")
    return fc, gpk, q, lo, hi


def _cross(f, g, level, rising):
    """Linear-interpolated frequency where g crosses `level`."""
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        if (rising and a < level <= b) or (not rising and a >= level > b):
            t = (level - a) / (b - a)
            return f[k] + t * (f[k + 1] - f[k])
    return None


def main():
    (AFE / "sim").mkdir(exist_ok=True)
    # RA range chosen to span roughly 50 Hz .. 8 kHz of ideal corner
    ra_values = np.geomspace(1.9e3, 3.3e5, 20)
    print(f"{'RA[k]':>8}{'f_c(sim)':>11}{'f_c(ideal)':>12}"
          f"{'sim/ideal':>10}{'gain[dB]':>10}{'Q':>7}")
    print("-" * 60)
    rows = []
    for ra in ra_values:
        fc, gpk, q, lo, hi = run_one(ra)
        ideal = 1.0 / (2 * math.pi * ra * CVAL)
        print(f"{ra/1e3:>8.2f}{fc:>11.0f}{ideal:>12.0f}"
              f"{fc/ideal:>10.3f}{gpk:>10.1f}{q:>7.2f}")
        rows.append((ra, fc, ideal, gpk, q))
    np.savetxt(AFE / "sim" / "filterbank_sweep.csv",
               np.array(rows), delimiter=",",
               header="RA,fc_sim,fc_ideal,gain_dB,Q", comments="")
    fcs = [r[1] for r in rows]
    print("-" * 60)
    print(f"커버 범위(sim f_c): {min(fcs):.0f} .. {max(fcs):.0f} Hz")
    print(f"목표 50-8000 Hz 커버 가능?  "
          f"low={'OK' if min(fcs)<=60 else 'NO'}  "
          f"high={'OK' if max(fcs)>=7500 else 'NO(GBW 한계?)'}")


if __name__ == "__main__":
    main()
