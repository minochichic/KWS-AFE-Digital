"""Characterize one GIC channel via ngspice .ac.

char(RA, C, R1) -> (f_c, Q, peak_gain_dB, H) where H is the full complex
response on a frequency grid. Used by the filterbank designer to fit each
channel to an ideal mel target. SPICE-in-the-loop because the ideal formula
f_c = 1/(2*pi*RA*C) is ~20% off at high frequency (finite-GBW OPA379).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

AFE = Path(__file__).resolve().parents[1]
BASE = (AFE / "netlists" / "gic_channel.cir").read_text()


def char(RA: float, C: float, R1: float, fmax: float = 100e3):
    """Run ngspice .ac for (RA, C, R1); return (f_c, Q, gain_dB, f, mag)."""
    net = BASE
    net = re.sub(r"\.param RA\s*=\s*\S+",   f".param RA = {RA:.6g}",   net)
    net = re.sub(r"\.param CVAL\s*=\s*\S+", f".param CVAL = {C:.6g}",  net)
    net = re.sub(r"\.param R1v\s*=\s*\S+",  f".param R1v = {R1:.6g}",  net)
    net = re.sub(r"ac dec \d+ \S+ \S+",     f"ac dec 100 10 {fmax:.6g}", net)
    tmp = AFE / "sim" / "tmp_char.cir"
    (AFE / "sim").mkdir(exist_ok=True)
    tmp.write_text(net)
    subprocess.run(["ngspice", "-b", str(tmp)], cwd=AFE,
                   capture_output=True, text=True, timeout=60)

    d = np.loadtxt(AFE / "sim" / "gic_ac.csv")
    f, g = d[:, 0], d[:, 1]                       # Hz, dB
    i = int(np.argmax(g))
    fc, gpk = float(f[i]), float(g[i])
    half = gpk - 3.0
    lo = _cross(f[:i + 1], g[:i + 1], half, True)
    hi = _cross(f[i:], g[i:], half, False)
    q = fc / (hi - lo) if (lo and hi and hi > lo) else float("nan")
    return fc, q, gpk, f, g


def _cross(f, g, level, rising):
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        if (rising and a < level <= b) or (not rising and a >= level > b):
            t = (level - a) / (b - a)
            return f[k] + t * (f[k + 1] - f[k])
    return None


if __name__ == "__main__":
    # exploratory: how do RA and R1 control (f_c, Q)?  C fixed at 10n.
    print("R1 scan (C=10n, RA=10k) -- does R1 set Q without moving f_c?")
    print(f"{'R1[k]':>8}{'f_c':>9}{'Q':>8}{'gain':>8}")
    for r1 in (5e3, 10e3, 20e3, 40e3, 80e3, 160e3):
        fc, q, g, *_ = char(10e3, 10e-9, r1)
        print(f"{r1/1e3:>8.1f}{fc:>9.0f}{q:>8.2f}{g:>8.1f}")
    print("\nRA scan (C=10n, R1=14.7k) -- f_c vs RA")
    print(f"{'RA[k]':>8}{'f_c':>9}{'Q':>8}{'gain':>8}")
    for ra in (3e3, 6e3, 12e3, 24e3, 48e3):
        fc, q, g, *_ = char(ra, 10e-9, 14.7e3)
        print(f"{ra/1e3:>8.1f}{fc:>9.0f}{q:>8.2f}{g:>8.1f}")
