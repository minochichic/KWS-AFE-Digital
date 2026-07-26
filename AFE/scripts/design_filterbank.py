"""Design 16 GIC channels to match the ideal mel filterbank (50-8000 Hz).

Control structure (verified in channel.py): f_c is set by RA (with C), and Q is
set by R1 (f_c-independent, Q ~ R1/RA). So each channel is designed by
(1) bisecting RA to the mel center f_c, then (2) setting R1 = Q_target*RA and
refining once. C is chosen per channel to keep RA in a sane range.

Outputs (AFE/sim/):
  filterbank_design.csv   per-channel (f_c target/sim, Q target/sim, RA, C, R1, gain)
  filterbank_matrix.csv   [16, 257] |H_k(f)| on the ML STFT grid (0..8000 Hz)
Run from repo root:  .venv/bin/python AFE/scripts/design_filterbank.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import char                                    # noqa: E402

AFE = Path(__file__).resolve().parents[1]
N_CH = 16
F_MIN, F_MAX = 50.0, 8000.0
SR, N_FFT = 16000, 512


def hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_targets():
    """torchaudio-style: n+2 mel points; centers + FWHM-based target Q."""
    pts = mel2hz(np.linspace(hz2mel(F_MIN), hz2mel(F_MAX), N_CH + 2))
    centers = pts[1:-1]
    # triangular filter k spans [pts[k], pts[k+2]]; FWHM = half the base
    fwhm = (pts[2:] - pts[:-2]) / 2.0
    q = centers / fwhm
    return centers, q


def fit_ra(fc_target, C):
    """Bisect RA (geometric) to hit fc_target. f_c is independent of R1."""
    lo, hi = 1.2e3, 4e5
    ra = math.sqrt(lo * hi)
    for _ in range(20):
        ra = math.sqrt(lo * hi)
        fc, *_ = char(ra, C, ra)           # R1=ra during search (irrelevant to f_c)
        if fc > fc_target:                 # too high -> need larger RA
            lo = ra
        else:
            hi = ra
    return ra


def design_channel(fc_t, q_t):
    C = 0.8 / (2 * math.pi * fc_t * 10e3)   # aim RA ~ 10k
    ra = fit_ra(fc_t, C)
    r1 = max(1e3, q_t * ra)
    # one Q refinement (Q ~ R1/RA, linear)
    for _ in range(2):
        fc, q, g, f, gdb = char(ra, C, r1)
        if not (q == q) or q <= 0:          # NaN/degenerate -> widen
            break
        r1 = max(1e3, r1 * q_t / q)
    fc, q, g, f, gdb = char(ra, C, r1)
    return dict(RA=ra, C=C, R1=r1, fc=fc, Q=q, gain=g, f=f, gdb=gdb)


def main():
    (AFE / "sim").mkdir(exist_ok=True)
    centers, qtar = mel_targets()
    print(f"{'ch':>3}{'fc_target':>11}{'fc_sim':>9}{'err%':>7}"
          f"{'Q_tgt':>7}{'Q_sim':>7}{'gain':>7}{'RA[k]':>8}{'C[nF]':>8}{'R1[k]':>8}")
    print("-" * 82)
    grid = np.linspace(0, SR / 2, N_FFT // 2 + 1)            # ML STFT freq grid
    matrix, rows = [], []
    for k in range(N_CH):
        d = design_channel(centers[k], qtar[k])
        err = 100 * (d["fc"] - centers[k]) / centers[k]
        print(f"{k:>3}{centers[k]:>11.0f}{d['fc']:>9.0f}{err:>7.1f}"
              f"{qtar[k]:>7.2f}{d['Q']:>7.2f}{d['gain']:>7.1f}"
              f"{d['RA']/1e3:>8.2f}{d['C']*1e9:>8.2f}{d['R1']/1e3:>8.1f}")
        # |H| linear, peak-normalized, on the ML grid
        mag = 10 ** (d["gdb"] / 20.0)
        Hk = np.interp(grid, d["f"], mag, left=0, right=0)
        Hk = Hk / (Hk.max() + 1e-12)
        matrix.append(Hk)
        rows.append([k, centers[k], d["fc"], qtar[k], d["Q"], d["gain"],
                     d["RA"], d["C"], d["R1"]])

    np.savetxt(AFE / "artifacts" / "filterbank_matrix.csv", np.array(matrix),
               delimiter=",")
    np.savetxt(AFE / "artifacts" / "filterbank_design.csv", np.array(rows),
               delimiter=",",
               header="ch,fc_target,fc_sim,Q_target,Q_sim,gain_dB,RA,C,R1",
               comments="")
    fcs = [r[2] for r in rows]
    errs = [abs(100 * (r[2] - r[1]) / r[1]) for r in rows]
    print("-" * 82)
    print(f"f_c 커버: {min(fcs):.0f}..{max(fcs):.0f} Hz | "
          f"평균 f_c 오차 {np.mean(errs):.1f}% | 최대 {max(errs):.1f}%")
    print(f"저장: sim/filterbank_matrix.csv [16,{len(grid)}], filterbank_design.csv")


if __name__ == "__main__":
    main()
