"""Add a measured low-frequency leak to some channels of the SPICE filterbank matrix.

    python -m experiments.fbank_leak --leak 7:500:2:2.0 --leak 12:800:2:1.0 \
        --out analog/AFE/artifacts/filterbank_matrix_board_leak.csv

Why (2026-11-17 bench): on the board, ch7 (700 Hz) and ch12 (2.4 kHz) fire whenever
the low bands do -- in setting-C captures, where the software AFE has ch7 off, the
board has it on 94-99 % of the frames in which ch0/1/3/4 are on, and 1-6 % otherwise.
A 200 Hz sine at 10-100 mVpp (offset 0.3-0.9 V) keeps both on. The SPICE matrix has
no such path, so the software image misses it, and putting the software ch7+ch12
rows into the board images alone takes bd_grid_setC from 49 % to 68 % on 197 captures.

The leak is modelled where the matrix lives, in power:
    |H_c(f)|^2  ->  |H_c(f)|^2 + (g * L(f))^2,   L = 1/sqrt(1 + (f/fc)^(2n))
an n-th order low-pass of corner fc at gain g relative to the channel's own peak
(rows are peak-normalized). Whether it is filter leakage, supply crosstalk or
distortion is not known; this only reproduces what the comparator does.
fc/n/g were picked by experiments/thr_calib agreement on the 11/17 captures
(423 captures, best threshold per voltage): ch7 0.934 -> 0.969 at 500 Hz n=2 g=2,
ch12 0.928 -> 0.975 at 800 Hz n=2 g=1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def leak(m: np.ndarray, sr: int, ch: int, fc: float, order: int, g: float) -> np.ndarray:
    f = np.arange(m.shape[1]) * sr / (2 * (m.shape[1] - 1))
    lp = 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    out = m.copy()
    out[ch] = np.sqrt(m[ch] ** 2 + (g * lp) ** 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path,
                    default=Path("analog/AFE/artifacts/filterbank_matrix_board.csv"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--leak", action="append", required=True,
                    help="ch:fc_hz:order:gain, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = np.loadtxt(args.matrix, delimiter=",")
    for spec in args.leak:
        c, fc, n, g = spec.split(":")
        m = leak(m, args.sr, int(c), float(fc), int(n), float(g))
    np.savetxt(args.out, m, delimiter=",")
    print(f"wrote {args.out} ({m.shape[0]}x{m.shape[1]}) from {args.matrix} with {args.leak}")


if __name__ == "__main__":
    main()
