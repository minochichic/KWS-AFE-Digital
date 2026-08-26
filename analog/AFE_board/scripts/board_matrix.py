"""Characterize the COLLEAGUE'S board and emit a filterbank matrix for training.

This is not a designer. `analog/AFE/scripts/design_filterbank.py` picks
components to hit a mel target; here the components are GIVEN (the colleague's
BOM) and we only measure what they do. Nothing in this file may change a
component value.

Source of truth (all three came from the colleague, 2026-08-25):
  netlists/netlist_preamp.cir              topology, incl. the x10 preamp
  artifacts/channel_components_*.csv       per-channel RA/R1/R2/C/C3/R7/R8
  artifacts/common_components.csv          mic + preamp, shared by all channels

The filter stage is linear, so .ac from the mic to /v_filt is the true channel
response. The detector and comparator sit downstream and do not shape it. Rows
are peak-normalized exactly as design_filterbank.py does, so a flat preamp gain
cancels and only the SHAPE reaches training.

Run from repo root (needs ngspice on PATH):
  .venv/bin/python analog/AFE_board/scripts/board_matrix.py

Writes analog/AFE/artifacts/filterbank_{matrix,design}_board.csv -- next to the
other banks, because train/config.py spice_matrix_path reads from there and
data/afe.py pairs matrix<suffix> with design<suffix> for spice_gain_restore.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parents[1]
AFE = BOARD.parent / "AFE"
SR, N_FFT = 16000, 512                       # must match train/config.py
N_CH = 16


def stft_grid() -> np.ndarray:
    return np.linspace(0, SR / 2, N_FFT // 2 + 1)          # 257 pts, 31.25 Hz


def load_bom(chan_csv: Path, common_csv: Path):
    ch = np.genfromtxt(chan_csv, delimiter=",", names=True, dtype=None,
                       encoding="utf-8")
    co = np.genfromtxt(common_csv, delimiter=",", names=True, encoding="utf-8")
    if ch.shape[0] != N_CH:
        raise ValueError(f"{chan_csv.name} has {ch.shape[0]} rows, expected {N_CH}")
    return ch, co


def run_channel(net_tmpl: str, row, co, out: Path) -> tuple:
    """.ac sweep on the ML STFT grid; returns (f, |H| linear at /v_filt)."""
    # 257-point grid = DC + 256 linear points at exactly 31.25 Hz spacing.
    # .ac cannot start at DC, so sweep 1..256 and prepend 0 (a bandpass is 0
    # at DC anyway).
    df = SR / N_FFT
    params = {
        "RA": row["RA_kohm"] * 1e3, "R1": row["R1_kohm"] * 1e3,
        "R2": row["R2_kohm"] * 1e3, "R4": row["R4_kohm"] * 1e3,
        "R5": row["R5_kohm"] * 1e3, "R6": row["R6_kohm"] * 1e3,
        "R7": row["R7_kohm"] * 1e3, "R8": row["R8_kohm"] * 1e3,
        "C1": row["C_nF"] * 1e-9, "C3": row["C3_nF"] * 1e-9,
        "VMIC": float(co["VMIC_V"]), "Rmic": float(co["Rmic_kohm"]) * 1e3,
        "Cc": float(co["Cc_nF"]) * 1e-9,
        "Rpre1": float(co["Rpre1_kohm"]) * 1e3,
        "Rpre2": float(co["Rpre2_kohm"]) * 1e3,
        "Rpre3": float(co["Rpre3_kohm"]) * 1e3,
    }
    # KiCad writes net names with a leading slash ("/v_filt"). ngspice accepts
    # them as node names but its expression parser reads the slash as division,
    # so vdb(/v_filt) cannot be written. Strip it in our COPY only -- the
    # colleague's tracked netlist keeps their names. The include lines carry
    # bare filenames, so no path is touched.
    net = re.sub(r"(?<![\w.\"'])/(?=[A-Za-z_])", "", net_tmpl)
    # Drop the colleague's .param defaults; we supply every value explicitly so
    # a missing key becomes an ngspice error instead of a silent stale default.
    net = re.sub(r"^\.param .*$", "", net, flags=re.M)
    decl = "\n".join(f".param {k} = {v:.10g}" for k, v in params.items())
    ctrl = (f"\n.control\n"
            f"  ac lin {N_FFT // 2} {df:.10g} {SR / 2:.10g}\n"
            f"  wrdata {out.as_posix()} vdb(v_filt)\n"
            f".endc\n")
    net = net.replace(".end", decl + ctrl + ".end")

    tmp = BOARD / "netlists" / "tmp_board.cir"     # ngspice -b has no stdin mode
    tmp.write_text(net)
    r = subprocess.run(["ngspice", "-b", tmp.name], cwd=BOARD / "netlists",
                       capture_output=True, text=True, timeout=120)
    if not out.exists():
        raise RuntimeError(f"ngspice produced no output:\n{r.stdout}\n{r.stderr}")
    d = np.loadtxt(out)
    f, gdb = d[:, 0], d[:, 1]
    return np.concatenate([[0.0], f]), np.concatenate([[0.0], 10 ** (gdb / 20.0)])


def measure(f: np.ndarray, mag: np.ndarray) -> tuple:
    """(f_c, Q, peak gain dB) from the swept response."""
    i = int(np.argmax(mag))
    fc, pk = float(f[i]), float(mag[i])
    half = pk / np.sqrt(2.0)
    lo = _cross(f[:i + 1], mag[:i + 1], half, True)
    hi = _cross(f[i:], mag[i:], half, False)
    q = fc / (hi - lo) if (lo and hi and hi > lo) else float("nan")
    return fc, q, 20 * np.log10(pk + 1e-30)


def _cross(f, g, level, rising):
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        if (rising and a < level <= b) or (not rising and a >= level > b):
            return float(f[k] + (level - a) / (b - a) * (f[k + 1] - f[k]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", default=None,
                    help="channel BOM csv (default: the newest in artifacts/)")
    ap.add_argument("--suffix", default="_board")
    args = ap.parse_args()

    art = BOARD / "artifacts"
    chan_csv = (Path(args.channels) if args.channels else
                sorted(art.glob("channel_components_*.csv"))[-1])
    ch, co = load_bom(chan_csv, art / "common_components.csv")
    net_tmpl = (BOARD / "netlists" / "netlist_preamp.cir").read_text()

    print(f"BOM: {chan_csv.name}\n")
    print(f"{'ch':>3}{'f_c BOM':>10}{'f_c sim':>10}{'err%':>7}"
          f"{'Q BOM':>7}{'Q sim':>7}{'gain':>8}{'Vthr':>8}")
    print("-" * 60)

    grid, matrix, rows = stft_grid(), [], []
    for k in range(N_CH):
        row = ch[k]
        f, mag = run_channel(net_tmpl, row, co, BOARD / "sim" / f"ac_{k}.csv")
        fc, q, gdb = measure(f, mag)
        tgt = float(row["f_c_hz"])
        vthr = 1.8 * row["R8_kohm"] / (row["R7_kohm"] + row["R8_kohm"])
        print(f"{k:>3}{tgt:>10.1f}{fc:>10.1f}{100*(fc-tgt)/tgt:>7.1f}"
              f"{row['Q']:>7.2f}{q:>7.2f}{gdb:>8.1f}{vthr:>8.3f}")

        Hk = np.interp(grid, f, mag, left=0, right=0)
        matrix.append(Hk / (Hk.max() + 1e-12))     # peak-norm, as the other banks
        rows.append([k, tgt, fc, float(row["Q"]), q, gdb,
                     row["RA_kohm"] * 1e3, row["C_nF"] * 1e-9, row["R1_kohm"] * 1e3])

    mp = AFE / "artifacts" / f"filterbank_matrix{args.suffix}.csv"
    dp = AFE / "artifacts" / f"filterbank_design{args.suffix}.csv"
    np.savetxt(mp, np.array(matrix), delimiter=",")
    np.savetxt(dp, np.array(rows), delimiter=",",
               header="ch,fc_target,fc_sim,Q_target,Q_sim,gain_dB,RA,C,R1",
               comments="")
    print(f"\n저장: {mp.relative_to(AFE.parent.parent)} [16,{len(grid)}]")
    print(f"      {dp.relative_to(AFE.parent.parent)}")
    print(f"\n학습:\n  .venv/bin/python -m train.train --config configs/base.yaml \\\n"
          f"    --tag bd_base afe.spice_matrix_path={mp.relative_to(AFE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
