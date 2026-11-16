"""The board's buildable comparator thresholds, as a table training can use.

    # provisional table from the circuit numbers (no board data needed)
    python -m experiments.thr_grid sim --out analog/AFE_board/artifacts/thr_grid_sim_20261116.csv

    # after training with afe.threshold_grid_path: which voltage each channel uses
    python -m experiments.thr_grid report --run runs/<tag>

The bench (2026-10-29..): the AWG puts every clip at the preamp-output node
`/GIC_0/vin` at 100 mVpp (offset 900 mV), so with afe.input_pp the software
sees every clip at one level too. Each comparator's V- picks one of the
reference voltages (2026-11: 900..996 mV, 16 mV apart, up to 4 channels per
voltage). For channel c and level V the table holds the RAW software envelope
(AFEFrontend.envelopes raw=True, after input_pp) at which that comparator trips:

    dV = V - V_Env,DC[c]                 measured DC at the detector output
    A  = swing_c^-1(dV)                  node amplitude [mV] that lifts V+ by dV
    raw = k_c * A / (Vpp_mV / input_pp)  k_c = raw envelope of a unit tone at f_c

`sim` takes swing_c from swing_board.md's two simulated points (1 and 4 mV at the
node) -- the same broken line as experiments/awg_thresholds.py, so the same
caveat: below 1 mV it assumes a straight line to the origin, above 4 mV it
extends the last slope (overestimates; the detector compresses). Cells outside
the two points are listed. experiments/thr_calib.py replaces the whole chain
with a fit to board captures.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import List

import numpy as np

from experiments.board_thresholds import DC_CSV, board_rows, measured, measured_dc

LEVELS = [900.0 + 16.0 * i for i in range(7)]          # 900..996 mV
PRE_MV = (1.0, 4.0)                                     # swing_board.md's node amplitudes


def swing_inverse(dv: float, s1: float, s4: float):
    """V+ rise [mV] -> node amplitude [mV], inverse of awg_thresholds.swing_at."""
    if dv <= 0:
        return 0.0, "below DC"
    if dv <= s1:
        return PRE_MV[0] * dv / s1, "lin<1mV"
    if dv <= s4:
        return PRE_MV[0] + (PRE_MV[1] - PRE_MV[0]) * (dv - s1) / (s4 - s1), ""
    slope = (s4 - s1) / (PRE_MV[1] - PRE_MV[0])
    return PRE_MV[1] + (dv - s4) / slope, "extrap>4mV"


def unit_tone_gain(afe, fcs: List[float], sr: int, n: int) -> np.ndarray:
    """k_c: raw envelope (median of the middle frames) of a unit-amplitude tone at f_c.

    compression="sqrt" makes the raw envelope linear in amplitude, so one tone
    per channel is enough; checked here at half amplitude."""
    import torch
    t = torch.arange(n, dtype=torch.float32) / sr
    k = []
    for c, f in enumerate(fcs):
        env = []
        for a in (1.0, 0.5):
            w = (a * torch.sin(2 * np.pi * f * t)).unsqueeze(0)
            with torch.no_grad():
                e = afe.envelopes(w, raw=True)[0, c]
            m = e.shape[-1]
            env.append(float(e[m // 5: 4 * m // 5].median()))
        if abs(env[1] / env[0] - 0.5) > 0.01:
            raise SystemExit(f"ch{c}: raw envelope not linear in amplitude "
                             f"({env[1] / env[0]:.3f} at half) -- compression must be sqrt")
        k.append(env[0])
    return np.array(k)


def _afe(config: str, overrides: dict):
    from data.afe import AFEFrontend
    from train.config import load_config
    cfg = load_config(config, overrides)
    a = cfg.afe
    if a.compression != "sqrt":
        raise SystemExit(f"compression={a.compression!r}: the table assumes sqrt (linear in amplitude)")
    a.input_pp = 0.0                                   # k_c is per PCM unit, before the level step
    a.threshold_grid_path = ""
    afe = AFEFrontend(a)
    afe.eval()
    return cfg, afe


def sim(args) -> None:
    from data.thr_grid import write_grid
    ov = {"afe.spice_matrix_path": args.spice_matrix}
    cfg, afe = _afe(args.config, ov)
    sr = int(cfg.afe.sample_rate)
    n = int(sr * cfg.afe.clip_ms / 1000)
    bom, board = board_rows()
    fcs = [board[c]["fc"] for c in range(16)]
    _, s1, up1 = measured(0.1)
    _, s4, up4 = measured(0.4)
    dc = measured_dc(args.dc_csv)
    k = unit_tone_gain(afe, fcs, sr, n)
    mv_per_unit = args.vpp_mv / args.input_pp

    levels = args.levels
    g = np.full((16, len(levels)), np.nan)
    amp = np.full_like(g, np.nan)
    flag = [[""] * len(levels) for _ in range(16)]
    for c in range(16):
        for j, v in enumerate(levels):
            A, how = swing_inverse(v - dc[c][1], s1[c], s4[c])
            flag[c][j] = how
            if how == "below DC":
                continue                              # V- under the idle V+: always on
            amp[c, j] = A
            g[c, j] = k[c] * A / mv_per_unit

    print(f"filterbank {args.spice_matrix}, BOM {bom}, DC {args.dc_csv.name}, swing swing_board.md (sim)")
    print(f"level: every clip at {args.vpp_mv:g} mVpp at the node == input_pp {args.input_pp} "
          f"-> 1 PCM unit = {mv_per_unit:.1f} mV\n")
    print("node amplitude [mV] that trips each comparator (* = outside the 1..4 mV simulated points)")
    print(f"{'ch':>3}{'f_c':>6}{'V_DC':>7} | " + "".join(f"{v:>8.0f}" for v in levels))
    for c in range(16):
        cells = "".join(
            f"{'':>8}" if not np.isfinite(amp[c, j]) else
            f"{amp[c, j]:>7.2f}{'*' if flag[c][j] else ' '}" for j in range(len(levels)))
        print(f"{c:>3}{fcs[c]:>6.0f}{dc[c][1]:>7.1f} | {cells}")
    out = Path(args.out)
    write_grid(out, levels, g, [
        f"experiments/thr_grid.py sim, {datetime.date.today()}",
        f"raw envelope (envelopes raw=True) at which ch trips, input_pp={args.input_pp}, "
        f"{args.vpp_mv:g} mVpp at /GIC_0/vin (1 PCM unit = {mv_per_unit:.1f} mV)",
        f"filterbank {args.spice_matrix}; DC {args.dc_csv.name}; swing swing_board.md 1/4 mV (simulated)",
        "cells outside the 1..4 mV points are extrapolated: "
        + ", ".join(f"ch{c}@{levels[j]:g}" for c in range(16) for j in range(len(levels)) if flag[c][j])[:400],
        f"train with afe.input_pp={args.input_pp} afe.threshold_grid_path=<this file>",
    ])
    print(f"\nwrote {out}")


def report(args) -> None:
    import torch
    from data.afe import AFEFrontend, load_afe_state
    from data.thr_grid import describe
    from train.config import load_config

    run = Path(args.run)
    cfg = load_config(str(run / "config.yaml"))
    if not cfg.afe.threshold_grid_path:
        raise SystemExit(f"{run}: trained without afe.threshold_grid_path")
    afe = AFEFrontend(cfg.afe)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    load_afe_state(afe, ck["afe"])
    lvl = afe.grid_choice()
    levels = afe.grid_levels
    print(f"{run}: grid {cfg.afe.threshold_grid_path}, capacity "
          f"{cfg.afe.threshold_grid_capacity or 'unlimited'}, input_pp {cfg.afe.input_pp}")
    print("  " + describe(levels, lvl))
    out = run / "vthr_levels.csv"
    out.write_text("ch,vthr_mV\n" + "".join(f"{c},{levels[l]:g}\n" for c, l in enumerate(lvl)))
    print(f"wrote {out}")


def raw_thresholds(run: Path):
    """(cfg, latent raw thresholds [C], used raw thresholds [C] or None) of a run.

    Raw = the envelope units of the grid table (envelopes raw=True). For a free
    run the two are the same; for a grid run the latent one is where training
    pushed the parameter (straight-through keeps pushing past an edge cell),
    the used one is the cell the board gets."""
    import torch
    from data.afe import _EPS, AFEFrontend, load_afe_state
    from train.config import load_config
    cfg = load_config(str(run / "config.yaml"))
    if cfg.afe.normalize != "fixed" or cfg.afe.input_pp <= 0:
        raise SystemExit(f"{run}: needs normalize=fixed and input_pp>0 "
                         f"(got {cfg.afe.normalize}, {cfg.afe.input_pp}) -- raw units differ otherwise")
    afe = AFEFrontend(cfg.afe)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    load_afe_state(afe, ck["afe"])
    lo, hi = float(afe.fixed_lo), float(afe.fixed_hi)
    to_raw = lambda t: t.detach().double().numpy() * (hi - lo + _EPS) + lo
    latent = to_raw(afe.threshold)
    used = to_raw(afe.grid_thresholds()) if afe.grid_levels is not None else None
    return cfg, latent, used


def equiv_mv(raw: float, row: np.ndarray, levels: List[float]) -> float:
    """The reference voltage that would put this channel's comparator at `raw`:
    linear along the table row, extended with the end slopes outside it."""
    ok = np.isfinite(row)
    v, r = np.asarray(levels, dtype=float)[ok], row[ok]
    if len(r) < 2:
        return float("nan")
    if raw <= r[0]:
        return float(v[0] + (raw - r[0]) * (v[1] - v[0]) / (r[1] - r[0]))
    if raw >= r[-1]:
        return float(v[-1] + (raw - r[-1]) * (v[-1] - v[-2]) / (r[-1] - r[-2]))
    return float(np.interp(raw, r, v))


def compare(args) -> None:
    """Where each run's thresholds sit on the board's voltage axis."""
    from data.thr_grid import load_grid
    levels, g = load_grid(Path(args.grid))
    cols, lines = [], []
    for run in args.run:
        cfg, latent, used = raw_thresholds(Path(run))
        name = Path(run).name
        cols.append((name, latent, used))
    head = f"{'ch':>3} | " + "".join(f"{n[:16]:>18}" for n, _, _ in cols)
    print(f"grid {args.grid}: {levels[0]:g}..{levels[-1]:g} mV")
    print("equivalent reference voltage [mV] of each run's threshold "
          "(latent -> used for a grid run; < > = outside the buildable range)")
    print(head)
    out_lo, out_hi = {n: [] for n, _, _ in cols}, {n: [] for n, _, _ in cols}
    for c in range(g.shape[0]):
        cells = []
        for n, latent, used in cols:
            m = equiv_mv(latent[c], g[c], levels)
            mark = "<" if m < levels[0] - 8 else (">" if m > levels[-1] + 8 else " ")
            if mark == "<":
                out_lo[n].append(c)
            elif mark == ">":
                out_hi[n].append(c)
            s = f"{m:7.0f}{mark}"
            if used is not None:
                s += f"->{equiv_mv(used[c], g[c], levels):4.0f}"
            cells.append(f"{s:>18}")
        print(f"{c:>3} | " + "".join(cells))
    for n, _, _ in cols:
        print(f"{n}: below {levels[0]:g} mV ch {out_lo[n]}, above {levels[-1]:g} mV ch {out_hi[n]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim", help="provisional table from DC + simulated swing")
    s.add_argument("--config", default="configs/base.yaml")
    s.add_argument("--spice-matrix", default="analog/AFE/artifacts/filterbank_matrix_board.csv",
                   help="the filter bank bd_base trained on")
    s.add_argument("--dc-csv", type=Path, default=DC_CSV)
    s.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    s.add_argument("--vpp-mv", type=float, default=100.0, help="AWG level at the node")
    s.add_argument("--input-pp", type=float, default=0.29, help="afe.input_pp the run will use")
    s.add_argument("--out", required=True)
    r = sub.add_parser("report", help="voltage per channel of a grid-trained run")
    r.add_argument("--run", required=True)
    c = sub.add_parser("compare", help="each run's thresholds as reference voltages on a grid")
    c.add_argument("--grid", required=True)
    c.add_argument("--run", nargs="+", required=True,
                   help="runs trained with the same input_pp and filter bank (free or grid)")
    args = ap.parse_args()
    {"sim": sim, "report": report, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
