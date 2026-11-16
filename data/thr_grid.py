"""Comparator thresholds the board can actually build: a per-channel grid.

The board's reference generator offers a few voltages (2026-11: 900..996 mV in
16 mV steps), every channel picks one, and one voltage can drive at most a few
comparators (4 today). A learned threshold that falls between two levels, or a
fifth channel wanting the same level, is a number nobody can solder. This
module turns those constraints into something training can respect:

  grid CSV      channel x level -> the RAW envelope value (AFEFrontend.envelopes
                raw=True units) at which that channel's comparator trips when its
                reference sits at that level. Raw, not normalized, so the table
                does not depend on any run's fixed_lo/hi. Written by
                experiments/thr_grid_sim.py (from the circuit model) or
                experiments/thr_calib.py (fitted to board captures).
  assign()      each channel -> one level, nearest to its learned threshold,
                with at most `capacity` channels per level (min-cost assignment).

AFEFrontend snaps its thresholds to assign()'s choice in the forward pass,
straight-through, so the network always trains on a buildable comparator bank.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_grid(path: str | Path) -> Tuple[List[float], np.ndarray]:
    """CSV -> (levels in mV, [C, L] raw thresholds; NaN = level unusable).

        # comments
        ch,900,916,932,948,964,980,996
        0,0.0123,0.0151,...
        1,,0.0098,...            <- empty cell = this channel cannot use 900
    """
    p = Path(path)
    rows = [l.strip() for l in p.read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    head = rows[0].split(",")
    if head[0].strip().lower() != "ch":
        raise ValueError(f"{p}: first column must be 'ch', got {head[0]!r}")
    levels = [float(h) for h in head[1:]]
    g = np.full((len(rows) - 1, len(levels)), np.nan)
    for i, r in enumerate(rows[1:]):
        cells = r.split(",")
        if int(cells[0]) != i:
            raise ValueError(f"{p}: row {i + 1} is channel {cells[0]}, expected {i}")
        for j, v in enumerate(cells[1:1 + len(levels)]):
            v = v.strip()
            if v and v.lower() != "nan":
                g[i, j] = float(v)
    if not np.isfinite(g).any(axis=1).all():
        bad = [i for i in range(len(g)) if not np.isfinite(g[i]).any()]
        raise ValueError(f"{p}: channels {bad} have no usable level")
    return levels, g


def write_grid(path: str | Path, levels: List[float], g: np.ndarray,
               comments: List[str] = ()) -> None:
    lines = [f"# {c}" for c in comments]
    lines.append("ch," + ",".join(f"{v:g}" for v in levels))
    for i, row in enumerate(g):
        lines.append(f"{i}," + ",".join("" if not np.isfinite(v) else f"{v:.6g}" for v in row))
    Path(path).write_text("\n".join(lines) + "\n")


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Min-cost assignment of every row to a distinct column (rows <= cols).

    The O(n^2 m) potentials form of the Hungarian method. Returns col per row.
    16 x 28 here, so speed is irrelevant; exactness is the point."""
    n, m = cost.shape
    if n > m:
        raise ValueError(f"{n} rows cannot go to {m} columns")
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)        # p[j] = row (1-based) on column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            col[p[j] - 1] = j - 1
    return col


def assign(thr: np.ndarray, grid: np.ndarray, capacity: int = 0) -> np.ndarray:
    """Each channel -> a level index, minimizing sum (thr_c - grid_c,l)^2.

    thr and grid in the SAME units. capacity <= 0: every channel takes its own
    nearest usable level. capacity k: at most k channels per level -- a min-cost
    assignment over k copies of every level, so the channels that move are the
    ones that lose least by moving."""
    thr = np.asarray(thr, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    C, L = grid.shape
    if thr.shape[0] != C:
        raise ValueError(f"{thr.shape[0]} thresholds for a {C}-channel grid")
    cost = (thr[:, None] - grid) ** 2
    big = 1e6 * (np.nanmax(cost) + 1.0)      # unusable, but keeps the solver finite
    cost = np.where(np.isfinite(grid), cost, big)
    if capacity <= 0:
        return cost.argmin(axis=1)
    if C > L * capacity:
        raise ValueError(f"{C} channels do not fit {L} levels x {capacity}")
    col = _hungarian(np.repeat(cost, capacity, axis=1))
    lvl = col // capacity
    if (cost[np.arange(C), lvl] >= big).any():
        raise ValueError("no assignment uses only usable levels -- raise the capacity")
    return lvl


def describe(levels: List[float], lvl: np.ndarray) -> str:
    """'ch0 900, ch1 948, ...' plus how many channels share each level."""
    per = ", ".join(f"ch{c} {levels[l]:g}" for c, l in enumerate(lvl))
    use = ", ".join(f"{levels[j]:g}:{int((lvl == j).sum())}" for j in range(len(levels)))
    return f"{per}\n  channels per level: {use}"


def _brute_force(thr, grid, capacity):         # tests only
    import itertools
    C, L = grid.shape
    best, arg = math.inf, None
    for combo in itertools.product(range(L), repeat=C):
        if capacity > 0 and max(combo.count(j) for j in range(L)) > capacity:
            continue
        if any(not np.isfinite(grid[c, l]) for c, l in enumerate(combo)):
            continue
        s = sum((thr[c] - grid[c, l]) ** 2 for c, l in enumerate(combo))
        if s < best:
            best, arg = s, combo
    return (None, best) if arg is None else (np.array(arg), best)
