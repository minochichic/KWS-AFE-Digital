"""One-call SPICE runs of the 16-channel AFE, for sweeping values from a cell.

    from analog.AFE.scripts.sim_afe import sim, table
    r = [sim("now",  r4=30e3, r5=350e3, tau_ms=0.7),
         sim("pre5", r4=30e3, r5=350e3, tau_ms=0.7, preamp=5.0)]
    table(r)

Everything the analog side is still moving is a keyword argument. Two of them
carry the design intent and the rest follow:

    gain = R5 / R4          what the detector does to the signal
    tau  = R5 * C3          how fast the envelope forgets

so R5 moves both, and C3 is derived from `tau_ms` unless you pass it directly.

V_ref is NOT a free parameter and is not defaulted. The quiescent envelope is
`da + (R5/R4)*(vf - da)`, so it moves whenever R4 or R5 does -- the 918.4 mV
in BUILD_TABLE.md is only right for the original 10k/47k. Every run here
measures the quiescent first with a throwaway .op, then sets V_ref from it, so
delta means what training means by it whatever the resistors are.
"""
from __future__ import annotations

import math
import pathlib
import re
import subprocess
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_afe16 as G  # noqa: E402

NETLISTS = pathlib.Path(__file__).resolve().parents[1] / "netlists"
ARTIFACTS = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
BURST = (0.020, 0.060)          # tone on/off in the generated netlist


def _run(text: str, tag: str) -> str:
    cir = NETLISTS / f"sim_{tag}.cir"
    cir.write_text(text, encoding="utf-8")
    p = subprocess.run(["ngspice", "-b", cir.name], cwd=NETLISTS,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ngspice failed for {tag}:\n{p.stderr[-2000:]}")
    return p.stdout


def _nodes(out: str) -> dict[str, float]:
    return {m.group(1).lower(): float(m.group(2))
            for m in re.finditer(r"^v\((\S+)\)\s*=\s*(\S+)", out, re.M)}


def sim(tag: str, *, r4: float = 30e3, r5: float = 350e3,
        tau_ms: float | None = 0.7, c3: float | None = None,
        preamp: float = 10.0, freq: float = 1349.0, amp: float = 2.52e-3,
        delta_mv: float = 0.0, hardmax: bool = False,
        tran: bool = True, verbose: bool = True) -> dict:
    """Build, run and measure one parameter set. ~60 s with tran, ~2 s without."""
    if c3 is None:
        if tau_ms is None:
            raise ValueError("pass tau_ms or c3")
        c3 = tau_ms * 1e-3 / r5
    a = types.SimpleNamespace(
        r4=r4, r5=r5, c3=c3, preamp=preamp, freq=freq, amp=amp,
        delta=delta_mv, hardmax=hardmax, no_tran=True, vref=None,
        tran_csv=f"../artifacts/sim_{tag}.csv")

    # pass 1: where does the envelope actually sit with THESE resistors
    a.vref = 0.9
    q = [v for k, v in _nodes(_run(G.build(a), tag)).items()
         if re.fullmatch(r"ve\d+", k)]
    quiescent = sum(q) / len(q)

    # pass 2: V_ref from the measured quiescent, so delta = what was asked for
    a.vref = quiescent + delta_mv * 1e-3
    a.no_tran = not tran
    d = _nodes(_run(G.build(a), tag))

    marg = [(d[f"vt{c}"] - d[f"ve{c}"]) * 1e3 for c in range(16)]
    r = dict(tag=tag, r4=r4, r5=r5, c3=c3, preamp=preamp, freq=freq,
             gain=r5 / r4, tau_nom_ms=r5 * c3 * 1e3,
             quiescent=quiescent, vmax=d["v_max"], vref=d["v_ref"],
             floor_mv=(d["v_max"] - quiescent) * 1e3,
             margins_mv=marg, min_margin_mv=min(marg),
             n_firing=sum(d[f"vo{c}"] > 0.9 for c in range(16)),
             scatter_gain=r5 / r4)
    r.update(rise_mv=math.nan, tau_ms=math.nan, ripple_mv=math.nan,
             fire_pct=math.nan)
    if tran:
        r.update(_measure(ARTIFACTS / f"sim_{tag}.csv", quiescent))
    if verbose:
        print(f"[{tag}] gain {r['gain']:.2f}  tau_nom {r['tau_nom_ms']:.2f} ms  "
              f"quiescent {quiescent*1e3:.1f} mV  floor {r['floor_mv']:.1f} mV  "
              f"min margin {r['min_margin_mv']:.2f} mV")
    return r


def _measure(csv: pathlib.Path, q: float) -> dict:
    rows = [[float(x) for x in ln.split()]
            for ln in csv.read_text().splitlines() if ln.strip()]
    t = [x[0] for x in rows]
    ve, vo = [x[1] for x in rows], [x[9] for x in rows]
    at = lambda ts: min(range(len(t)), key=lambda k: abs(t[k] - ts))  # noqa: E731
    on, off = BURST
    seg = ve[at(off - 0.015):at(off - 0.001)]        # settled part of the burst
    tau = math.nan
    i0 = at(off + 0.0005)
    y0 = ve[i0] - q
    for k in range(i0, at(off + 0.028)):
        if ve[k] <= q + y0 / math.e:
            tau = (t[k] - t[i0]) * 1e3
            break
    lo, hi = at(on), at(off)
    return dict(rise_mv=(max(seg) - q) * 1e3, tau_ms=tau,
                ripple_mv=(max(seg) - min(seg)) * 1e3,
                fire_pct=sum(vo[k] > 0.9 for k in range(lo, hi)) / (hi - lo) * 100)


def table(rows: list[dict]) -> None:
    """Compare runs. The last two columns are the ones that trade off."""
    h = (f"{'tag':>10} {'R4':>6} {'R5':>7} {'C3':>7} {'pre':>4} {'gain':>6} "
         f"{'quiesc':>8} {'rise':>7} {'tau':>6} {'floor':>6} {'minMrg':>7} "
         f"{'scatter':>8} {'head':>5}")
    print(h)
    print("-" * len(h))
    for r in rows:
        # +-0.5 mV OPA379 Vos, amplified by R5/R4, plus ~1 mV comparator offset
        scatter = 0.5 * r["scatter_gain"] + 1.0
        print(f"{r['tag']:>10} {r['r4']/1e3:>5.0f}k {r['r5']/1e3:>6.0f}k "
              f"{r['c3']*1e9:>6.2f}n {r['preamp']:>4.0f} {r['gain']:>6.2f} "
              f"{r['quiescent']*1e3:>7.1f}m {r['rise_mv']:>6.1f}m "
              f"{r['tau_ms']:>5.2f}m {r['floor_mv']:>5.1f}m "
              f"{r['min_margin_mv']:>6.2f}m {scatter:>7.2f}m "
              f"{r['min_margin_mv']/scatter:>5.2f}x")
    print()
    print("floor   = LSE bottom T*ln16. Fixed by n*V_T -- does NOT scale with gain.")
    print("minMrg  = worst channel's silence margin = alpha_min * floor.")
    print("scatter = per-channel quiescent spread, (R5/R4)*Vos + comparator offset.")
    print("head    = minMrg / scatter. Below 1x, the quietest channels free-run")
    print("          at silence. THIS is what gain costs -- the floor is fixed,")
    print("          so raising R5/R4 grows the scatter without growing the margin.")


if __name__ == "__main__":
    table([sim("now", r4=30e3, r5=350e3, tau_ms=0.7)])
