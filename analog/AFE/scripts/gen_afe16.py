"""Emit analog/AFE/netlists/afe16_xlse.cir -- the whole 16-channel AFE.

The per-channel chain is copied from AFE_tuning/netlists/v15_channel.cir, which
matches the colleague's board node for node. Only the comparator reference is
new, so anything that already validated on that netlist still validates here.

The old reference was R7/R8, a fixed divider across the full 1.8 V rail. The new
one is:

    16 envelopes -> diode-OR + current sink        -> V_or = LSE(env) - V_d
    buffer with a MATCHED diode in the feedback    -> V_max = V_or + V_d = LSE
    per-channel divider Ra/Rb between V_max, V_ref -> V_thr,c
    V_ref = envelope quiescent, i.e. delta = 0

Two things are worth knowing before reading the numbers it produces:

  * the sink must be a CURRENT source, not a resistor. The log-sum-exp only
    comes out because the sum of the diode currents is held constant; through a
    resistor the sum would follow V_or and the algebra collapses.
  * V_ref is an ideal source here on purpose, so delta stays one sweepable knob
    matching training's floor_frac. A replica detector was tried and does not
    work -- see SPICE_FINDINGS.md section 4 for why nothing passive can track it.

Component values come from filterbank_design.csv and the alphas from a trained
run, so regenerating cannot drift from either source.

    python3 analog/AFE/scripts/gen_afe16.py --no-tran
    cd analog/AFE/netlists && ngspice -b afe16_xlse.cir

Findings are written up in analog/AFE/SPICE_FINDINGS.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
DESIGN = ROOT / "analog/AFE/artifacts/filterbank_design.csv"
OUT = ROOT / "analog/AFE/netlists/afe16_xlse.cir"
TRAN_CSV = ROOT / "analog/AFE/artifacts/afe16_tran.csv"
TONE_CH = 6             # channel the default test tone lands in

# runs/xl_g12, read through effective_alpha(). Regenerate after any retrain --
# the raw parameter is NOT this, it floats outside [0,1] (straight-through).
ALPHA = [0.8745, 0.7804, 0.4788, 0.4482, 0.5423, 0.4072, 0.1725, 0.1326,
         0.1799, 0.1242, 0.2096, 0.2005, 0.1244, 0.0788, 0.0572, 0.3787]

# Envelope quiescent, from this netlist's own .op (all 16 agree to 5 uV).
# The colleague's board measures 917.83 mV, so the sim tracks it to 0.55 mV.
VQUIESCENT = 0.9183840

RTOT = 1.0e6            # Ra + Rb per channel [ohm]
ISINK = 1.0e-6          # diode-OR sink [A]. 1 uA -> V_d ~ 0.16 V, ~1.8 uW.
                        # Does NOT set the LSE temperature: that is n*V_T
                        # regardless of current. Only V_d moves, and the
                        # matched diode cancels it.


def eng(v: float, unit: str = "") -> str:
    """SPICE-friendly value. ngspice reads 'meg' but not 'M' for 1e6."""
    for scale, suf in ((1e6, "meg"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
                       (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(v) >= scale or scale == 1e-12:
            return f"{v / scale:.4f}{suf}{unit}"
    return f"{v}{unit}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=2.52e-3,
                    help="mic amplitude [V peak]. 2.52 mV = 74 dB SPL "
                         "(ANALOG.md 6-2)")
    ap.add_argument("--freq", type=float, default=1349.0,
                    help="test tone [Hz]. default = channel 6 centre")
    ap.add_argument("--no-tran", action="store_true",
                    help="operating point only. The transient is ~50 s.")
    ap.add_argument("--delta", type=float, default=0.0,
                    help="V_ref - quiescent [mV]. 0 = what training assumes.")
    ap.add_argument("--hardmax", action="store_true",
                    help="replace the diode-OR with an ideal max(), i.e. the "
                         "'xmax' normalisation, to see what the LSE floor buys")
    ap.add_argument("--r4", type=float, default=10e3,
                    help="detector input resistor [ohm]. Sets gain R5/R4, NOT tau.")
    ap.add_argument("--r5", type=float, default=47e3,
                    help="detector feedback resistor [ohm]. Sets BOTH the gain "
                         "R5/R4 and tau = R5*C3.")
    ap.add_argument("--preamp", type=float, default=10.0,
                    help="mic preamp gain (Rf = (G-1)*10k).")
    ap.add_argument("--c3", type=float, default=100e-9,
                    help="detector smoothing cap [F]. Sets tau = R5*C3.")
    ap.add_argument("-o", "--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    args.out.write_text(build(args), encoding="utf-8")
    print(f"wrote {args.out}")


def build(args) -> str:
    """Netlist text for one parameter set. `args` needs the argparse fields.

    Every value the analog side is still moving -- R4, R5, C3, preamp gain,
    the reference -- is a parameter here, because they are moving and a
    hard-coded netlist would go stale between one conversation and the next.

    `vref` is the one that cannot be defaulted honestly: the quiescent point
    is `da + (R5/R4)*(vf - da)`, so it MOVES when R4 or R5 does. Pass it
    explicitly (sim_afe.py measures it first), or accept VQUIESCENT, which is
    only correct for the original R4=10k / R5=47k.
    """
    vref = getattr(args, "vref", None)
    if vref is None:
        vref = VQUIESCENT + args.delta * 1e-3
    r5 = getattr(args, "r5", 47e3)
    preamp = getattr(args, "preamp", 10.0)
    tran_csv = getattr(args, "tran_csv", f"../artifacts/{TRAN_CSV.name}")

    with DESIGN.open(newline="", encoding="utf-8") as fh:
        d = list(csv.DictReader(fh))
    assert len(d) == 16, f"{DESIGN} has {len(d)} channels, expected 16"
    L = []
    A = L.append

    A(".title AFE 16ch -- diode-OR soft-max threshold (xlse), delta = 0")
    A("* Generated by analog/AFE/scripts/gen_afe16.py -- do not hand-edit.")
    A("* Per-channel chain identical to AFE_tuning/netlists/v15_channel.cir.")
    A("* New vs the existing board: the comparator reference only (see below).")
    A("")
    A(".include ../models/OPA379.LIB")
    A("* BAT54 as a 2-terminal .model: the vendor subckt carries a parametric")
    A("* 'mfg=' that ngspice rejects. Same params the verified netlists use.")
    A(".model Dbat54wt1 D(Is=2.2n Rs=2 N=1.03 Cjo=10p M=0.4 Vj=0.4 Bv=30")
    A("+ Ibv=10u Eg=0.69 Xti=2)")
    A("")
    A("* ---- supplies -------------------------------------------------------")
    A("Vpos  vpos  0  DC 1.8")
    A("Vmid  vmid  0  DC 0.9        $ bias rail")
    A("")
    A(f"* ---- microphone + preamp (G = 1 + Rf/Rg = {preamp:g}) ---------------")
    A("* Tone BURST (on 20-60 ms, off after) so the transient shows both the")
    A("* attack and the decay. u() is ngspice's unit step.")
    A(f"Bmic  vmic  vmid  V = {args.amp:.6g}*sin(6.283185307*{args.freq:.6g}*time)")
    A("+ *(u(time-20m)-u(time-60m))")
    A("XUPRE vmic  npre  vpos 0 vin  OPA379")
    A(f"Rf    vin   npre  {eng((preamp - 1.0) * 10e3)}")
    A("Rg    npre  vmid  10k")
    A("")
    A("* ---- shared threshold generator (THE NEW PART) ----------------------")
    if args.hardmax:
        A("* xmax variant: an IDEAL max(), which is what the diode-OR would be")
        A("* if n*V_T were zero. Not buildable as written -- it exists to isolate")
        A("* what the LSE floor T*ln(16) is worth. See SPICE_FINDINGS.md 2.")
        expr = "v(ve0)"
        for c in range(1, 16):
            expr = f"max({expr},v(ve{c}))"
        A(f"Bmax  v_max 0  V = {expr}")
        A("")
    else:
        A("* A) diode-OR: 16 BAT54 into one node held at constant current.")
        A("*    The sink MUST be a current source. With a resistor the sum of the")
        A("*    diode currents follows V_or and the log-sum-exp does not appear.")
        A(f"Isink v_or  0     DC {eng(ISINK)}")
        A("")
        A("* B) buffer with a MATCHED diode in the feedback, carrying the same")
        A("*    current as the sink, so it adds back exactly the V_d the OR lost:")
        A("*      op-amp holds n_fb = v_or, Dcomp drops V_d(Iref)")
        A("*      => v_max = v_or + V_d = LSE(env)")
        A("XUBUF v_or  n_fb  vpos 0 v_max  OPA379")
        A("Dcomp v_max n_fb  Dbat54wt1")
        A(f"Iref  n_fb  0     DC {eng(ISINK)}")
        A("")
    A("* C) V_ref, the low end of every divider. Training fixes it at the")
    A("*    envelope quiescent, i.e. delta = 0 (docs/EXPERIMENT_MAP.md A-3).")
    A("*")
    A("*    Kept as an ideal source on purpose, so delta is one sweepable knob")
    A("*    here exactly as floor_frac is in training. On the board it is a")
    A("*    divider off the 1.8 V rail through a unity-gain OPA379; the buffer")
    A("*    is NOT optional, because at quiescent D2 is reverse biased and the")
    A("*    envelope node is high impedance -- unbuffered, the 16 dividers drew")
    A("*    ~0.9 uA and moved the reference 40 mV.")
    A("*")
    A("*    A replica detector was tried first and does not work. Its R4 has to")
    A("*    sit at the filter output DC level, not at vmid, and that level is")
    A("*    itself 2 op-amp offsets below vmid. Worse, the quiescent envelope is")
    A("*    da + (R5/R4)*(vf - da): each channel's own offset, amplified 4.7x,")
    A("*    random per device. Nothing passive tracks that, so a replica buys")
    A("*    less than it costs (3 op-amps) and this is a bring-up trim instead.")
    A(f".param VREFSET={vref:.7f}   $ delta = {args.delta:+.1f} mV")
    A("Vref  v_ref 0  DC {VREFSET}")
    A("")

    for c in range(16):
        row = d[c]
        fc, q = float(row["fc_sim"]), float(row["Q_sim"])
        ra, cv, r1 = float(row["RA"]), float(row["C"]), float(row["R1"])
        a = ALPHA[c]
        A(f"* ==== channel {c}: f_c {fc:.0f} Hz, Q {q:.2f}, alpha {a:.4f} ====")
        # GIC bandpass
        A(f"R1_{c}   vin    np{c}   {eng(r1)}")
        A(f"C1_{c}   vmid   np{c}   {eng(cv)}")
        A(f"RA1_{c}  c2p{c} np{c}   {eng(ra)}")
        A(f"XU1_{c}  np{c}  nn{c}   vpos 0 vf{c}  OPA379")
        A(f"C2_{c}   c2p{c} nn{c}   {eng(cv)}")
        A(f"RA2_{c}  nn{c}  vf{c}   {eng(ra)}")
        A(f"R2_{c}   nu2{c} vf{c}   100k")
        A(f"R3_{c}   vmid   nu2{c}  100k")
        A(f"XU2_{c}  nu2{c} nn{c}   vpos 0 c2p{c}  OPA379")
        # active detector
        A(f"R4_{c}   vf{c}  da{c}   {eng(args.r4)}")
        A(f"R5_{c}   da{c}  ve{c}   {eng(r5)}")
        A(f"R6_{c}   vmid   np3{c}  8.25k")
        A(f"D1_{c}   da{c}  dk{c}   Dbat54wt1")
        A(f"XU3_{c}  np3{c} da{c}   vpos 0 dk{c}  OPA379")
        A(f"D2_{c}   dk{c}  ve{c}   Dbat54wt1")
        A(f"C3_{c}   ve{c}  0       {eng(args.c3)}")
        # into the OR bus
        A(f"DOR_{c}  ve{c}  v_or    Dbat54wt1")
        # threshold divider: alpha = Rb/(Ra+Rb) -> V_thr = a*V_max + (1-a)*V_ref
        A(f"Ra_{c}   v_max  vt{c}   {eng((1.0 - a) * RTOT)}")
        A(f"Rb_{c}   vt{c}  v_ref   {eng(a * RTOT)}")
        # behavioural comparator (LPV7215 model uses if()/switch ngspice rejects)
        A(f"Bcmp_{c} vo{c}  0  V = 0.9 + 0.9*tanh(2e4*(v(ve{c}) - v(vt{c})))")
        A("")

    A("* ---- analyses -------------------------------------------------------")
    A(".control")
    A("op")
    A("echo ''")
    A("echo '--- shared nodes -------------------------------------'")
    A("print v(vin) v(v_max) v(v_ref)" + ("" if args.hardmax else " v(v_or)"))
    A("echo ''")
    A("echo '--- per channel: envelope, threshold, comparator ------'")
    for c in range(16):
        A(f"print v(ve{c}) v(vt{c}) v(vo{c})")
    A("")
    if not args.no_tran:
        A("* Tone burst. The tail is what to watch: R5 discharges C3 back toward")
        A("* the detector bias, and that is the tau CLAUDE.md 2.5 says the paper")
        A(f"* never gives a number for. R5*C3 = {r5 * args.c3 * 1e3:.2f} ms nominal;")
        A("* the OR diode pulls in parallel so the measured tau comes out lower.")
        A("tran 20u 90m")
        A(f"wrdata {tran_csv} v(ve{TONE_CH}) v(v_max)"
          f" v(v_ref) v(vt{TONE_CH}) v(vo{TONE_CH})")
    A(".endc")
    A("")
    A(".end")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
