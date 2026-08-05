"""Fully-wired one-channel AFE schematic (paper Fig.1 style) from full_chain.cir.

Real connecting wires (not net-label stubs). Values shown as design VARIABLES
(RA, CVAL=C, R1v=R1, R2v=R2/R3, VREF via R7/R8) + designators (R4/R5/R6/C3);
no baked-in constants. 16 channels = this circuit with per-channel RA/C/R1.

Netlist nodes reproduced exactly:
  GIC: R1 in-np, C1 mid-np, RA1 c2p-np, U1(+np,-nn,->vfilt), RA2 nn-vfilt,
       C2 c2p-nn, U2(+u2p,-nn,->c2p), R2 u2p-vfilt, R3 mid-u2p
  DET: R4 vfilt-da, U3(+u3p,-da,->dk), R6 mid-u3p, R5 da-vdet, D1 da->dk,
       D2 dk->vdet, C3 vdet-gnd
  CMP: R7 1.8-vminus, R8 vminus-gnd, CMP(+vdet,-vminus,->vout)

Uses explicit coordinates (y up) so nothing overlaps. Run:
  .venv/bin/python AFE/scripts/draw_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import schemdraw
from schemdraw import elements as elm

AFE = Path(__file__).resolve().parents[1]


def opamp(d, *, in1, in2, out_at, sign_top, name, face_left=False):
    """Place an opamp so its in1 pin lands at `in1` (a coordinate). Pins get
    manual +/- (sign_top = '+' means the top input in1 is non-inverting).
    Returns the element (anchors: .in1 .in2 .out .vd .vs)."""
    op = elm.Opamp(sign=False).scale(0.55)
    if face_left:
        op = op.left()
    op = op.anchor("in1").at(in1)
    d += op
    top = "+" if sign_top == "+" else "−"
    bot = "−" if sign_top == "+" else "+"
    dx = 0.16 if not face_left else -0.16
    d += elm.Label().at(op.in1).label(top, ofst=(dx, 0.16), fontsize=15)
    d += elm.Label().at(op.in2).label(bot, ofst=(dx, -0.16), fontsize=15)
    d += elm.Label().at(op.center).label(name, ofst=(0, 0), fontsize=12)
    d += elm.Vdd().at(op.vd).label("1.8V", fontsize=10)
    d += elm.Ground().at(op.vs)
    return op


def wire(d, pts):
    for a, b in zip(pts[:-1], pts[1:]):
        d += elm.Line().at(a).to(b)


def R(d, a, b, label, loc="top"):
    d += elm.Resistor().at(a).to(b).label(label, loc=loc, fontsize=11)


def C(d, a, b, label, loc="top"):
    d += elm.Capacitor().at(a).to(b).label(label, loc=loc, fontsize=11)


def main():
    schemdraw.config(unit=1.0, fontsize=12, lw=1.6)
    d = schemdraw.Drawing()

    # ---- node coordinates (y up) ----
    Yt, Ym = 2.0, 0.0                       # top rail, mid rail
    IN = (0, Yt); NP = (2.2, Yt)
    C2P = (2.2, Ym); NN = (4.4, Ym); VF = (7.4, Ym)
    U2P = (7.4, -2.2)
    DA = (10.6, Ym); DK = (13.6, Ym); VDET = (15.6, Ym)
    VM = (18.0, -0.9)                       # comparator V- (divider mid)

    # ============ GIC band-pass ============
    R(d, IN, NP, "R1 =R1v")
    d += elm.Label().at(IN).label("Vin", ofst=(-0.5, 0.0), fontsize=11)
    d += elm.Dot().at(NP)
    # C1: np -> 0.9V (vertical cap in the bias stub)
    d += elm.Capacitor().at(NP).up().length(1.4).label("C1 =CVAL", loc="right", fontsize=11)
    d += elm.Vdd().label("0.9V", fontsize=10)

    # U1: + top (in1)=np, - bottom(in2)=nn, out->vfilt
    U1 = opamp(d, in1=(3.7, Yt), in2=None or (3.7, Yt-1.3),
               out_at=None, sign_top="+", name="U1")
    wire(d, [NP, U1.in1])
    wire(d, [U1.in2, (U1.in2[0], Ym), NN])            # - input down to nn
    wire(d, [U1.out, (VF[0], U1.out[1]), VF])         # out down to vfilt
    d += elm.Dot().at(VF); d += elm.Label().at(VF).label("Vfilt", ofst=(0.1, 0.3), fontsize=11)
    # RA1 np->c2p, C2 c2p->nn, RA2 nn->vfilt
    R(d, NP, C2P, "RA1 =RA", loc="left")
    d += elm.Dot().at(C2P)
    C(d, C2P, NN, "C2 =CVAL")
    d += elm.Dot().at(NN)
    R(d, NN, VF, "RA2 =RA")
    # R2 vfilt->u2p, R3 u2p->0.9V
    R(d, VF, U2P, "R2 =R2v", loc="right")
    d += elm.Dot().at(U2P)
    wire(d, [U2P, (2.2, U2P[1])]);
    R(d, (2.2, U2P[1]), (2.2, U2P[1]-1.6), "R3 =R2v", loc="left")
    d += elm.Vss().at((2.2, U2P[1]-1.6)).label("0.9V", fontsize=10)
    # U2: - top(in1)=nn, + bottom(in2)=u2p, out->c2p  (faces left)
    U2 = opamp(d, in1=(5.6, -1.7), in2=None or (5.6, -3.0),
               out_at=None, sign_top="−", name="U2", face_left=True)
    wire(d, [U2.in1, (5.6, -0.4), (NN[0], -0.4), NN])          # - to nn
    wire(d, [U2.in2, (U2P[0]-0.0, U2.in2[1]), (U2P[0], U2P[1])])  # + to u2p (via right/up)
    wire(d, [U2.out, (1.4, U2.out[1]), (1.4, Ym), C2P])         # out to c2p

    # ============ active detector ============
    R(d, VF, DA, "R4")
    d += elm.Dot().at(DA)
    U3 = opamp(d, in1=(11.6, Ym), in2=None or (11.6, Ym-1.3),
               out_at=None, sign_top="−", name="U3")
    wire(d, [DA, U3.in1])
    wire(d, [U3.in2, (10.9, U3.in2[1])])
    R(d, (10.9, U3.in2[1]), (10.9, U3.in2[1]-1.5), "R6", loc="left")
    d += elm.Vss().at((10.9, U3.in2[1]-1.5)).label("0.9V", fontsize=10)
    wire(d, [U3.out, (DK[0], U3.out[1]), DK]); d += elm.Dot().at(DK)  # up to mid rail
    # D2 dk->vdet (horizontal)
    d += elm.Diode().at(DK).to(VDET).label("D2", fontsize=11)
    d += elm.Dot().at(VDET); d += elm.Label().at(VDET).label("V+", ofst=(0.35, 0.35), fontsize=11)
    # D1 da->dk (feedback, above)
    wire(d, [DA, (DA[0], 1.3)])
    d += elm.Diode().at((DA[0], 1.3)).to((DK[0], 1.3)).label("D1", fontsize=11)
    wire(d, [(DK[0], 1.3), DK])
    # R5 da->vdet (top feedback)
    wire(d, [DA, (DA[0], 2.4)])
    R(d, (DA[0], 2.4), (VDET[0], 2.4), "R5")
    wire(d, [(VDET[0], 2.4), VDET])
    # C3 vdet->gnd
    d += elm.Capacitor().at(VDET).to((VDET[0], VDET[1]-1.6)).label("C3", loc="right", fontsize=11)
    d += elm.Ground().at((VDET[0], VDET[1]-1.6))

    # ============ comparator ============
    CMP = opamp(d, in1=(20.0, Ym), in2=None or (20.0, Ym-1.3),
                out_at=None, sign_top="+", name="CMP")
    wire(d, [VDET, CMP.in1])
    d += elm.Line().at(CMP.out).right().length(0.8).label("Vout", loc="right", fontsize=12)
    # R7/R8 divider -> V- (in2)
    wire(d, [CMP.in2, (VM[0], CMP.in2[1]), VM]); d += elm.Dot().at(VM)
    R(d, VM, (VM[0], VM[1]+1.7), "R7", loc="left")
    d += elm.Vdd().at((VM[0], VM[1]+1.7)).label("1.8V", fontsize=10)
    R(d, VM, (VM[0], VM[1]-1.7), "R8", loc="left")
    d += elm.Ground().at((VM[0], VM[1]-1.7))
    d += elm.Label().at(VM).label("V−", ofst=(0.3, 0.28), fontsize=11)

    # stage separators / titles
    for x, t in [(3.6, "GIC filter"), (12.4, "active detector"), (20.0, "comparator")]:
        d += elm.Label().at((x, 4.3)).label(t, fontsize=15, color="#0057b8")

    d.save(str(AFE / "artifacts" / "afe_schematic.svg"))
    d.save(str(AFE / "artifacts" / "afe_schematic.png"), dpi=150)
    print("저장: artifacts/afe_schematic.svg / .png")


if __name__ == "__main__":
    main()
