"""Precise per-layer op count for BinaryMatchboxNet (grounds the FPGA analysis).

Hooks every conv in the real model and counts MACs from the ACTUAL tensor shapes
(no hand-derived formulas), splitting them by precision:
  * binary  -> XNOR + popcount  (1-bit weights AND 1-bit activations)
  * int8    -> integer MAC      (conv3); conv1 is int8 weights x +-1 input, so it
               is a signed accumulate (no multiplier) -- reported separately
  * fixed   -> conv4 epilogue

Also derives the folding budget: at a 2-10 Hz classification rate an FPGA has
~1e7-1e8 cycles per inference, so it prints how few MACs/cycle actually suffice.

Run from repo root: .venv/bin/python troubleshooting/scripts/count_ops.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.binary_matchboxnet import BinaryMatchboxNet          # noqa: E402
from train.config import load_config                             # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CLOCK_HZ = 100e6
RATES = [2, 10]                    # classifications per second


def precision_of(mod: nn.Module) -> str:
    n = type(mod).__name__
    if "Binary" in n:
        return "binary"
    if "Quant" in n:
        return "int8"
    return "fixed"


def main():
    cfg = load_config(str(REPO / "configs" / "base.yaml"))
    model = BinaryMatchboxNet(cfg.model).eval()

    rows = []

    def hook(name):
        def f(mod, inp, out):
            cin_g = mod.in_channels // mod.groups
            k = mod.kernel_size[0]
            L_out = out.shape[-1]
            macs = mod.out_channels * L_out * cin_g * k
            rows.append(dict(name=name, prec=precision_of(mod), macs=macs,
                             cin=mod.in_channels, cout=mod.out_channels, k=k,
                             groups=mod.groups, L=L_out,
                             params=sum(p.numel() for p in mod.parameters())))
        return f

    handles = [m.register_forward_hook(hook(n))
               for n, m in model.named_modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model(torch.sign(torch.randn(1, cfg.model.in_channels, cfg.model.T)))
    for h in handles:
        h.remove()

    # ---- per-layer table ----
    print(f"{'layer':<34}{'prec':>7}{'Cin':>5}{'Cout':>6}{'k':>4}{'grp':>5}"
          f"{'L':>5}{'MACs':>12}")
    tot = {}
    for r in rows:
        print(f"{r['name']:<34}{r['prec']:>7}{r['cin']:>5}{r['cout']:>6}"
              f"{r['k']:>4}{r['groups']:>5}{r['L']:>5}{r['macs']:>12,}")
        tot[r['prec']] = tot.get(r['prec'], 0) + r['macs']

    total = sum(tot.values())
    print(f"\n{'':<34}{'':>7}{'':>5}{'':>6}{'':>4}{'':>5}{'TOTAL':>5}{total:>12,}")
    print("\n=== by precision ===")
    for p, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {p:<8} {v:>12,} MACs  ({100*v/total:5.1f}%)")

    # conv1: int8 weights x +-1 activations -> signed accumulate, no multiplier
    c1 = sum(r['macs'] for r in rows if '.conv1.' in r['name'])
    print(f"\n  conv1 (int8 w x +-1 act = signed accumulate, NO multiplier): "
          f"{c1:,} ({100*c1/total:.1f}%)")
    true_mult = tot.get('int8', 0) - c1 + tot.get('fixed', 0)
    print(f"  -> TRUE multipliers needed: {true_mult:,} MACs "
          f"({100*true_mult/total:.1f}%)  [conv3 + conv4 only]")

    # binary depthwise vs pointwise (CLAUDE.md 3.4: keep the two PEs separate)
    dw = sum(r['macs'] for r in rows if r['prec'] == 'binary' and r['groups'] > 1)
    pw = tot.get('binary', 0) - dw
    print(f"\n  binary depthwise : {dw:>10,} ({100*dw/tot['binary']:4.1f}% of binary)"
          f"  <- packing gain small")
    print(f"  binary pointwise : {pw:>10,} ({100*pw/tot['binary']:4.1f}% of binary)"
          f"  <- packing gain max")

    # ---- folding budget ----
    print("\n=== folding budget (why fully-unrolled is over-provisioned) ===")
    for rate in RATES:
        cyc = CLOCK_HZ / rate
        print(f"  {rate:>2} Hz @100MHz -> {cyc:,.0f} cycles/inference; "
              f"need {total/cyc:8.3f} MACs/cycle to finish in time")
    print("  (i.e. a single-MAC engine already meets the real-time requirement)")

    # ---- folding design points ----
    # Three engine classes, folded independently (they need different hardware):
    #   P_bin : XNOR+popcount lanes (bit-packed)          -> binary layers
    #   P_acc : signed-accumulate lanes (int8 w x +-1 act) -> conv1, no multiplier
    #   P_dsp : true int8 multipliers                      -> conv3 + conv4
    n_bin, n_acc, n_dsp = tot.get('binary', 0), c1, true_mult
    print("\n=== folding design points (cycles @100MHz) ===")
    print(f"{'design':<12}{'P_bin':>7}{'P_acc':>7}{'P_dsp':>7}"
          f"{'bin cyc':>11}{'acc cyc':>10}{'dsp cyc':>10}{'total':>11}{'ms':>7}")
    for name, pb, pa, pd in [("minimal", 8, 1, 1), ("balanced", 64, 8, 4),
                             ("fast", 256, 32, 16)]:
        cb, ca, cd = n_bin / pb, n_acc / pa, n_dsp / pd
        tt = cb + ca + cd
        print(f"{name:<12}{pb:>7}{pa:>7}{pd:>7}{cb:>11,.0f}{ca:>10,.0f}"
              f"{cd:>10,.0f}{tt:>11,.0f}{1e3*tt/CLOCK_HZ:>7.1f}")
    print(f"  budget: {1e3/RATES[1]:.0f} ms @ {RATES[1]} Hz, "
          f"{1e3/RATES[0]:.0f} ms @ {RATES[0]} Hz")

    # ---- weight memory ----
    print("\n=== weight memory (bit-packed) ===")
    bits = {"binary": 1, "int8": 8, "fixed": 16}
    memtot = 0
    for p in ("binary", "int8", "fixed"):
        pr = sum(r['params'] for r in rows if r['prec'] == p)
        B = pr * bits[p] / 8
        memtot += B
        print(f"  {p:<8} {pr:>8,} params x {bits[p]:>2}b = {B/1024:7.2f} KB")
    print(f"  {'TOTAL':<8} {'':>8}          {memtot/1024:7.2f} KB")


if __name__ == "__main__":
    main()
