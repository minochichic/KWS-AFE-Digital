"""Pre-synthesis FPGA resource estimate for the folded BinaryMatchboxNet.

Weight + ACTIVATION memory are measured from the real model (hooks). LUT/FF/DSP
are ANALYTICAL estimates from the folding factors times per-unit costs; those
per-unit costs are ASSUMPTIONS listed in COST below and must be confirmed by
synthesis. Device capacities are ballpark and flagged as needing datasheet check.

Run from repo root: .venv/bin/python troubleshooting/scripts/estimate_resources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.binary_matchboxnet import BinaryMatchboxNet          # noqa: E402
from train.config import load_config                             # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# --- ASSUMPTIONS (per-unit logic cost; confirm by synthesis) ---
COST = {
    "bin_lane_lut": 3.0,     # XNOR+popcount compressor tree, LUT6 per input bit
    "bin_acc_lut": 24,       # per binary PE: accumulator + threshold compare
    "bin_acc_ff": 24,
    "acc_lane_lut": 22,      # int8 conditional-negate + add (conv1, no multiplier)
    "acc_lane_ff": 20,
    "dsp_per_mult": 1,       # one int8xint8 MAC per DSP slice
    "ctrl_lut": 1200,        # FSM / address generation / buffers for a folded design
    "ctrl_ff": 900,
}

DESIGNS = [("minimal", 8, 1, 1), ("balanced", 64, 8, 4), ("fast", 256, 32, 16)]

# ballpark device capacities -- CONFIRM against datasheets
DEVICES = [
    ("Lattice iCE40 UP5K", 5280, 5280, 8, 1_000 + 120),      # LUT4, FF, DSP, kbit(SPRAM+EBR)
    ("Xilinx Artix-7 35T", 20800, 41600, 90, 1800),          # LUT6, FF, DSP48, kbit BRAM
]


def measure_memory(cfg):
    """Weight bits per precision + peak activation bits, from real tensor shapes."""
    model = BinaryMatchboxNet(cfg.model).eval()
    acts, wbits = [], {"binary": 0, "int8": 0, "fixed": 0}

    def prec_of(m):
        n = type(m).__name__
        return "binary" if "Binary" in n else ("int8" if "Quant" in n else "fixed")

    def hook(mod, inp, out):
        p = prec_of(mod)
        bits = {"binary": 1, "int8": 8, "fixed": 16}[p]
        wbits[p] += sum(q.numel() for q in mod.parameters()) * bits
        # activation stored between layers: binary layers emit 1-bit, others int8
        abits = 1 if p == "binary" else 8
        acts.append((mod.out_channels * out.shape[-1] * abits, p))

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model(torch.sign(torch.randn(1, cfg.model.in_channels, cfg.model.T)))
    for h in hs:
        h.remove()
    return wbits, acts


def main():
    cfg = load_config(str(REPO / "configs" / "base.yaml"))
    wbits, acts = measure_memory(cfg)

    wtot = sum(wbits.values())
    print("=== memory (measured from the model) ===")
    for p, b in wbits.items():
        print(f"  weights {p:<7} {b/8/1024:7.2f} KB")
    print(f"  weights TOTAL   {wtot/8/1024:7.2f} KB")

    peak = max(a for a, _ in acts)
    two_largest = sorted((a for a, _ in acts), reverse=True)[:2]
    act_buf = sum(two_largest)          # ping-pong between consecutive layers
    print(f"  activations peak layer {peak/8/1024:.2f} KB; "
          f"ping-pong buffer {act_buf/8/1024:.2f} KB")
    total_kbit = (wtot + act_buf) / 1000
    print(f"  => on-chip memory needed ~ {(wtot+act_buf)/8/1024:.1f} KB "
          f"({total_kbit:.0f} kbit)")

    print("\n=== logic estimate (ANALYTICAL -- confirm by synthesis) ===")
    print(f"{'design':<10}{'P_bin':>6}{'P_acc':>6}{'P_dsp':>6}"
          f"{'LUT':>9}{'FF':>8}{'DSP':>6}")
    est = {}
    for name, pb, pa, pd in DESIGNS:
        lut = (pb * COST["bin_lane_lut"] + COST["bin_acc_lut"]
               + pa * COST["acc_lane_lut"] + COST["ctrl_lut"])
        ff = COST["bin_acc_ff"] + pa * COST["acc_lane_ff"] + COST["ctrl_ff"]
        dsp = pd * COST["dsp_per_mult"]
        est[name] = (lut, ff, dsp)
        print(f"{name:<10}{pb:>6}{pa:>6}{pd:>6}{lut:>9,.0f}{ff:>8,.0f}{dsp:>6}")

    print("\n=== fit check (device capacities are BALLPARK -- confirm) ===")
    for dev, LUT, FF, DSP, kbit in DEVICES:
        print(f"  {dev} ({LUT:,} LUT / {DSP} DSP / {kbit:,} kbit)")
        for name, _, _, _ in DESIGNS:
            l, f, d = est[name]
            ok = (l <= LUT) and (f <= FF) and (d <= DSP) and (total_kbit <= kbit)
            why = []
            if l > LUT: why.append("LUT")
            if d > DSP: why.append("DSP")
            if total_kbit > kbit: why.append("MEM")
            print(f"      {name:<9} {'FIT' if ok else 'NO ':<4}"
                  f" LUT {100*l/LUT:5.1f}%  DSP {100*d/DSP:5.1f}%  "
                  f"MEM {100*total_kbit/kbit:5.1f}%"
                  + (f"   <- {','.join(why)}" if why else ""))

    print("\n=== what dominates ===")
    print(f"  memory: int8 weights {wbits['int8']/8/1024:.1f} KB = "
          f"{100*wbits['int8']/wtot:.0f}% of weights "
          f"-> 4-bit quantization of conv1/conv3 saves "
          f"{wbits['int8']/2/8/1024:.1f} KB")
    print("  logic : control/addressing dominates at small P (assumption ctrl_lut="
          f"{COST['ctrl_lut']}) -> a bigger P is nearly free until LUTs run out")


if __name__ == "__main__":
    main()
