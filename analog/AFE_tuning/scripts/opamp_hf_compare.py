"""Stage 1 step 2: compare U3 detector op-amps at HIGH frequency (real models).

The colleague's VALIDATION numbers (OPA379 / TLV9041D / TLV9042) were all taken
at 1 kHz. But the whole point of picking an op-amp for the HF channels is its
behaviour AT high frequency, where GBW / slew rate decide how much envelope swing
survives. So drive the two highest channels at their OWN f_c (4898, 6761 Hz),
10 mVpp, and measure the steady-state v_env swing for each op-amp.

Uses the real vendor libs now in ngspice_v15_2608001/lib (ngbehavior=pski).
Output: printed table. Run from repo root.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parents[1]
REAL = TU / "real"
LIB = "../../ngspice_v15_2608001/lib"
BASE = (REAL / "op_test.cir").read_text()

OPAMPS = {"OPA379": "OPA379.LIB", "TLV9041D": "TLV9041D.lib", "TLV9042": "TLV9042.lib"}
CHANNELS = {13: (4898, "11.10k", "2.57n", "56.6k"),
            15: (6761, "10.61k", "1.86n", "53.4k")}
AMP = 5e-3            # 10 mVpp


def run(ch, opamp):
    fc, ra, c1, r1 = CHANNELS[ch]
    n = BASE
    # set channel filter values
    n = re.sub(r"\.param RA=\S+ R1=\S+", f".param RA={ra} R1={r1}", n)
    n = re.sub(r"\.param C1=\S+", f".param C1={c1}", n)
    # swap U3 op-amp + its library
    if opamp != "OPA379":
        n = n.replace('.include "%s/OPA379.LIB"' % LIB,
                      '.include "%s/OPA379.LIB"\n.include "%s/%s"' % (LIB, LIB, OPAMPS[opamp]))
        n = re.sub(r"(XU3 .*?GND Net-_D1-K_ )OPA379", r"\1" + opamp, n)
    # drive at own f_c, measure steady-state v_env
    n = n.replace("V4 /vin Net-_V3-Pad1_ DC 0 SIN(0 10m 1k 0 0 0) AC 1",
                  f"V4 /vin Net-_V3-Pad1_ DC 0 SIN(0 {AMP:g} {fc} 0 0 0)")
    n = n.replace("""  op
  print v(/v_env) v(/v_filt) v(/v_thr) v(/vin)""",
"""  tran 2u 45m
  meas tran vmin MIN v(/v_env) from=28m to=45m
  meas tran vmax MAX v(/v_env) from=28m to=45m
  meas tran vfmin MIN v(/v_filt) from=28m to=45m
  meas tran vfmax MAX v(/v_filt) from=28m to=45m
  print vmin vmax vfmin vfmax
.endc""")
    (REAL / "hf.cir").write_text(n)
    r = subprocess.run(["ngspice", "-b", str(REAL / "hf.cir")],
                       cwd=REAL, capture_output=True, text=True, timeout=600)
    d = {k: float(v) for k, v in re.findall(r"(vmin|vmax|vfmin|vfmax)\s*=\s*(\S+)", r.stdout)}
    if "vmax" not in d:
        print(r.stdout[-600:]); raise SystemExit(f"failed ch{ch} {opamp}")
    return d


def main():
    print(f"{'ch':>3} {'f_c':>6} {'opamp':>9} {'v_filt swing':>13} {'v_env swing':>12} "
          f"{'v_env DC~':>10}")
    for ch in CHANNELS:
        for op in OPAMPS:
            d = run(ch, op)
            vsw = (d["vmax"] - d["vmin"]) * 1e3
            fsw = (d["vfmax"] - d["vfmin"]) * 1e3
            dc = d["vmin"] * 1e3
            print(f"{ch:>3} {CHANNELS[ch][0]:>6} {op:>9} {fsw:>11.2f}mV {vsw:>10.2f}mV "
                  f"{dc:>8.1f}mV")
        print()


if __name__ == "__main__":
    main()
