"""Verify DC Vmargin = Vthr - Venv_DC is positive for every channel.

Runs the .op on each channel with the real vendor models and the proposed
real-measured R7/R8 (channel_components_v_realfix.csv), using that channel's
detector op-amp. Vmargin>0 means the comparator is OFF at rest (fires only on
signal) -- the fix for the 13/16-always-ON problem.
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

TU = Path(__file__).resolve().parents[1]
REAL = TU / "real"
LIB = "../../ngspice_v15_2608001/lib"
BASE = (REAL / "op_test.cir").read_text()
CSV = TU.parent / "ngspice_v15_2608001" / "component_versions" / "channel_components_v_realfix.csv"
OPLIB = {"OPA379": None, "TLV9041D": "TLV9041D.lib", "TLV9042": "TLV9042.lib"}


def op(row):
    n = BASE
    n = re.sub(r"\.param RA=\S+ R1=\S+ R2=\S+ R4=\S+ R5=\S+ R6=\S+ R7=\S+ R8=\S+",
               f".param RA={row['RA_kohm']}k R1={row['R1_kohm']}k R2=100k R4=10k "
               f"R5=47k R6=8.25k R7={row['R7_kohm']}k R8={row['R8_kohm']}k", n)
    n = re.sub(r"\.param C1=\S+ C3=\S+", f".param C1={row['C_nF']}n C3=100n", n)
    op_name = row["detector_opamp"]
    if OPLIB[op_name]:
        n = n.replace('.include "%s/OPA379.LIB"' % LIB,
                      '.include "%s/OPA379.LIB"\n.include "%s/%s"' % (LIB, LIB, OPLIB[op_name]))
        n = re.sub(r"(XU3 .*?GND Net-_D1-K_ )OPA379", r"\1" + op_name, n)
    n = n.replace("""  op
  print v(/v_env) v(/v_filt) v(/v_thr) v(/vin)""",
                  "  op\n  print v(/v_env) v(/v_thr)")
    (REAL / "vm.cir").write_text(n)
    r = subprocess.run(["ngspice", "-b", str(REAL / "vm.cir")],
                       cwd=REAL, capture_output=True, text=True, timeout=180)
    venv = float(re.search(r"v\(/v_env\)\s*=\s*(\S+)", r.stdout).group(1))
    vthr = float(re.search(r"v\(/v_thr\)\s*=\s*(\S+)", r.stdout).group(1))
    return venv, vthr


def main():
    rows = list(csv.DictReader(CSV.open()))
    print(f"{'ch':>2} {'f_c':>5} {'opamp':>9} {'Venv_DC':>9} {'Vthr':>9} {'Vmargin':>9}")
    allpos = True
    for row in rows:
        venv, vthr = op(row)
        vm = (vthr - venv) * 1e3
        allpos &= vm > 0
        flag = "OK" if vm > 0 else "❌ 음수"
        print(f"{int(row['ch']):>2} {int(float(row['f_c_hz'])):>5} {row['detector_opamp']:>9} "
              f"{venv*1e3:>8.2f} {vthr*1e3:>8.2f} {vm:>+8.2f}  {flag}")
    print("\n결과:", "16채널 모두 Vmargin 양수 ✅" if allpos else "일부 음수 ❌")


if __name__ == "__main__":
    main()
