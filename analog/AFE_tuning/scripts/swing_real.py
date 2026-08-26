"""Stage 1 step 3: per-channel envelope swing with the REAL vendor models.

Supersedes swing_per_channel.py (which k-calibrated our behavioral models onto
the colleague's one 1 kHz point). Now that lib/ is available we measure directly.

For each channel, drive at its own f_c with 10 mVpp and measure:
  * Venv_DC : quiescent v_env with no signal (channel-INDEPENDENT, so measured
              once per op-amp)
  * swing   : (steady-state v_env max) - Venv_DC  -> how far the envelope rises
Low/mid channels use OPA379; the upper channels are measured with BOTH OPA379 and
TLV9041D (the HF winner from step 2) and the larger-swing op-amp is chosen.

Threshold: Vthr = Venv_DC + f*swing, f=0.38, floored at 3 mV; a channel whose
swing is at comparator-offset level is parked always-OFF. R8 = 1000k*Vthr/1.8,
R7 = 1000k - R8, rounded to 0.01 kOhm.

Output: artifacts/r7r8_real.md   Run from repo root.

THREE THINGS ARE ARGUMENTS NOW, and any of them moving forces a new --out.

  --design   read RA/C/R1 from a filterbank_design*.csv instead of the table
             below, so a re-solved band can be measured
  --opamp    force ONE part on every channel instead of picking the larger
             swing per channel -- this is the "can OPA379 do all sixteen"
             question, and it is a detector swap: XU3 only. The GIC filter is
             OPA379 in every case.
  --out      the artifacts filename

r7r8_real.md is the shipped table and analog/README.md marks the tuning folder
frozen, so it must not be written by an exploratory run. A margin table that
silently describes a different part or a different band is worse than no table.

  python analog/AFE_tuning/scripts/swing_real.py \
      --design analog/AFE/artifacts/filterbank_design_125_5000.csv \
      --opamp OPA379 --out r7r8_opa379_125_5000.md
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REAL = Path(__file__).resolve().parents[1] / "real"
ART = Path(__file__).resolve().parents[1] / "artifacts"
LIB = "../../ngspice_v15_2608001/lib"
BASE = (REAL / "op_test.cir").read_text()

# ch: (f_c, RA, C1, R1)
CH = {
 0:(166,"12.34k","76.56n","16.5k"), 1:(295,"12.50k","42.35n","26.1k"),
 2:(447,"12.48k","27.93n","34.1k"), 3:(631,"12.26k","20.05n","40.2k"),
 4:(832,"12.27k","15.12n","45.9k"), 5:(1072,"12.16k","11.77n","50.0k"),
 6:(1349,"12.05k","9.38n","53.3k"), 7:(1660,"12.01k","7.59n","56.0k"),
 8:(2042,"11.80k","6.23n","57.1k"), 9:(2455,"11.74k","5.15n","58.4k"),
 10:(2951,"11.57k","4.30n","58.5k"),11:(3467,"11.59k","3.61n","59.3k"),
 12:(4169,"11.24k","3.04n","57.4k"),13:(4898,"11.10k","2.57n","56.6k"),
 14:(5754,"10.89k","2.19n","55.2k"),15:(6761,"10.61k","1.86n","53.4k"),
}
UPPER = range(10, 16)      # also try TLV9041D here


def eng(x, unit):
    """SI float -> the netlist's k/n notation, e.g. 12340.0 -> '12.34k'."""
    for scale, suf in ((1e-9, "n"), (1e-6, "u"), (1e-3, "m"),
                       (1.0, ""), (1e3, "k"), (1e6, "meg")):
        if abs(x) < scale * 1000 or suf == "meg":
            return f"{x / scale:.2f}{suf}"
    return f"{x:.2f}"


def load_design(path):
    """ch -> (f_c, RA, C, R1) from a filterbank_design*.csv (SI units)."""
    import csv as _csv
    out = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            ch = int(float(row["ch"]))
            out[ch] = (int(round(float(row["fc_sim"]))),
                       eng(float(row["RA"]), "ohm"),
                       eng(float(row["C"]), "F"),
                       eng(float(row["R1"]), "ohm"))
    if len(out) != 16:
        raise SystemExit(f"{path}: 16채널이 아니라 {len(out)}채널이다")
    return out
OPAMP_LIB = {"OPA379": None, "TLV9041D": "TLV9041D.lib"}
AMP, F_TRIP, FLOOR_MV = 5e-3, 0.38, 3.0
SUP, TOT = 1.8, 1000.0


def netlist(ch, opamp, signal):
    fc, ra, c1, r1 = CH[ch]
    n = re.sub(r"\.param RA=\S+ R1=\S+", f".param RA={ra} R1={r1}", BASE)
    n = re.sub(r"\.param C1=\S+", f".param C1={c1}", n)
    if OPAMP_LIB[opamp]:
        n = n.replace('.include "%s/OPA379.LIB"' % LIB,
                      '.include "%s/OPA379.LIB"\n.include "%s/%s"' % (LIB, LIB, OPAMP_LIB[opamp]))
        n = re.sub(r"(XU3 .*?GND Net-_D1-K_ )OPA379", r"\1" + opamp, n)
    src = (f"SIN(0 {AMP:g} {fc} 0 0 0)" if signal else "DC 0")
    n = n.replace("V4 /vin Net-_V3-Pad1_ DC 0 SIN(0 10m 1k 0 0 0) AC 1",
                  f"V4 /vin Net-_V3-Pad1_ DC 0 {src}")
    if signal:
        ctrl = ("  tran 2u 45m\n  meas tran vmax MAX v(/v_env) from=28m to=45m\n"
                "  print vmax")
    else:
        ctrl = "  op\n  print v(/v_env)"
    return n.replace("""  op
  print v(/v_env) v(/v_filt) v(/v_thr) v(/vin)""", ctrl)


def sim(ch, opamp, signal):
    (REAL / "s.cir").write_text(netlist(ch, opamp, signal))
    r = subprocess.run(["ngspice", "-b", str(REAL / "s.cir")],
                       cwd=REAL, capture_output=True, text=True, timeout=600)
    key = "vmax" if signal else r"v\(/v_env\)"
    m = re.search(key + r"\s*=\s*(\S+)", r.stdout)
    if not m:
        print(r.stdout[-500:]); raise SystemExit(f"fail ch{ch} {opamp} sig={signal}")
    return float(m.group(1))


def main():
    global CH, UPPER
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", help="filterbank_design*.csv for RA/C/R1")
    ap.add_argument("--opamp", choices=sorted(OPAMP_LIB),
                    help="force this part on every channel (detector, XU3)")
    ap.add_argument("--out", default="r7r8_real.md")
    args = ap.parse_args()
    if (args.design or args.opamp) and args.out == "r7r8_real.md":
        raise SystemExit(
            "--design/--opamp change what is being measured, so --out must "
            "change too. r7r8_real.md is the shipped table (analog/README.md "
            "marks AFE_tuning frozen) and a margin table that quietly "
            "describes a different part or band is worse than none.")
    if args.design:
        CH = load_design(args.design)
        print(f"부품표: {args.design}")
    if args.opamp:
        UPPER = ()                     # no per-channel choice; one part wins
        print(f"검출기 op-amp 를 {args.opamp} 로 고정 (GIC 필터는 늘 OPA379)")

    venv_dc = {op: sim(0, op, False) for op in OPAMP_LIB}   # channel-independent
    print("Venv_DC:", {k: round(v*1e3, 1) for k, v in venv_dc.items()})

    rows = []
    for ch in CH:
        cands = ([args.opamp] if args.opamp
                 else ["OPA379"] + (["TLV9041D"] if ch in UPPER else []))
        best = None
        for op in cands:
            vmax = sim(ch, op, True)
            sw = vmax - venv_dc[op]
            if best is None or sw > best[1]:
                best = (op, sw)
        op, sw = best
        rows.append((ch, CH[ch][0], op, venv_dc[op], sw))
        print(f"ch{ch:2d} f_c={CH[ch][0]:6d}  {op:9s}  swing {sw*1e3:7.2f} mV")

    def divider(v):
        r8 = round(TOT*v/SUP, 2); return round(TOT-r8, 2), r8, SUP*r8/(TOT and (TOT-r8)+r8)
    floor = FLOOR_MV*1e-3
    md = ["# 채널별 R7/R8 — 실제 벤더 모델 실측 (Stage 1 step 3)\n",
          "`swing_per_channel.py`(k-보정 추정)를 대체한다. lib/ 실모델로 각 채널을 자기 f_c,",
          f"10 mVpp로 구동해 직접 측정. 트립 분율 f={F_TRIP}, 마진 하한 {FLOOR_MV:.0f} mV.\n",
          "HF opamp: step 2에서 TLV9041D가 승자(ch13 4.07 vs OPA379 1.16 mV). 상위 채널은",
          "OPA379/TLV9041D 둘 다 측정해 swing 큰 쪽 채택.\n",
          f"Venv_DC (무신호, 채널무관): OPA379 {venv_dc['OPA379']*1e3:.1f} mV, "
          f"TLV9041D {venv_dc['TLV9041D']*1e3:.1f} mV\n",
          "| ch | f_c[Hz] | opamp | swing[mV] | 마진 | Vthr[mV] | R7[kΩ] | R8[kΩ] | 상태 |",
          "|---:|---:|---|---:|---:|---:|---:|---:|:--|"]
    csv = []
    for ch, fc, op, dc, sw in rows:
        m_ideal = F_TRIP*sw
        if m_ideal >= floor: margin, st = m_ideal, "✅"
        elif sw > floor*1.3: margin, st = floor, f"△ 하한(f={floor/sw:.2f})"
        else: margin, st = sw*1.10, "❌ 상시OFF (프리앰프 필요)"
        vthr = dc + margin
        r8 = round(TOT*vthr/SUP, 2); r7 = round(TOT-r8, 2)
        md.append(f"| {ch} | {fc} | {op} | {sw*1e3:.2f} | +{margin*1e3:.2f} | {vthr*1e3:.2f} "
                  f"| **{r7:.2f}** | **{r8:.2f}** | {st} |")
        csv.append((ch, r7, r8, vthr, op))
    md += ["", "```csv", "ch,R7_kohm,R8_kohm,Vthr_V,detector_opamp"]
    md += [f"{c},{r7:.2f},{r8:.2f},{v:.5f},{op}" for c, r7, r8, v, op in csv]
    md += ["```", "",
           "추정판(swing_per_channel.md) 대비: 저역 swing이 실측에서 더 크고, HF는 TLV9041D로",
           "OPA379 대비 3~4배 개선되나 ch14/15는 여전히 오프셋 수준 → 프리앰프 필요."]
    ok = sum(1 for r in md if r.startswith("| ") and r.endswith("✅ |"))
    md += ["", f"이 표의 조건: 부품표 {args.design or '내장(50-8000)'}, "
               f"검출기 op-amp {args.opamp or '채널별 큰 쪽 선택'}.",
           f"마진 하한 {FLOOR_MV:.0f} mV 를 넘긴 채널 {ok}/16."]
    ART.joinpath(args.out).write_text("\n".join(md) + "\n")
    print(f"\n하한 통과 {ok}/16")
    print(f"저장: artifacts/{args.out}")


if __name__ == "__main__":
    main()
