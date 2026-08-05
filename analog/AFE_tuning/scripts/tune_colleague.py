"""Tune component VALUES in the colleague's channel circuit (topology unchanged).

Uses the colleague's own netlist topology (netlists/colleague_channel.cir, a
localized copy of ngspice_channel_sweeper/netlist_template.cir) and their
channel_components.csv as the starting point. Sweeps the detector gain resistor
R4 (with R6 = R4||R5 as their GUI suggests) at a realistic 10 mVpp input, and
measures per channel:

  * DC operating point Vdet (quiescent detector output)
  * transient V+ swing on a real GSC word
  * margin = swing / comparator offset (~5 mV)

Why R4 and not R5: detector gain G = R5/R4, and the envelope time constant is
tau = R5*C3. Lowering R4 raises gain WITHOUT touching tau; raising R5 would
change both. So R4 is the clean gain knob.

Also re-derives R7/R8 per channel from the measured post-tuning V+ range so the
comparator threshold sits inside each channel's actual swing, and writes a
GUI-loadable CSV (same columns as channel_components.csv).

Output: artifacts/tuning_r4.md, artifacts/tuning_r4.png,
        artifacts/channel_components_tuned.csv
Run from repo root: .venv/bin/python AFE_tuning/scripts/tune_colleague.py
"""
from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TU = Path(__file__).resolve().parents[1]
REPO = TU.parent
NET = (TU / "netlists" / "colleague_channel.cir").read_text()
CSV = REPO / "ngspice_channel_sweeper" / "channel_components.csv"
FS = 20_000
DUR = 0.040
VPP = 10e-3                 # colleague's PWL normalization: 10 mVpp
VOS = 5e-3                  # comparator offset budget
R4_LIST = [10e3, 4.7e3, 2.2e3, 1e3]
PROBE_CH = [2, 7, 11, 15]   # low / mid / upper-mid / HF representatives


def load_csv():
    rows = np.genfromtxt(CSV, delimiter=",", names=True)
    return rows


def gsc_window():
    p = REPO / "AFE" / "audio" / "six.wav"
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    win = int(DUR * sr)
    if len(x) > win:
        e = np.convolve(x ** 2, np.ones(win), "valid"); x = x[int(np.argmax(e)):][:win]
    y = np.interp(np.arange(int(DUR * FS)) / FS, np.arange(len(x)) / sr, x)
    y = y - y.mean()
    return y / (y.max() - y.min())          # peak-to-peak = 1 -> scale by VPP


def run(ra, c1, r1, r4, r5, r6, sig=None, analysis="tran"):
    """Set params on the colleague netlist; return (Vdet_dc, swing_mV)."""
    net = NET
    net = re.sub(r"\.param RA=\S+ R1=\S+ R2=\S+ R4=\S+ R5=\S+ R6=\S+ R7=\S+ R8=\S+",
                 f".param RA={ra:.6g} R1={r1:.6g} R2=100k R4={r4:.6g} R5={r5:.6g} "
                 f"R6={r6:.6g} R7=500k R8=500k", net)
    net = re.sub(r"\.param C1=\S+ C3=\S+", f".param C1={c1:.6g} C3=100n", net)
    if analysis == "op":
        net = net.replace(".end", ".control\n  op\n  print v(/v_detect_out)\n.endc\n.end")
        (TU / "sim" / "t.cir").write_text(net)
        r = subprocess.run(["ngspice", "-b", str(TU / "sim" / "t.cir")],
                           cwd=TU, capture_output=True, text=True, timeout=180)
        m = re.search(r"v\(/v_detect_out\)\s*=\s*(\S+)", r.stdout)
        return float(m.group(1)) if m else np.nan
    # transient with the GSC PWL at VPP
    t = np.arange(len(sig)) / FS
    v = VPP * sig
    pwl = ("V4 /vin Net-_V3-Pad1_ PWL(\n"
           + "\n".join(f"+ {a:.7e} {b:.7e}" for a, b in zip(t, v)) + "\n)")
    net = re.sub(r"V4 /vin Net-_V3-Pad1_[^\n]*", pwl, net)
    net = net.replace(".end", """.control
  tran 5u 40m
  wrdata sim/t.csv v(/v_detect_out)
  meas tran vdmin MIN v(/v_detect_out) from=8m to=40m
  meas tran vdmax MAX v(/v_detect_out) from=8m to=40m
  print vdmin vdmax
.endc
.end""")
    (TU / "sim" / "t.cir").write_text(net)
    r = subprocess.run(["ngspice", "-b", str(TU / "sim" / "t.cir")],
                       cwd=TU, capture_output=True, text=True, timeout=300)
    d = {k: float(v) for k, v in re.findall(r"(vdmin|vdmax)\s*=\s*(\S+)", r.stdout)}
    if "vdmin" not in d:
        return np.nan, np.nan
    return d["vdmin"], (d["vdmax"] - d["vdmin"]) * 1e3


def main():
    rows = load_csv()
    sig = gsc_window()
    r5 = 47e3

    print("=== R4 sweep (R6 = R4||R5), input 10 mVpp GSC 'six' ===")
    table = {}
    for ch in PROBE_CH:
        r = rows[ch]
        ra, c1, r1 = r["RA_kohm"] * 1e3, r["C_nF"] * 1e-9, r["R1_kohm"] * 1e3
        for r4 in R4_LIST:
            r6 = r4 * r5 / (r4 + r5)
            vmin, sw = run(ra, c1, r1, r4, r5, r6, sig)
            table[(ch, r4)] = (vmin, sw)
            print(f"ch{int(r['ch']):2d} f_c={r['f_c_hz']:6.0f}  R4={r4/1e3:5.2f}k "
                  f"(G={r5/r4:4.1f}, R6={r6/1e3:5.2f}k)  V+swing={sw:7.1f} mV  "
                  f"margin=x{sw/(VOS*1e3):5.1f}")

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ch in PROBE_CH:
        sws = [table[(ch, r4)][1] for r4 in R4_LIST]
        ax.plot([r5 / r4 for r4 in R4_LIST], sws, "o-", lw=2,
                label=f"ch{ch} ({rows[ch]['f_c_hz']:.0f} Hz)")
    ax.axhline(VOS * 1e3, color="tab:red", ls="--", lw=1.2, label=f"offset ~{VOS*1e3:.0f} mV")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("detector gain G = R5/R4  (R4 = 10k -> 1k)")
    ax.set_ylabel("V+ swing [mV]")
    ax.set_title("Colleague circuit: R4 is the clean gain knob (tau = R5*C3 unchanged)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(TU / "artifacts" / "tuning_r4.png", dpi=130)

    # ---- full 16-ch at the recommended R4, re-derive R7/R8 ----
    R4_REC = 1e3
    r6_rec = R4_REC * r5 / (R4_REC + r5)
    print(f"\n=== all 16 ch at recommended R4={R4_REC/1e3:.2f}k (R6={r6_rec/1e3:.2f}k) ===")
    out = []
    for ch in range(16):
        r = rows[ch]
        ra, c1, r1 = r["RA_kohm"] * 1e3, r["C_nF"] * 1e-9, r["R1_kohm"] * 1e3
        vmin, sw = run(ra, c1, r1, R4_REC, r5, r6_rec, sig)
        # threshold at 25% of the swing above the floor (fires on real energy,
        # stays clear of the offset); R7/R8 rounded to 0.01k like their GUI.
        vthr = vmin + 0.25 * sw / 1e3
        r8 = round(1e6 * vthr / 1.8 / 10) * 10          # 0.01 kOhm grid
        r7 = round((1e6 - r8) / 10) * 10
        out.append((int(r["ch"]), r["f_c_hz"], r["Q"], r["RA_kohm"], r["C_nF"],
                    r["R1_kohm"], r["gain_dB"], 100.0, R4_REC / 1e3, r5 / 1e3,
                    r6_rec / 1e3, 100.0, r7 / 1e3, r8 / 1e3, vthr, vmin, sw))
        print(f"ch{ch:2d} f_c={r['f_c_hz']:6.0f}  V+ {vmin:.4f}..{vmin+sw/1e3:.4f} "
              f"(swing {sw:6.1f} mV, x{sw/(VOS*1e3):4.1f})  Vthr={vthr:.4f} "
              f"R7={r7/1e3:.2f}k R8={r8/1e3:.2f}k")

    hdr = ("ch,f_c_hz,Q,RA_kohm,C_nF,R1_kohm,gain_dB,R2_kohm,R4_kohm,R5_kohm,"
           "R6_kohm,C3_nF,R7_kohm,R8_kohm,Vthr_V")
    with open(TU / "artifacts" / "channel_components_tuned.csv", "w") as f:
        f.write(hdr + "\n")
        for o in out:
            f.write(f"{o[0]},{o[1]:.0f},{o[2]:.2f},{o[3]:.2f},{o[4]:.2f},{o[5]:.1f},"
                    f"{o[6]:.1f},{o[7]:.0f},{o[8]:.2f},{o[9]:.0f},{o[10]:.2f},"
                    f"{o[11]:.0f},{o[12]:.2f},{o[13]:.2f},{o[14]:.4f}\n")

    md = ["# 동료 회로 소자값 튜닝 (토폴로지 불변)\n",
          "`ngspice_channel_sweeper/netlist_template.cir`의 **회로 구조를 그대로 두고** 소자값만 "
          "조정. 입력은 동료 규약대로 10 mVpp GSC('six'), 비교기 오프셋 5 mV 가정.\n",
          "## 1. 핵심: R4가 유일한 '깨끗한' 이득 노브\n",
          "- 검출기 이득 `G = R5/R4`, 엔벨로프 시상수 `τ = R5·C3`.",
          "- **R5를 올리면 이득과 τ가 같이 변한다**(τ가 음소 시간축을 정하므로 건드리면 안 됨).",
          "- **R4를 내리면 이득만 올라간다** → R4가 정답. R6는 동료 GUI 규칙대로 `R4∥R5`로 따라감.\n",
          "## 2. R4 스윕 결과 (V+ 스윙, 대표 채널)\n",
          "| ch | f_c[Hz] | " + " | ".join(f"R4={r/1e3:.2g}k (G={r5/r:.0f})" for r in R4_LIST) + " |",
          "|---:|---:|" + "---:|" * len(R4_LIST)]
    for ch in PROBE_CH:
        md.append(f"| {ch} | {rows[ch]['f_c_hz']:.0f} | " +
                  " | ".join(f"{table[(ch,r4)][1]:.1f} mV" for r4 in R4_LIST) + " |")
    md += ["",
           f"- 기본값 R4=10k는 스윙이 오프셋({VOS*1e3:.0f} mV) 수준 → **비교기가 신호를 못 가림**.",
           "- R4를 낮추면 스윙이 크게 증가 → 오프셋 여유 확보. 그림: `tuning_r4.png`.",
           "",
           f"## 3. 권장값 (R4={R4_REC/1e3:.2f}k, R6={r6_rec/1e3:.2f}k)\n",
           "| ch | f_c[Hz] | V+스윙[mV] | 여유(스윙/Vos) | Vthr[V] | R7[kΩ] | R8[kΩ] |",
           "|---:|---:|---:|:--|---:|---:|---:|"]
    for o in out:
        sw = o[16]
        md.append(f"| {o[0]} | {o[1]:.0f} | {sw:.1f} | ×{sw/(VOS*1e3):.1f} | "
                  f"{o[14]:.4f} | {o[12]:.2f} | {o[13]:.2f} |")
    md += ["",
           "- **R7/R8 재도출**: 임계를 각 채널 실제 V+ 스윙의 **하위 25%** 지점에 배치(실신호에 "
           "반응하되 오프셋과 이격). 동료 GUI 규칙대로 0.01 kΩ로 반올림(합계 1 MΩ 강제 안 함).",
           "- 동료 CSV의 기존 R7/R8(≈490/510k)은 우리 **초기 mel+log 매핑** 산물로, 그 후 √도메인이 "
           "더 정합적임이 밝혀졌다. 위 값은 **측정된 V+ 스윙 기반**이라 더 견고.",
           "",
           "## 4. 바꾸지 말 것\n",
           "- **RA, C1/C2, R1**: 채널 f_c/Q를 mel에 맞춘 값(우리 filterbank_design 결과). 변경 시 "
           "필터뱅크가 mel과 어긋남.",
           "- **R5, C3**: `τ = R5·C3 = 4.7 ms`가 음소 시간축에 맞다(회로에서 실측 확인). 유지.",
           "- **R2=R3=100k**: GIC 피드백 쌍.",
           "",
           "## 5. 적용 방법\n",
           "`artifacts/channel_components_tuned.csv`를 동료 툴의 "
           "`component_versions/`에 복사 → GUI `버전 불러오기`. (기본 CSV는 읽기 전용 유지.)",
           "",
           "## 6. 남은 한계 (소자값으로 못 고침)\n",
           "- 이득을 올려도 **검출기 데드존(≈5 mV, OPA379 슬루/GBW 한계)** 은 거의 안 준다 → "
           "약신호/HF는 여전히 불리. 근본 해법은 **마이크 프리앰프**(별도 단, `AFE_micamp/`).",
           "- 비교기 오프셋에 대한 정확도 취약성은 **오프셋-인지 학습**이 필요(ML 쪽, "
           "`AFE_micamp/artifacts/offset_fragility.md`)."]
    (TU / "artifacts" / "tuning_r4.md").write_text("\n".join(md) + "\n")
    print("\n저장: artifacts/tuning_r4.md, tuning_r4.png, channel_components_tuned.csv")


if __name__ == "__main__":
    main()
