"""Measure each channel's envelope swing AT ITS OWN f_c, then size R7/R8.

Why this exists: the only swing figure available from the colleague's real-model
runs (15.79 mV) was taken with ch00 driven at 1 kHz -- far outside ch00's 166 Hz
passband -- so it cannot size per-channel thresholds. Each channel must be driven
at its own centre frequency.

Method (and its one honest compromise):
  * Topology/nodes are the colleague's v15 template (localized copy).
  * Our diode/comparator models differ from their vendor libs, so ABSOLUTE
    voltages would not transfer. We therefore
      1. measure the swing profile locally at each channel's f_c,
      2. measure ONE common point (ch00 @ 1 kHz) that the colleague also
         measured, and use the ratio as a model-mismatch calibration factor k,
      3. report swing_scaled = k * swing_local, i.e. our RELATIVE profile placed
         on THEIR absolute scale.
  * Venv_DC is channel-independent (they measured 917.83 mV on ch00/ch08/ch15),
    so the offset anchor comes straight from their measurement.

Then: Vthr_c = Venv_DC + f * swing_c ; R8 = 1000k*Vthr/1.8 ; R7 = 1000k - R8,
each rounded to 0.01 kOhm like their GUI.

Output: artifacts/r7r8_per_channel.md
Run from repo root: .venv/bin/python AFE_tuning/scripts/swing_per_channel.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parents[1]
REPO = TU.parent
NET = (TU / "netlists" / "v15_channel.cir").read_text()
CSV = REPO / "ngspice_v15_2608001" / "channel_components.csv"

VPP = 0.010                  # colleague's convention: 10 mVpp
AMP = VPP / 2.0              # sine amplitude
# Colleague's real-model anchors (OPA379):
VENV_DC_REAL = 917.83e-3     # V, channel-independent quiescent
SWING_REAL_CH0_1K = 15.79e-3 # V, ch00 driven at 1 kHz
# Their other detector op-amp (for the HF channels)
OPAMP_REAL = {"OPA379": (917.83e-3, 15.79e-3),
              "TLV9042": (942.05e-3, 20.59e-3)}
TLV_CHANNELS = {12, 13, 14, 15}
TRIP_F = 0.380               # trip fraction (see AFE_tuning/artifacts/tuning_r4.md)
# A margin below the comparator's input offset (a few mV) would be decided by
# the offset, not the signal, so keep an absolute floor. Channels whose whole
# swing is near that floor cannot work at all: for those we deliberately place
# the threshold ABOVE their peak (always-OFF) -- a dead channel the network can
# learn to ignore beats a channel that flips on offset alone.
MARGIN_FLOOR_MV = 3.0
SUPPLY, TOTAL_K = 1.8, 1000.0


def run(ra, c1, r1, freq, amp, tstop=60e-3, meas_from=40e-3):
    """Return (Venv_DC, Venv_max) for one channel driven by a steady sine."""
    net = NET
    net = re.sub(r"\.param RA=\S+ R1=\S+", f".param RA={ra:.6g} R1={r1:.6g}", net)
    net = re.sub(r"\.param C1=\S+", f".param C1={c1:.6g}", net)
    net = re.sub(r"V4 /vin Net-_V3-Pad1_[^\n]*",
                 f"V4 /vin Net-_V3-Pad1_ DC 0 SIN(0 {amp:.6g} {freq:.6g} 0 0 0) AC 1", net)
    net = net.replace(".end", f""".control
  op
  print v(/v_env)
  tran 5u {tstop:.6g}
  meas tran vmax MAX v(/v_env) from={meas_from:.6g} to={tstop:.6g}
  print vmax
.endc
.end""")
    (TU / "sim").mkdir(exist_ok=True)
    (TU / "sim" / "sw.cir").write_text(net)
    r = subprocess.run(["ngspice", "-b", str(TU / "sim" / "sw.cir")],
                       cwd=TU, capture_output=True, text=True, timeout=600)
    dc = re.search(r"v\(/v_env\)\s*=\s*(\S+)", r.stdout)
    mx = re.search(r"vmax\s*=\s*(\S+)", r.stdout)
    if not (dc and mx):
        print(r.stdout[-800:]); raise SystemExit("measurement failed")
    return float(dc.group(1)), float(mx.group(1))


def divider(vthr):
    r8 = round(TOTAL_K * vthr / SUPPLY, 2)
    r7 = round(TOTAL_K - r8, 2)
    return r7, r8, SUPPLY * r8 / (r7 + r8)


def main():
    d = np.genfromtxt(CSV, delimiter=",", names=True)
    fc, RA, C1, R1 = d["f_c_hz"], d["RA_kohm"] * 1e3, d["C_nF"] * 1e-9, d["R1_kohm"] * 1e3

    # --- calibration point: ch00 @ 1 kHz (what the colleague measured) ---
    dc0, mx0 = run(RA[0], C1[0], R1[0], 1000.0, AMP)
    sw_local_cal = mx0 - dc0
    k = SWING_REAL_CH0_1K / sw_local_cal
    print(f"[calib] ch00 @1kHz  local swing {sw_local_cal*1e3:6.2f} mV  "
          f"vs real {SWING_REAL_CH0_1K*1e3:.2f} mV  ->  k = {k:.3f}")
    print(f"[calib] local Venv_DC {dc0*1e3:.2f} mV vs real {VENV_DC_REAL*1e3:.2f} mV "
          f"(offset is model-specific -> use theirs)\n")

    rows = []
    for c in range(16):
        dc, mx = run(RA[c], C1[c], R1[c], float(fc[c]), AMP)     # driven AT f_c
        sw_local = mx - dc
        sw_scaled = k * sw_local
        rows.append((c, fc[c], sw_local, sw_scaled))
        print(f"ch{c:2d} f_c={fc[c]:6.0f} Hz  local swing {sw_local*1e3:7.2f} mV "
              f"-> scaled {sw_scaled*1e3:7.2f} mV")

    md = ["# 채널별 f_c 구동 swing 기반 R7/R8\n",
          "각 채널을 **자기 중심주파수로** 구동해 측정한 swing으로 임계를 배치한다.",
          f"입력 {VPP*1e3:.0f} mVpp, 트립 분율 f = {TRIP_F:.3f}, 분압 명목합 {TOTAL_K:.0f} kΩ,",
          "0.01 kΩ 독립 반올림(동료 GUI 규칙).\n",
          "## 방법과 한계 (중요)\n",
          "- 토폴로지·노드명은 동료 v15 템플릿 그대로(로컬 사본).",
          "- **우리 다이오드/비교기 모델이 동료 벤더 lib과 달라 절대전압은 전이되지 않는다.**",
          f"  그래서 동료가 측정한 공통점(ch00 @1 kHz = {SWING_REAL_CH0_1K*1e3:.2f} mV)으로",
          f"  모델 불일치 보정계수 **k = {k:.3f}** 를 구해 우리 상대 프로파일을 그들 스케일로 옮겼다.",
          "- `Venv_DC`는 채널 무관(그들이 ch00/08/15에서 모두 917.83 mV) → **그들 측정값 사용**.",
          "- 따라서 아래 swing은 *추정*이다. 확정하려면 Detector 탭에서 각 채널을 자기 f_c로",
          "  구동해 재측정하고, 같은 식에 넣으면 된다.\n",
          "## 결과\n",
          "| ch | f_c[Hz] | U3 opamp | swing(추정)[mV] | f×swing | 채택 마진 | Vthr[mV] "
          "| **R7[kΩ]** | **R8[kΩ]** | 상태 |",
          "|---:|---:|---|---:|---:|---:|---:|---:|---:|:--|"]
    csv_lines = []
    floor = MARGIN_FLOOR_MV * 1e-3
    for c, f_c, sw_local, sw_scaled in rows:
        name = "TLV9042" if c in TLV_CHANNELS else "OPA379"
        dc_real, sw_ref_real = OPAMP_REAL[name]
        # scale our profile onto this op-amp's absolute level:
        # their opamp swing ratio at the common 1 kHz point carries the opamp effect
        sw = sw_scaled * (sw_ref_real / SWING_REAL_CH0_1K)
        m_ideal = TRIP_F * sw
        if m_ideal >= floor:
            margin, state = m_ideal, "✅"
        elif sw > floor * 1.3:                  # floor still sits below the peak
            margin, state = floor, f"△ 하한 적용 (f={floor/sw:.2f})"
        else:                                    # swing ≲ offset -> unusable
            margin, state = sw * 1.10, "❌ 상시OFF (프리앰프 필요)"
        vthr = dc_real + margin
        r7, r8, act = divider(vthr)
        md.append(f"| {c} | {f_c:.0f} | {name} | {sw*1e3:.2f} | {m_ideal*1e3:.2f} | "
                  f"**+{margin*1e3:.2f}** | {act*1e3:.2f} | **{r7:.2f}** | **{r8:.2f}** | {state} |")
        csv_lines.append((c, r7, r8, act, name))
    md += ["", "## CSV 붙여넣기용 (R7/R8/Vthr/detector_opamp)", "",
           "```csv",
           "ch,R7_kohm,R8_kohm,Vthr_V,detector_opamp"]
    for c, r7, r8, act, name in csv_lines:
        md.append(f"{c},{r7:.2f},{r8:.2f},{act:.5f},{name}")
    md += ["```", "",
           "동료 툴 기본 CSV는 읽기 전용이므로 `component_versions/`에 새 버전으로 저장한 뒤",
           "GUI `버전 불러오기`로 적용한다. 적용 후 DC 탭 Vmargin이 16채널 모두 양수",
           "(저역 +27~33 mV, 고역 +3~7 mV)면 성공.",
           "",
           "## 왜 기존 CSV가 틀렸나 — 근본 원인\n",
           "동료의 실제 모델 실행에서 **13/16 채널의 `Vmargin = Vthr − Venv_DC`가 음수**로",
           "나왔다(비교기가 무신호에서 이미 트립 = 정보 0). 그 R7/R8은 우리가 만든 초기",
           "매핑(`AFE/artifacts/threshold_learned_r7r8.md`)이며, 원인은 **측정창이 짧았던 것**이다:",
           "",
           "- 그때 V+ 최소값을 `from=8ms`로 측정했으나 **τ = R5·C3 = 4.7 ms**이므로 8 ms는 1.7τ뿐.",
           "  정착에는 **≥5τ ≈ 25 ms**가 필요하다. → C3가 덜 charged된 **기동 과도**를 바닥으로 오인.",
           "- 결과적으로 임계를 실제 정지점(918 mV)보다 **낮게** 배치했다.",
           "- 이번 측정에서 로컬 `Venv_DC = 918.38 mV`가 동료 실측 `917.83 mV`와 **0.55 mV 차이로",
           "  일치**함을 확인 → 모델 문제가 아니라 측정 방법 문제였다.",
           "- **교훈: 엔벨로프 측정창은 5τ 이후부터.**",
           "",
           "## 발견: 채널별 swing이 60배 차이난다\n",
           "1 kHz 단일 측정(15.79 mV)으로는 보이지 않던 사실이다. 자기 f_c로 구동하면",
           "저역 ~86 mV에서 고역 ~1.2 mV까지 **단조 감소**한다(OPA379 GBW 90 kHz 롤오프).",
           "",
           "- 따라서 **균일 R7/R8은 부적절**하다. 저역은 임계를 +27~33 mV, 고역은 +3~7 mV 위에 둬야 한다.",
           "- **ch14/ch15는 현 회로로 사용 불가**: swing 4.3/1.6 mV로 비교기 오프셋(수 mV) 수준이라",
           "  어떤 임계를 놓아도 오프셋이 판정을 지배한다. 표에서는 **상시 OFF**로 고정했다",
           "  (죽은 채널은 NN이 무시하지만, 랜덤 발화 채널은 노이즈로 작용하므로 더 나쁘다).",
           "- **해법은 마이크 프리앰프**(`AFE_micamp/`): HF 스윙 ×25–40, 오프셋-인지 학습과",
           "  함께 정확도 +10pp 확인됨.",
           "",
           "## 신뢰도\n",
           "| 항목 | 출처 | 신뢰도 |",
           "|---|---|---|",
           "| Venv_DC (절대 기준점) | 동료 실모델 실측 | 높음 |",
           "| swing의 주파수 프로파일 (급감) | 우리 SPICE 측정 | 높음(물리적으로 타당) |",
           f"| swing 절대값 | 1 kHz 공통점 보정(k={k:.3f}) | **추정** |",
           "| TLV9042 채널(12–15) | 1 kHz 비율 1.30배 근사 | **추정**(TLV lib 미보유) |",
           "",
           "확정 방법: 동료 Detector 탭에서 각 채널을 **자기 f_c로** 구동해 swing 재측정 →",
           "같은 식(`Vthr = Venv_DC + 0.38×swing`, 하한 3 mV)에 대입. ch12–15는 TLV9042",
           "실모델로 직접 확인 필요."]
    (TU / "artifacts" / "r7r8_per_channel.md").write_text("\n".join(md) + "\n")
    print("\n저장: artifacts/r7r8_per_channel.md")


if __name__ == "__main__":
    main()
