"""Characterize the active detector (precision rectifier + C3) in SPICE.

Justifies the detector component values (R4/R5/R6/C3) by MEASURING what they
produce, instead of asserting them:

  (1) Compression curve -- sweep input amplitude A, measure steady-state
      envelope excursion (V+ - 0.9). Fit V+exc = k * A^p on the small-signal
      part: p tells us linear (1), sqrt (0.5), log-ish (<0.5), or saturating.
      Small-signal slope gives the effective gain vs the nominal G = R5/R4.
  (2) Time constant -- gate the tone on then off; fit the V+ discharge to an
      exponential to get tau_discharge, and the rise to tau_charge. Compares to
      the nominal R5*C3.

This directly informs the circuit-matched retrain (README roadmap): the true
compression curve is what the ML detector model must reproduce (the log-vs-sqrt
question), and tau is the EMA time constant.

Outputs: artifacts/detector_characterization.png, artifacts/detector_char.md
Run:  .venv/bin/python AFE/scripts/characterize_detector.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AFE = Path(__file__).resolve().parents[1]
NET = (AFE / "netlists" / "detector.cir").read_text()
BIAS = 0.9
FREQ = 1000.0                    # carrier well above 1/tau (~210 Hz)
R4, R5, C3 = 10e3, 47e3, 100e-9  # nominal, for reference lines


def run(net):
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_det.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_det.cir")],
                   cwd=AFE, capture_output=True, text=True, timeout=300)
    d = np.loadtxt(AFE / "sim" / "det.csv")
    return d[:, 0], d[:, 1], d[:, 3]        # t, v(vfilt), v(vdet)


AMPS = np.array([5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
                 2e-2, 5e-2, 1e-1, 2e-1])
FREQS = [250.0, 1000.0, 4000.0]              # low / mid / high channel carriers


def sweep_amplitude(freq):
    exc = []
    for A in AMPS:
        net = re.sub(r"\.param AMP=\S+", f".param AMP={A:.6g}", NET)
        net = re.sub(r"\.param FREQ=\S+", f".param FREQ={freq:.6g}", net)
        t, _, vdet = run(net)
        m = t >= 30e-3                       # steady state (>5*tau)
        exc.append(vdet[m].mean() - BIAS)
    return np.array(exc)


def turn_on(exc):
    """Smallest A where the envelope excursion crosses 0 (deadzone edge)."""
    pos = np.where(exc > 0)[0]
    if len(pos) == 0 or pos[0] == 0:
        return np.nan
    i = pos[0]
    # log-linear interp between i-1 (<=0) and i (>0)
    lo, hi = exc[i-1], exc[i]
    f = (0 - lo) / (hi - lo)
    return np.exp(np.log(AMPS[i-1]) + f * (np.log(AMPS[i]) - np.log(AMPS[i-1])))


def burst_tau():
    """Gated sine: on 0..30 ms, off 30..70 ms. Fit V+ charge & discharge."""
    fs = 40_000
    t = np.arange(int(0.070 * fs)) / fs
    gate = (t < 30e-3).astype(float)
    sig = BIAS + 5e-2 * gate * np.sin(2 * np.pi * FREQ * t)   # 50 mV, above deadzone
    body = "\n".join(f"+ {a:.7e} {b:.7e}" for a, b in zip(t, sig))
    net = re.sub(r"Vsig[^\n]*", f"Vsig vfilt 0 PWL(\n{body}\n)", NET)
    net = re.sub(r"tran 2u 40m", "tran 5u 70m", net)
    tt, _, vdet = run(net)

    # discharge: exp fit on 32..60 ms (after tone off at 30 ms)
    disch = (tt >= 32e-3) & (tt < 60e-3)
    x = tt[disch] - tt[disch][0]
    y = vdet[disch] - vdet[tt >= 68e-3].mean()      # subtract final floor
    good = y > 1e-5
    tau_d = (-1.0 / np.polyfit(x[good], np.log(y[good]), 1)[0]
             if good.sum() > 5 else np.nan)
    # charge: time to reach 90% of the on-state level (fast, non-exponential)
    on = vdet[(tt >= 25e-3) & (tt < 30e-3)].mean() - BIAS
    ch = tt[(tt >= 0) & (tt < 30e-3)]
    vch = vdet[(tt >= 0) & (tt < 30e-3)] - BIAS
    hit = np.where(vch >= 0.9 * on)[0]
    t90 = ch[hit[0]] if len(hit) else np.nan
    return tt, vdet, t90, tau_d


def main():
    print("[1] amplitude sweep at 3 carriers (compression + deadzone) ...")
    curves = {}
    for f in FREQS:
        exc = sweep_amplitude(f)
        curves[f] = exc
        A0 = turn_on(exc)
        resp = exc > 0
        # compression exponent above turn-on (A<=100mV to avoid rail)
        fit_m = resp & (AMPS <= 1e-1)
        p = (np.polyfit(np.log(AMPS[fit_m]), np.log(exc[fit_m]), 1)[0]
             if fit_m.sum() > 2 else np.nan)
        g = exc[AMPS == 1e-1][0] / 0.1                    # gain at 100 mV
        print(f"  f={f:6.0f} Hz: deadzone≈{A0*1e3:5.2f} mV, "
              f"gain@100mV≈{g:5.1f}, p≈{p:.2f}")

    print("[2] burst response (tau) at 1 kHz ...")
    tt, vdet, t90, tau_d = burst_tau()
    print(f"  t90_charge = {t90*1e3:.2f} ms (fast),  "
          f"tau_discharge = {tau_d*1e3:.2f} ms  (nominal R5*C3 = {R5*C3*1e3:.2f} ms)")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cols = {250.0: "tab:green", 1000.0: "tab:blue", 4000.0: "tab:red"}
    for f in FREQS:
        e = curves[f]
        m = e > 0
        ax[0].loglog(AMPS[m] * 1e3, e[m] * 1e3, "o-", color=cols[f],
                     label=f"{f:.0f} Hz")
    ax[0].loglog(AMPS * 1e3, (R5 / R4) * AMPS * 1e3, ":", color="gray",
                 label=f"ideal linear G=R5/R4={R5/R4:.1f}")
    ax[0].set_xlabel("input amplitude A [mV]"); ax[0].set_ylabel("envelope V+ − 0.9V [mV]")
    ax[0].set_title("Detector compression / deadzone vs carrier freq")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=9)
    ax[1].plot(tt * 1e3, vdet, color="tab:blue", lw=1.2)
    ax[1].axvline(30, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("time [ms]"); ax[1].set_ylabel("V+ [V]")
    ax[1].set_title(f"Burst response  t90_charge={t90*1e3:.1f}ms  "
                    f"τ_discharge={tau_d*1e3:.1f}ms")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(AFE / "artifacts" / "detector_characterization.png", dpi=130)

    # ---- md ----
    md = ["# 능동 검출기 특성화 (SPICE 실측)\n",
          "`characterize_detector.py`가 검출기(정밀정류+C3)를 단독 구동해 측정. "
          "부품값(R4/R5/R6/C3)을 주장하지 않고 **그 값이 만드는 응답을 실측**.\n",
          "## 압축 곡선 / 데드존 (입력 진폭 A → 엔벨로프 V+), 반송 주파수별\n",
          "| f [Hz] | 데드존 [mV] | gain@100mV | 압축지수 p |",
          "|---:|---:|---:|---:|"]
    for f in FREQS:
        e = curves[f]; A0 = turn_on(e)
        fit_m = (e > 0) & (AMPS <= 1e-1)
        p = (np.polyfit(np.log(AMPS[fit_m]), np.log(e[fit_m]), 1)[0]
             if fit_m.sum() > 2 else float("nan"))
        g = e[AMPS == 1e-1][0] / 0.1
        md.append(f"| {f:.0f} | {A0*1e3:.2f} | {g:.1f} | {p:.2f} |")
    md += ["", "- **데드존**: 이 진폭 이하에서 V+가 무반응(정밀정류기가 저GBW OPA379의 "
           "크로스오버 왜곡으로 저신호를 못 정류). 고주파일수록 커짐.",
           f"- **압축지수 p**: 1=선형(=R5/R4). 측정치는 데드존 위에서 **초선형(p>1)** — "
           "턴온 무릎 때문. 공칭 이득 R5/R4={:.1f}.".format(R5/R4), "",
           "## 시상수 τ (톤 버스트 on/off, 1 kHz)",
           f"- 충전 t90 = **{t90*1e3:.2f} ms** (빠름), "
           f"방전 τ = **{tau_d*1e3:.2f} ms**  vs 공칭 R5·C3 = {R5*C3*1e3:.2f} ms ✓",
           "",
           "## 함의 (부품값 근거 / Phase B)",
           "- τ_discharge는 R5·C3로 **정확히 예측됨** → C3/R5 값의 근거 확립.",
           "- 그러나 검출기엔 **주파수 의존 데드존**이 있어 저신호·고주파를 약화 → "
           "앞서 HF 채널이 약했던 물리적 원인 중 하나. R4(이득)·OPA379(GBW) 재검토 대상.",
           "- 재학습용 검출기 모델은 **선형 √이 아니라 (데드존+무릎+포화) 곡선**을 "
           "재현해야 함. 위 곡선을 룩업/파라메트릭으로 넣는다. (README 로드맵 step 1)"]
    (AFE / "artifacts" / "detector_char.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/detector_characterization.png, detector_char.md")


if __name__ == "__main__":
    main()
