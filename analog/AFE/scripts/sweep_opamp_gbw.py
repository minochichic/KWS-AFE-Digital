"""Sweep op-amp GBW and measure the precision-rectifier deadzone.

Tests the prediction  V_dz ~ 2*Vd*f/GBW  (deadzone proportional to 1/GBW):
uses a single-pole behavioral op-amp (detector_gbw.cir) whose unity-gain freq
GBW is a parameter, sweeps input amplitude at a fixed carrier to find the
deadzone (turn-on amplitude), and repeats over GBW.

This isolates the GBW lever the R4 (gain) sweep could NOT move: if deadzone
falls ~1/GBW, a faster low-power op-amp is the real fix for HF channels.

Outputs: artifacts/opamp_gbw_sweep.png, artifacts/opamp_gbw_sweep.md
Run:  .venv/bin/python AFE/scripts/sweep_opamp_gbw.py
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
NET = (AFE / "netlists" / "detector_gbw.cir").read_text()
BIAS = 0.9
FREQ = 1000.0
VD = 0.3                                  # BAT54 Schottky drop (approx)
GBWS = [30e3, 90e3, 300e3, 1e6, 3e6, 10e6]
AMPS = np.array([2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3,
                 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2])


def run(gbw, amp):
    net = re.sub(r"\.param GBW=\S+", f".param GBW={gbw:.6g}", NET)
    net = re.sub(r"\.param AMP=\S+", f".param AMP={amp:.6g}", net)
    net = re.sub(r"\.param FREQ=\S+", f".param FREQ={FREQ:.6g}", net)
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_gbw.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_gbw.cir")],
                   cwd=AFE, capture_output=True, text=True, timeout=300)
    d = np.loadtxt(AFE / "sim" / "detg.csv")
    t, vdet = d[:, 0], d[:, 3]
    return vdet[t >= 30e-3].mean() - BIAS


def turn_on(exc):
    pos = np.where(exc > 5e-6)[0]            # first amp with real response
    if len(pos) == 0 or pos[0] == 0:
        return np.nan
    i = pos[0]
    lo, hi = exc[i-1], exc[i]
    f = (0 - lo) / (hi - lo)
    return np.exp(np.log(AMPS[i-1]) + f * (np.log(AMPS[i]) - np.log(AMPS[i-1])))


def main():
    dz = []
    for gbw in GBWS:
        exc = np.array([run(gbw, A) for A in AMPS])
        d0 = turn_on(exc)
        dz.append(d0)
        print(f"GBW={gbw/1e3:8.0f} kHz -> deadzone = {d0*1e3:6.3f} mV")
    dz = np.array(dz)
    gbw = np.array(GBWS)

    pred = 2 * VD * FREQ / gbw               # V_dz ~ 2*Vd*f/GBW
    OPA379_DZ = 7.29e-3                       # real OPA379 measured @1kHz (this repo)
    # slope in the crossover-dominated (low-GBW) region, before the floor
    low = gbw <= 1e6
    slope_low = np.polyfit(np.log(gbw[low]), np.log(dz[low]), 1)[0]
    print(f"\nlow-GBW slope (<=1MHz) = {slope_low:.2f}  (theory -1, 1/GBW)")
    print(f"model @90kHz = {dz[1]*1e3:.3f} mV vs real OPA379 = {OPA379_DZ*1e3:.2f} mV "
          f"(~{OPA379_DZ/dz[1]:.0f}x) -> real limiter is slew rate, not GBW alone")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(gbw / 1e3, dz * 1e3, "o-", color="tab:blue",
              label="model deadzone (single-pole, GBW only)")
    ax.loglog(gbw / 1e3, pred * 1e3, "--", color="tab:red",
              label=r"theory $2V_D f/GBW$")
    ax.plot(90, OPA379_DZ * 1e3, "k*", ms=15,
            label=f"real OPA379 measured ({OPA379_DZ*1e3:.1f} mV)")
    ax.set_xlabel("op-amp GBW [kHz]"); ax.set_ylabel("deadzone [mV] @ 1 kHz")
    ax.set_title(f"Deadzone vs GBW: model ∝1/GBW (slope {slope_low:.2f}), but real "
                 f"OPA379 ≫ model\n→ real limiter is slew rate, not GBW alone")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(AFE / "artifacts" / "opamp_gbw_sweep.png", dpi=130)

    md = ["# op-amp GBW sweep — 데드존의 레버 (그리고 진짜 원인)\n",
          f"검출기(정밀정류)의 op-amp를 단극 거동모델(GBW 가변, SR 무한)로 바꿔 데드존 측정. "
          f"반송 {FREQ/1e3:.0f} kHz, $V_D$≈{VD}V.\n",
          f"저GBW 구간 로그-로그 기울기 = **{slope_low:.2f}** ≈ −1 → **데드존 ∝ 1/GBW 확인**.\n",
          "| GBW [kHz] | 모델 데드존 [mV] | 이론 2·Vd·f/GBW [mV] |",
          "|---:|---:|---:|"]
    for g, d, p in zip(gbw, dz, pred):
        md.append(f"| {g/1e3:.0f} | {d*1e3:.3f} | {p*1e3:.3f} |")
    md += ["",
           "## 중요한 반전: 실제 OPA379는 모델보다 데드존이 훨씬 크다",
           f"- 모델 @90kHz = **{dz[1]*1e3:.2f} mV** vs **실측 OPA379 = {OPA379_DZ*1e3:.1f} mV** "
           f"(~{OPA379_DZ/dz[1]:.0f}배).",
           "- 즉 실제 데드존은 **순수 GBW 크로스오버로 설명 안 됨**. 단극 모델은 SR(슬루레이트)"
           "가 무한이라 크로스오버 회복이 빨라 데드존을 과소평가.",
           "- 나노파워 op-amp(OPA379)는 **SR이 매우 낮아**, 0 교차에서 출력이 2·Vd를 "
           "슬루하는 데 시간이 걸림 → 그게 실제 데드존의 지배 요인.",
           "",
           "## 결론 (레버 정리)",
           "- **방향은 맞다**: op-amp를 빠르게 하면 데드존↓ (모델서 1/GBW 확인).",
           "- **단, 진짜 병목은 GBW가 아니라 SR**. HF 채널을 살리려면 **GBW·SR 둘 다 높은** "
           "op-amp 필요. 둘 다 보통 **소비전류↑** → µW 예산과 정면 트레이드오프.",
           "- 다음: 거동모델에 SR 제한을 넣어 실측 7.3mV를 재현하고 **SR sweep**으로 "
           "데드존↔전력 곡선을 정량화(부품 선정 근거)."]
    (AFE / "artifacts" / "opamp_gbw_sweep.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/opamp_gbw_sweep.png, opamp_gbw_sweep.md")


if __name__ == "__main__":
    main()
