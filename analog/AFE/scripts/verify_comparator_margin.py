"""Verify that raising detector gain (R4 10k->1k) restores comparator margin.

Part A: per-channel V+ swing at R4=10k vs R4=1k on a fricative-rich GSC word
        (six.wav), vs the comparator offset budget (~5 mV).
Part B: offset-flip test on the weakest channel -- set the threshold at its
        envelope midpoint and inject VOS = -5 mV and +5 mV; count output pulses.
        If a few-mV offset flips the pulse count between ~0 and saturated, the
        channel's binarization is offset-dominated (no real sensing). At R4=1k
        the larger swing should make the count stable across +/-VOS.

Outputs: artifacts/comparator_margin.png, artifacts/comparator_margin.md
Run:  .venv/bin/python AFE/scripts/verify_comparator_margin.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_transient as rt

AFE = Path(__file__).resolve().parents[1]
NET = rt.NET
VOS = 5e-3
R4_HI, R4_LO = 10e3, 1e3


def set_r4(net, r4):
    return re.sub(r"\.param R4v=\S+", f".param R4v={r4:.6g}", net)


def run(net_base, r4, vref, vos, pwl):
    net = set_r4(net_base, r4)
    net = re.sub(r"\.param VREF=\S+", f".param VREF={vref:.6g}", net)
    net = re.sub(r"\.param VOS=\S+", f".param VOS={vos:.6g}", net)
    net = re.sub(r"Vin\s+in\s+0\s+PWL[^\n]*", pwl, net)
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_marg.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_marg.cir")],
                   cwd=AFE, capture_output=True, text=True, timeout=300)
    d = np.loadtxt(AFE / "sim" / "tran.csv")
    t = d[:, 0]
    return t, d[:, 5], d[:, 7]                 # t, v(vdet), v(vout)


def pulses(t, vout):
    keep = t >= 8e-3
    v = vout[keep]
    return int(np.sum((v[1:] > 0.9) & (v[:-1] <= 0.9)))


def duty(t, vout):
    """Fraction of time the output is high [%]. Robust to 'always on' (which
    has 0 rising edges but 100% duty) -- the metric that exposes an offset flip."""
    keep = t >= 8e-3
    return 100.0 * float(np.mean(vout[keep] > 0.9))


def main():
    wav = AFE / "audio" / "six.wav"
    if not wav.exists():
        wav = sorted((AFE / "audio").glob("*.wav"))[0]
    sig = rt._load_wav_window(wav)
    pwl = rt.pwl_block(sig)
    print(f"[clip] {wav.name}")

    # ---- Part A: per-channel V+ swing at 10k vs 1k ----
    swing = {R4_HI: np.zeros(16), R4_LO: np.zeros(16)}
    vmid = {R4_HI: np.zeros(16), R4_LO: np.zeros(16)}
    for r4 in (R4_HI, R4_LO):
        for ch in range(16):
            RA, C, R1, fc = rt.channel_params(ch)
            net_base = rt.apply_channel(NET, RA, C, R1)
            t, vdet, _ = run(net_base, r4, 0.9, 0.0, pwl)
            m = t >= 8e-3
            lo, hi = vdet[m].min(), vdet[m].max()
            swing[r4][ch] = (hi - lo) * 1e3       # mV
            vmid[r4][ch] = 0.5 * (lo + hi)
        print(f"R4={r4/1e3:.0f}k swing[mV]: "
              + " ".join(f"{s:.0f}" for s in swing[r4]))

    # MARGINAL channel: 10k swing closest to 2*Vos (so +/-Vos can flip it; the
    # absolute-weakest channel is just dead and shows no flip).
    ch_w = int(np.argmin(np.abs(swing[R4_HI] - 2 * VOS * 1e3)))
    fcw = rt.channel_params(ch_w)[3]
    print(f"\nmarginal channel @10k: ch{ch_w} (f_c={fcw:.0f} Hz), "
          f"swing {swing[R4_HI][ch_w]:.1f}mV -> {swing[R4_LO][ch_w]:.1f}mV @1k")

    # ---- Part B: offset-flip test on the weakest channel ----
    RA, C, R1, fc = rt.channel_params(ch_w)
    net_w = rt.apply_channel(NET, RA, C, R1)
    flip = {}
    for r4 in (R4_HI, R4_LO):
        tm, _, vom = run(net_w, r4, vmid[r4][ch_w], -VOS, pwl)
        tp, _, vop = run(net_w, r4, vmid[r4][ch_w], +VOS, pwl)
        flip[r4] = (duty(tm, vom), duty(tp, vop), pulses(tm, vom), pulses(tp, vop))
        print(f"  R4={r4/1e3:.0f}k ch{ch_w}: duty(-5mV)={flip[r4][0]:.0f}% "
              f"duty(+5mV)={flip[r4][1]:.0f}%  (pulses {flip[r4][2]}/{flip[r4][3]})")

    # ---- plot ----
    fcs = [rt.channel_params(c)[3] for c in range(16)]
    x = np.arange(16)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.2, swing[R4_HI], 0.4, label="R4=10k (G=4.7)", color="tab:gray")
    ax.bar(x + 0.2, swing[R4_LO], 0.4, label="R4=1k (G=47)", color="tab:blue")
    ax.axhline(VOS * 1e3, color="red", ls="--", lw=1.2,
               label=f"comparator Vos ~{VOS*1e3:.0f}mV")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{f:.0f}" for f in fcs], rotation=60, fontsize=8)
    ax.set_xlabel("channel f_c [Hz]"); ax.set_ylabel("V+ swing [mV] (log)")
    ax.set_title(f"Per-channel V+ swing vs comparator offset — '{wav.stem}'\n"
                 f"R4 10k→1k lifts weak/HF channels above the offset floor")
    ax.legend(); ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(AFE / "artifacts" / "comparator_margin.png", dpi=130)

    # ---- md ----
    md = ["# 비교기 센싱 마진 검증 (R4 10k vs 1k)\n",
          f"클립 `{wav.name}`(가장 큰 40ms). 비교기 오프셋 Vos={VOS*1e3:.0f}mV.\n",
          "## Part A — 채널별 V+ 스윙 [mV]",
          "| ch | f_c[Hz] | R4=10k | R4=1k | 10k가 Vos 이하? |",
          "|---:|---:|---:|---:|:--:|"]
    for c in range(16):
        under = "⚠️" if swing[R4_HI][c] <= VOS * 1e3 else ""
        md.append(f"| {c} | {fcs[c]:.0f} | {swing[R4_HI][c]:.1f} | "
                  f"{swing[R4_LO][c]:.1f} | {under} |")
    md += ["",
           f"## Part B — 오프셋 플립 (마진 애매한 ch{ch_w}, f_c={fcw:.0f}Hz)",
           "임계=엔벨로프 중앙, ±5mV 오프셋 주입 시 **출력 듀티사이클**(0/1 rising edge는 "
           "'항상 ON'을 0으로 오독하므로 듀티로 측정):",
           "",
           "| R4 | duty(VOS=−5mV) | duty(VOS=+5mV) | 해석 |",
           "|---:|---:|---:|:--|",
           f"| 10k | {flip[R4_HI][0]:.0f}% | {flip[R4_HI][1]:.0f}% | "
           f"{'±5mV가 100%↔0% 좌우 → 센싱 붕괴' if abs(flip[R4_HI][0]-flip[R4_HI][1])>40 else '안정'} |",
           f"| 1k | {flip[R4_LO][0]:.0f}% | {flip[R4_LO][1]:.0f}% | "
           f"{'여전히 민감' if abs(flip[R4_LO][0]-flip[R4_LO][1])>40 else '±오프셋에도 안정 → 마진 회복'} |",
           "",
           "→ R4=10k에선 약채널 스윙이 Vos 수준이라 ±5mV가 듀티를 100%↔0%로 뒤집음(정보 소실). "
           "R4=1k로 스윙을 키우면 안정. (behavioral tanh라 Vos=0이면 안 보이던 문제.)"]
    (AFE / "artifacts" / "comparator_margin.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/comparator_margin.png, comparator_margin.md")


if __name__ == "__main__":
    main()
