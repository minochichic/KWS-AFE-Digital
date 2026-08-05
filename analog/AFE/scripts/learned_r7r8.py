"""Map the ML-learned per-channel thresholds -> comparator divider R7/R8.

The learned thresholds live in the AFE's normalized log-mel domain: envelopes
are per-clip min-max normalized with a SINGLE lo/hi shared across all 16
channels (data/afe.py `amin/amax dim=(1,2)`), so each threshold t_k in [0,1]
is a position in that GLOBAL dynamic range. To turn it into a fixed comparator
voltage we place it at the same normalized position inside the GLOBAL SPICE
envelope range [V_LO, V_HI] measured across all channels:

    V+_thr_k = V_LO + clip(t_k, 0, 1) * (V_HI - V_LO)
    R8 = RTOTAL * V+_thr_k / 1.8 ,  R7 = RTOTAL - R8

Assumptions (both stated in the README/output):
  (1) the log-mel -> V+ map is ~linear over the narrow operating band (V+ swing
      ~25 mV, so log ~= linear -- negligible error);
  (2) the ML global [lo,hi] corresponds to the SPICE global [V_LO,V_HI].
This is a first-cut for SIMULATION; a fully faithful divider would retrain the
ML with normalize="none" (absolute threshold) or add hardware AGC.

t_k outside [0,1] means the channel learned to (almost) never fire (t_k>1) or
always fire (t_k<0). ch0 here is t=1.166 -> clamped to V_HI -> effectively off.

Run from repo root:  .venv/bin/python AFE/scripts/learned_r7r8.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import run_transient as rt
from threshold_table import envelope_range

AFE = Path(__file__).resolve().parents[1]
VPOS = 1.8
RTOTAL = 1e6

# Learned thresholds from best.pt (run sc_v2_dense, epoch 94, val 0.8715),
# normalize=minmax, threshold_trainable=true. Channel order = mel low->high.
LEARNED = [1.165686, 0.753501, 0.758362, 0.373970, 0.657004, 0.827481,
           0.671781, 0.638798, 0.499882, 0.321111, 0.555065, 0.385823,
           0.520926, 0.565629, 0.668642, 0.742312]


def main():
    sigs = rt.calib_signals()                 # [(name, signal), ...]
    pwls = [(nm, rt.pwl_block(s)) for nm, s in sigs]

    # per-channel SPICE envelope swing, aggregated over ALL calibration signals
    # (vmin = min over signals, vmax = max over signals): the operating range
    # the hardware sees across the calibration set.
    swings = []
    for ch in range(16):
        RA, C, R1, fc = rt.channel_params(ch)
        net_base = rt.apply_channel(rt.NET, RA, C, R1)
        vmins, vmaxs = [], []
        for nm, pwl in pwls:
            a, b = envelope_range(net_base, pwl)
            vmins.append(a); vmaxs.append(b)
        vmin, vmax = min(vmins), max(vmaxs)
        swings.append((ch, fc, vmin, vmax))
        print(f"ch{ch:2d}  f_c={fc:6.0f}  V+ {vmin:.4f}..{vmax:.4f}")

    v_lo = min(s[2] for s in swings)          # global SPICE envelope min
    v_hi = max(s[3] for s in swings)          # global SPICE envelope max
    print(f"\nGLOBAL SPICE envelope range: V_LO={v_lo:.4f}  V_HI={v_hi:.4f}  "
          f"(span {1e3*(v_hi-v_lo):.1f} mV)")

    def divider(vthr):
        r8 = RTOTAL * vthr / VPOS
        return RTOTAL - r8, r8                          # (R7, R8)

    # Two placement philosophies:
    #  GLOBAL  -- faithful to the ML's global min-max: place t in the shared
    #             [V_LO,V_HI]. Weak (HF) channels end up above their own swing
    #             (OFF), because the ML's log+global-AGC lifts them but a linear
    #             fixed divider does not. Exposes the representation mismatch.
    #  LOCAL   -- faithful to the hardware: each channel has its OWN R7/R8, so
    #             place t within THAT channel's swing [vmin,vmax]. Every channel
    #             is live. Approximates the ML (mixes global-learned t with a
    #             per-channel range), but is the natural per-channel divider.
    rows = []
    print(f"\n{'ch':>3} {'f_c':>6} {'t':>7}  {'GLOBAL':^22}  {'LOCAL':^22}")
    for (ch, fc, vmin, vmax), t_raw in zip(swings, LEARNED):
        t = float(np.clip(t_raw, 0.0, 1.0))
        vg = v_lo + t * (v_hi - v_lo)                   # global placement
        vl = vmin + t * (vmax - vmin)                   # local placement
        r7g, r8g = divider(vg)
        r7l, r8l = divider(vl)
        state = "OFF" if vg > vmax else ("ON" if vg < vmin else "active")
        rows.append((ch, fc, t_raw, vg, r7g, r8g, state, vl, r7l, r8l))
        print(f"ch{ch:2d} {fc:6.0f} {t_raw:+.4f}  "
              f"V+{vg:.4f} R8={r8g/1e3:5.1f}k [{state:>6}]  "
              f"V+{vl:.4f} R8={r8l/1e3:5.1f}k")

    calib = ", ".join(nm for nm, _ in sigs)
    L = ["# AFE 학습 threshold 기반 비교기 분압 R7/R8\n"]
    L.append("best.pt (sc_v2_dense, epoch 94, val 0.8715)의 학습 threshold를 매핑. "
             "normalize=minmax(전역), trainable=true.")
    L.append(f"캘리브레이션: 실제 GSC {len(sigs)}단어 ({calib}).")
    L.append(f"SPICE 전역 엔벨로프 V_LO={v_lo:.4f}V, V_HI={v_hi:.4f}V "
             f"(span {1e3*(v_hi-v_lo):.1f} mV). R7+R8={RTOTAL/1e3:.0f}kΩ, "
             f"V-=1.8·R8/(R7+R8).\n")
    L.append("**두 가지 배치**: GLOBAL = ML 전역정규화 충실(약한 HF는 OFF로 → "
             "log+AGC vs 선형분압 불일치 노출). LOCAL = 하드웨어 충실(채널별 "
             "자기 스윙에 배치 → 전 채널 활성, ML 근사).\n")
    L.append("| ch | f_c [Hz] | t | V+_thr(G) | R7(G) | R8(G) | 상태(G) | V+_thr(L) | R7(L) | R8(L) |")
    L.append("|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|")
    for ch, fc, t_raw, vg, r7g, r8g, state, vl, r7l, r8l in rows:
        L.append(f"| {ch} | {fc:.0f} | {t_raw:+.4f} | {vg:.4f} | {r7g/1e3:.1f} | "
                 f"{r8g/1e3:.1f} | {state} | {vl:.4f} | {r7l/1e3:.1f} | {r8l/1e3:.1f} |")
    (AFE / "artifacts" / "threshold_learned_r7r8.md").write_text("\n".join(L) + "\n")
    print("\n저장: artifacts/threshold_learned_r7r8.md")


if __name__ == "__main__":
    main()
