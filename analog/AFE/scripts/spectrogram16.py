"""16-channel AFE spectrogram from the ACTUAL SPICE circuit.

Runs the full chain (GIC filter -> active detector -> comparator) over a whole
GSC word for every one of the 16 designed channels, collects the detector
envelope V+(t), discretizes it into 10 ms windows (max), and binarizes with the
per-channel learned threshold placed in that channel's own V+ swing on this clip.

Output: artifacts/afe_spectrogram16.png
  top   -- continuous V+ envelope [16 x time]  (what the comparator sees)
  bottom-- binary AFE image [16 x T]           (what the NN sees)

Run from repo root:  .venv/bin/python AFE/scripts/spectrogram16.py --wav six.wav
"""
from __future__ import annotations

import argparse
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_transient as rt
from learned_r7r8 import LEARNED           # 16 learned thresholds t_k in [0,1]

AFE = Path(__file__).resolve().parents[1]
FS = 16_000                                # GSC native; PWL sample rate
STEP = "12u"                               # tran internal step (resolves 8 kHz)
WIN_MS = 10.0                              # envelope discretization window


def load_full(path):
    """Full mono clip in [-1,1] at FS."""
    with wave.open(str(path), "rb") as wf:
        sr, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x = x / (np.iinfo(dt).max + 1.0)
    t_new = np.arange(int(len(x) / sr * FS)) / FS
    y = np.interp(t_new, np.arange(len(x)) / sr, x)
    return y / (np.abs(y).max() + 1e-9)


def run_channel(net_base, pwl, dur):
    net = re.sub(r"Vin\s+in\s+0\s+PWL[^\n]*", pwl, net_base)
    net = re.sub(r"tran\s+\S+\s+\S+", f"tran {STEP} {dur:.4f}", net)
    net = re.sub(r"wrdata\s+\S+[^\n]*", "wrdata sim/spec.csv v(vdet)", net)
    net = re.sub(r"meas tran[^\n]*\n", "", net)
    net = re.sub(r"print[^\n]*\n", "", net)
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_spec.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_spec.cir")],
                   cwd=AFE, capture_output=True, text=True, timeout=600)
    d = np.loadtxt(AFE / "sim" / "spec.csv")
    return d[:, 0], d[:, 1]                 # t, v(vdet)


def maxpool_to(env_t, env_v, dur, n_win):
    """Max of V+ within each WIN_MS window -> [n_win]."""
    out = np.full(n_win, np.nan)
    edges = np.linspace(0, dur, n_win + 1)
    for i in range(n_win):
        m = (env_t >= edges[i]) & (env_t < edges[i + 1])
        if m.any():
            out[i] = env_v[m].max()
    # fill any empty window by nearest
    idx = np.where(~np.isnan(out))[0]
    return np.interp(np.arange(n_win), idx, out[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="six.wav", help="wav name in AFE/audio/")
    args = ap.parse_args()

    wav = AFE / "audio" / args.wav
    if not wav.exists():
        wav = sorted((AFE / "audio").glob("*.wav"))[0]
    sig = load_full(wav)
    dur = len(sig) / FS
    n_win = int(round(dur * 1e3 / WIN_MS))
    print(f"[wav] {wav.name}  dur={dur*1e3:.0f} ms  ->  {n_win} windows ({WIN_MS:.0f} ms)")

    # PWL at FS over the full clip
    t = np.arange(len(sig)) / FS
    v = rt.BIAS + rt.AMP * sig
    pwl = "Vin in 0 PWL(\n" + "\n".join(f"+ {a:.7e} {b:.7e}"
                                        for a, b in zip(t, v)) + "\n)"

    env = np.zeros((16, n_win))
    for ch in range(16):
        RA, C, R1, fc = rt.channel_params(ch)
        net_base = rt.apply_channel(rt.NET, RA, C, R1)
        et, ev = run_channel(net_base, pwl, dur)
        env[ch] = maxpool_to(et, ev, dur, n_win)
        print(f"  ch{ch:2d}  f_c={fc:6.0f}  V+ {env[ch].min():.4f}..{env[ch].max():.4f}")

    # per-channel threshold: learned t_k placed in this clip's own V+ swing
    binimg = np.zeros_like(env)
    for ch in range(16):
        lo, hi = env[ch].min(), env[ch].max()
        thr = lo + float(np.clip(LEARNED[ch], 0, 1)) * (hi - lo)
        binimg[ch] = (env[ch] > thr).astype(float)

    fcs = [rt.channel_params(ch)[3] for ch in range(16)]
    yt = [0, 4, 8, 12, 15]
    ylab = [f"{fcs[i]:.0f}" for i in yt]
    extent = [0, dur * 1e3, 15.5, -0.5]

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    im0 = ax[0].imshow(env, aspect="auto", origin="upper", extent=extent,
                       cmap="viridis", interpolation="nearest")
    ax[0].set_title(f"AFE 16-ch spectrogram from SPICE circuit — '{wav.stem}'\n"
                    f"top: V+ envelope (continuous)   bottom: binary AFE output "
                    f"(NN input)")
    ax[0].set_ylabel("f_c [Hz]"); ax[0].set_yticks(yt); ax[0].set_yticklabels(ylab)
    fig.colorbar(im0, ax=ax[0], label="V+ [V]", pad=0.01)

    im1 = ax[1].imshow(binimg, aspect="auto", origin="upper", extent=extent,
                       cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax[1].set_ylabel("f_c [Hz]"); ax[1].set_yticks(yt); ax[1].set_yticklabels(ylab)
    ax[1].set_xlabel("time [ms]")
    fig.colorbar(im1, ax=ax[1], label="pulse (0/1)", pad=0.01)

    fig.tight_layout()
    out = AFE / "artifacts" / "afe_spectrogram16.png"
    fig.savefig(out, dpi=130)
    print(f"저장: {out}   (활성 픽셀 {int(binimg.sum())}/{binimg.size})")


if __name__ == "__main__":
    main()
