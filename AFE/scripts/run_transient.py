"""Full-chain AFE transient on a speech clip -> paper-style 4-panel plot.

Reproduces Cerutti Fig.1's V_in / V_filt / V_+ (envelope) / V_out for one
channel. Uses a real GSC .wav if one is dropped in AFE/audio/, else a
speech-like synthetic signal. Two passes: first to measure the envelope V_+
range, then with the comparator threshold set to its midpoint (real hardware:
the R7/R8 divider, which must be tuned to this range -- not the 0.92 V it
computes to as drawn).

Pick any of the 16 designed channels with --ch (reads RA/C/R1 from
artifacts/filterbank_design.csv); default is channel 6 (~1.36 kHz).

Run from repo root:  .venv/bin/python AFE/scripts/run_transient.py --ch 3
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AFE = Path(__file__).resolve().parents[1]
NET = (AFE / "netlists" / "full_chain.cir").read_text()
DUR = 0.040                 # 40 ms window (like the paper's 20 ms, a bit longer)
FS = 20_000                 # PWL sample rate (>2x 8 kHz; keeps point count sane)
BIAS = 0.9                  # mid-rail
AMP = 4e-3                  # input amplitude around bias (small, like the mic)


def load_or_synth():
    """Return a mono signal in [-1,1] at FS. Real wav if present, else synth."""
    wavs = sorted((AFE / "audio").glob("*.wav")) if (AFE / "audio").is_dir() else []
    if wavs:
        import soundfile as sf
        x, sr = sf.read(wavs[0])
        if x.ndim > 1:
            x = x[:, 0]
        # resample to FS (linear) and take a voiced-looking DUR window
        t_new = np.arange(int(DUR * FS)) / FS
        t_old = np.arange(len(x)) / sr
        # pick the loudest 40 ms window
        w = int(DUR * sr)
        e = np.convolve(x ** 2, np.ones(w), "valid")
        s = int(np.argmax(e))
        seg = x[s:s + w]
        y = np.interp(t_new, np.arange(len(seg)) / sr, seg)
        print(f"[audio] {wavs[0].name} (loudest {DUR*1e3:.0f} ms window)")
        return y / (np.abs(y).max() + 1e-9)
    # synthetic voiced speech-like: pitch-modulated carrier near ~1.3 kHz + formants
    t = np.arange(int(DUR * FS)) / FS
    pitch = 130.0
    glottal = (np.sin(2 * np.pi * pitch * t) > 0).astype(float)   # pulse train-ish
    glottal = np.maximum(0, np.sin(2 * np.pi * pitch * t)) ** 2
    car = (np.sin(2 * np.pi * 1300 * t) + 0.6 * np.sin(2 * np.pi * 800 * t)
           + 0.4 * np.sin(2 * np.pi * 2400 * t))
    y = glottal * car + 0.03 * np.random.default_rng(0).standard_normal(len(t))
    print("[audio] synthetic voiced speech-like signal (no wav in AFE/audio/)")
    return y / (np.abs(y).max() + 1e-9)


def pwl_block(sig):
    """Inline PWL source line (ngspice's `PWL file=` parsing is unreliable)."""
    t = np.arange(len(sig)) / FS
    v = BIAS + AMP * sig
    body = "\n".join(f"+ {ti:.7e} {vi:.7e}" for ti, vi in zip(t, v))
    return f"Vin in 0 PWL(\n{body}\n)"


def channel_params(ch):
    """Read (RA, C, R1, f_c) for channel `ch` from the design table."""
    d = np.loadtxt(AFE / "artifacts" / "filterbank_design.csv",
                   delimiter=",", skiprows=1)
    r = d[int(ch)]                      # cols: ch,fc_t,fc_sim,Q_t,Q_sim,gain,RA,C,R1
    return r[6], r[7], r[8], r[2]       # RA, C, R1, f_c_sim


def apply_channel(net, RA, C, R1):
    net = re.sub(r"\.param RA\s*=\s*\S+",   f".param RA={RA:.6g}", net)
    net = re.sub(r"\.param CVAL\s*=\s*\S+", f".param CVAL={C:.6g}", net)
    net = re.sub(r"\.param R1v\s*=\s*\S+",  f".param R1v={R1:.6g}", net)
    return net


def run(vref, pwl, net_base):
    net = re.sub(r"\.param VREF=\S+", f".param VREF={vref:.6g}", net_base)
    net = re.sub(r"Vin\s+in\s+0\s+PWL[^\n]*", pwl, net)
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_full.cir").write_text(net)
    r = subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_full.cir")],
                       cwd=AFE, capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


def read_tran():
    d = np.loadtxt(AFE / "sim" / "tran.csv")
    # wrdata writes (t, v(in)) (t, v(vfilt)) ... columns interleaved
    t = d[:, 0]
    return t, d[:, 1], d[:, 3], d[:, 5], d[:, 7]   # in, vfilt, vdet, vout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=6,
                    help="channel index 0..15 (RA/C/R1 from design CSV)")
    args = ap.parse_args()

    RA, C, R1, fc = channel_params(args.ch)
    net_base = apply_channel(NET, RA, C, R1)
    print(f"channel {args.ch}: f_c={fc:.0f} Hz  RA={RA/1e3:.2f}k  "
          f"C={C*1e9:.2f}n  R1={R1/1e3:.1f}k")

    sig = load_or_synth()
    pwl = pwl_block(sig)

    print("[pass 1] measure envelope V_+ range ...")
    out = run(0.9004, pwl, net_base)
    m = {k: float(v) for k, v in re.findall(r"(vdmin|vdmax)\s*=\s*(\S+)", out)}
    if "vdmin" not in m:
        print(out[-1500:]); raise SystemExit("pass 1 failed")
    vref = 0.5 * (m["vdmin"] + m["vdmax"])
    print(f"  V_+ range {m['vdmin']:.5f}..{m['vdmax']:.5f} V -> VREF={vref:.5f}")

    print("[pass 2] run with tuned threshold ...")
    run(vref, pwl, net_base)
    t, vin, vfilt, vdet, vout = read_tran()
    keep = t >= 8e-3                              # drop the startup settling
    t, vin, vfilt, vdet, vout = (a[keep] for a in (t, vin, vfilt, vdet, vout))
    t = t - t[0]

    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for a, y, lab, c in [(ax[0], vin, "V_in [V]", "tab:blue"),
                         (ax[1], vfilt, "V_filt [V]", "tab:blue"),
                         (ax[2], vdet, "V_+ [V]", "tab:blue")]:
        a.plot(t * 1e3, y, c, lw=1.3); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[2].axhline(vref, color="tab:red", ls="--", lw=1, label=f"threshold {vref:.4f}")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[3].plot(t * 1e3, vout, "tab:red", lw=1.3)
    ax[3].set_ylabel("V_out [V]"); ax[3].set_xlabel("time [ms]"); ax[3].grid(alpha=0.3)
    ax[0].set_title(f"AFE full-chain transient -- channel {args.ch} "
                    f"(f_c = {fc:.0f} Hz)")
    fig.tight_layout()
    out_png = AFE / "artifacts" / f"afe_transient_ch{args.ch}.png"
    (AFE / "artifacts").mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=130)
    n_pulse = int(np.sum((vout[1:] > 0.9) & (vout[:-1] <= 0.9)))
    print(f"저장: {out_png}  (비교기 펄스 {n_pulse}개 / {DUR*1e3:.0f} ms)")


if __name__ == "__main__":
    main()
