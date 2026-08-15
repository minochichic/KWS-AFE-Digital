"""16-channel AFE spectrogram from the ACTUAL SPICE circuit.

Runs the full chain (GIC filter -> active detector -> comparator) over a whole
GSC word for every one of the 16 designed channels, collects the detector
envelope V+(t), discretizes it into 10 ms windows (max), and binarizes it with
the comparator reference of a CHOSEN TRACK.

  --track xlse   track 1: diode-OR soft max (log-sum-exp) as the shared
                 reference, per-channel divider tap. The confirmed model.
  --track fixed  track 2: no diode-OR; each channel against its own absolute
                 divider. Needs --v-lo/--v-hi (a dataset-level range).
  --track minmax NOT BUILDABLE -- reference only. See below.

The picture used to be drawn with per-clip, per-channel min-max, labelled "what
the NN sees". It was neither: min-max needs the whole clip's extremes, so it
reads the future and no real-time circuit can produce it -- which is why the
project dropped it (docs/experiments_log.md: minmax 0.802, 구현 불가). The
top panel was always right (it is the circuit), but a binary panel drawn that
way does not correspond to any buildable comparator, so it must not be shown as
the network's input. The track now goes in the title AND the filename.

Output: artifacts/afe_spectrogram16_<track>.png
  top   -- continuous V+ envelope [16 x time]  (the circuit itself)
  bottom-- binary AFE image [16 x T]           (that track's comparator)

Thresholds and the LSE temperature come from a trained run via
`export/afe_constants.py` (--afe-json). Without it the script falls back to the
older hardcoded learned_r7r8.LEARNED.

Run from repo root:
  python analog/AFE/scripts/spectrogram16.py --wav six.wav \
      --afe-json runs/xl_g12/afe.json
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


def load_afe(path):
    """Constants exported by export/afe_constants.py, or None."""
    if not path:
        return None
    import json
    d = json.loads(Path(path).read_text())
    print(f"[afe] {path}  normalize={d['normalize']}  "
          f"{len(d['threshold'])} thresholds")
    return d


def _lse(swing, T):
    """T * log sum_j exp(swing_j / T) over channels, computed stably."""
    m = swing.max(axis=0, keepdims=True)
    return (m + T * np.log(np.exp((swing - m) / T).sum(axis=0,
                                                       keepdims=True)))[0]


def binarize(env, args):
    """V+ envelope [16, n_win] in volts -> binary image, plus a caption.

    The comparator is the same in every track -- what changes is the reference
    it is handed. That is the ONLY place the two tracks differ, so it is the
    only thing this function branches on.
    """
    afe = load_afe(args.afe_json)
    track = args.track
    if afe is not None and args.track == "auto":
        track = afe["normalize"]
    elif args.track == "auto":
        track = "minmax"

    # The detector rests at its quiescent point and swings UP from there; the
    # diode-OR and the divider are both referenced to that rest point (this is
    # the same convention delta is defined in). So the comparison happens in the
    # swing domain, not on absolute node volts.
    q = args.quiescent if args.quiescent is not None else float(
        np.percentile(env, 1.0))
    swing = env - q
    print(f"[quiescent] {q * 1e3:.1f} mV  ->  swing 0..{swing.max() * 1e3:.1f} mV")

    n = env.shape[0]
    thr = (np.clip(np.asarray(afe["threshold"], float), 0, 1) if afe is not None
           else np.clip(np.asarray(LEARNED, float), 0, 1))
    if thr.size != n:
        raise ValueError(f"{thr.size} thresholds for {n} channels")

    if track == "xlse":
        # Track 1. The diode-OR builds a SOFT max (log-sum-exp), each channel is
        # compared against its own divider tap off that node:
        #     fire_k  <=>  (swing_k - d) / (LSE_T(swing) - d)  >  alpha_k
        # T is taken from lse_temp_frac, NOT from the exported lse_temp: the
        # fraction is dimensionless and re-derives correctly here, while the
        # absolute value belongs to the software envelope domain. See
        # export/afe_constants.py.
        frac = float(afe["lse_temp_frac"]) if afe else args.lse_temp_frac
        d = float(afe.get("xmax_floor", 0.0)) if afe else 0.0
        typ = float(np.median(swing.max(axis=0)))
        T = max(frac * typ, 1e-12)
        s = _lse(swing, T)
        norm = np.where(s > d, (swing - d) / np.maximum(s - d, 1e-12), -1.0)
        binimg = (np.clip(norm, -1, 1) > thr[:, None]).astype(float)
        sub = (f"track 1 (xlse): diode-OR soft max, T = {frac:.2f} x typical "
               f"peak = {T * 1e3:.1f} mV, delta = {d:.3g}")

    elif track == "fixed":
        # Track 2. No diode-OR at all: each channel has its own divider set to
        # an ABSOLUTE voltage. That reference is dataset-level, so a single clip
        # cannot establish it -- it has to be supplied.
        if args.v_lo is None or args.v_hi is None:
            lo, hi = float(swing.min()), float(swing.max())
            print("[warn] --v-lo/--v-hi not given, so the absolute reference is "
                  "taken from THIS CLIP. The real dividers are set from the "
                  "dataset range; this picture is indicative only.")
            note = " (clip range -- not the dataset range)"
        else:
            lo, hi = args.v_lo, args.v_hi
            note = ""
        norm = (swing - lo) / max(hi - lo, 1e-12)
        binimg = (norm > thr[:, None]).astype(float)
        sub = (f"track 2 (fixed): per-channel absolute divider, "
               f"range {lo * 1e3:.1f}..{hi * 1e3:.1f} mV{note}")

    elif track == "minmax":
        # NOT BUILDABLE. Kept only because it is what this script used to draw,
        # so the old picture stays reproducible. min-max needs the whole clip's
        # extremes, i.e. it reads the future -- no real-time circuit can do it,
        # which is exactly why the project dropped it (docs/experiments_log.md:
        # "minmax 0.802, 구현 불가"). Channel-shared, matching data/afe.py's
        # amin/amax over dim=(1,2); the older per-channel form here was further
        # still from the software model.
        lo, hi = swing.min(), swing.max()
        binimg = ((swing - lo) / max(hi - lo, 1e-12) > thr[:, None]).astype(float)
        sub = ("minmax: NOT BUILDABLE -- needs the whole clip's max "
               "(reads the future). Reference only.")
    else:
        raise ValueError(f"unknown track {track!r}")

    print(f"[track] {track}  ->  {sub}")
    return binimg, sub, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="six.wav", help="wav name in AFE/audio/")
    ap.add_argument("--track", default="auto",
                    choices=["auto", "xlse", "fixed", "minmax"],
                    help="auto = read it from --afe-json")
    ap.add_argument("--afe-json", default=None,
                    help="export/afe_constants.py output for the trained run")
    ap.add_argument("--quiescent", type=float, default=None,
                    help="detector rest point [V]; default = 1st percentile")
    ap.add_argument("--lse-temp-frac", type=float, default=0.78,
                    help="used only when --afe-json is absent")
    ap.add_argument("--v-lo", type=float, default=None,
                    help="track 2: dataset-level swing floor [V]")
    ap.add_argument("--v-hi", type=float, default=None,
                    help="track 2: dataset-level swing ceiling [V]")
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

    binimg, sub, track = binarize(env, args)

    fcs = [rt.channel_params(ch)[3] for ch in range(16)]
    yt = [0, 4, 8, 12, 15]
    ylab = [f"{fcs[i]:.0f}" for i in yt]
    extent = [0, dur * 1e3, 15.5, -0.5]

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    im0 = ax[0].imshow(env, aspect="auto", origin="upper", extent=extent,
                       cmap="viridis", interpolation="nearest")
    unbuildable = track == "minmax"
    ax[0].set_title(
        f"AFE 16-ch spectrogram from SPICE circuit — '{wav.stem}'\n"
        f"top: V+ envelope (continuous, the circuit itself)\n"
        f"bottom: {'⚠ ' if unbuildable else ''}{sub}",
        fontsize=10)
    ax[0].set_ylabel("f_c [Hz]"); ax[0].set_yticks(yt); ax[0].set_yticklabels(ylab)
    fig.colorbar(im0, ax=ax[0], label="V+ [V]", pad=0.01)

    im1 = ax[1].imshow(binimg, aspect="auto", origin="upper", extent=extent,
                       cmap="Reds" if unbuildable else "Greys",
                       interpolation="nearest", vmin=0, vmax=1)
    ax[1].set_ylabel("f_c [Hz]"); ax[1].set_yticks(yt); ax[1].set_yticklabels(ylab)
    ax[1].set_xlabel("time [ms]")
    fig.colorbar(im1, ax=ax[1], label="pulse (0/1)", pad=0.01)

    fig.tight_layout()
    # the track is in the filename, so a picture can never be mistaken for one
    # drawn under a different comparator reference
    out = AFE / "artifacts" / f"afe_spectrogram16_{track}.png"
    fig.savefig(out, dpi=130)
    print(f"저장: {out}   (활성 픽셀 {int(binimg.sum())}/{binimg.size}, "
          f"{100 * binimg.mean():.1f}%)")


if __name__ == "__main__":
    main()
