"""Silence firing rate vs speech firing rate, for fixed / xmax(floor sweep).

Event rate == interrupt rate == power in the event-driven AFE.  A normalization
that fires in silence breaks the whole low-power premise, so measure it before
committing to a floor value.

Usage:
    python experiments/xmax_event_rate.py [path/to/speech_commands_v0.02]
    # or set SPEECH_COMMANDS_ROOT
"""
import array
import glob
import os
import sys
import wave

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.afe import AFEFrontend                                    # noqa: E402
from data.speech_commands import synthesize_silence                 # noqa: E402
from train.config import load_config                                # noqa: E402

ROOT = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("SPEECH_COMMANDS_ROOT",
                            "datasets/SpeechCommands/speech_commands_v0.02"))
SR, N = 16000, 96


def wav_read(path):
    """16-bit PCM mono -> float32 in [-1, 1]. Avoids needing a torchaudio backend."""
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        a = array.array("h", w.readframes(w.getnframes()))
    return torch.tensor(a, dtype=torch.float32) / 32768.0


if not os.path.isdir(ROOT):
    sys.exit(f"Speech Commands not found at {ROOT!r} -- pass the path as argv[1] "
             f"or set SPEECH_COMMANDS_ROOT.")

torch.manual_seed(0)

# --- speech: N clips across the 10 keywords -------------------------------- #
kw = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
speech = []
for w in kw:
    for f in sorted(glob.glob(f"{ROOT}/{w}/*.wav"))[: N // len(kw) + 1]:
        x = wav_read(f)
        speech.append(torch.nn.functional.pad(x, (0, max(0, SR - x.numel())))[:SR])
speech = torch.stack(speech[:N])

# --- silence: background-noise crops (exactly how the dataset builds it) ---- #
noise = []
for f in sorted(glob.glob(f"{ROOT}/_background_noise_/*.wav")):
    noise.append(wav_read(f))
sil = synthesize_silence(noise, N, SR, seed=777, zero_fraction=0.1)
sil_noise, sil_zero = sil[int(N * 0.1):], sil[: int(N * 0.1)]

print(f"speech {tuple(speech.shape)}  silence(noise) {tuple(sil_noise.shape)}  "
      f"silence(zero) {tuple(sil_zero.shape)}")
print(f"rms  speech {speech.pow(2).mean().sqrt():.5f}   "
      f"silence {sil_noise.pow(2).mean().sqrt():.5f}\n")


def rate(cfg_over, calib):
    cfg = load_config("configs/base.yaml", cfg_over)
    afe = AFEFrontend(cfg.afe)
    afe.init_fixed_scale(calib)
    afe.init_thresholds(calib)
    out = {}
    for name, w in [("speech", speech), ("sil-noise", sil_noise),
                    ("sil-zero", sil_zero)]:
        b = afe(w)                                  # {-1,+1}
        out[name] = float((b > 0).float().mean())
    fl = float(afe.xmax_floor) if hasattr(afe, "xmax_floor") else float("nan")
    return out, fl


BASE = {"afe.filterbank_source": "spice", "afe.compression": "sqrt"}
calib = torch.cat([speech[:64], sil_noise[:32]])    # same mix the loader sees

rows = [("fixed q=0.75", {**BASE, "afe.normalize": "fixed",
                          "afe.fixed_scale_quantile": 0.75})]
for f in (0.30, 0.10, 0.02):
    rows.append((f"xmax f={f:.2f}", {**BASE, "afe.normalize": "xmax",
                                     "afe.xmax_floor_frac": f}))

print(f"{'setting':<14} {'floor':>8} {'speech':>8} {'sil-noise':>10} "
      f"{'sil-zero':>9}   silence/speech")
print("-" * 68)
for name, over in rows:
    r, fl = rate(over, calib)
    ratio = r["sil-noise"] / max(r["speech"], 1e-9)
    print(f"{name:<14} {fl:8.4f} {r['speech']:8.3f} {r['sil-noise']:10.3f} "
          f"{r['sil-zero']:9.3f}   {ratio:6.2f}x")
