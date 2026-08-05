"""Compare the SPICE-designed filterbank to the ideal (torchaudio) mel bank.

Loads sim/filterbank_matrix.csv (16 SPICE channels on the STFT grid), builds
the exact mel filterbank the ML uses, quantifies the match (cosine similarity
per channel), and saves an overlay plot (paper Fig.3 style: Simulation solid,
Mel dashed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AFE = Path(__file__).resolve().parents[1]
SR, N_FFT, N_MELS, FMIN, FMAX = 16000, 512, 16, 50.0, 8000.0


def main():
    spice = np.loadtxt(AFE / "artifacts" / "filterbank_matrix.csv", delimiter=",")  # [16,257]
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)

    # exact mel bank used by the ML (htk, MelSpectrogram default)
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1, f_min=FMIN, f_max=FMAX, n_mels=N_MELS,
        sample_rate=SR, norm=None, mel_scale="htk").numpy().T          # [16,257]
    mel = fb / (fb.max(axis=1, keepdims=True) + 1e-12)                 # peak-norm
    sp = spice / (spice.max(axis=1, keepdims=True) + 1e-12)

    # per-channel cosine similarity
    cos = [(float(np.dot(sp[k], mel[k]) /
            (np.linalg.norm(sp[k]) * np.linalg.norm(mel[k]) + 1e-12)))
           for k in range(N_MELS)]
    print("채널별 cosine 유사도 (SPICE vs mel):")
    for k in range(N_MELS):
        print(f"  ch{k:2d}: {cos[k]:.3f}")
    print(f"평균 cosine 유사도: {np.mean(cos):.3f}  (1.0 = 완전 일치)")

    # overlay plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(N_MELS):
        ax.plot(freqs, sp[k], color="tab:blue", lw=1.4,
                label="Simulation (SPICE)" if k == 0 else None)
        ax.plot(freqs, mel[k], color="tab:orange", lw=1.2, ls="--",
                label="Ideal Mel" if k == 0 else None)
    ax.set_xlim(0, 8200); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Gain [linear, peak-norm]")
    ax.set_title(f"SPICE-designed AFE filterbank vs ideal Mel "
                 f"(16 ch, mean cos {np.mean(cos):.3f})")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    out = AFE / "artifacts" / "filterbank_vs_mel.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
