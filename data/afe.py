"""Software simulation of Cerutti et al.'s analog front end (AFE).

Pipeline (Cerutti III / IV-A):

    waveform [B, L]
      -> STFT power spectrogram (25 ms window / 10 ms hop)
      -> mel filterbank with n_channels triangular filters, corner frequencies
         equally spaced in the mel domain between f_min and f_max
         (CLAUDE.md 3: the AFE's 16 bandpass filters, NOT the 64-mel analysis
         resolution of the full-precision baseline)
      -> log compression
      -> envelope: max over 10/25 ms windows        (Cerutti IV-A)
      -> per-clip min-max normalization to [0, 1]
      -> per-channel learnable threshold comparison  (the comparator)
      -> {-1, +1}, shape [B, n_channels, native_T]

Design decisions, and why:

* Thresholds are the ONLY parameters here (16 scalars). The real AFE's
  comparator reference voltage is set by a resistor divider; one number per
  channel is all the hardware can realize, so one nn.Parameter per channel is
  all we allow ourselves.
* The comparison goes through `sign_ste`, so the thresholds train end-to-end
  with the network (CLAUDE.md 2.4). Envelopes live in [0, 1] and thresholds
  near it, so |env - thr| <= 1 -- every element is inside the STE clip window
  and receives gradient.
* Min-max is per clip, global across channels (not per channel): per-channel
  scaling would erase inter-channel level differences, which is exactly the
  information the per-channel thresholds are supposed to absorb.
* Time padding after binarization uses -1, not 0. 0 is not a value in the
  binary domain; -1 is "channel off", which is what silence looks like.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.binary_ops import sign_ste
from train.config import AFEConfig

_EPS = 1e-6


def pad_or_crop(x: torch.Tensor, target_T: int,
                pad_value: float = -1.0) -> torch.Tensor:
    """Symmetrically pad (with `pad_value`) or center-crop the time axis.

    MatchboxNet 4.1 pads symmetrically to 128 frames; we do the same but with
    -1 because the tensor is already binarized.
    """
    t = x.shape[-1]
    if t == target_T:
        return x
    if t < target_T:
        total = target_T - t
        left = total // 2
        return F.pad(x, (left, total - left), value=pad_value)
    start = (t - target_T) // 2
    return x[..., start:start + target_T]


class AFEFrontend(nn.Module):
    """Waveform -> binary time-frequency image in {-1,+1}."""

    def __init__(self, cfg: AFEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        self.clip_samples = int(round(sr * cfg.clip_ms / 1000.0))

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg.n_fft,
            win_length=int(round(sr * cfg.stft_win_ms / 1000.0)),
            hop_length=int(round(sr * cfg.stft_hop_ms / 1000.0)),
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_channels,   # the AFE filterbank itself
            power=2.0,
        )

        # One comparator reference per channel. 0.5 is a placeholder; call
        # init_thresholds() with training data before real training
        # (Cerutti IV-A initializes to the per-channel average).
        self.threshold = nn.Parameter(
            torch.full((cfg.n_channels,), 0.5),
            requires_grad=cfg.threshold_trainable,
        )

    # ------------------------------------------------------------------ #
    def _fix_length(self, wave: torch.Tensor) -> torch.Tensor:
        """Accept [B, L] or [B, 1, L]; zero-pad/crop the wave to 1 clip."""
        if wave.dim() == 3:
            wave = wave.squeeze(1)
        if wave.dim() != 2:
            raise ValueError(f"expected [B, L] or [B, 1, L], got {tuple(wave.shape)}")
        L = wave.shape[-1]
        if L < self.clip_samples:
            wave = F.pad(wave, (0, self.clip_samples - L))
        elif L > self.clip_samples:
            wave = wave[:, :self.clip_samples]
        return wave

    def envelopes(self, wave: torch.Tensor) -> torch.Tensor:
        """Normalized full-precision envelopes [B, C, native_T] in [0, 1].

        This is the value the comparator sees; exposed separately for
        threshold initialization and for inspection plots.
        """
        wave = self._fix_length(wave)
        mel = self.melspec(wave)                       # [B, C, frames]
        mel = torch.log(mel + _EPS)

        # Envelope over 10/25 ms windows. adaptive pooling to native_T bins is
        # exactly "max of the spectrogram in windows of X ms" and stays correct
        # when the STFT frame grid does not divide the envelope window.
        if self.cfg.envelope_reduce == "max":
            env = F.adaptive_max_pool1d(mel, self.cfg.native_T)
        elif self.cfg.envelope_reduce == "mean":
            env = F.adaptive_avg_pool1d(mel, self.cfg.native_T)
        else:
            raise ValueError(f"unknown envelope_reduce {self.cfg.envelope_reduce!r}")

        if self.cfg.normalize == "minmax":
            lo = env.amin(dim=(1, 2), keepdim=True)
            hi = env.amax(dim=(1, 2), keepdim=True)
            env = (env - lo) / (hi - lo + _EPS)
        elif self.cfg.normalize != "none":
            raise ValueError(f"unknown normalize {self.cfg.normalize!r}")
        return env

    def forward(self, wave: torch.Tensor,
                target_T: Optional[int] = None) -> torch.Tensor:
        env = self.envelopes(wave)
        # The comparator: sign(env - thr) with a straight-through gradient.
        # Gradient w.r.t. threshold is -1 * upstream inside the clip window,
        # which is how the thresholds learn (CLAUDE.md 2.4).
        out = sign_ste(env - self.threshold.view(1, -1, 1), self.cfg.ste_clip)
        if target_T is not None:
            out = pad_or_crop(out, target_T, pad_value=-1.0)
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_thresholds(self, waves: torch.Tensor) -> None:
        """Set each threshold to its channel's mean envelope (Cerutti IV-A).

        Pass a representative batch of training waveforms. The normalization
        inside envelopes() already applies the same min-max scaling to the
        features, so the mean is directly in threshold coordinates.
        """
        env = self.envelopes(waves)
        self.threshold.copy_(env.mean(dim=(0, 2)))

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"channels={c.n_channels}, native_T={c.native_T}, "
                f"envelope={c.envelope_reduce}@{c.envelope_win_ms:.0f}ms, "
                f"trainable_thr={c.threshold_trainable}")
