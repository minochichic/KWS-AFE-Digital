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

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.binary_ops import sign_ste
from train.config import AFEConfig

_EPS = 1e-6


def discretize(pulse_times_ms, window_ms: float, n_windows: int,
               reduce: str = "max"):
    """Reference implementation of the AFE discretization rule (CLAUDE.md 2.8).

    Continuous comparator events (pulse times, sub-ms) -> [n_windows] bins.
    This is executable documentation, NOT the production path: the real AFE
    sim max-pools continuous log-mel envelopes, which is EQUIVALENT for
    reduce='max' (max-then-threshold == threshold-then-OR). It exists so the
    2.8 table can be unit-tested directly, since the full STFT pipeline works
    at 10 ms frame resolution and cannot ingest a sub-ms pulse train.

    reduce:
      "max"   -> 1 if any pulse fell in the window (the baseline rule).
      "count" -> number of pulses (preserves info; breaks binary-ness).
    ('mean'/activity-fraction would need pulse durations, which we don't model.)
    """
    counts = [0] * n_windows
    for t in pulse_times_ms:
        i = int(t // window_ms)
        if 0 <= i < n_windows:
            counts[i] += 1
    if reduce == "max":
        return [1 if c > 0 else 0 for c in counts]
    if reduce == "count":
        return counts
    raise ValueError(
        f"reduce {reduce!r} unsupported by this reference (max/count only)")


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

        win = int(round(sr * cfg.stft_win_ms / 1000.0))
        hop = int(round(sr * cfg.stft_hop_ms / 1000.0))
        self.filterbank_source = getattr(cfg, "filterbank_source", "mel")
        if self.filterbank_source == "mel":
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=cfg.n_fft,
                win_length=win,
                hop_length=hop,
                f_min=cfg.f_min,
                f_max=cfg.f_max,
                n_mels=cfg.n_channels,   # the AFE filterbank itself
                power=2.0,
            )
        elif self.filterbank_source == "spice":
            # power spectrogram + the SPICE-extracted GIC filterbank matrix.
            self._spectro = torchaudio.transforms.Spectrogram(
                n_fft=cfg.n_fft, win_length=win, hop_length=hop, power=2.0)
            p = Path(cfg.spice_matrix_path)
            if not p.is_absolute():
                p = Path(__file__).resolve().parents[1] / p
            m = np.loadtxt(p, delimiter=",")           # [C, n_fft//2+1]
            n_freqs = cfg.n_fft // 2 + 1
            if m.shape != (cfg.n_channels, n_freqs):
                raise ValueError(
                    f"SPICE filterbank {p} is {m.shape}, expected "
                    f"({cfg.n_channels}, {n_freqs}); regenerate on this STFT grid.")
            if getattr(cfg, "spice_gain_restore", False):
                # undo the per-channel peak-norm: weight |H| by the true linear
                # passband gain (gain_dB col in the design table) -> restores the
                # real cross-channel spectral tilt.
                dp = p.parent / "filterbank_design.csv"
                gdb = np.loadtxt(dp, delimiter=",", skiprows=1)[:, 5]   # gain_dB
                m = m * (10.0 ** (gdb / 20.0))[:, None]                 # amplitude
            self.register_buffer("spice_fbank",
                                 torch.tensor(m, dtype=torch.float32))
        else:
            raise ValueError(f"unknown filterbank_source {self.filterbank_source!r}")

        # One comparator reference per channel. 0.5 is a placeholder; call
        # init_thresholds() with training data before real training
        # (Cerutti IV-A initializes to the per-channel average).
        self.threshold = nn.Parameter(
            torch.full((cfg.n_channels,), 0.5),
            requires_grad=cfg.threshold_trainable,
        )

        # Stage-2: learnable per-channel detector deadzone (0 = no-op baseline).
        self.use_deadzone = getattr(cfg, "spice_deadzone", False)
        if self.use_deadzone:
            self.deadzone = nn.Parameter(torch.zeros(cfg.n_channels))

        # normalize="fixed"/"agc": dataset-level lo/hi, set once by
        # init_fixed_scale(). "fixed" uses them as the affine scale; "agc" uses
        # fixed_hi as the reference for the max-gain floor.
        # Buffers (not params) so they land in the checkpoint and move with .to().
        if cfg.normalize in ("fixed", "agc"):
            self.register_buffer("fixed_lo", torch.zeros(1))
            self.register_buffer("fixed_hi", torch.ones(1))

        # AGC divides by a running level, which is only meaningful in a
        # multiplicative (amplitude) domain. Under log compression a gain is a
        # SUBTRACTION and values go negative, so division is wrong there.
        # compression="sqrt" is also the circuit-faithful choice (V+ ~ amplitude).
        if cfg.normalize == "agc" and getattr(cfg, "compression", "log") != "sqrt":
            raise ValueError(
                'normalize="agc" requires compression="sqrt": AGC divides by a '
                'level, but under log compression gain is additive and envelopes '
                'can be negative.')

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

    def _bands(self, wave: torch.Tensor) -> torch.Tensor:
        """Per-channel power spectrogram [B, C, frames] from the chosen bank."""
        if self.filterbank_source == "spice":
            spec = self._spectro(wave)                  # [B, n_freqs, frames] POWER
            # filter output power = |H(f)|^2 * input power. spice_fbank stores
            # |H(f)| (voltage magnitude), so square it before applying to power.
            return torch.einsum("cf,bft->bct", self.spice_fbank ** 2, spec)
        return self.melspec(wave)                       # [B, C, frames]

    def envelopes(self, wave: torch.Tensor, raw: bool = False) -> torch.Tensor:
        """Normalized full-precision envelopes [B, C, native_T] in [0, 1].

        `raw=True` skips the normalization step (used by init_fixed_scale to
        measure the dataset-level lo/hi before they exist).

        This is the value the comparator sees; exposed separately for
        threshold initialization and for inspection plots.
        """
        wave = self._fix_length(wave)
        mel = self._bands(wave)                         # [B, C, frames] power
        comp = getattr(self.cfg, "compression", "log")
        if comp == "log":
            mel = torch.log(mel + _EPS)
        elif comp == "sqrt":                            # V+ ~ amplitude = sqrt(power)
            mel = torch.sqrt(mel + _EPS)                # circuit-faithful detector
        else:
            raise ValueError(f"unknown compression {comp!r}")

        # Stage-2 detector deadzone: cut each channel's low-amplitude tail
        # (relu(amp - dz_c)); dz_c >= 0, learned end-to-end. Sharpens the broad
        # SPICE filters. dz init 0 -> exact no-op.
        if getattr(self, "use_deadzone", False):
            mel = F.relu(mel - self.deadzone.clamp(min=0).view(1, -1, 1))

        # Optional tau smoothing (active-detector C3 model, CLAUDE.md 2.10/3.2).
        # Applied on the log-mel FRAMES, BEFORE discretization (do not reorder).
        # tau=0 is an exact no-op (baseline). Software approximation only.
        mel = self._smooth(mel)

        # Envelope over 10/25 ms windows. adaptive pooling to native_T bins is
        # exactly "max of the spectrogram in windows of X ms" and stays correct
        # when the STFT frame grid does not divide the envelope window.
        if self.cfg.envelope_reduce == "max":
            env = F.adaptive_max_pool1d(mel, self.cfg.native_T)
        elif self.cfg.envelope_reduce == "mean":
            env = F.adaptive_avg_pool1d(mel, self.cfg.native_T)
        else:
            raise ValueError(f"unknown envelope_reduce {self.cfg.envelope_reduce!r}")

        if raw or self.cfg.normalize == "none":
            return env
        if self.cfg.normalize == "minmax":
            lo = env.amin(dim=(1, 2), keepdim=True)   # per clip, channel-shared
            hi = env.amax(dim=(1, 2), keepdim=True)
            env = (env - lo) / (hi - lo + _EPS)
        elif self.cfg.normalize == "fixed":
            # dataset-level constants -> affine map == absolute threshold
            env = (env - self.fixed_lo) / (self.fixed_hi - self.fixed_lo + _EPS)
        elif self.cfg.normalize == "agc":
            env = self._agc(env)
        else:
            raise ValueError(f"unknown normalize {self.cfg.normalize!r}")
        return env

    def _agc(self, env: torch.Tensor) -> torch.Tensor:
        """Causal, channel-shared AGC -- the hardware-realizable normalization.

        per-clip min-max divides by the max over the WHOLE clip, i.e. it peeks at
        the future; no analog circuit can do that. A real AGC tracks a running
        level with fast attack / slow release and divides by it:

            lev[t] = lev[t-1] + a * (x[t] - lev[t-1]),   a = a_att if rising
                                                            a_rel otherwise
            out[:, c, t] = env[:, c, t] / max(lev[t], floor)

        Two properties that make it a faithful hardware model:
        * `x[t] = max over CHANNELS` -> all 16 channels share ONE gain, so the
          cross-channel spectrum shape survives (the whole point of using a
          channel-shared min-max in the first place).
        * `floor = fixed_hi / 10^(max_gain_db/20)` caps the gain, like a real
          AGC's noise gate; without it silence would be amplified into noise.

        Runs on the envelope grid (envelope_win_ms per step), so the loop is
        ~100 steps -- same cost as the tau EMA above, and autograd-friendly.
        """
        cfg = self.cfg
        dt = cfg.envelope_win_ms
        a_att = 1.0 - math.exp(-dt / max(cfg.agc_attack_ms, 1e-6))
        a_rel = 1.0 - math.exp(-dt / max(cfg.agc_release_ms, 1e-6))
        x = env.amax(dim=1)                          # [B, T] channel-shared level
        lev = x[:, 0]
        levels = [lev]
        for t in range(1, x.shape[1]):
            xt = x[:, t]
            rising = (xt > lev).to(env.dtype)        # fast up, slow down
            a = a_rel + (a_att - a_rel) * rising
            lev = lev + a * (xt - lev)
            levels.append(lev)
        floor = self.fixed_hi / (10.0 ** (cfg.agc_max_gain_db / 20.0))
        gain_ref = torch.maximum(torch.stack(levels, dim=1), floor)   # [B, T]
        return env / (gain_ref.unsqueeze(1) + _EPS)

    def _smooth(self, mel: torch.Tensor) -> torch.Tensor:
        """Causal exponential (EMA) smoothing along the STFT-frame axis.

        Models the analog active detector's C3 (fast charge / slow discharge)
        as a software approximation on log-mel frames. For a frame interval
        dt = stft_hop_ms and time constant tau, the discrete EMA coefficient is
        `alpha = 1 - exp(-dt/tau)`; the recurrence is
            y[t] = alpha * x[t] + (1 - alpha) * y[t-1].
        As tau -> 0, alpha -> 1 and y == x. tau == 0.0 is treated as an EXACT
        no-op (returns mel unchanged) -- CLAUDE.md 3.2 requires the baseline to
        be untouched, and this also avoids a div-by-zero.
        """
        tau = self.cfg.envelope_tau_ms
        if tau <= 0.0:
            return mel                                   # exact identity (baseline)
        dt = self.cfg.stft_hop_ms                        # frame interval (ms)
        alpha = 1.0 - math.exp(-dt / tau)
        # sequential scan over frames (~100); autograd-friendly.
        frames = mel.unbind(dim=2)
        y = frames[0]
        out = [y]
        for x in frames[1:]:
            y = alpha * x + (1.0 - alpha) * y
            out.append(y)
        return torch.stack(out, dim=2)

    def forward(self, wave: torch.Tensor,
                target_T: Optional[int] = None) -> torch.Tensor:
        env = self.envelopes(wave)
        # The comparator: sign(env - thr) with a straight-through gradient.
        # Gradient w.r.t. threshold is -1 * upstream inside the clip window,
        # which is how the thresholds learn (CLAUDE.md 2.4).
        thr = self.threshold.view(1, -1, 1)
        # Optional comparator input offset Vos (real LPV7215 has ~mV; ideal
        # sign() has none). Random per (clip, channel) -> eval Monte-Carlos over
        # the offset distribution. vos=0 is an exact no-op (baseline unchanged).
        vos = getattr(self.cfg, "comparator_vos", 0.0)
        if vos > 0.0:
            thr = thr + vos * torch.randn(env.shape[0], env.shape[1], 1,
                                          device=env.device, dtype=env.dtype)
        out = sign_ste(env - thr, self.cfg.ste_clip)
        if target_T is not None:
            out = pad_or_crop(out, target_T, pad_value=-1.0)
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_fixed_scale(self, waves: torch.Tensor) -> None:
        """Set the dataset-level lo/hi for normalize="fixed" (Cerutti IV-A).

        Call ONCE with a representative training batch, BEFORE init_thresholds()
        (which needs the scale to already exist). No-op for other normalize modes.
        Since lo/hi are constants, the later `env >= thr` decision is an absolute
        threshold -> maps straight to a fixed R7/R8 divider.
        """
        if self.cfg.normalize not in ("fixed", "agc"):
            return
        env = self.envelopes(waves, raw=True)
        self.fixed_lo.fill_(env.min())
        self.fixed_hi.fill_(env.max())

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
