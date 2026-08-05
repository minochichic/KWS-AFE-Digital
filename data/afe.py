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
import warnings
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


def _resolve_spice_path(path: str) -> Path:
    """Locate the SPICE filterbank, tolerating the pre-`analog/` layout.

    The analog folders moved under analog/ after runs and checkpoints had
    already recorded "AFE/artifacts/...". Those configs must keep loading, so
    try the recorded path first and then the same path with analog/ added or
    removed. Anything found this way is reported, because a config pointing at
    a path that no longer exists means the tree and the config disagree.
    """
    root = Path(__file__).resolve().parents[1]
    p = Path(path)
    first = p if p.is_absolute() else root / p
    if first.exists():
        return first
    parts = p.parts
    alts = [root / Path("analog", *parts)]
    if parts and parts[0] == "analog":
        alts.append(root / Path(*parts[1:]))
    for a in alts:
        if a.exists():
            warnings.warn(
                f"spice_matrix_path={path!r} does not exist; using {a} instead. "
                f"The analog folders moved under analog/ -- update the config "
                f"(train/config.py: spice_matrix_path) to silence this.")
            return a
    raise FileNotFoundError(
        f"SPICE filterbank not found. Tried {first} and {[str(a) for a in alts]}. "
        f"If the repo was just reorganized, `git pull`; the file lives at "
        f"analog/AFE/artifacts/filterbank_matrix.csv.")


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
            p = _resolve_spice_path(cfg.spice_matrix_path)
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
        if cfg.normalize in ("fixed", "agc", "xmax", "xmix"):
            self.register_buffer("fixed_lo", torch.zeros(1))
            self.register_buffer("fixed_hi", torch.ones(1))
        if cfg.normalize in ("xmax", "xmix"):
            self.register_buffer("xmax_floor", torch.zeros(1))

        # AGC divides by a running level, which is only meaningful in a
        # multiplicative (amplitude) domain. Under log compression a gain is a
        # SUBTRACTION and values go negative, so division is wrong there.
        # compression="sqrt" is also the circuit-faithful choice (V+ ~ amplitude).
        if cfg.normalize in ("agc", "xmax", "xmix") and \
                getattr(cfg, "compression", "log") != "sqrt":
            raise ValueError(
                f'normalize="{cfg.normalize}" requires compression="sqrt": it '
                'divides by a level, but under log compression gain is additive '
                'and envelopes can be negative.')

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
        elif self.cfg.normalize == "xmax":
            env = self._xmax(env)
        elif self.cfg.normalize == "xmix":
            env = self._xmix(env)
        else:
            raise ValueError(f"unknown normalize {self.cfg.normalize!r}")
        return env

    def _xmax(self, env: torch.Tensor) -> torch.Tensor:
        """Cross-channel relative threshold -- level invariance without any loop.

        Each channel is judged against the INSTANTANEOUS max across the 16
        channels rather than a fixed voltage:

            out[:, c, t] = env[:, c, t] / max( max_j env[:, j, t], floor )

        A gain g scales numerator and denominator alike, so it cancels exactly:
        loudness invariance is STRUCTURAL here, not something the net has to
        learn (gain augmentation tried that and lost 9pp, because with a fixed
        threshold a gain does not perturb the binary image, it erases it).

        Hardware: 16 diodes OR-ing the envelope outputs give that max passively
        into one shared divider. Compared with an AGC there is no feedback loop,
        no attack/release, hence no oscillation and no first-word problem -- and
        it replaces the 16 per-channel dividers, whose bias current was ~52 uW.

        The floor is what keeps silence quiet: dividing noise by noise would fire
        at random, so below it the comparison reverts to an absolute threshold.
        Real hardware gets this for free -- the diode-OR node cannot sink below
        the detector's quiescent level. It is set by init_fixed_scale() as a
        QUANTILE OF THE PER-FRAME cross-channel max, not as a fraction of the
        dataset peak: the dataset peak is the loudest frame of the loudest clip
        and sits ~200x above a median frame, so anchoring there made the floor
        dominate 87% of frames and collapsed this mode back into "fixed".
        """
        xmax = env.amax(dim=1, keepdim=True)                 # [B, 1, T]
        return env / (torch.maximum(xmax, self.xmax_floor) + _EPS)

    def _xmix(self, env: torch.Tensor) -> torch.Tensor:
        """`xmax` in the form the CIRCUIT actually produces.

        A resistor divider between the cross-channel max node and a shared
        reference cannot take a max(); it can only mix the two linearly:

            V_thr,c = a_c * V_max + (1 - a_c) * V_ref
              <=>   env_c > a_c * S + (1 - a_c) * d          (d = V_ref - quiescent)
              <=>   (env_c - d) / (S - d) > a_c

        so the comparison stays "normalized value > constant per-channel alpha"
        and only the denominator changes: max(S, floor) -> (S - d). alpha is
        still the divider ratio Rb/(Ra+Rb) and still lands in [0, 1].

        Why this matters even though `xmax` scores well: the two forms agree
        while S >> d but their SILENCE thresholds depend on alpha in opposite
        directions -- a_c*floor here vs (1-a_c)*d in the circuit -- so thresholds
        learned under `xmax` would be mis-set per channel once built. That is the
        same class of trap as the R7/R8 mapping, caught before committing to
        hardware rather than after.

        Two consequences worth naming:
          * invariance is now only approximate, because d does NOT scale with the
            input -- which is exactly the circuit's own behaviour, so this is
            fidelity, not regression;
          * the sqrt-guard degeneracy cannot happen here. In silence every
            channel sits on the guard, so the numerator goes NEGATIVE (guard - d)
            while the clamped denominator stays positive -- strongly "off",
            instead of `xmax` dividing the guard by itself and firing everything.
        """
        s = env.amax(dim=1, keepdim=True)                    # [B, 1, T] diode-OR
        d = self.xmax_floor                                  # V_ref - quiescent
        # S <= d is silence, and there the rearrangement is invalid: dividing by
        # a non-positive (S - d) flips the inequality. Decide it directly
        # instead. a*S + (1-a)*d is a convex combination of S and d, so with
        # S <= d it is >= S >= env_c and NO channel can fire -- emit a constant
        # "off" rather than a division blow-up. Clamping the denominator alone is
        # not enough: it produced values near -6e3, which dragged the mean that
        # init_thresholds() uses down to alpha ~ -200 and fired 93% of bits.
        out = torch.where(s > d, (env - d) / (s - d).clamp_min(_EPS),
                          torch.full_like(env, -1.0))
        # Bound it. alpha is a divider ratio Rb/(Ra+Rb), so only [0, 1] is
        # physical, and the comparator cannot tell "just below V_ref" from "far
        # below" -- the tail carries nothing it can output. Left unbounded it
        # reaches -59 (a frame whose S barely exceeds d has a tiny denominator),
        # 89% of values go negative, and the channel mean init_thresholds() uses
        # lands at alpha = -15, i.e. outside the range any resistor pair can
        # build. Clamping costs no information and keeps alpha buildable.
        return out.clamp(-1.0, 1.0)

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
        if not getattr(self.cfg, "binarize", True):
            # Diagnostic path: no comparator, so the network gets the continuous
            # envelope. Measures what the 1-bit input costs, holding the
            # filterbank, normalization and the network itself fixed.
            return env if target_T is None else pad_or_crop(env, target_T,
                                                            pad_value=0.0)
        # The comparator: sign(env - thr) with a straight-through gradient.
        # Gradient w.r.t. threshold is -1 * upstream inside the clip window,
        # which is how the thresholds learn (CLAUDE.md 2.4).
        thr = self.threshold.view(1, -1, 1)
        if self.cfg.normalize == "xmix":
            # alpha is a divider ratio Rb/(Ra+Rb): outside [0, 1] no resistor
            # pair can build it, and above 1 the channel is simply dead, since
            # the normalized value cannot exceed 1. Clamping only at init is not
            # enough -- nothing stopped training from walking back out, and a run
            # came back with alpha up to 1.344, i.e. a silently dead channel that
            # also could not have been soldered.
            # Straight-through: the forward pass uses the buildable value while
            # the gradient passes unchanged, so a channel parked on a boundary
            # can still walk back in instead of sticking there.
            thr = thr + (thr.clamp(0.0, 1.0) - thr).detach()
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
        if self.cfg.normalize not in ("fixed", "agc", "xmax", "xmix"):
            return
        waves = waves.to(self.threshold.device)      # dataloader batches are CPU
        env = self.envelopes(waves, raw=True)
        self.fixed_lo.fill_(env.min())
        q = getattr(self.cfg, "fixed_scale_quantile", 1.0)
        if q >= 1.0:
            self.fixed_hi.fill_(env.max())
        else:
            # Quantile over PER-CLIP maxima instead of the global max. The global
            # max is one outlier clip, which squeezes every typical clip into a
            # sliver of [0,1]: measured env std 0.034 vs 0.138 for per-clip
            # min-max, so the learned thresholds land at 0.002-0.025 and a single
            # Adam step (lr 1e-3) moves them ~15% instead of ~3%. The achievable
            # optimum is unchanged (an affine scale is absorbed by the learnable
            # thresholds) -- this only fixes the CONDITIONING of that optimization.
            # Still a constant, so it remains an absolute threshold -> fixed R7/R8.
            self.fixed_hi.fill_(env.amax(dim=(1, 2)).quantile(q))
        if self.cfg.normalize in ("xmax", "xmix"):
            # Silence floor for _xmax: a low quantile of the PER-FRAME
            # cross-channel max, so only genuinely quiet frames fall back to the
            # absolute comparison and speech frames stay gain-invariant.
            # <=0 means NO floor at all -> exact gain invariance everywhere.
            f = float(self.cfg.xmax_floor_frac)
            self.xmax_floor.fill_(
                env.amax(dim=1).flatten().quantile(f) if f > 0.0 else 0.0)
            # compression="sqrt" bottoms out at sqrt(_EPS), so a silent or
            # zero-padded frame is NOT zero -- every channel sits on that guard
            # together. Dividing that by itself gives 1.0 in all 16 channels, so
            # xmax turns "no signal" into the MAXIMALLY ACTIVE image, which is
            # backwards and unreproducible in hardware. Measured at
            # xmax_floor_frac=0.02 the floor landed exactly on the guard: a
            # zeroed clip fired 100% of bits and a x0.01 clip fired MORE than
            # real speech (0.538 vs 0.386). The floor must sit clear of it.
            # "xmix" is immune: it SUBTRACTS d, so a frame sitting on the guard
            # gives a non-positive numerator and stays off. Only "xmax", which
            # divides the guard by itself, needs the warning.
            guard = (_EPS ** 0.5 if self.cfg.compression == "sqrt" else 0.0)
            if self.cfg.normalize == "xmax" and \
                    0.0 < float(self.xmax_floor) < 2.0 * guard:
                warnings.warn(
                    f"afe.xmax_floor_frac={f} puts xmax_floor at "
                    f"{float(self.xmax_floor):.2e}, at the compression guard "
                    f"({guard:.2e}); silent/zero-padded frames will fire ALL "
                    f"channels. Raise xmax_floor_frac (0.05 clears it).")

    @torch.no_grad()
    def init_thresholds(self, waves: torch.Tensor) -> None:
        """Set each threshold to its channel's mean envelope (Cerutti IV-A).

        Pass a representative batch of training waveforms. The normalization
        inside envelopes() already applies the same min-max scaling to the
        features, so the mean is directly in threshold coordinates.
        """
        waves = waves.to(self.threshold.device)      # dataloader batches are CPU
        env = self.envelopes(waves)
        thr = env.mean(dim=(0, 2))
        if self.cfg.normalize == "xmix":
            # alpha is the divider ratio Rb/(Ra+Rb), so only [0, 1] is buildable.
            # It also removes a degenerate start: a channel that never rose above
            # V_ref in the init batch averages to exactly -1, the same value
            # _xmix() emits for silence, and sign(0) is +1 -- that channel would
            # fire on EVERY frame including digital silence (measured: one full
            # row, 4.9% of all bits).
            thr = thr.clamp(0.0, 1.0)
        self.threshold.copy_(thr)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"channels={c.n_channels}, native_T={c.native_T}, "
                f"envelope={c.envelope_reduce}@{c.envelope_win_ms:.0f}ms, "
                f"trainable_thr={c.threshold_trainable}")
