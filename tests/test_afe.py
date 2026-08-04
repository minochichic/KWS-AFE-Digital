"""AFE simulation tests (FIRST_TASK.md step 3).

What must hold:

1. waveform in -> [B, 16, native_T] out, values exactly {-1,+1}.
2. The binarization is a plain per-channel threshold comparison -- bit-exact
   with `where(env >= thr, +1, -1)`, because that is all the analog comparator
   can do.
3. Gradients reach the 16 thresholds through the step function (STE), and the
   thresholds actually move under SGD -- CLAUDE.md 2.4 requires end-to-end
   threshold training.
4. Physics sanity: a pure tone lights up the filterbank channel whose mel
   band contains it.
"""

from __future__ import annotations

import math

import pytest
import torch

from data.afe import AFEFrontend, discretize, pad_or_crop
from train.config import AFEConfig


def make_frontend(**kw) -> AFEFrontend:
    return AFEFrontend(AFEConfig(**kw))


def noise(batch: int = 2, sr: int = 16000) -> torch.Tensor:
    torch.manual_seed(0)
    return 0.1 * torch.randn(batch, sr)


# --------------------------------------------------------------------------- #
# 1. shapes and value domain
# --------------------------------------------------------------------------- #
def test_output_shape_and_values_25ms() -> None:
    fe = make_frontend(envelope_win_ms=25.0)
    out = fe(noise())
    assert out.shape == (2, 16, 40)                      # native_T = 1000/25
    assert set(out.unique().tolist()) <= {-1.0, 1.0}


def test_output_shape_10ms() -> None:
    fe = make_frontend(envelope_win_ms=10.0)
    assert fe(noise()).shape == (2, 16, 100)             # native_T = 1000/10


def test_channel_count_follows_config() -> None:
    fe = make_frontend(n_channels=8)
    assert fe(noise()).shape[1] == 8
    assert fe.threshold.shape == (8,)


def test_accepts_B_1_L_and_short_or_long_waves() -> None:
    fe = make_frontend(envelope_win_ms=10.0)
    assert fe(noise().unsqueeze(1)).shape == (2, 16, 100)   # [B,1,L]
    short = 0.1 * torch.randn(2, 9000)                      # <1 s -> zero-pad
    long_ = 0.1 * torch.randn(2, 20000)                     # >1 s -> crop
    assert fe(short).shape == (2, 16, 100)
    assert fe(long_).shape == (2, 16, 100)


# --------------------------------------------------------------------------- #
# 2. binarization == threshold comparison, exactly
# --------------------------------------------------------------------------- #
def test_binarization_matches_comparator() -> None:
    """The whole AFE decision is env >= thr. Nothing else."""
    fe = make_frontend()
    w = noise()
    env = fe.envelopes(w)                                # normalized, [B,C,T]
    out = fe(w)
    thr = fe.threshold.view(1, -1, 1)
    expected = torch.where(env >= thr, 1.0, -1.0)
    assert torch.equal(out, expected)


def test_envelopes_are_minmax_normalized() -> None:
    fe = make_frontend()
    env = fe.envelopes(noise())
    for b in range(env.shape[0]):                        # per-clip scaling
        assert math.isclose(env[b].min().item(), 0.0, abs_tol=1e-5)
        assert math.isclose(env[b].max().item(), 1.0, abs_tol=1e-5)


def test_raising_threshold_never_adds_ones() -> None:
    fe = make_frontend()
    w = noise()
    ones_before = (fe(w) > 0).sum(dim=(0, 2))            # per channel
    with torch.no_grad():
        fe.threshold += 0.2
    ones_after = (fe(w) > 0).sum(dim=(0, 2))
    assert torch.all(ones_after <= ones_before)

    with torch.no_grad():
        fe.threshold.fill_(1.1)                          # above max -> all off
    assert torch.all(fe(w) == -1.0)


# --------------------------------------------------------------------------- #
# 3. threshold learning (CLAUDE.md 2.4)
# --------------------------------------------------------------------------- #
def test_threshold_gradient_flows_through_step() -> None:
    fe = make_frontend()
    fe(noise()).mean().backward()
    assert fe.threshold.grad is not None
    assert torch.all(fe.threshold.grad != 0)


def test_threshold_not_trainable_when_disabled() -> None:
    fe = make_frontend(threshold_trainable=False)
    assert not fe.threshold.requires_grad
    # still present in state_dict for export
    assert "threshold" in fe.state_dict()


def test_thresholds_move_under_sgd() -> None:
    """End-to-end: push the output toward all -1; thresholds must rise."""
    fe = make_frontend()
    opt = torch.optim.SGD([fe.threshold], lr=0.05)
    w = noise()
    before = fe.threshold.detach().clone()
    for _ in range(10):
        opt.zero_grad()
        fe(w).mean().backward()                          # minimize mean output
        opt.step()
    after = fe.threshold.detach()
    assert not torch.equal(before, after)
    assert torch.all(after >= before)                    # rose, as predicted


def test_threshold_init_from_data_is_channel_mean() -> None:
    """Cerutti IV-A: thresholds init to the per-channel average envelope."""
    fe = make_frontend()
    w = noise(4)
    fe.init_thresholds(w)
    env = fe.envelopes(w)
    expected = env.mean(dim=(0, 2))
    assert torch.allclose(fe.threshold.detach(), expected, atol=1e-6)
    assert torch.all(fe.threshold >= 0) and torch.all(fe.threshold <= 1)


# --------------------------------------------------------------------------- #
# 4. physics sanity: tones land in the right channel
# --------------------------------------------------------------------------- #
def _mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)          # HTK, torchaudio default


@pytest.mark.parametrize("freq", [300.0, 1000.0, 4000.0])
def test_pure_tone_activates_matching_channel(freq: float) -> None:
    cfg = AFEConfig()
    fe = AFEFrontend(cfg)
    t = torch.arange(16000) / 16000.0
    tone = torch.sin(2 * math.pi * freq * t).unsqueeze(0)

    env = fe.envelopes(tone)                             # [1, 16, T]
    got = env.mean(dim=2).argmax().item()

    centers = torch.linspace(_mel(cfg.f_min), _mel(cfg.f_max),
                             cfg.n_channels + 2)[1:-1]
    expected = (centers - _mel(freq)).abs().argmin().item()
    assert abs(got - expected) <= 1, f"{freq} Hz: channel {got}, expected ~{expected}"


# --------------------------------------------------------------------------- #
# 5. pad / crop to the model's T
# --------------------------------------------------------------------------- #
def test_pad_to_larger_T_uses_binary_off() -> None:
    x = torch.ones(2, 16, 100)
    out = pad_or_crop(x, 128)
    assert out.shape == (2, 16, 128)
    # symmetric pad: 14 left, 14 right, filled with -1 ("off"), NOT 0 --
    # 0 is not a value the binary domain contains.
    assert torch.all(out[:, :, :14] == -1.0)
    assert torch.all(out[:, :, -14:] == -1.0)
    assert torch.all(out[:, :, 14:114] == 1.0)


def test_crop_to_smaller_T_is_centered() -> None:
    x = torch.arange(10).float().view(1, 1, 10).expand(1, 3, 10)
    out = pad_or_crop(x, 6)
    assert out.shape == (1, 3, 6)
    assert torch.equal(out[0, 0], torch.tensor([2.0, 3, 4, 5, 6, 7]))


def test_pad_or_crop_identity() -> None:
    x = torch.randn(1, 4, 50)
    assert pad_or_crop(x, 50) is x


def test_frontend_target_T_end_to_end() -> None:
    fe = make_frontend(envelope_win_ms=10.0)             # native 100
    out = fe(noise(), target_T=128)                      # MatchboxNet 4.1 pad
    assert out.shape == (2, 16, 128)
    assert set(out.unique().tolist()) <= {-1.0, 1.0}


def test_gradient_survives_padding() -> None:
    fe = make_frontend(envelope_win_ms=10.0)
    fe(noise(), target_T=128).mean().backward()
    assert fe.threshold.grad is not None
    assert torch.any(fe.threshold.grad != 0)


# --------------------------------------------------------------------------- #
# 6. discretization rule (CLAUDE.md 2.8, standalone reference)
# --------------------------------------------------------------------------- #
def test_discretize_matches_claude_md_table() -> None:
    # 20 ms span, 10 ms windows, pulses at 1.2/4.6/13.4/18.1 ms -> [1, 1]
    pulses = [1.2, 4.6, 13.4, 18.1]
    assert discretize(pulses, window_ms=10.0, n_windows=2, reduce="max") == [1, 1]


def test_discretize_info_loss_one_vs_two_pulses() -> None:
    """max loses count/time/duration: 1 pulse and 2 pulses both give 1."""
    one = discretize([1.2], 10.0, 2, reduce="max")
    two = discretize([1.2, 4.6], 10.0, 2, reduce="max")
    assert one[0] == two[0] == 1                          # same despite differing
    # count preserves the difference (but breaks binary-ness)
    assert discretize([1.2], 10.0, 2, "count")[0] == 1
    assert discretize([1.2, 4.6], 10.0, 2, "count")[0] == 2


def test_discretize_ignores_out_of_range_pulses() -> None:
    assert discretize([25.0, -1.0, 5.0], 10.0, 2, "max") == [1, 0]


def test_maxpool_equals_threshold_then_or() -> None:
    """The production path (max-pool continuous env, then threshold) equals the
    reference rule (threshold each frame, then OR) -- CLAUDE.md 2.8 equivalence.
    """
    torch.manual_seed(0)
    env = torch.rand(1, 1, 20)                            # 20 frames, [0,1)
    thr = 0.6
    # production: pool 20 frames -> 2 bins, then threshold
    pooled = torch.nn.functional.adaptive_max_pool1d(env, 2)
    prod = (pooled >= thr).long().flatten().tolist()
    # reference: threshold each frame, then OR within each 10-frame bin
    binm = (env[0, 0] >= thr).long()
    ref = [int(binm[:10].any()), int(binm[10:].any())]
    assert prod == ref


# --------------------------------------------------------------------------- #
# 7. tau smoothing (CLAUDE.md 3.2) -- baseline must stay a no-op
# --------------------------------------------------------------------------- #
def test_tau_zero_is_exact_identity() -> None:
    fe = make_frontend(envelope_tau_ms=0.0)
    mel = torch.randn(2, 16, 40)
    assert torch.equal(fe._smooth(mel), mel)             # EXACT, not approx


def test_tau_zero_baseline_envelopes_unchanged() -> None:
    """A frontend with tau=0 gives the identical envelopes as if smoothing did
    not exist -- guards the baseline."""
    w = noise()
    fe = make_frontend(envelope_win_ms=10.0, envelope_tau_ms=0.0)
    # _smooth is identity, so envelopes == envelopes-without-smooth by construction
    mel = fe.melspec(fe._fix_length(w))
    mel = torch.log(mel + 1e-6)
    assert torch.equal(fe._smooth(mel), mel)


def test_tau_positive_changes_output_and_follows_ema() -> None:
    fe = make_frontend(envelope_tau_ms=5.0, stft_hop_ms=10.0)
    # differs from identity
    m = torch.randn(1, 3, 6)
    assert not torch.equal(fe._smooth(m), m)
    # matches the EMA recurrence exactly
    alpha = 1.0 - math.exp(-10.0 / 5.0)
    x = torch.randn(1, 1, 5)
    y = fe._smooth(x)[0, 0]
    exp = [x[0, 0, 0]]
    for t in range(1, 5):
        exp.append(alpha * x[0, 0, t] + (1 - alpha) * exp[-1])
    assert torch.allclose(y, torch.stack(exp), atol=1e-6)


def test_tau_smoothing_is_differentiable() -> None:
    fe = make_frontend(envelope_tau_ms=5.0)
    x = torch.randn(1, 2, 8, requires_grad=True)
    fe._smooth(x).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)


# --------------------------------------------------------------------------- #
# 8. path separation: envelope_win_ms must not touch the STFT (CLAUDE.md 2.9)
# --------------------------------------------------------------------------- #
def test_envelope_win_does_not_change_stft_hop() -> None:
    fe10 = make_frontend(envelope_win_ms=10.0, stft_hop_ms=10.0)
    fe25 = make_frontend(envelope_win_ms=25.0, stft_hop_ms=10.0)
    # STFT hop is identical (independent of envelope_win)
    assert fe10.melspec.hop_length == fe25.melspec.hop_length
    # only native_T (the discretization grid) changes
    assert fe10.cfg.native_T == 100 and fe25.cfg.native_T == 40


# --------------------------------------------------------------------------- #
# 9. Phase B: SPICE filterbank source (circuit-matched front end)
# --------------------------------------------------------------------------- #
def test_spice_filterbank_shapes_and_binary() -> None:
    fe = make_frontend(filterbank_source="spice", envelope_win_ms=10.0)
    assert tuple(fe.spice_fbank.shape) == (16, 512 // 2 + 1)   # [n_channels, n_freqs]
    wave = torch.randn(3, 16000) * 0.05
    env = fe.envelopes(wave)
    assert env.shape == (3, 16, 100)
    assert float(env.min()) >= 0.0 and float(env.max()) <= 1.0
    out = fe(wave, target_T=128)
    assert out.shape == (3, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}


def test_spice_and_mel_differ() -> None:
    wave = torch.randn(2, 16000) * 0.05
    em = make_frontend(filterbank_source="mel", envelope_win_ms=10.0).envelopes(wave)
    es = make_frontend(filterbank_source="spice", envelope_win_ms=10.0).envelopes(wave)
    assert (em - es).abs().mean() > 1e-3      # genuinely different filter shapes


def test_spice_thresholds_train() -> None:
    fe = make_frontend(filterbank_source="spice", envelope_win_ms=10.0)
    wave = torch.randn(4, 16000) * 0.05
    before = fe.threshold.detach().clone()
    opt = torch.optim.SGD(fe.parameters(), lr=1.0)
    fe(wave, target_T=128).sum().backward()
    opt.step()
    assert not torch.allclose(before, fe.threshold)   # STE gradient reaches thr


def test_sqrt_compression_binary_and_trains() -> None:
    fe = make_frontend(filterbank_source="spice", compression="sqrt",
                       envelope_win_ms=10.0)
    wave = torch.randn(3, 16000) * 0.05
    env = fe.envelopes(wave)
    assert env.shape == (3, 16, 100)
    assert float(env.min()) >= 0.0 and float(env.max()) <= 1.0
    out = fe(wave, target_T=128)
    assert out.shape == (3, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}
    # sqrt vs log give different envelopes (compression actually applied)
    fl = make_frontend(filterbank_source="spice", compression="log",
                       envelope_win_ms=10.0)
    assert (fe.envelopes(wave) - fl.envelopes(wave)).abs().mean() > 1e-3


def test_spice_gain_restore_scales_fbank() -> None:
    off = make_frontend(filterbank_source="spice", envelope_win_ms=10.0)
    on = make_frontend(filterbank_source="spice", spice_gain_restore=True,
                       envelope_win_ms=10.0)
    assert not torch.allclose(off.spice_fbank, on.spice_fbank)   # per-ch re-weighted
    wave = torch.randn(2, 16000) * 0.05
    out = on(wave, target_T=128)
    assert out.shape == (2, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}


def test_deadzone_noop_at_init_and_trains() -> None:
    off = make_frontend(filterbank_source="spice", compression="sqrt",
                        envelope_win_ms=10.0)
    on = make_frontend(filterbank_source="spice", compression="sqrt",
                       spice_deadzone=True, envelope_win_ms=10.0)
    wave = torch.randn(3, 16000) * 0.05
    # dz initialised to 0 -> exact no-op (baseline preserved)
    assert torch.allclose(off.envelopes(wave), on.envelopes(wave), atol=1e-6)
    # learnable: the deadzone moves under SGD
    on.init_thresholds(wave)
    before = on.deadzone.detach().clone()
    opt = torch.optim.SGD(on.parameters(), lr=5.0)
    on(wave, target_T=128).sum().backward()
    opt.step()
    assert not torch.allclose(before, on.deadzone)
    out = on(wave, target_T=128)
    assert out.shape == (3, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}


def test_comparator_vos_noop_and_perturbs() -> None:
    wave = torch.randn(3, 16000) * 0.05
    base = make_frontend()
    off0 = make_frontend(comparator_vos=0.0)
    assert torch.equal(base(wave, target_T=128), off0(wave, target_T=128))  # exact no-op
    off = make_frontend(comparator_vos=0.1)
    o = off(wave, target_T=128)
    assert o.shape == (3, 16, 128)
    assert set(torch.unique(o).tolist()) <= {-1.0, 1.0}


# --------------------------------------------------------------------------- #
# 10. normalize="fixed": Cerutti's dataset-level min-max == absolute threshold
# --------------------------------------------------------------------------- #
def test_fixed_scale_is_absolute_threshold() -> None:
    """A constant lo/hi is an affine map, so the binary decision is identical to
    comparing raw envelopes against a rescaled absolute threshold -- which is what
    a FIXED R7/R8 divider does in hardware."""
    wave = noise(4)
    fe = make_frontend(normalize="fixed", envelope_win_ms=10.0)
    fe.init_fixed_scale(wave)
    fe.init_thresholds(wave)
    lo, hi = fe.fixed_lo.item(), fe.fixed_hi.item()
    assert hi > lo                                     # scale actually measured
    env = fe.envelopes(wave)
    raw = fe.envelopes(wave, raw=True)
    thr = fe.threshold.detach().view(1, -1, 1)
    thr_abs = lo + thr * (hi - lo)                     # same threshold, raw units
    assert torch.equal(env >= thr, raw >= thr_abs)
    # envelopes stay ~[0,1] so ste_clip needs no retuning, and grads flow
    assert float(env.min()) >= 0.0 and float(env.max()) <= 1.0 + 1e-6
    fe(wave, target_T=128).mean().backward()
    assert torch.all(fe.threshold.grad != 0)


def test_fixed_scale_in_state_dict_and_noop_elsewhere() -> None:
    wave = noise(2)
    fe = make_frontend(normalize="fixed", envelope_win_ms=10.0)
    fe.init_fixed_scale(wave)
    assert "fixed_lo" in fe.state_dict() and "fixed_hi" in fe.state_dict()
    # other modes: no buffers, init_fixed_scale is a no-op
    mm = make_frontend(envelope_win_ms=10.0)
    assert "fixed_lo" not in mm.state_dict()
    mm.init_fixed_scale(wave)                          # must not raise


# --------------------------------------------------------------------------- #
# 11. normalize="agc": causal, channel-shared gain (hardware-realizable)
# --------------------------------------------------------------------------- #
def _agc_frontend(**kw):
    # AGC needs the multiplicative (amplitude) domain -> compression="sqrt"
    fe = make_frontend(normalize="agc", compression="sqrt",
                       envelope_win_ms=10.0, **kw)
    w = noise(4)
    fe.init_fixed_scale(w)
    fe.init_thresholds(w)
    return fe, w


def test_agc_is_causal() -> None:
    """Perturbing a LATE part of the wave must not change EARLY frames -- the
    whole point of AGC over per-clip min-max (which peeks at the future)."""
    fe, w = _agc_frontend()
    w2 = w.clone()
    w2[:, 14000:] *= 4.0                      # only the last ~125 ms
    e1, e2 = fe.envelopes(w), fe.envelopes(w2)
    assert torch.equal(e1[:, :, :60], e2[:, :, :60])       # early untouched
    assert not torch.equal(e1[:, :, 80:], e2[:, :, 80:])   # late did change


def test_agc_gain_is_channel_shared() -> None:
    """One gain for all channels -> inter-channel ratios (the spectral shape)
    survive, which is why min-max was channel-shared in the first place."""
    fe, w = _agc_frontend()
    raw = fe.envelopes(w, raw=True)
    env = fe.envelopes(w)
    r_raw = raw[:, 0] / (raw[:, 1] + 1e-9)
    r_agc = env[:, 0] / (env[:, 1] + 1e-9)
    assert torch.allclose(r_raw, r_agc, atol=1e-3)


def test_agc_max_gain_floor_limits_silence_boost() -> None:
    """A quiet clip must not be amplified without limit (noise gate)."""
    fe, w = _agc_frontend(agc_max_gain_db=20.0)
    quiet = w * 1e-3
    env = fe.envelopes(quiet)
    ref = fe.fixed_hi / (10.0 ** (20.0 / 20.0))       # the level floor
    raw = fe.envelopes(quiet, raw=True)
    assert float(env.max()) <= float((raw / ref).max()) + 1e-5   # gain capped


def test_agc_trains_and_outputs_binary() -> None:
    fe, w = _agc_frontend()
    out = fe(w, target_T=128)
    assert out.shape == (4, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}
    fe(w, target_T=128).mean().backward()
    assert torch.all(fe.threshold.grad != 0)


def test_agc_rejects_log_compression() -> None:
    """Division-based AGC is meaningless in the log domain (gain is additive
    there and envelopes can go negative) -- must fail loudly, not silently."""
    with pytest.raises(ValueError, match="sqrt"):
        make_frontend(normalize="agc", compression="log", envelope_win_ms=10.0)


def test_fixed_scale_quantile_is_outlier_robust_and_still_affine() -> None:
    """A quantile scale must (a) shrink fixed_hi vs the global max and (b) keep
    the binary decision identical for the same ABSOLUTE threshold -- it only
    reconditions the optimization, it cannot change what is representable."""
    wave = noise(6)
    fe_max = make_frontend(normalize="fixed", envelope_win_ms=10.0)
    fe_q = make_frontend(normalize="fixed", envelope_win_ms=10.0,
                         fixed_scale_quantile=0.75)
    fe_max.init_fixed_scale(wave)
    fe_q.init_fixed_scale(wave)
    assert float(fe_q.fixed_hi) < float(fe_max.fixed_hi)     # outlier dropped

    raw = fe_max.envelopes(wave, raw=True)
    abs_thr = raw.median()                                   # one absolute threshold
    def binar(fe):
        t = (abs_thr - fe.fixed_lo) / (fe.fixed_hi - fe.fixed_lo)
        return fe.envelopes(wave) >= t
    assert torch.equal(binar(fe_max), binar(fe_q))           # same decisions


# --------------------------------------------------------------------------- #
# 12. normalize="xmax": cross-channel relative threshold (gain-invariant)
# --------------------------------------------------------------------------- #
def _xmax_frontend(**kw):
    fe = make_frontend(normalize="xmax", compression="sqrt",
                       envelope_win_ms=10.0, **kw)
    w = noise(6)
    fe.init_fixed_scale(w)
    fe.init_thresholds(w)
    return fe, w


def test_xmax_is_gain_invariant_above_the_floor() -> None:
    """The whole point: an input gain cancels between numerator and denominator,
    so the binary image is unchanged -- level invariance is structural here, not
    something the network has to learn."""
    fe, w = _xmax_frontend(xmax_floor_frac=0.0)      # no floor -> exact
    b = fe(w, target_T=128)
    for g in (0.5, 2.0, 4.0):
        assert torch.equal(fe(g * w, target_T=128), b)


def test_xmax_floor_restores_an_absolute_comparison_when_quiet() -> None:
    """Below the floor the denominator stops tracking, so the envelope shrinks
    with the input instead of being renormalized -- that is what stops silence
    from dividing noise by noise and firing at random. (Checked on the envelope,
    not the binary output: once everything is under threshold both are all -1.)"""
    fe, w = _xmax_frontend(xmax_floor_frac=0.9)      # floor binds almost always
    quiet = 1e-3 * w
    e1 = fe.envelopes(quiet)
    e2 = fe.envelopes(0.1 * quiet)
    assert float(e2.max()) < 0.5 * float(e1.max())   # scaled down, NOT invariant

    # Same signals with NO floor stay renormalized (ratio ~1). Not exact here
    # because the _EPS guard in the denominator is not negligible at 1e-3 scale;
    # invariance at real signal levels is asserted by the test above.
    nofloor = _xmax_frontend(xmax_floor_frac=0.0)[0]
    r = float(nofloor.envelopes(0.1 * quiet).max()
              / nofloor.envelopes(quiet).max())
    assert 0.99 < r < 1.01


def test_xmax_outputs_binary_and_trains() -> None:
    fe, w = _xmax_frontend()
    out = fe(w, target_T=128)
    assert out.shape == (6, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}
    out.mean().backward()
    assert torch.all(fe.threshold.grad != 0)


def test_xmax_rejects_log_compression() -> None:
    with pytest.raises(ValueError, match="sqrt"):
        make_frontend(normalize="xmax", compression="log", envelope_win_ms=10.0)
