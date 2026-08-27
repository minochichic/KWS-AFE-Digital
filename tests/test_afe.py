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
import warnings

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


def test_minmax_is_gain_invariant_and_fixed_is_not() -> None:
    """The hardware threshold is a FIXED R7/R8 divider. It cannot track how loud
    the room is, so a louder sound MUST fire more comparators.

    normalize="minmax" rescales per clip, so doubling the volume leaves the
    binary image bit-identical -- a perfect AGC, and unbuildable. "fixed" divides
    by dataset-level constants, so level survives. This is the whole reason
    track 2 exists; it fails loudly if base.yaml is ever pointed at "minmax".
    """
    wave = noise(4)
    quiet, loud = wave * 0.5, wave * 2.0

    # sqrt to match configs/base.yaml; xlse requires it, and minmax is
    # invariant under either (log makes gain additive, sqrt multiplicative,
    # and a per-clip affine rescale cancels both).
    mm = make_frontend(normalize="minmax", compression="sqrt", envelope_win_ms=10.0)
    mm.init_thresholds(wave)
    with torch.no_grad():
        assert torch.equal(mm(quiet, target_T=128), mm(loud, target_T=128))

    for mode in ("fixed", "xlse"):                     # both absolute-threshold
        fe = make_frontend(normalize=mode, compression="sqrt",
                           envelope_win_ms=10.0)
        fe.init_fixed_scale(wave)
        fe.init_thresholds(wave)
        with torch.no_grad():
            a, b = fe(quiet, target_T=128), fe(loud, target_T=128)
        assert not torch.equal(a, b), f"{mode} is gain-invariant, so it is an AGC"
        assert (b > 0).float().mean() > (a > 0).float().mean(), \
            f"{mode}: louder input must not fire FEWER comparators"


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


def test_xmax_silence_does_not_fire_every_channel() -> None:
    """A zeroed clip must stay quiet, not saturate.

    sqrt compression bottoms out at sqrt(_EPS)=1e-3, so silence puts all 16
    channels on that guard TOGETHER; xmax divides it by itself and gets 1.0
    everywhere, i.e. no signal becomes the maximally active image. Measured on
    real Speech Commands at xmax_floor_frac=0.02 (floor == the guard exactly):
    a zeroed clip fired 100% of bits and a x0.01 clip fired MORE than speech
    (0.538 vs 0.386). A floor clear of the guard is what prevents it.
    """
    fe, _ = _xmax_frontend(xmax_floor_frac=0.5)
    out = fe(torch.zeros(2, fe.cfg.sample_rate), target_T=128)
    assert float((out > 0).float().mean()) == 0.0


def test_xmax_warns_when_the_floor_lands_on_the_compression_guard() -> None:
    """Guard the failure above: the floor is a quantile of real frames, so a
    dataset with enough silent frames can put it right on sqrt(_EPS) and
    silently disable the protection. That must be loud, not silent."""
    fe = make_frontend(normalize="xmax", compression="sqrt",
                       envelope_win_ms=10.0, xmax_floor_frac=0.1)
    with pytest.warns(UserWarning, match="compression guard"):
        fe.init_fixed_scale(torch.zeros(6, fe.cfg.sample_rate))


# --------------------------------------------------------------------------- #
# 12b. normalize="xmix": the same idea in the form the circuit produces
# --------------------------------------------------------------------------- #
def _xmix_frontend(**kw):
    fe = make_frontend(normalize="xmix", compression="sqrt",
                       envelope_win_ms=10.0, **kw)
    w = noise(6)
    fe.init_fixed_scale(w)
    fe.init_thresholds(w)
    return fe, w


def test_xmix_matches_the_resistor_divider_algebra() -> None:
    """The circuit can only mix V_max and V_ref linearly, so the firing rule is
    env_c > a*S + (1-a)*d. Check the implementation is exactly that rearranged,
    since a mismatch here is what would mis-set every divider on the board."""
    fe, w = _xmix_frontend(xmax_floor_frac=0.1)
    raw = fe.envelopes(w, raw=True)
    s = raw.amax(dim=1, keepdim=True)
    d = float(fe.xmax_floor)
    a = fe.threshold.view(1, -1, 1)

    fired = fe(w) > 0                                    # what the model does
    hw = raw > (a * s + (1.0 - a) * d)                    # what the board does
    # ties aside, the two rules must agree everywhere the denominator is valid
    ok = (s > d).expand_as(fired)
    assert torch.equal(fired[ok], hw[ok])


def test_xmix_silence_threshold_depends_on_alpha_the_other_way() -> None:
    """The whole reason this mode exists. Under xmax a silent frame's effective
    threshold is a*floor (rises with alpha); in the circuit it is (1-a)*d (falls
    with alpha). Alphas learned under xmax would therefore be mis-set per channel
    once built -- the same class of trap as the R7/R8 mapping."""
    fe, _ = _xmix_frontend(xmax_floor_frac=0.1)
    d = float(fe.xmax_floor)
    quiet = torch.full((1, fe.cfg.n_channels, 4), 0.5 * d)   # every channel < d
    out = fe._xmix(quiet)
    assert bool((out < 0).all())          # numerator negative -> firmly "off"


def test_xmix_is_immune_to_the_sqrt_guard_degeneracy() -> None:
    """xmax needed a floor because guard/guard = 1 fires everything. Subtracting
    d instead makes the numerator negative there, so silence is off by
    construction rather than by picking the floor correctly."""
    fe, _ = _xmix_frontend(xmax_floor_frac=0.1)
    out = fe(torch.zeros(2, fe.cfg.sample_rate), target_T=128)
    assert float((out > 0).float().mean()) == 0.0


def test_xmix_agrees_with_xmax_on_loud_frames() -> None:
    """Both forms reduce to env/S once S >> d, so switching modes must not move
    the binary image where the signal actually lives.

    The floor has to be set explicitly rather than left at its quantile: white
    noise has near-uniform frame energy, so a 10% quantile floor lands right on
    the typical S and the S >> d regime this test is about never happens.
    """
    w = noise(6)
    fes = {}
    for mode in ("xmax", "xmix"):
        fe = make_frontend(normalize=mode, compression="sqrt",
                           envelope_win_ms=10.0, xmax_floor_frac=0.1)
        fe.init_fixed_scale(w)
        fe.init_thresholds(w)
        fes[mode] = fe
    typical_s = float(fes["xmax"].envelopes(w, raw=True).amax(dim=1).median())
    for fe in fes.values():                              # d = S / 100 -> S >> d
        fe.xmax_floor.fill_(typical_s / 100.0)
    fes["xmix"].init_thresholds(w)
    fes["xmax"].init_thresholds(w)
    fes["xmix"].threshold.data.copy_(fes["xmax"].threshold.data)
    agree = (fes["xmax"](w) == fes["xmix"](w)).float().mean()
    assert float(agree) > 0.98


def test_xmix_rejects_log_compression() -> None:
    with pytest.raises(ValueError, match="sqrt"):
        make_frontend(normalize="xmix", compression="log", envelope_win_ms=10.0)


def test_binarize_false_passes_the_continuous_envelope() -> None:
    """Diagnostic switch: without the comparator the network must see the real
    envelope, so that "the network is weak" can be told apart from "the 1-bit
    input is lossy". Everything else (filterbank, normalization) is untouched."""
    fe = make_frontend(normalize="minmax", envelope_win_ms=10.0, binarize=False)
    w = noise(4)
    out = fe(w, target_T=128)
    assert out.shape == (4, 16, 128)
    assert len(torch.unique(out)) > 2                    # NOT binary
    assert torch.allclose(fe(w), fe.envelopes(w))        # unpadded: passthrough
    # No comparator means the threshold is unused, so the front end has no
    # trainable parameter left -- it becomes a fixed feature extractor and only
    # the network learns. That is the point: the network is the thing under test.
    assert not out.requires_grad


def test_binarize_true_is_the_unchanged_default() -> None:
    fe = make_frontend(normalize="minmax", envelope_win_ms=10.0)
    assert fe.cfg.binarize is True
    assert set(torch.unique(fe(noise(4), target_T=128)).tolist()) <= {-1.0, 1.0}


def test_xmax_rejects_log_compression() -> None:
    with pytest.raises(ValueError, match="sqrt"):
        make_frontend(normalize="xmax", compression="log", envelope_win_ms=10.0)


# --------------------------------------------------------------------------- #
# 14. spice_matrix_path survives the analog/ reorganization
# --------------------------------------------------------------------------- #
def test_spice_path_falls_back_to_the_pre_analog_layout() -> None:
    """Runs and checkpoints recorded "AFE/artifacts/..." before the analog
    folders moved under analog/. Those configs must keep loading -- otherwise a
    repo reorganization silently breaks every saved run."""
    from data.afe import _resolve_spice_path
    good = _resolve_spice_path("analog/AFE/artifacts/filterbank_matrix.csv")
    assert good.exists()
    with pytest.warns(UserWarning, match="moved under analog/"):
        old = _resolve_spice_path("AFE/artifacts/filterbank_matrix.csv")
    assert old == good


def test_spice_path_missing_says_how_to_fix_it() -> None:
    from data.afe import _resolve_spice_path
    with pytest.raises(FileNotFoundError, match="git pull"):
        _resolve_spice_path("nowhere/filterbank_matrix.csv")


def test_xmix_alpha_stays_buildable_during_training() -> None:
    """alpha is a resistor ratio Rb/(Ra+Rb); outside [0,1] nothing can build it,
    and above 1 the channel is dead because the normalized value cannot exceed 1.
    Clamping at init was not enough -- a real run came back with alpha at 1.344.
    """
    fe = make_frontend(normalize="xmix", compression="sqrt", envelope_win_ms=10.0)
    w = noise(4)
    fe.init_fixed_scale(w)
    fe.init_thresholds(w)
    with torch.no_grad():                                # walk it out of range
        fe.threshold.copy_(torch.linspace(-0.6, 1.6, fe.cfg.n_channels))
    env = fe.envelopes(w)
    eff = torch.where(env >= fe.threshold.view(1, -1, 1).clamp(0, 1), 1.0, -1.0)
    assert torch.equal(fe(w), eff)                        # forward uses [0,1]

    # and the gradient still reaches the out-of-range channels, so they can
    # return -- a hard clamp would strand them on the boundary forever.
    fe.threshold.grad = None
    fe(w).mean().backward()
    assert torch.all(fe.threshold.grad != 0)


def test_effective_alpha_is_what_the_comparator_uses() -> None:
    """xmix clamps alpha straight-through, so the raw parameter drifts outside
    [0,1] while the circuit only sees the clamped value. Reading the parameter
    directly made a finished run look unbuildable (alpha 1.341) when it was
    pinned at 1.0. Export must go through effective_alpha()."""
    fe = make_frontend(normalize="xmix", compression="sqrt", envelope_win_ms=10.0)
    w = noise(4)
    fe.init_fixed_scale(w); fe.init_thresholds(w)
    with torch.no_grad():
        fe.threshold.copy_(torch.linspace(-0.6, 1.6, fe.cfg.n_channels))
    a = fe.effective_alpha()
    assert torch.all(a >= 0) and torch.all(a <= 1)
    assert torch.equal(a, fe.threshold.detach().clamp(0, 1))
    # and it matches the decision the model actually makes
    env = fe.envelopes(w)
    assert torch.equal(fe(w), torch.where(env >= a.view(1, -1, 1), 1.0, -1.0))


def test_effective_alpha_is_a_passthrough_for_other_modes() -> None:
    fe = make_frontend(normalize="minmax", envelope_win_ms=10.0)
    fe.init_thresholds(noise(4))
    assert torch.equal(fe.effective_alpha(), fe.threshold.detach())


# --------------------------------------------------------------------------- #
# 15. k comparators per channel -> k-bit thermometer code
# --------------------------------------------------------------------------- #
def test_one_comparator_is_the_untouched_default() -> None:
    """k=1 must be byte-identical to before this existed, including the
    parameter shape, so every checkpoint already on disk still loads."""
    fe = make_frontend(envelope_win_ms=10.0)
    assert fe.cfg.comparators_per_channel == 1
    assert fe.threshold.shape == (fe.cfg.n_channels,)
    assert fe(noise()).shape == (2, 16, 100)


def test_k_comparators_give_k_rows_per_channel() -> None:
    fe = make_frontend(envelope_win_ms=10.0, comparators_per_channel=2)
    out = fe(noise(), target_T=128)
    assert out.shape == (2, 32, 128)                      # 16 channels x 2 bits
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}  # still 1-bit each
    assert fe.threshold.shape == (32,)


def test_comparator_rows_are_channel_major() -> None:
    """A channel's bits must be adjacent (ch0_t0, ch0_t1, ch1_t0, ...), so the
    two rows of one physical channel stay together for the FPGA and so the
    per-channel view in export is a plain reshape(C, k)."""
    fe = make_frontend(envelope_win_ms=10.0, comparators_per_channel=2)
    w = noise()
    with torch.no_grad():                                 # bit0 always on, bit1 off
        fe.threshold.copy_(torch.tensor([-10.0, 10.0] * fe.cfg.n_channels))
    out = fe(w)
    assert torch.all(out[:, 0::2] == 1.0)
    assert torch.all(out[:, 1::2] == -1.0)


def test_two_comparators_do_not_start_on_top_of_each_other() -> None:
    """Identical thresholds would encode one bit twice and waste the second
    comparator (and its 1.04 uW). Init spreads them over the channel's own
    distribution."""
    fe = make_frontend(envelope_win_ms=10.0, comparators_per_channel=2)
    fe.init_thresholds(noise(6))
    per_ch = fe.threshold.detach().view(fe.cfg.n_channels, 2)
    assert torch.all(per_ch[:, 1] > per_ch[:, 0])         # ordered, distinct


def test_all_comparator_thresholds_receive_gradient() -> None:
    fe = make_frontend(envelope_win_ms=10.0, comparators_per_channel=2)
    fe.init_thresholds(noise(4))
    fe(noise(4)).mean().backward()
    assert torch.all(fe.threshold.grad != 0)


def test_config_requires_in_channels_to_follow_comparator_count() -> None:
    from train.config import Config, ModelConfig
    import pytest as _pytest
    with _pytest.raises(ValueError, match="comparators_per_channel"):
        Config(afe=AFEConfig(n_channels=16, comparators_per_channel=2),
               model=ModelConfig(in_channels=16)).validate()


# --------------------------------------------------------------------------- #
# 16. normalize="xlse": the SOFT max a diode-OR actually produces
# --------------------------------------------------------------------------- #
def _xlse_frontend(**kw):
    fe = make_frontend(normalize="xlse", compression="sqrt",
                       envelope_win_ms=10.0, xmax_floor_frac=0.02, **kw)
    w = noise(6)
    fe.init_fixed_scale(w)
    fe.init_thresholds(w)
    return fe, w


def test_xlse_collapses_to_xmix_as_temperature_goes_to_zero() -> None:
    """log-sum-exp IS the max at T->0, so a vanishing temperature must reproduce
    xmix exactly -- that is the check that the soft-max is the same rule with a
    knob, not a different one."""
    fe, w = _xlse_frontend(lse_temp_frac=1e-6)
    ref = make_frontend(normalize="xmix", compression="sqrt",
                        envelope_win_ms=10.0, xmax_floor_frac=0.02)
    ref.init_fixed_scale(w); ref.init_thresholds(w)
    ref.threshold.data.copy_(fe.threshold.data)
    assert torch.allclose(fe.envelopes(w), ref.envelopes(w), atol=1e-4)
    assert torch.equal(fe(w), ref(w))


def test_xlse_denominator_never_undershoots_the_max() -> None:
    """LSE >= max, always. That is what keeps the normalized value <= 1 and so
    keeps alpha a buildable [0,1] divider ratio even when the max is soft."""
    fe, w = _xlse_frontend(lse_temp_frac=0.52)
    raw = fe.envelopes(w, raw=True)
    T = fe.lse_temp
    lse = torch.logsumexp(raw / T, dim=1) * T
    assert torch.all(lse >= raw.amax(dim=1) - 1e-5)
    assert float(fe.envelopes(w).max()) <= 1.0 + 1e-5


def test_xlse_temperature_scales_with_a_typical_frame_not_the_dataset_peak() -> None:
    """How soft the diode-OR is depends on T relative to the signal it compares.
    Anchoring to the dataset peak would repeat the floor's mistake -- that peak
    sits ~200x above a median frame."""
    fe, w = _xlse_frontend(lse_temp_frac=0.5)
    raw = fe.envelopes(w, raw=True)
    typical = float(raw.amax(dim=1).median())
    assert abs(float(fe.lse_temp) - 0.5 * typical) < 1e-6
    assert float(fe.lse_temp) < 0.5 * float(raw.max())     # far below the peak


def test_xlse_softer_temperature_inflates_the_denominator() -> None:
    """The failure this mode exists to model: a warmer diode makes the OR node
    overshoot the true max, so the threshold every channel is judged against
    rises. Measured on real speech it reaches 1.9-2.5x at realistic swings."""
    fe_hard, w = _xlse_frontend(lse_temp_frac=0.01)
    fe_soft, _ = _xlse_frontend(lse_temp_frac=0.8)
    raw = fe_hard.envelopes(w, raw=True)
    mx = raw.amax(dim=1)
    r = lambda fe: float(((torch.logsumexp(raw / fe.lse_temp, dim=1)
                           * fe.lse_temp) / mx).median())
    assert r(fe_hard) < 1.02 < 1.5 < r(fe_soft)


def test_xlse_outputs_binary_and_keeps_alpha_buildable() -> None:
    fe, w = _xlse_frontend(lse_temp_frac=0.52)
    out = fe(w, target_T=128)
    assert out.shape == (6, 16, 128)
    assert set(torch.unique(out).tolist()) <= {-1.0, 1.0}
    a = fe.effective_alpha()
    assert torch.all(a >= 0) and torch.all(a <= 1)


def test_xlse_rejects_log_compression() -> None:
    with pytest.raises(ValueError, match="sqrt"):
        make_frontend(normalize="xlse", compression="log", envelope_win_ms=10.0)


# --------------------------------------------------------------------------- #
# 17. load_afe_state: old checkpoints must stay re-scorable, but only exactly
# --------------------------------------------------------------------------- #
def _with_deadzone(**kw):
    """A frontend that HAS the deadzone parameter, to mint a legacy state_dict."""
    return make_frontend(spice_deadzone=True, envelope_win_ms=10.0, **kw)


def test_load_afe_state_drops_a_zero_deadzone_under_sqrt() -> None:
    """The key exists only because it used to be registered unconditionally.
    All-zero + sqrt means relu(mel) is the identity, so dropping is exact."""
    from data.afe import load_afe_state
    old = _with_deadzone(compression="sqrt")
    new = make_frontend(compression="sqrt", envelope_win_ms=10.0)
    assert "deadzone" in old.state_dict() and "deadzone" not in new.state_dict()

    w = noise(4)
    old.init_fixed_scale(w); old.init_thresholds(w)
    dropped = load_afe_state(new, old.state_dict())
    assert dropped == ["deadzone"]
    # exact, not approximate: the two frontends must agree bit for bit
    assert torch.equal(new(w), old(w))


def test_load_afe_state_refuses_a_nonzero_deadzone() -> None:
    """A trained deadzone is part of the model. Dropping it would silently
    re-score the run through a different front end."""
    from data.afe import load_afe_state
    old = _with_deadzone(compression="sqrt")
    old.deadzone.data.fill_(0.01)
    new = make_frontend(compression="sqrt", envelope_win_ms=10.0)
    with pytest.raises(RuntimeError, match="deadzone"):
        load_afe_state(new, old.state_dict())


def test_load_afe_state_refuses_log_where_relu_is_not_identity() -> None:
    """Under log compression mel goes negative, so relu(mel) clips instead of
    passing through -- and the checkpoint cannot tell us whether it was on."""
    from data.afe import load_afe_state
    old = _with_deadzone(compression="log")
    new = make_frontend(compression="log", envelope_win_ms=10.0)
    with pytest.raises(RuntimeError, match="deadzone"):
        load_afe_state(new, old.state_dict())


def test_load_afe_state_still_catches_a_genuinely_wrong_checkpoint() -> None:
    """Tolerance is scoped to one key. A shape mismatch must still fail."""
    from data.afe import load_afe_state
    src = make_frontend(n_channels=16, envelope_win_ms=10.0)
    dst = make_frontend(n_channels=8, envelope_win_ms=10.0)
    with pytest.raises(RuntimeError):
        load_afe_state(dst, src.state_dict())


# --------------------------------------------------------------------------- #
# 18. delta stability: the number the analog side has to build
# --------------------------------------------------------------------------- #
def test_collect_init_batch_gathers_enough_clips() -> None:
    """One batch is enough for a channel mean and not for a 2% quantile."""
    from data.afe import collect_init_batch
    batches = [(torch.randn(8, 16000), torch.zeros(8, dtype=torch.long))
               for _ in range(10)]
    w = collect_init_batch(batches, n_clips=40)
    assert w.shape[0] == 40                      # whole batches, exactly 5 of 8
    assert collect_init_batch(batches, n_clips=1).shape[0] == 8   # 최소 한 배치
    assert collect_init_batch(batches, n_clips=10**6).shape[0] == 80  # 있는 만큼


def test_more_clips_make_the_floor_hold_still() -> None:
    """The point of the change: delta must stop depending on which clips the
    shuffle happened to hand over."""
    from data.afe import collect_init_batch

    def floor_from(waves):
        fe = make_frontend(normalize="xmix", compression="sqrt",
                           envelope_win_ms=10.0, xmax_floor_frac=0.05)
        fe.init_fixed_scale(waves)
        return float(fe.xmax_floor)

    torch.manual_seed(0)
    pool = [(torch.randn(16, 16000) * torch.rand(16, 1), None)
            for _ in range(24)]                  # clips of widely varying level
    small = [floor_from(pool[i][0]) for i in range(6)]
    big = [floor_from(collect_init_batch(pool[i * 6:(i + 1) * 6], 10 ** 6))
           for i in range(4)]
    spread = lambda v: max(v) / max(min(v), 1e-12)
    assert spread(big) < spread(small)


def test_floor_on_the_guard_warns_for_divider_forms() -> None:
    """xmix/xlse do not misfire on the guard -- but delta then measures zero
    padding instead of quiet speech, and it is a circuit spec."""
    fe = make_frontend(normalize="xmix", compression="sqrt",
                       envelope_win_ms=10.0, xmax_floor_frac=0.30)
    w = noise(8)
    w[:6] = 0.0                                  # 75% silent -> quantile in the atom
    with pytest.warns(UserWarning, match="guard"):
        fe.init_fixed_scale(w)


def test_no_warning_when_the_floor_clears_the_guard() -> None:
    fe = make_frontend(normalize="xmix", compression="sqrt",
                       envelope_win_ms=10.0, xmax_floor_frac=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fe.init_fixed_scale(noise(8))


def test_chunking_does_not_change_the_estimates() -> None:
    """Init now sees thousands of clips, so the STFT is chunked. The numbers
    must be identical -- every statistic is taken after concatenation."""
    w = noise(40)
    a = make_frontend(normalize="xmix", compression="sqrt",
                      envelope_win_ms=10.0, xmax_floor_frac=0.05)
    b = make_frontend(normalize="xmix", compression="sqrt",
                      envelope_win_ms=10.0, xmax_floor_frac=0.05)
    assert torch.equal(a._envelopes_chunked(w, raw=True, chunk=7),
                       a.envelopes(w, raw=True))
    a.init_fixed_scale(w); a.init_thresholds(w)
    b.init_fixed_scale(w); b.init_thresholds(w)
    assert float(a.xmax_floor) == float(b.xmax_floor)
    assert torch.equal(a.threshold.data, b.threshold.data)


# ---- threshold_init, which was declared and never read until 2026-08-23 ----

def _tiny_afe(**over):
    """An AFE small enough to init in a test, with overrides applied."""
    from train.config import AFEConfig
    from data.afe import AFEFrontend
    cfg = AFEConfig(n_channels=4, normalize="minmax", **over)
    return AFEFrontend(cfg).eval()


def _skewed_batch(n: int = 24, sr: int = 16000, seed: int = 0):
    """Speech-like in the way that matters here: mostly quiet, rarely loud."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, sr, generator=g) * 0.01
    for i in range(n):                       # a short burst in each clip
        a = int(torch.randint(0, sr - 2000, (1,), generator=g))
        w[i, a:a + 1500] += torch.randn(1500, generator=g) * 0.5
    return w


def test_threshold_init_mean_and_quantile_differ_on_skewed_data() -> None:
    """The whole reason the option exists. If these came out equal the choice
    would be cosmetic, and the fx_d0 measurement says they are not."""
    w = _skewed_batch()
    a = _tiny_afe(threshold_init="channel_mean")
    a.init_thresholds(w)
    mean_thr = a.threshold.detach().clone()

    b = _tiny_afe(threshold_init="quantile", threshold_init_quantile=0.5)
    b.init_thresholds(w)
    med_thr = b.threshold.detach().clone()

    assert torch.all(mean_thr > med_thr), (
        "on right-skewed data the mean must sit above the median")


def test_the_quantile_lands_where_it_says() -> None:
    """A threshold at quantile q must leave 1-q of the envelopes above it."""
    w = _skewed_batch(seed=1)
    for q in (0.25, 0.5, 0.9):
        a = _tiny_afe(threshold_init="quantile", threshold_init_quantile=q)
        a.init_thresholds(w)
        env = a.envelopes(w, raw=False)
        for c in range(env.shape[1]):
            below = float((env[:, c] <= a.threshold[c]).double().mean())
            assert abs(below - q) < 0.02, f"channel {c} at q={q}: {below:.3f}"


def test_the_default_is_byte_identical_to_before_the_option_existed() -> None:
    """Every recorded number came from the mean. Adding a branch must not move
    the baseline, or the run table stops meaning what it says."""
    w = _skewed_batch(seed=2)
    a = _tiny_afe()                                   # no override at all
    a.init_thresholds(w)
    env = a.envelopes(w, raw=False)
    assert torch.allclose(a.threshold, env.mean(dim=(0, 2)))


def test_an_unknown_threshold_init_is_refused_not_ignored() -> None:
    """The failure this replaces: the field was never read, so a config asking
    for something else got the mean and said nothing."""
    w = _skewed_batch(seed=3)
    a = _tiny_afe()
    a.cfg.threshold_init = "median"                   # plausible, and wrong
    with pytest.raises(ValueError, match="threshold_init"):
        a.init_thresholds(w)


def test_config_rejects_a_quantile_outside_the_open_interval() -> None:
    from train.config import Config
    import dataclasses
    for bad in (0.0, 1.0, -0.1, 1.5):
        cfg = Config()
        cfg.afe = dataclasses.replace(cfg.afe, threshold_init="quantile",
                                      threshold_init_quantile=bad)
        with pytest.raises(ValueError, match="threshold_init_quantile"):
            cfg.validate()


def test_measured_thresholds_are_loaded_and_frozen(tmp_path) -> None:
    """A built board's trip points go in as-is, and training must not move them.

    This is what makes per-board weights the answer to comparator offset: the
    offset need not be cancelled, only known. Whatever the divider and the
    comparator's own Vos add up to is one constant per channel once the board
    exists, so it is measured and trained against.
    """
    vals = [0.05 + 0.01 * i for i in range(16)]
    p = tmp_path / "trip.csv"
    p.write_text("\n".join(str(v) for v in vals))

    wave = noise(4)
    fe = make_frontend(normalize="fixed", compression="sqrt",
                       envelope_win_ms=10.0, threshold_init="measured",
                       threshold_measured_path=str(p),
                       threshold_trainable=False)
    fe.init_fixed_scale(wave)
    fe.init_thresholds(wave)
    assert torch.allclose(fe.threshold.detach(),
                          torch.tensor(vals, dtype=torch.float32), atol=1e-6)
    assert not fe.threshold.requires_grad          # hardware decided these

    # wrong count and out-of-range values must fail loudly, not be padded or
    # rescaled: both mean the volts->normalised conversion went wrong.
    bad = tmp_path / "short.csv"
    bad.write_text("\n".join(str(v) for v in vals[:8]))
    with pytest.raises(ValueError, match="8개"):
        fe2 = make_frontend(normalize="fixed", compression="sqrt",
                            envelope_win_ms=10.0, threshold_init="measured",
                            threshold_measured_path=str(bad))
        fe2.init_fixed_scale(wave)
        fe2.init_thresholds(wave)

    mv = tmp_path / "mv.csv"                       # mV pasted in by mistake
    mv.write_text("\n".join(str(v * 1000) for v in vals))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        fe3 = make_frontend(normalize="fixed", compression="sqrt",
                            envelope_win_ms=10.0, threshold_init="measured",
                            threshold_measured_path=str(mv))
        fe3.init_fixed_scale(wave)
        fe3.init_thresholds(wave)
