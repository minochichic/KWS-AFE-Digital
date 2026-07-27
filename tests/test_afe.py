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
