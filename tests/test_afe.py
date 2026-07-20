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

from data.afe import AFEFrontend, pad_or_crop
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
