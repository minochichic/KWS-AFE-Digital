"""BinaryMatchboxNet assembly tests (FIRST_TASK.md step 4).

What must hold:

1. [B, 16, T] in -> [B, 12] out, for every (C, T) sweep point -- all sizes
   come from the config, none from the code.
2. The precision map is structurally real: conv1/conv3 are QuantConv1d,
   B1-B3 and conv2 contain only BinaryConv1d, conv4 is a plain conv.
3. Every stage boundary feeding a binary stage carries {-1,+1} exactly.
4. The residual joins the pointwise accumulator in the *integer* domain and
   is followed by one BN/sign -- the FPGA can only add ints and threshold once.
5. Gradients reach every trainable parameter through the full STE stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.binary_matchboxnet import BinaryMatchboxNet, BinaryTCSBlock
from models.binary_ops import BinaryConv1d
from models.quant_ops import QuantConv1d
from train.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"


def build(**overrides) -> BinaryMatchboxNet:
    cfg = load_config(BASE, overrides or None)
    return BinaryMatchboxNet(cfg.model)


def binary_image(batch: int, ch: int, T: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 2, (batch, ch, T)).float() * 2 - 1


# --------------------------------------------------------------------------- #
# 1. shapes, config-driven sizing
# --------------------------------------------------------------------------- #
def test_forward_shape_base_config() -> None:
    model = build()
    out = model(binary_image(2, 16, 128))
    assert out.shape == (2, 12)


@pytest.mark.parametrize("C,T", [(16, 40), (32, 64), (64, 96)])
def test_forward_shape_sweep_points(C: int, T: int) -> None:
    model = build(**{"model.C": C, "model.T": T})
    assert model(binary_image(2, 16, T)).shape == (2, 12)


def test_widths_follow_C() -> None:
    model = build(**{"model.C": 32})
    assert model.stages["conv1"].conv.out_channels == 64      # 2*C
    b1 = model.stages["b1"]
    assert b1.subs[0].pw.out_channels == 32                   # C
    assert model.stages["conv4"].conv.out_channels == 12


def test_param_count_independent_of_T() -> None:
    """Conv sizes must never depend on T -- catches accidental T hardcoding."""
    p64 = sum(p.numel() for p in build(**{"model.T": 64}).parameters())
    p128 = sum(p.numel() for p in build(**{"model.T": 128}).parameters())
    assert p64 == p128


def test_conv1_param_count_exact() -> None:
    conv = build().stages["conv1"].conv
    assert conv.weight.numel() == 16 * 128 * 11
    assert conv.bias is None                                  # BN follows


# --------------------------------------------------------------------------- #
# 2. precision map is structurally real
# --------------------------------------------------------------------------- #
def test_precision_module_types() -> None:
    model = build()
    assert isinstance(model.stages["conv1"].conv, QuantConv1d)
    assert isinstance(model.stages["conv3"].conv, QuantConv1d)
    # conv4: plain full-precision conv (fixed-point conversion is export's job)
    conv4 = model.stages["conv4"].conv
    assert type(conv4) is torch.nn.Conv1d
    for name in ("b1", "b2", "b3"):
        blk = model.stages[name]
        for sub in blk.subs:
            assert isinstance(sub.dw, BinaryConv1d)
            assert isinstance(sub.pw, BinaryConv1d)
    assert isinstance(model.stages["conv2"].conv, BinaryConv1d)


def test_no_binary_conv_hides_in_first_or_last_stage() -> None:
    model = build()
    for stage_name in ("conv1", "conv4"):
        for m in model.stages[stage_name].modules():
            assert not isinstance(m, BinaryConv1d), (
                f"{stage_name} must never contain a binary conv (CLAUDE.md 2.2)"
            )


def test_tcs_block_rejects_non_binary_precision() -> None:
    cfg = load_config(BASE)
    cfg.model.stages[1].precision = "int8"     # b1 -- validate() allows it,
    with pytest.raises(ValueError, match="binary"):
        BinaryMatchboxNet(cfg.model)           # the model must not.


# --------------------------------------------------------------------------- #
# 3. binary domain at stage boundaries
# --------------------------------------------------------------------------- #
def test_binary_stages_receive_pm1_inputs() -> None:
    model = build().eval()
    seen: dict = {}

    def grab(name):
        def hook(_mod, inputs):
            seen[name] = inputs[0].detach()
        return hook

    for name in ("b1", "b2", "b3", "conv2"):
        model.stages[name].register_forward_pre_hook(grab(name))
    model(binary_image(2, 16, 128))

    for name, x in seen.items():
        vals = set(x.unique().tolist())
        assert vals <= {-1.0, 1.0}, f"{name} input contains {vals}"


def test_conv3_receives_continuous_epilogue_features() -> None:
    """conv2 -> conv3 stays in a continuous domain (ReLU, not sign).

    Cerutti IV-D feeds the last binary layer's fixed-point values straight to
    the classifier; binarizing here would throw away exactly the information
    the non-binary epilogue exists to preserve (CLAUDE.md 2.2 rule 1).
    """
    model = build().eval()
    seen = {}
    model.stages["conv3"].register_forward_pre_hook(
        lambda _m, inp: seen.setdefault("x", inp[0].detach()))
    model(binary_image(2, 16, 128))
    x = seen["x"]
    assert torch.all(x >= 0)                     # ReLU output
    assert x.unique().numel() > 2                # not collapsed to binary


# --------------------------------------------------------------------------- #
# 4. residual semantics (CLAUDE.md 2.3)
# --------------------------------------------------------------------------- #
def test_residual_projection_only_where_channels_change() -> None:
    model = build()
    assert isinstance(model.stages["b1"].skip, BinaryConv1d)   # 128 -> 64
    assert isinstance(model.stages["b2"].skip, torch.nn.Identity)
    assert isinstance(model.stages["b3"].skip, torch.nn.Identity)


def test_residual_accumulator_is_integer_domain() -> None:
    """Input to the final BN of each block = pw accumulator + residual.

    The FPGA adds two integer values there; if the software version is not
    integer-valued, BN-to-threshold fusion (Cerutti eq. 3) breaks. This is
    why the final pointwise and the skip projection get scale=False even when
    alpha scaling is on elsewhere.
    """
    model = build().eval()
    grabbed = {}
    for name in ("b1", "b2", "b3"):
        blk = model.stages[name]
        blk.post_bns[-1].register_forward_pre_hook(
            lambda _m, inp, key=name: grabbed.setdefault(key, inp[0].detach()))
    model(binary_image(2, 16, 128))

    for name, acc in grabbed.items():
        assert torch.allclose(acc, acc.round(), atol=1e-4), (
            f"{name}: residual accumulator is not integer-valued"
        )


def test_residual_joins_before_the_final_bn() -> None:
    """input(final BN) == pw_accumulator + skip, exactly.

    Guards the ordering itself: adding the residual after BN would still show
    an integer-valued accumulator at the BN input, but the block would then
    need a second threshold in hardware, violating CLAUDE.md 2.3's
    "add in the integer accumulator, then threshold once".
    """
    model = build().eval()
    blk = model.stages["b1"]
    got = {}
    blk.subs[-1].register_forward_hook(
        lambda _m, _i, out: got.setdefault("acc", out.detach()))
    blk.skip.register_forward_hook(
        lambda _m, _i, out: got.setdefault("res", out.detach()))
    blk.post_bns[-1].register_forward_pre_hook(
        lambda _m, inp: got.setdefault("bn_in", inp[0].detach()))

    model(binary_image(2, 16, 128))
    assert torch.allclose(got["bn_in"], got["acc"] + got["res"], atol=1e-5)


def test_final_pw_and_skip_are_unscaled() -> None:
    model = build()
    b1 = model.stages["b1"]
    assert b1.subs[-1].pw.scale is False
    assert b1.skip.scale is False
    # non-final pointwise may keep alpha (its own BN absorbs it exactly)
    assert b1.subs[0].pw.scale is True


# --------------------------------------------------------------------------- #
# 5. gradient flow end-to-end
# --------------------------------------------------------------------------- #
def test_gradients_reach_every_parameter() -> None:
    model = build()
    out = model(binary_image(4, 16, 128))
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 3, 7, 11]))
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert missing == [], f"no gradient reached: {missing}"


def test_afe_to_model_pipeline() -> None:
    """Full chain: waveform -> AFE -> model -> logits, gradients everywhere."""
    from data.afe import AFEFrontend
    cfg = load_config(BASE)
    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)

    torch.manual_seed(0)
    wave = 0.1 * torch.randn(2, 16000)
    # base.yaml is normalize="fixed" now, whose lo/hi are dataset constants.
    # Skipping the init leaves them at 0/1 and the placeholder threshold 0.5,
    # so with ste_clip=0.003 no sample lands inside the STE window and the
    # threshold gradient is silently zero. train/train.py runs these two in
    # this order; a test that skips them is testing a path nobody trains on.
    afe.init_fixed_scale(wave)
    afe.init_thresholds(wave)
    logits = model(afe(wave, target_T=cfg.model.T))
    assert logits.shape == (2, 12)

    logits.sum().backward()
    assert afe.threshold.grad is not None and torch.any(afe.threshold.grad != 0)
    assert model.stages["conv1"].conv.weight.grad is not None


# --------------------------------------------------------------------------- #
# 6. conv2 ablation knob (CLAUDE.md 2.2)
# --------------------------------------------------------------------------- #
def test_conv2_int8_ablation_builds_and_runs() -> None:
    cfg = load_config(BASE)
    conv2 = next(s for s in cfg.model.stages if s.name == "conv2")
    conv2.precision = "int8"
    cfg.validate()
    model = BinaryMatchboxNet(cfg.model)
    assert isinstance(model.stages["conv2"].conv, QuantConv1d)
    assert model(binary_image(2, 16, 128)).shape == (2, 12)


def test_conv2_default_is_separable_dense_is_fallback() -> None:
    """Decision 2026-07-21 (user-approved, see CLAUDE.md 2.2): conv2 defaults
    to separable (~96.5K total, matching the paper's 93K); dense (324K) stays
    available as the first ablation if accuracy misses the 85% target."""
    cfg = load_config(BASE)
    conv2 = next(s for s in cfg.model.stages if s.name == "conv2")
    assert conv2.separable is True

    sep_params = sum(p.numel() for p in BinaryMatchboxNet(cfg.model).parameters())

    conv2.separable = False                      # the recorded fallback
    model = BinaryMatchboxNet(cfg.model)
    dense_params = sum(p.numel() for p in model.parameters())

    assert model(binary_image(2, 16, 128)).shape == (2, 12)
    assert sep_params < dense_params


# --------------------------------------------------------------------------- #
# 7. summary
# --------------------------------------------------------------------------- #
def test_summary_accounts_for_all_parameters() -> None:
    model = build()
    rows = model.summary()
    assert [r[0] for r in rows] == ["conv1", "b1", "b2", "b3",
                                    "conv2", "conv3", "conv4"]
    assert sum(r[2] for r in rows) == sum(p.numel() for p in model.parameters())
