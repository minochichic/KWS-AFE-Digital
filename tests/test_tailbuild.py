"""The tail read out of a real model: naming, chaining, and the two apply paths.

export/tailbuild.py sits between the torch-free arithmetic (tailfmt) and the two
things that consume it (emit's ROMs, golden's vectors). The failures it can have
are all of the quiet kind:

* `tail_plan` names a site differently from `Emitter._plain`, so the ROM and the
  manifest layer never meet. The same class of bug tests/test_golden.py exists
  to catch for the binary layers.
* `apply_site` (vectorized, int64 tensors) and `AffineFold.apply` (scalar, pure
  Python) drift apart, so the RTL's reference model and the golden vectors
  disagree about what the hardware does.
* the format threading is off by one layer, which multiplies a whole layer's
  output by 64 -- and after a relu and a clamp that does not look like a scale
  error, it looks like a dead network.
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from export.emit import Emitter                                  # noqa: E402
from export.tailbuild import (TAIL_FORMATS, apply_site, check_site,  # noqa: E402
                              fixed_weights, int8_weights, tail_plan)
from export.tailfmt import FMT_CONV2_PW, FMT_CONV3, FRAC_BITS     # noqa: E402
from models.binary_matchboxnet import BinaryMatchboxNet           # noqa: E402
from train.config import load_config                              # noqa: E402


@pytest.fixture(scope="module")
def model():
    cfg = load_config("configs/base.yaml")
    m = BinaryMatchboxNet(cfg.model).eval()
    # BN needs running stats before it can be folded; a few random batches is
    # enough to make them non-degenerate, which is all these tests need
    m.train()
    with torch.no_grad():
        for _ in range(4):
            x = torch.randint(0, 2, (8, cfg.model.in_channels, cfg.model.T),
                              dtype=torch.float32) * 2 - 1
            m(x)
    return m.eval()


@pytest.fixture(scope="module")
def plan(model):
    return tail_plan(model)


# --------------------------------------------------------------------------- #
# naming and structure
# --------------------------------------------------------------------------- #

def test_the_tail_is_the_three_layers_whose_bn_survives(plan):
    assert [s.name for s in plan] == ["conv2_pw", "conv3", "conv4"]
    assert [s.kind for s in plan] == ["bn_relu", "bn_relu", "logits"]


def test_every_site_names_a_layer_the_emitter_also_names(model, tmp_path):
    """A site with no manifest layer means the ROM has nothing to attach to, and
    a layer with no site means its BN never reached the hardware."""
    cfg = load_config("configs/base.yaml")
    man = Emitter(model, tmp_path).run(cfg, "test")
    layers = {l["name"] for l in man["layers"]}
    for s in tail_plan(model):
        assert s.name in layers, s.name
    tail_layers = {l["name"] for l in man["layers"]
                   if l["epilogue"] in ("bn_relu", "logits")}
    assert tail_layers == {s.name for s in tail_plan(model)}


def test_the_formats_thread_forward_in_dataflow_order(plan):
    """Each site divides out its INPUT's fixed-point scale. conv2_pw is fed +-1
    so it has none; conv3 is fed conv2_pw's output; conv4 is fed conv3's."""
    assert plan[0].in_fmt is None
    assert plan[1].in_fmt is FMT_CONV2_PW
    assert plan[2].in_fmt is FMT_CONV3
    assert [s.out_fmt for s in plan] == [TAIL_FORMATS[s.name] for s in plan]


def test_conv4_shares_one_gain_across_the_classes(plan):
    """argmax compares the twelve outputs against each other, so a per-class
    gain would reorder them."""
    c4 = plan[-1]
    assert len(set(c4.fold.gain)) == 1
    assert c4.fold.gain_real[0] == pytest.approx(
        1.0 / (1 << (FRAC_BITS + FMT_CONV3.frac_bits)))
    assert not c4.fold.relu           # logits are signed


def test_every_site_passes_its_own_fold_check(plan):
    for s in plan:
        check_site(s)


def test_the_accumulator_bounds_are_the_widths_the_readme_claims(plan):
    by = {s.name: s for s in plan}
    assert by["conv2_pw"].acc_bits == 8          # 64 binary terms
    assert by["conv3"].acc_bits == 28            # README 3-07
    assert by["conv4"].acc_bits <= 25


# --------------------------------------------------------------------------- #
# the two apply paths must agree
# --------------------------------------------------------------------------- #

def test_apply_site_agrees_with_the_scalar_fold(plan):
    """The vectorized path is what golden.py uses and the scalar path is the
    RTL's reference model. If they drift, the testbench compares the RTL against
    one of them while the ROM was built for the other."""
    rng = random.Random(19)
    for s in plan:
        n = s.fold.n_channels
        bound = min(1 << (s.acc_bits - 1), 1 << 20)
        a = torch.tensor(
            [[[rng.randint(-bound, bound) for _ in range(5)] for _ in range(n)]],
            dtype=torch.int64)
        got = apply_site(s, a)
        for c in range(n):
            for t in range(5):
                assert int(got[0, c, t]) == s.fold.apply(c, int(a[0, c, t])), \
                    (s.name, c, t)


def test_apply_site_saturates_rather_than_wrapping(plan):
    s = plan[0]
    n = s.fold.n_channels
    big = 1 << (s.acc_bits - 1)
    a = torch.full((1, n, 2), big, dtype=torch.int64)
    a[:, :, 1] = -big
    y = apply_site(s, a)
    assert bool(((y >= s.out_fmt.lo) & (y <= s.out_fmt.hi)).all())
    if s.fold.relu:
        assert bool((y >= 0).all())


def test_apply_site_refuses_the_wrong_rank(plan):
    with pytest.raises(ValueError):
        apply_site(plan[0], torch.zeros(4, 4, dtype=torch.int64))


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #

def test_conv4_weights_land_on_the_grid_the_sweep_measured(model):
    """experiments/tail_fixedpoint.py rounded conv4's weights to frac=6 with one
    shared scale. Per-channel int8 would be finer and would also mean the
    measured 0.8445 no longer certifies this format."""
    conv = model.stages["conv4"].conv
    q = fixed_weights(conv)
    assert q.dtype == torch.int64
    back = q.double() / float(1 << FRAC_BITS)
    assert float((back - conv.weight.detach().double()).abs().max()) \
        <= 0.5 / float(1 << FRAC_BITS)


def test_int8_weights_are_what_the_module_itself_quantized(model):
    from models.quant_ops import QuantConv1d
    conv = model.stages["conv3"].conv
    assert isinstance(conv, QuantConv1d)
    q, s = int8_weights(conv)
    assert int(q.abs().max()) <= conv.qmax
    assert s.numel() == conv.out_channels
    assert bool((s > 0).all())


def test_conv3_gain_carries_the_input_scale(model, plan):
    """conv3's gain is bn_g * int8_scale / 2^in_frac.

    The last factor is the one that is easy to drop: conv3 sums int8 weights
    against integers that represent x * 2^F, so the accumulator is 2^F too large
    and the gain has to divide it back out. Without it conv3's output is 64x too
    big, and after a relu and a clamp that does not read as a scale error -- it
    reads as a dead network. So the gain is rebuilt here from its three named
    factors and compared against what tailbuild produced.
    """
    from export.fuse import bn_affine

    c3 = plan[1]
    g, _ = bn_affine(model.stages["conv3"].bn)
    _, scale = int8_weights(c3.conv)
    want = (g * scale / float(1 << c3.in_fmt.frac_bits)).tolist()
    assert c3.fold.gain_real == pytest.approx(want, rel=1e-12)

    # and the factor really is there: without it every gain would be 64x larger
    naive = (g * scale).tolist()
    assert max(abs(v) for v in naive) == pytest.approx(
        max(abs(v) for v in want) * (1 << c3.in_fmt.frac_bits), rel=1e-9)


def test_conv2_pw_gain_is_bn_times_alpha(model, plan):
    """A binary layer's accumulator is pre-alpha, so alpha belongs in the gain.
    Leaving it out is a per-channel error of one to two orders of magnitude."""
    from export.fuse import bn_affine, conv_alpha

    c2 = plan[0]
    g, b = bn_affine(model.stages["conv2"].bn)
    a = conv_alpha(model.stages["conv2"].pw)
    alpha = (torch.ones_like(g) if a is None
             else a.detach().double().reshape(-1))
    assert c2.fold.gain_real == pytest.approx((g * alpha).tolist(), rel=1e-12)
    assert c2.fold.bias_real == pytest.approx(b.tolist(), rel=1e-12)
