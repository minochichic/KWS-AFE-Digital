"""The manifest must describe the network that actually runs.

tests/test_export.py already pins the primitives -- BN fusion and bit packing.
What emit.py adds on top is a *description* of the whole network, and a wrong
description is worse than a wrong primitive: RTL built from it computes
something plausible and silently different.

Two claims are worth pinning because I derived them rather than read them off:

  1. The accumulator widths come from +-n_terms. If n_terms is wrong for any
     layer, the bound stops being a bound and the datapath wraps.
  2. Which layers fuse to an integer compare. Stage activations are chosen by
     the NEXT stage's precision, so conv2/conv3 end in relu, not sign, and do
     NOT collapse to a threshold. Labelling them "threshold" would drop a real
     BN from the design.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from export.emit import Emitter, write_parameters_vh, signed_bits
from export.fuse import binary_accumulator
from models.binary_ops import BinaryConv1d
from models.binary_matchboxnet import BinaryMatchboxNet, PlainStage
from train.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    cfg = load_config(str(ROOT / "configs" / "base.yaml"))
    model = BinaryMatchboxNet(cfg.model).eval()
    # BN needs running stats that are not the identity, or fusion is trivial
    # and the thresholds all collapse to the same value.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.normal_(0.0, 2.0)
                m.running_var.uniform_(0.5, 3.0)
                m.weight.normal_(0.0, 1.0)
                m.bias.normal_(0.0, 0.5)
    return cfg, model


@pytest.fixture(scope="module")
def manifest(built, tmp_path_factory):
    cfg, model = built
    out = tmp_path_factory.mktemp("rtl")
    man = Emitter(model, out).run(cfg, "test")
    write_parameters_vh(man, out / "parameters.vh")
    return man, out, cfg, model


def test_every_conv_appears_exactly_once(manifest):
    man, _, _, model = manifest
    convs = {n for n, m in model.named_modules() if isinstance(m, nn.Conv1d)}
    # one manifest layer per conv, plus the residual adds which are not convs
    n_add = sum(1 for l in man["layers"] if l["op"] == "residual_add")
    assert len(man["layers"]) - n_add == len(convs), (
        f"{len(convs)} convs in the model, "
        f"{len(man['layers']) - n_add} non-add layers in the manifest")


def test_accumulator_bounds_hold(manifest):
    """+-n_terms must actually bound the integer accumulator.

    Random +-1 input is the adversarial case here, not a soft one: it is far
    more likely to drive a binary accumulator toward its extreme than speech is.
    """
    man, _, cfg, model = manifest
    by_shape = {}
    for l in man["layers"]:
        if l["op"].startswith("binary"):
            by_shape.setdefault((l["in_ch"], l["out_ch"], l["kernel"],
                                 l["groups"]), []).append(l["n_terms"])

    seen = 0
    hooks = []

    def check(mod, inp):
        nonlocal seen
        acc = binary_accumulator(mod, inp[0]).detach()
        key = (mod.in_channels, mod.out_channels, mod.kernel_size[0], mod.groups)
        bounds = by_shape.get(key)
        assert bounds, f"no manifest layer for {key}"
        b = max(bounds)
        assert float(acc.abs().max()) <= b, (
            f"accumulator {float(acc.abs().max()):.0f} exceeded bound {b} "
            f"for {key}")
        seen += 1

    for m in model.modules():
        if isinstance(m, BinaryConv1d):
            hooks.append(m.register_forward_pre_hook(check))
    try:
        x = torch.where(torch.rand(4, cfg.afe.n_channels, cfg.model.T) > 0.5,
                        1.0, -1.0)
        model(x)
    finally:
        for h in hooks:
            h.remove()
    assert seen > 0, "no binary conv was exercised"


def test_epilogue_matches_the_real_activation(manifest):
    """conv2/conv3 end in relu, so they must NOT be labelled `threshold`."""
    man, _, _, model = manifest
    want = {"sign": "threshold", "relu": "bn_relu", "none": "logits"}
    by_name = {l["name"]: l for l in man["layers"]}

    checked = 0
    for sname, mod in model.stages.items():
        if not isinstance(mod, PlainStage):
            continue
        # the stage's defining conv carries the stage activation; for a
        # separable stage that is the pointwise half
        key = f"{sname}_pw" if mod.dw is not None else sname
        assert key in by_name, f"{key} missing from the manifest"
        assert by_name[key]["epilogue"] == want[mod.activation], (
            f"{key}: activation {mod.activation!r} -> manifest says "
            f"{by_name[key]['epilogue']!r}")
        checked += 1
    assert checked >= 3

    # and the specific claim that motivated the test
    assert by_name["conv2_pw"]["epilogue"] == "bn_relu"
    assert by_name["conv3"]["epilogue"] == "bn_relu"
    assert by_name["conv4"]["epilogue"] == "logits"
    assert by_name["conv1"]["epilogue"] == "threshold"


def test_threshold_rom_exists_for_every_fusing_layer(manifest):
    man, out, _, _ = manifest
    for l in man["layers"]:
        if l["epilogue"] == "threshold":
            assert l["thresholds"], f"{l['name']} fuses but has no threshold ROM"
            assert (out / f"{l['thresholds']}.hex").is_file()
            rom = man["roms"][l["thresholds"]]
            assert rom["n_words"] == 2 * l["out_ch"], (
                "threshold ROM is n thresholds + n polarity words")
        else:
            assert not l["thresholds"], (
                f"{l['name']} does not fuse but carries a threshold ROM")


def test_the_tail_carries_an_affine_rom_and_not_a_threshold_one(manifest):
    """The two epilogues must not share a field name.

    A tail layer's epilogue is `(gain, offset) >> shift`, not a compare. If it
    were announced under `thresholds`, an RTL author would wire it into a
    comparator -- and a gain read as a threshold does not fail, it produces
    plausible garbage.
    """
    man, out, _, _ = manifest
    seen = 0
    for l in man["layers"]:
        if l["epilogue"] in ("bn_relu", "logits"):
            seen += 1
            assert not l["thresholds"], f"{l['name']} is not a compare"
            assert l["affine"], f"{l['name']} has no affine ROM"
            rom = man["roms"][l["affine"]]
            assert rom["kind"] == "affine"
            assert rom["n_words"] == 2 * l["out_ch"], "n gains + n offsets"
            assert (out / rom["file"]).is_file()
        else:
            assert not l["affine"], f"{l['name']} is a compare, not arithmetic"
    assert seen == 3, f"{seen} tail layers, expected conv2_pw/conv3/conv4"


def test_every_referenced_rom_was_written(manifest):
    man, out, _, _ = manifest
    for l in man["layers"]:
        for key in ("weights", "thresholds", "affine"):
            if l[key]:
                assert l[key] in man["roms"], f"{l['name']}.{key} not in roms"
                assert (out / man["roms"][l[key]]["file"]).is_file()


def test_parameters_vh_agrees_with_manifest(manifest):
    man, out, cfg, _ = manifest
    vh = (out / "parameters.vh").read_text()
    assert f"`define KWS_N_CH        {cfg.afe.n_channels}" in vh
    assert f"`define KWS_T           {cfg.model.T}" in vh
    assert f"`define KWS_ACC_BITS    {man['acc_bits_widest']}" in vh
    # the padding value is the thing most likely to be assumed as 0
    assert "-1, not 0" in vh
    for l in man["layers"]:
        p = f"KWS_L{man['layers'].index(l)}_{l['name'].upper()}_ACC_BITS"
        assert p in vh, f"{p} missing from parameters.vh"


def test_only_binary_fed_layers_carry_a_bound(manifest):
    """sum|q| bounds an int8 layer only when its input is +-1.

    conv1 is fed the AFE output and is bounded that way. conv3 is fed
    relu(BN(conv2_pw)), a real number, so the same formula bounds nothing there
    -- n_terms must stay 0 rather than printing a width resting on an
    assumption that does not hold.

    conv3 still gets an acc_bits, from a DIFFERENT bound: integer weights known
    at export time times a clamped fixed-point input (export/tailfmt.py). Both
    are real bounds; they just come from different pairs of extremes.
    """
    man, _, _, _ = manifest
    by_name = {l["name"]: l for l in man["layers"]}

    assert by_name["conv1"]["n_terms"] > 0, "conv1 IS fed +-1 and is bounded"
    assert by_name["conv1"]["acc_bits"] > 0

    for name in ("conv3", "conv4"):
        assert by_name[name]["n_terms"] == 0, f"{name} is not fed +-1"
        assert by_name[name]["acc_bits"] > 0, f"{name} still needs a width"
        assert name not in man["unsized_layers"]
        assert "tailfmt" in by_name[name]["notes"]


def test_the_headline_width_covers_the_tail_too(manifest):
    """KWS_ACC_BITS must be the widest register ANYWHERE.

    It used to be the widest among layers with n_terms, which excluded conv3 --
    whose accumulator is twice as wide as anything in the binary engine. A
    define called "widest accumulator" that is half the real widest is a trap:
    a tail register sized from it wraps at a quarter of conv3's range, and
    wrapping looks like plausible data.
    """
    man, _, _, _ = manifest
    assert man["acc_bits_widest"] == max(l["acc_bits"] for l in man["layers"])
    binary = [l["acc_bits"] for l in man["layers"] if l["n_terms"]]
    assert man["acc_bits_widest_binary"] == max(binary)
    assert man["acc_bits_widest"] >= man["acc_bits_widest_binary"]


def test_widths_are_the_bound_not_a_measurement(manifest):
    """Both bounds, checked as arithmetic rather than as a docstring.

    This used to assert a prefix on acc_bits_source, which broke the moment the
    tail gave that string a second sentence -- a test of the prose, not of the
    widths. What it should check is that every width is reproducible from the
    layer's own declared shape, which is what makes it a bound rather than
    something measured off a few clips.
    """
    from export.tailfmt import FixedFormat, acc_bits_for_real_input

    man, _, _, _ = manifest
    for l in man["layers"]:
        if l["n_terms"]:
            assert l["acc_bits"] == signed_bits(-l["n_terms"], l["n_terms"])

    checked = 0
    for l in man["layers"]:
        if not l["affine"]:
            continue
        rom = man["roms"][l["affine"]]
        if not rom.get("in_format"):
            continue                      # conv2_pw is fed +-1, covered above
        i, f = (int(v) for v in rom["in_format"].split("."))
        checked += 1
        assert l["acc_bits"] == acc_bits_for_real_input(
            l["in_ch"] * l["kernel"], rom["weight_absmax"],
            FixedFormat(i, f), in_nonneg=True), l["name"]
    assert checked == 2, "conv3 and conv4 both have a real-valued input"

    assert "measur" not in man["acc_bits_source"]
    assert "analytic bound" in man["acc_bits_source"]
