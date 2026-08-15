"""Golden vectors are only useful if they line up with the manifest.

The failure this guards against is silent: if golden.py names a layer
`b1_sub0_dw` while the manifest calls it `b1_s0_dw`, the testbench looks for a
file that does not exist, compares nothing, and passes.

The second thing worth pinning is that the dumped accumulator really is the
integer RTL computes, and that thresholding it reproduces the network's own
+-1 output -- that is the whole contract between these files and the hardware.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from export.emit import Emitter
from export.golden import dump_golden, layer_sites
from export.pack import unpack_pm1, PackedBits, WORD_BITS
from models.binary_matchboxnet import BinaryMatchboxNet
from train.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    cfg = load_config(str(ROOT / "configs" / "base.yaml"))
    model = BinaryMatchboxNet(cfg.model).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.normal_(0.0, 2.0)
                m.running_var.uniform_(0.5, 3.0)
                m.weight.normal_(0.0, 1.0)
                m.bias.normal_(0.0, 0.5)
    x = torch.where(torch.rand(3, cfg.afe.n_channels, cfg.model.T) > 0.5,
                    1.0, -1.0)
    return cfg, model, x


@pytest.fixture(scope="module")
def dumped(built, tmp_path_factory):
    cfg, model, x = built
    out = tmp_path_factory.mktemp("golden")
    man = dump_golden(model, x, out, "test")
    return man, out, cfg, model, x


def test_layer_names_match_the_manifest(dumped, tmp_path_factory):
    """The one failure mode that passes silently."""
    _man, _out, cfg, model, _x = dumped
    rtl = tmp_path_factory.mktemp("rtl")
    emitted = Emitter(model, rtl).run(cfg, "test")

    manifest_names = {l["name"] for l in emitted["layers"]}
    golden_names = {s.name for s in layer_sites(model)}
    conv_layers = manifest_names          # golden covers the adds too
    assert golden_names == conv_layers, (
        f"only in golden: {sorted(golden_names - conv_layers)}; "
        f"only in manifest: {sorted(conv_layers - golden_names)}")


def test_every_declared_file_exists(dumped):
    man, out, _, _, _ = dumped
    for key, meta in man["files"].items():
        assert (out / meta["file"]).is_file(), f"{key}: {meta['file']} missing"
    for extra in ("logits.txt", "predictions.txt", "golden.json"):
        assert (out / extra).is_file()


def test_accumulators_are_integers(dumped):
    """A non-integer accumulator means alpha leaked into the dump."""
    man, out, _, _, _ = dumped
    n_acc = 0
    for key, meta in man["files"].items():
        if not key.endswith("_acc"):
            continue
        n_acc += 1
        words = (out / meta["file"]).read_text().split()
        assert len(words) == 1 or all(len(w) == 8 for w in words)
        # round-trip a few through the signed decode the testbench will use
        for w in words[:64]:
            v = int(w, 16)
            v -= 1 << 32 if v >= 1 << 31 else 0
            assert -(1 << 31) <= v < (1 << 31)
    assert n_acc >= 15, f"only {n_acc} accumulator dumps -- hooks missed layers"


def test_binary_outputs_round_trip(dumped):
    """The packed +-1 output must decode back to legal +-1 of the right shape."""
    man, out, _, model, x = dumped
    seen = 0
    for key, meta in man["files"].items():
        if not key.endswith("_out") or not meta["file"].endswith(".hex"):
            continue
        seen += 1
        n_clip, ch, T = meta["shape"]
        words = (out / meta["file"]).read_text().split()
        n_words = (ch + WORD_BITS - 1) // WORD_BITS
        assert len(words) == n_clip * T * n_words, (
            f"{key}: {len(words)} words for [{n_clip},{ch},{T}]")
        packed = PackedBits(
            words=torch.tensor([int(w, 16) for w in words],
                               dtype=torch.int64).reshape(-1, n_words),
            n_valid=ch, shape=(n_clip * T, ch))
        vals = unpack_pm1(packed)
        assert set(torch.unique(vals).tolist()) <= {-1.0, 1.0}
    assert seen >= 10, f"only {seen} binary output dumps"


def test_input_hex_decodes_to_the_input(dumped):
    man, out, _, _, x = dumped
    n_clip, ch, T = man["files"]["input"]["shape"]
    words = (out / "input.hex").read_text().split()
    n_words = (ch + WORD_BITS - 1) // WORD_BITS
    assert len(words) == n_clip * T * n_words
    packed = PackedBits(
        words=torch.tensor([int(w, 16) for w in words],
                           dtype=torch.int64).reshape(-1, n_words),
        n_valid=ch, shape=(n_clip * T, ch))
    back = unpack_pm1(packed).reshape(n_clip, T, ch).transpose(1, 2)
    assert torch.equal(back, x), "input.hex does not decode to the fed tensor"


def test_fused_output_equals_the_models_own_activation(dumped):
    """End-to-end check of accumulator -> threshold -> packing.

    test_export pins FusedThreshold against sign(BN(alpha*n)) on synthetic
    input; this pins the whole chain as golden.py assembles it, on the tensor
    the network actually produced, and through the bit packing. If any link
    were reversed the decoded bits would stop matching here.
    """
    man, out, _, model, x = dumped
    sub = model.stages["b1"].subs[0]

    captured = {}
    h = sub.bn.register_forward_hook(lambda m, i, o: captured.setdefault("y", o))
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    want = torch.where(captured["y"] > 0, 1.0, -1.0)   # sign(0) -> +1, as fused

    meta = man["files"]["b1_s0_dw_out"]
    n_clip, ch, T = meta["shape"]
    words = (out / meta["file"]).read_text().split()
    n_words = (ch + WORD_BITS - 1) // WORD_BITS
    packed = PackedBits(
        words=torch.tensor([int(w, 16) for w in words],
                           dtype=torch.int64).reshape(-1, n_words),
        n_valid=ch, shape=(n_clip * T, ch))
    got = unpack_pm1(packed).reshape(n_clip, T, ch).transpose(1, 2)
    n_bad = int((got != want).sum())
    assert n_bad == 0, f"{n_bad}/{want.numel()} bits differ from sign(BN(dw(x)))"


def test_predictions_match_the_logits(dumped):
    man, out, _, _, _ = dumped
    logits = [[float(v) for v in line.split()]
              for line in (out / "logits.txt").read_text().splitlines()]
    pred = [int(v) for v in (out / "predictions.txt").read_text().split()]
    assert len(pred) == len(logits) == man["n_clips"]
    for row, p in zip(logits, pred):
        assert p == max(range(len(row)), key=lambda i: row[i])
