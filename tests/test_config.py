"""Config loading + guardrail tests.

These lock in CLAUDE.md 2.2's two absolute rules so a future YAML edit cannot
quietly binarize the first or last layer.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from train.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"


def test_base_config_loads_and_validates() -> None:
    cfg = load_config(BASE)
    assert cfg.model.C == 64
    assert cfg.model.n_classes == 12
    assert cfg.model.in_channels == cfg.afe.n_channels == 16
    assert [s.name for s in cfg.model.stages] == [
        "conv1", "b1", "b2", "b3", "conv2", "conv3", "conv4"
    ]


def test_stage_kernels_match_matchboxnet_table1() -> None:
    cfg = load_config(BASE)
    kernels = {s.name: s.kernel for s in cfg.model.stages}
    assert kernels == {
        "conv1": 11, "b1": 13, "b2": 15, "b3": 17,
        "conv2": 29, "conv3": 1, "conv4": 1,
    }
    conv2 = next(s for s in cfg.model.stages if s.name == "conv2")
    assert conv2.dilation == 2


def test_channel_widths_scale_with_C() -> None:
    cfg = load_config(BASE, overrides={"model.C": 32})
    widths = {s.name: s.out_channels(cfg.model.C, cfg.model.n_classes)
              for s in cfg.model.stages}
    assert widths["conv1"] == 64      # 2 * C
    assert widths["b1"] == 32         # C
    assert widths["conv4"] == 12      # n_classes, independent of C


def test_precision_assignment_follows_claude_md() -> None:
    cfg = load_config(BASE)
    prec = {s.name: s.precision for s in cfg.model.stages}
    assert prec["conv1"] == "int8"
    assert prec["b1"] == prec["b2"] == prec["b3"] == "binary"
    assert prec["conv4"] == "fixed"


def _write(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(BASE.read_text())
    mutate(raw)
    p = tmp_path / "mutated.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_binarizing_first_layer_is_rejected(tmp_path: Path) -> None:
    def mutate(raw):
        raw["model"]["stages"][0]["precision"] = "binary"

    with pytest.raises(ValueError, match="first layer"):
        load_config(_write(tmp_path, mutate))


def test_binarizing_last_layer_is_rejected(tmp_path: Path) -> None:
    def mutate(raw):
        raw["model"]["stages"][-1]["precision"] = "binary"

    with pytest.raises(ValueError, match="last layer"):
        load_config(_write(tmp_path, mutate))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    def mutate(raw):
        raw["model"]["widht"] = 64      # typo

    with pytest.raises(ValueError, match="unknown config key"):
        load_config(_write(tmp_path, mutate))


def test_afe_channel_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(raw):
        raw["model"]["in_channels"] = 8

    with pytest.raises(ValueError, match="must equal"):
        load_config(_write(tmp_path, mutate))


def test_native_T_follows_envelope_window() -> None:
    cfg = load_config(BASE)
    assert cfg.afe.native_T == 100                 # 1000 ms / 10 ms
    # 100 -> 128 zero-pad is MatchboxNet 4.1's documented behavior, not a bug.
    assert "zero-padded" in cfg.warnings()[0]

    cfg25 = load_config(BASE, overrides={"afe.envelope_win_ms": 25.0,
                                         "model.T": 40})
    assert cfg25.afe.native_T == 40
    # T matches the native rate, so no pad/crop note (the conv2 kernel-span
    # note is separate and asserted below).
    assert not any("padded" in w or "cropped" in w for w in cfg25.warnings())


def test_short_T_flags_oversized_conv2_kernel() -> None:
    """T=40 leaves 20 frames after conv1's stride 2, below conv2's span of 57."""
    cfg = load_config(BASE, overrides={"afe.envelope_win_ms": 25.0,
                                       "model.T": 40})
    _, notes = cfg.time_axis_report()
    assert any("conv2" in n and "exceeds" in n for n in notes)


def test_baseline_T_has_no_oversized_kernels() -> None:
    _, notes = load_config(BASE).time_axis_report()
    assert notes == []


def test_time_axis_lengths_track_stride() -> None:
    lengths, _ = load_config(BASE).time_axis_report()
    by_name = {n: (t_in, t_out) for n, t_in, t_out, _ in lengths}
    assert by_name["conv1"] == (128, 64)           # stride 2
    assert by_name["b1"] == (64, 64)               # stride 1 throughout
    assert by_name["conv4"] == (64, 64)


def test_sweep_overrides_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(BASE, overrides={"model.C": 16, "model.T": 96,
                                       "afe.envelope_win_ms": 10.0})
    assert (cfg.model.C, cfg.model.T) == (16, 96)
    out = tmp_path / "saved.yaml"
    cfg.save(out)
    assert load_config(out).model.C == 16


def test_defaults_alone_are_incomplete() -> None:
    """A bare Config() has no stages -- YAML is mandatory, by design."""
    with pytest.raises(ValueError, match="stages is empty"):
        Config().validate()
