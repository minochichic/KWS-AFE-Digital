"""The SPICE picture must binarize the way the trained model does.

analog/AFE/scripts/spectrogram16.py re-implements the comparator in numpy so it
can run without torch. A re-implementation that drifts from data/afe.py draws a
picture of a circuit we are not building -- which is exactly what happened
before: the script used per-clip min-max, a rule that reads the whole clip's
extremes and therefore cannot be built at all.

So these tests pin the script against the REAL forward path rather than against
a second re-derivation of the formula.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "analog" / "AFE" / "scripts"


@pytest.fixture(scope="module")
def spec():
    """Import spectrogram16 with its ngspice/plot-side deps stubbed out."""
    sys.path.insert(0, str(SCRIPTS))
    for name, attrs in [
        ("run_transient", {"BIAS": 0.9, "AMP": 1.0, "NET": "",
                           "channel_params": lambda c: (0, 0, 0, 0),
                           "apply_channel": lambda *a: ""}),
        ("learned_r7r8", {"LEARNED": [0.5] * 16}),
    ]:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)
    import matplotlib
    matplotlib.use("Agg")
    import spectrogram16
    return spectrogram16


class _Args:
    track = "auto"
    afe_json = None
    quiescent = None
    lse_temp_frac = 0.78
    v_lo = None
    v_hi = None


QUIESCENT = 0.9178          # detector rest point [V]
FRAC = 0.78


def _swing(seed=0, n_ch=16, n_win=100):
    rng = np.random.default_rng(seed)
    return rng.random((n_ch, n_win)) ** 2 * 0.05      # 0..50 mV, speech-ish


def _write_json(tmp_path, normalize, thr, **extra):
    p = tmp_path / "afe.json"
    p.write_text(json.dumps({"normalize": normalize,
                             "threshold": [float(t) for t in thr], **extra}))
    return str(p)


def test_xlse_matches_data_afe_exactly(spec, tmp_path):
    """Track 1 in numpy == data/afe.py `_xlse` + comparator, bit for bit."""
    from data.afe import AFEFrontend
    from train.config import AFEConfig

    swing = _swing()
    env = QUIESCENT + swing
    rng = np.random.default_rng(1)
    alphas = rng.random(16) * 0.8 + 0.1

    args = _Args()
    args.quiescent = QUIESCENT
    args.afe_json = _write_json(tmp_path, "xlse", alphas,
                                lse_temp_frac=FRAC, xmax_floor=0.0)
    got, _, track = spec.binarize(env, args)
    assert track == "xlse"

    # the reference: the actual training-time normalization, on the same data.
    # `mel` (not `spice`) because _xlse never touches the filterbank -- it is
    # handed envelopes -- and this keeps the test off the SPICE artifact files.
    # compression="sqrt" is not a choice: AFEFrontend rejects xlse without it,
    # since dividing by a level is only meaningful in an amplitude domain.
    # .double() matters. The numpy side runs in float64; a float32 reference
    # would disagree in the last bit on any pixel sitting near its threshold,
    # and "bit for bit" would then be luck rather than a property.
    afe = AFEFrontend(AFEConfig(normalize="xlse", filterbank_source="mel",
                                compression="sqrt")).double()
    x = torch.tensor(swing, dtype=torch.float64).unsqueeze(0)   # [1, 16, T]
    # the script derives T from the data it was handed; give the model the same
    typ = float(np.median(swing.max(axis=0)))
    afe.lse_temp.fill_(FRAC * typ)
    afe.xmax_floor.fill_(0.0)
    ref_norm = afe._xlse(x)[0].numpy()
    ref = (ref_norm > alphas[:, None]).astype(float)

    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} pixels differ from data/afe.py._xlse")
    assert 0.0 < got.mean() < 1.0, "degenerate image -- test says nothing"


def test_minmax_is_not_the_same_picture(spec, tmp_path):
    """The old rule really did draw something else, so relabelling mattered."""
    swing = _swing()
    env = QUIESCENT + swing
    alphas = np.random.default_rng(1).random(16) * 0.8 + 0.1

    args = _Args()
    args.quiescent = QUIESCENT
    args.afe_json = _write_json(tmp_path, "xlse", alphas,
                                lse_temp_frac=FRAC, xmax_floor=0.0)
    xlse, _, _ = spec.binarize(env, args)
    args.track = "minmax"
    mm, _, track = spec.binarize(env, args)
    assert track == "minmax"
    assert not np.array_equal(xlse, mm), (
        "min-max and xlse agree on this input; the test cannot detect a "
        "regression that swaps them")


def test_minmax_reads_the_future(spec, tmp_path):
    """Why min-max is unbuildable, as an executable statement.

    Change ONLY the last frame and an early frame's bits move. A causal circuit
    cannot do that; a diode-OR reference cannot either, and that asymmetry is
    the whole argument for dropping min-max.
    """
    swing = _swing()
    alphas = np.random.default_rng(1).random(16) * 0.8 + 0.1
    args = _Args()
    args.quiescent = QUIESCENT
    args.track = "minmax"
    args.afe_json = _write_json(tmp_path, "xlse", alphas,
                                lse_temp_frac=FRAC, xmax_floor=0.0)

    a, _, _ = spec.binarize(QUIESCENT + swing, args)
    loud = swing.copy()
    loud[:, -1] = 0.5                      # a shout at the very end
    b, _, _ = spec.binarize(QUIESCENT + loud, args)
    assert not np.array_equal(a[:, :-1], b[:, :-1]), (
        "the future did not leak backwards -- min-max would then be causal")

    # xlse, by contrast, is per-frame: earlier frames must be untouched
    args.track = "xlse"
    c, _, _ = spec.binarize(QUIESCENT + swing, args)
    d, _, _ = spec.binarize(QUIESCENT + loud, args)
    assert np.array_equal(c[:, :-1], d[:, :-1]), (
        "xlse changed an earlier frame -- it must depend on that frame only")


def test_fixed_uses_the_supplied_absolute_range(spec, tmp_path):
    """Track 2's reference is dataset-level, so it must come from outside."""
    swing = _swing()
    env = QUIESCENT + swing
    alphas = np.full(16, 0.5)
    args = _Args()
    args.quiescent = QUIESCENT
    args.afe_json = _write_json(tmp_path, "fixed", alphas)

    args.v_lo, args.v_hi = 0.0, 0.05
    got, _, track = spec.binarize(env, args)
    assert track == "fixed"
    ref = ((swing - 0.0) / 0.05 > alphas[:, None]).astype(float)
    assert np.array_equal(got, ref)

    # a different assumed range must move the picture, or the argument is dead
    args.v_lo, args.v_hi = 0.0, 0.10
    other, _, _ = spec.binarize(env, args)
    assert not np.array_equal(got, other)
