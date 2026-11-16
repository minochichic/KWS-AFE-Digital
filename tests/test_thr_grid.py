"""Buildable thresholds: the per-channel grid and the bench input level.

What must hold:
1. assign() is the true min-cost assignment, with at most `capacity` channels
   per level, and never picks an unusable (empty) cell.
2. With a grid the comparator is exactly `env_raw >= grid_raw[c, chosen level]`
   -- whatever fixed_lo/hi the run has -- and the threshold still gets a
   gradient (straight-through), so it can walk to another level.
3. input_pp makes the front end level-invariant (the AWG sends every clip at the
   same Vpp) and leaves digital zero at zero.
4. A checkpoint written without a grid loads into a grid front end unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from data.afe import AFEFrontend
from data.thr_grid import _brute_force, assign, load_grid, write_grid
from train.config import AFEConfig

LEVELS = [900, 916, 932, 948, 964, 980, 996]


def test_assign_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(200):
        C, L = int(rng.integers(2, 7)), int(rng.integers(2, 4))
        cap = int(rng.integers(1, 3))
        if C > L * cap:
            cap = -(-C // L)
        g = np.sort(rng.random((C, L)), axis=1)
        g[rng.random((C, L)) < 0.15] = np.nan
        g[~np.isfinite(g).any(axis=1), 0] = 0.5
        thr = rng.random(C)
        want, best = _brute_force(thr, g, cap)
        if want is None:                                  # no buildable assignment exists
            with pytest.raises(ValueError):
                assign(thr, g, cap)
            continue
        got = assign(thr, g, cap)
        assert np.bincount(got, minlength=L).max() <= cap
        assert np.isfinite(g[np.arange(C), got]).all()
        assert sum((thr[c] - g[c, l]) ** 2 for c, l in enumerate(got)) == pytest.approx(best)


def test_capacity_moves_the_cheapest_channel():
    # five channels all nearest to level 0; capacity 4 -> exactly one moves,
    # and it is the one sitting closest to level 1
    g = np.tile([0.0, 1.0], (5, 1))
    thr = np.array([0.1, 0.2, 0.45, 0.05, 0.3])
    lvl = assign(thr, g, capacity=4)
    assert lvl.tolist() == [0, 0, 1, 0, 0]
    assert assign(thr, g, capacity=0).tolist() == [0] * 5


def test_grid_csv_roundtrip(tmp_path):
    g = np.arange(16 * 7, dtype=float).reshape(16, 7) / 100
    g[3, 0] = np.nan
    write_grid(tmp_path / "g.csv", LEVELS, g, ["test"])
    levels, back = load_grid(tmp_path / "g.csv")
    assert levels == LEVELS
    assert np.isnan(back[3, 0]) and np.allclose(np.nan_to_num(back), np.nan_to_num(g))


def _grid_frontend(tmp_path, cap=4, **kw):
    """A fixed-normalized front end whose grid sits on the channels' own envelope quantiles."""
    torch.manual_seed(0)
    wave = 0.1 * torch.randn(8, 16000)
    base = AFEFrontend(AFEConfig(normalize="fixed", **kw))
    base.init_fixed_scale(wave)
    raw = base.envelopes(wave, raw=True)                          # [B, C, T]
    flat = raw.permute(1, 0, 2).reshape(16, -1)
    q = torch.quantile(flat, torch.linspace(0.3, 0.9, len(LEVELS)), dim=1).T.double().numpy()
    # nudge every cell between data points: a cell ON an envelope value is a tie
    # that the normalized and the raw comparison may round differently
    vals = np.sort(flat.double().numpy(), axis=1)
    for c in range(16):
        i = np.searchsorted(vals[c], q[c])
        q[c] = (vals[c][np.clip(i - 1, 0, None)] + vals[c][np.clip(i, None, vals.shape[1] - 1)]) / 2
    write_grid(tmp_path / "g.csv", LEVELS, q)
    fe = AFEFrontend(AFEConfig(normalize="fixed", threshold_grid_path=str(tmp_path / "g.csv"),
                               threshold_grid_capacity=cap, **kw))
    fe.load_state_dict(base.state_dict())                         # 4. no grid -> grid: loads
    fe.init_fixed_scale(wave)
    fe.init_thresholds(wave)
    return fe, wave, q


def test_grid_comparator_is_raw_envelope_against_the_chosen_cell(tmp_path):
    fe, wave, q = _grid_frontend(tmp_path)
    lvl = fe.grid_choice()
    assert np.bincount(lvl, minlength=len(LEVELS)).max() <= 4
    raw = fe.envelopes(wave, raw=True)
    want = torch.where(raw >= torch.tensor(q[np.arange(16), lvl], dtype=raw.dtype).view(1, -1, 1),
                       1.0, -1.0)
    assert torch.equal(fe(wave), want)
    # independent of the run's normalization constants
    fe.fixed_hi.mul_(3.0)
    with torch.no_grad():                                         # same levels, new units
        fe.threshold.copy_(fe._grid_norm()[torch.arange(16), torch.as_tensor(lvl)])
    assert (fe.grid_choice() == lvl).all()
    assert torch.equal(fe(wave), want)


def test_grid_threshold_still_gets_a_gradient(tmp_path):
    fe, wave, _ = _grid_frontend(tmp_path, ste_clip=1.0)
    fe(wave).sum().backward()
    assert fe.threshold.grad is not None and fe.threshold.grad.abs().sum() > 0


def test_input_pp_is_level_invariant_and_keeps_zero():
    torch.manual_seed(0)
    wave = 0.05 * torch.randn(3, 16000)
    fe = AFEFrontend(AFEConfig(normalize="fixed", input_pp=0.29))
    fe.init_fixed_scale(wave)
    fe.init_thresholds(wave)
    assert torch.allclose(fe.envelopes(wave), fe.envelopes(7.0 * wave), atol=1e-5)
    assert (fe(torch.zeros(1, 16000)) == -1).all()


def test_input_pp_off_is_a_no_op():
    torch.manual_seed(0)
    wave = 0.05 * torch.randn(3, 16000)
    a = AFEFrontend(AFEConfig(normalize="none"))
    b = AFEFrontend(AFEConfig(normalize="none", input_pp=0.0))
    assert torch.equal(a.envelopes(wave), b.envelopes(wave))


def test_grid_rejects_relative_modes(tmp_path):
    from train.config import Config
    write_grid(tmp_path / "g.csv", LEVELS, np.ones((16, 7)))
    with pytest.raises(ValueError, match="normalize"):
        Config(afe=AFEConfig(normalize="minmax", threshold_grid_path=str(tmp_path / "g.csv"))).validate()
