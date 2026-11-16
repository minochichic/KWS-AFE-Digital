"""experiments/thr_calib.py recovers known comparator thresholds from captures.

Synthetic board: the software AFE itself with chosen per-(channel, voltage)
thresholds, captures cut at random places in the ARB timeline, 1% of bits
flipped. The fit must find the offsets and thresholds that reproduce the
captures, and best_threshold must be the exact optimum.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.afe_clip import words_of
from experiments.thr_calib import (NATIVE, PAD, SoftwareAFE, arb_wave, best_threshold,
                                   calibrate, to_grid)


def test_best_threshold_is_exact():
    rng = np.random.default_rng(0)
    v = rng.random(300)
    y = (v + 0.2 * rng.standard_normal(300)) > 0.6
    t, acc = best_threshold(v, y)
    grid = np.concatenate([[v.min() - 1, v.max() + 1], v])
    brute = max(((v >= s) == y).mean() for s in grid)
    assert acc == pytest.approx(brute)
    assert ((v >= t) == y).mean() == pytest.approx(acc)


def _burst(rng, n):
    """Speech-like: a few noise bursts with different spectra and levels, silence around."""
    x = np.zeros(n)
    for _ in range(3):
        a, b = sorted(rng.integers(n // 5, 4 * n // 5, 2))
        seg = rng.standard_normal(b - a)
        k = rng.integers(1, 6)
        seg = np.convolve(seg, np.ones(k) / k, mode="same") * rng.uniform(0.2, 1.0)
        x[a:b] += seg * np.hanning(b - a)
    return x


def test_calibrate_recovers_thresholds_and_offsets():
    rng = np.random.default_rng(3)
    sw = SoftwareAFE("configs/base.yaml", {
        "afe.spice_matrix_path": "analog/AFE/artifacts/filterbank_matrix_board.csv",
        "afe.input_pp": 0.29})
    envs = {f"arb{i}": sw.raw(_burst(rng, 16000 * 2)) for i in range(3)}
    levels = [900.0, 948.0]
    allv = np.concatenate([e for e in envs.values()], axis=1)
    truth = {(c, mv): float(np.quantile(allv[c], q))
             for c in range(16) for mv, q in zip(levels, (0.6, 0.8))}
    recs, true_off = [], []
    for ref, e in envs.items():
        pe = np.pad(e, ((0, 0), (PAD, PAD)), mode="edge")
        for mv in levels:
            for _ in range(3):
                s = int(rng.integers(PAD // 2, e.shape[1] - NATIVE + PAD // 2))
                t = np.array([truth[(c, mv)] for c in range(16)])
                bits = (pe[:, s:s + NATIVE] >= t[:, None]).T.astype(np.uint8)
                flip = rng.random(bits.shape) < 0.01
                bits[flip] ^= 1
                recs.append({"ref": ref, "vthr_mv": [mv] * 16, "clip": words_of(bits)})
                true_off.append(s - PAD)
    fit, offs, agree = calibrate(recs, envs)
    assert np.median(agree) > 0.97
    assert np.mean(np.array(offs) == np.array(true_off)) > 0.8
    # the fitted threshold reproduces the true comparator on this data
    for (c, mv), (t, acc, n, rate) in fit.items():
        assert acc > 0.95, (c, mv, acc)
        v = allv[c]
        same = ((v >= t) == (v >= truth[(c, mv)])).mean()
        assert same > 0.97, (c, mv, same)
    g, src = to_grid(fit, levels)
    assert np.isfinite(g).all() and all(s == "board" for row in src for s in row)


def test_arb_wave_reads_keysight_text(tmp_path):
    x = (np.sin(np.arange(6400) / 10) * 30000).astype(int)
    body = ["File Format:1.10", "Sample Rate:64000.000000", "High Level:0.050000",
            "Low Level:-0.050000", "Data Points:%d" % (len(x) + 1), "Data:", "0"]
    p = tmp_path / "t.arb"
    p.write_text("\n".join(body + [str(v) for v in x]) + "\n")
    w = arb_wave(p)
    assert abs(len(w) - (len(x) + 1) / 4) <= 1
    assert abs(w[0]) < 0.01 * np.abs(w).max() and np.abs(w).max() > 1.0   # mV, offset removed
