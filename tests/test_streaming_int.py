"""The integer streaming reference must agree with the float-side vote semantics.

IntVote mirrors rtl/kws_vote.v. ConsecutiveVote is the policy the validation
numbers were chosen with. With the gate expressed as 'rejected -> quiet class',
the two must emit the same detections on every sequence -- otherwise the chip
would be checked against a policy that was never evaluated.
"""
import numpy as np

from experiments.eval_streaming_int import IntVote, crc16_windows, expected_word, keyword_and_margin
from experiments.streaming_window import VoteConfig, vote_sequence


def _int_dets(idx, margin, m, n, cd):
    v = IntVote(m, n, cd)
    out = []
    for w, (k, g) in enumerate(zip(idx, margin)):
        d, _ = v.step(int(k), int(g))
        if d is not None:
            out.append((w, d))
    return out


def _float_dets(idx, margin, m, n, cd):
    gated = [int(k) if g >= m else 10 for k, g in zip(idx, margin)]
    steps = vote_sequence(gated, range(len(gated)), VoteConfig(n, cd, 12, (10, 11)))
    return [(w, s.detection) for w, s in enumerate(steps) if s.detection is not None]


def test_intvote_matches_consecutive_vote_random():
    rng = np.random.default_rng(0)
    for trial in range(3000):
        n_win = int(rng.integers(5, 60))
        # few keyword values and margins clustered around the threshold -> many streaks
        idx = rng.integers(0, 3, n_win)
        margin = rng.integers(-3, 4, n_win) + 100
        n = int(rng.integers(1, 6))
        cd = int(rng.integers(0, 12))
        assert _int_dets(idx, margin, 100, n, cd) == _float_dets(idx, margin, 100, n, cd), trial


def test_keyword_margin_ties_and_quiet():
    pooled = np.zeros((2, 12), dtype=np.int64)
    pooled[0, [3, 7]] = 50          # tie -> lower index
    pooled[0, 11] = 20
    pooled[1, 10] = 90              # quiet wins -> negative margin
    pooled[1, 2] = 10
    idx, margin = keyword_and_margin(pooled)
    assert idx.tolist() == [3, 2]
    assert margin.tolist() == [30, -80]


def test_crc_depends_on_every_field_and_order():
    base = crc16_windows([1, 2, 3], [5, -6, 7])
    assert crc16_windows([1, 2, 3], [5, -6, 7]) == base
    assert crc16_windows([1, 2, 4], [5, -6, 7]) != base
    assert crc16_windows([1, 2, 3], [5, -6, 8]) != base
    assert crc16_windows([2, 1, 3], [-6, 5, 7]) != base


def test_expected_word_layout():
    w = expected_word(target=9, dets=[(4, 9), (18, 2)], crc=0xBEEF)
    assert w >> 24 == 0xBEEF
    assert (w >> 20) & 0xF == 9
    assert (w >> 10) & 0x3FF == (1 << 9) | (18 << 4) | 2
    assert w & 0x3FF == (1 << 9) | (4 << 4) | 9
    assert expected_word(10, [], 0) & 0xFFFFFF == (10 << 20)
