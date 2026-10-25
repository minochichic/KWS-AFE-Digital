import numpy as np
import pytest

from experiments.eval_trigger import (
    NATIVE,
    Capture,
    TriggerPolicy,
    _padded,
    hot_mask,
    score_case,
    simulate_triggers,
    summarize,
)

ALL = np.ones(16, dtype=bool)


def stream(frames: int = 400) -> np.ndarray:
    return np.zeros((frames, 16), dtype=np.uint8)


def test_silence_never_triggers():
    assert simulate_triggers(stream(), TriggerPolicy(), ALL) == []


def test_m1_captures_from_the_first_sound_frame():
    s = stream()
    s[130, 3] = 1
    caps = simulate_triggers(s, TriggerPolicy(k=1), ALL)
    assert caps == [Capture(onset=130, start=130, end=130 + NATIVE)]


def test_k_counts_channels_in_one_frame():
    s = stream()
    s[120, :2] = 1          # two channels: not enough for k=3
    s[150, :3] = 1
    caps = simulate_triggers(s, TriggerPolicy(k=3), ALL)
    assert [c.onset for c in caps] == [150]


def test_run_needs_consecutive_frames_and_onset_is_the_first():
    s = stream()
    s[110, 0] = 1           # lone frame
    s[140:143, 0] = 1
    caps = simulate_triggers(s, TriggerPolicy(run=3), ALL)
    assert [(c.onset, c.start) for c in caps] == [(140, 140)]


def test_pre_reaches_back_and_clamps_at_zero():
    s = stream()
    s[150, 0] = 1
    s[5, 0] = 1
    caps = simulate_triggers(s, TriggerPolicy(pre=20, busy=0), ALL)
    assert [(c.onset, c.start, c.end) for c in caps] == [(5, 0, 100), (150, 130, 230)]


def test_busy_blocks_rescan_after_a_capture():
    s = stream()
    s[0, 0] = 1
    s[103, 0] = 1           # inside busy (100..107): ignored
    s[110, 0] = 1
    caps = simulate_triggers(s, TriggerPolicy(busy=8), ALL)
    assert [c.onset for c in caps] == [0, 110]


def test_capture_running_off_the_stream_is_dropped():
    s = stream(250)
    s[200, 0] = 1
    assert simulate_triggers(s, TriggerPolicy(), ALL) == []


def test_hot_channels_are_not_counted():
    sil = np.zeros((1000, 16), dtype=np.uint8)
    sil[:, 15] = 1                       # always on in silence
    counted = hot_mask(sil, 0.5)
    assert not counted[15] and counted[:15].all()
    s = stream()
    s[:, 15] = 1
    assert simulate_triggers(s, TriggerPolicy(hot=0.5), counted) == []
    assert simulate_triggers(s, TriggerPolicy(), ALL)[0].onset == 0


def test_score_keyword_hit_wrong_and_false_alarm():
    word = (120, 160)
    caps = [Capture(0, 0, 100), Capture(110, 110, 210)]
    # first capture misses the word and says keyword 2 -> FA; second hits
    r = score_case(caps, [2, 5], label=5, word=word, n_keywords=10)
    assert (r["hit"], r["wrong"], r["fa"], r["wakes"]) == (1, 0, 1, 2)
    assert r["cover"] == pytest.approx(1.0)
    r = score_case(caps, [10, 3], label=5, word=word, n_keywords=10)
    assert (r["hit"], r["wrong"], r["fa"]) == (0, 1, 0)


def test_score_partial_cover_and_quiet_streams():
    r = score_case([Capture(150, 150, 250)], [5], 5, (120, 160), 10)
    assert r["cover"] == pytest.approx(10 / 40)
    r = score_case([Capture(0, 0, 100)], [4], label=10, word=(100, 200), n_keywords=10)
    assert (r["keyword"], r["fa"]) == (0, 1)


def test_summarize_rates():
    rows = [score_case([Capture(110, 110, 210)], [5], 5, (120, 160), 10),
            score_case([], [], 6, (120, 160), 10),
            score_case([Capture(0, 0, 100)], [3], 11, (100, 200), 10)]
    s = summarize(rows)
    assert s["recall"] == 0.5 and s["missed_word"] == 0.5
    assert s["fa_per_stream"] == pytest.approx(1 / 3)


def test_padded_matches_training_layout():
    w = np.zeros((NATIVE, 16), dtype=np.uint8)
    w[0, 1] = 1
    x = _padded(w)
    assert x.shape == (16, 128)
    assert (x[:, :14] == -1).all() and (x[:, -14:] == -1).all()
    assert x[1, 14] == 1 and x[0, 14] == -1


def test_policy_mode():
    assert TriggerPolicy().mode == "M1"
    assert TriggerPolicy(pre=10).mode == "M2"
    assert TriggerPolicy(run=2).mode == "M2"
