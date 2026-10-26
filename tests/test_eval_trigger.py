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
END = 300


def stream(frames: int = 400) -> np.ndarray:
    return np.zeros((frames, 16), dtype=np.uint8)


def sim(s, pol, counted=ALL, bg0=None):
    return simulate_triggers(s, pol, counted, END, bg0)


def test_silence_never_triggers():
    assert sim(stream(), TriggerPolicy()) == []


def test_m1_captures_from_the_first_sound_frame():
    s = stream()
    s[130, 3] = 1
    assert sim(s, TriggerPolicy(k=1)) == [Capture(onset=130, start=130, end=130 + NATIVE)]


def test_k_counts_channels_in_one_frame():
    s = stream()
    s[120, :2] = 1          # two channels: not enough for k=3
    s[150, :3] = 1
    assert [c.onset for c in sim(s, TriggerPolicy(k=3))] == [150]


def test_run_needs_consecutive_frames_and_onset_is_the_first():
    s = stream()
    s[110, 0] = 1           # lone frame
    s[140:143, 0] = 1
    assert [(c.onset, c.start) for c in sim(s, TriggerPolicy(run=3))] == [(140, 140)]


def test_pre_reaches_back_and_clamps_at_zero():
    s = stream()
    s[150, 0] = 1
    s[5, 0] = 1
    caps = sim(s, TriggerPolicy(pre=20, busy=0))
    assert [(c.onset, c.start, c.end) for c in caps] == [(5, 0, 100), (150, 130, 230)]


def test_busy_blocks_rescan_after_a_capture():
    s = stream()
    s[0, 0] = 1
    s[103, 0] = 1           # inside busy (100..107): ignored
    s[110, 0] = 1
    assert [c.onset for c in sim(s, TriggerPolicy(busy=8))] == [0, 110]


def test_capture_running_off_the_stream_is_dropped():
    s = stream(250)
    s[200, 0] = 1
    assert sim(s, TriggerPolicy()) == []


def test_scan_end_stops_new_triggers():
    s = stream()
    s[250, 0] = 1
    assert simulate_triggers(s, TriggerPolicy(), ALL, scan_end=200) == []


def test_hot_channels_are_not_counted():
    sil = np.zeros((1000, 16), dtype=np.uint8)
    sil[:, 15] = 1                       # always on in silence
    counted = hot_mask(sil, 0.5)
    assert not counted[15] and counted[:15].all()
    s = stream()
    s[:, 15] = 1
    assert sim(s, TriggerPolicy(hot=0.5), counted) == []
    assert sim(s, TriggerPolicy())[0].onset == 0


def test_delta_ignores_a_steady_noise_floor_and_fires_on_a_rise():
    """The measured case: noise keeps several comparators on all the time."""
    s = stream()
    s[:, :6] = 1                         # floor: 6 of 16 on, every frame
    s[200:210, 6:10] = 1                 # word: 4 more come on
    assert sim(s, TriggerPolicy(k=1))[0].onset == 0          # absolute: fires at once
    caps = sim(s, TriggerPolicy(k=1, delta=3), bg0=6.0)
    assert [c.onset for c in caps] == [200]


def test_background_tracks_only_quiet_frames():
    s = stream()
    s[:, :2] = 1
    s[100:400, 2:12] = 1                 # sustained loud section
    caps = sim(s, TriggerPolicy(k=1, delta=4, busy=0), bg0=2.0)
    assert caps[0].onset == 100          # the rise is caught, not learned away


def test_score_keyword_hit_wrong_and_false_alarm():
    word = (120, 160)
    caps = [Capture(0, 0, 100), Capture(110, 110, 210)]
    r = score_case(caps, [2, 5], label=5, word=word, n_keywords=10, scanned=300)
    assert (r["hit"], r["wrong"], r["fa"], r["wakes"]) == (1, 0, 1, 2)
    assert r["cover"] == pytest.approx(1.0)
    assert r["duty"] == pytest.approx(200 / 300)
    r = score_case(caps, [10, 3], label=5, word=word, n_keywords=10, scanned=300)
    assert (r["hit"], r["wrong"], r["fa"]) == (0, 1, 0)


def test_duty_counts_only_the_scanned_part():
    r = score_case([Capture(250, 250, 350)], [11], 11, (100, 200), 10, scanned=300)
    assert r["duty"] == pytest.approx(50 / 300)


def test_summarize_rates_and_quiet_duty():
    rows = [score_case([Capture(110, 110, 210)], [5], 5, (120, 160), 10, 300),
            score_case([], [], 6, (120, 160), 10, 300),
            score_case([Capture(0, 0, 100)], [3], 10, (100, 200), 10, 300)]
    s = summarize(rows, quiet=[False, False, True])
    assert s["recall"] == 0.5 and s["missed_word"] == 0.5
    assert s["fa_per_stream"] == pytest.approx(1 / 3)
    assert s["quiet_duty"] == pytest.approx(100 / 300)
    assert s["quiet_wakes"] == 1


def test_padded_matches_training_layout():
    w = np.zeros((NATIVE, 16), dtype=np.uint8)
    w[0, 1] = 1
    x = _padded(w)
    assert x.shape == (16, 128)
    assert (x[:, :14] == -1).all() and (x[:, -14:] == -1).all()
    assert x[1, 14] == 1 and x[0, 14] == -1


def test_policy_mode_and_label():
    assert TriggerPolicy().mode == "M1"
    assert TriggerPolicy(pre=10).mode == "M2"
    assert TriggerPolicy(delta=3).mode == "M2"     # the background needs 10 ms wakes
    assert TriggerPolicy(k=1, delta=3, run=2, pre=10).label == "k1_d3_run2_pre10_all"


def test_quiet_waves_levels():
    torch = pytest.importorskip("torch")
    from experiments.eval_trigger import quiet_waves
    noise = [torch.randn(48000), torch.randn(20000) * 5]
    z = quiet_waves(noise, 4, 16000, None, seed=0)
    assert z.shape == (4, 16000) and not z.any()
    q = quiet_waves(noise, 8, 16000, 0.01, seed=0)
    rms = q.pow(2).mean(1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 0.01), rtol=1e-3)
    assert torch.equal(q, quiet_waves(noise, 8, 16000, 0.01, seed=0))   # seeded
