"""Executable specification for continuous capture, snapshots, and voting."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.streaming_window import (
    ConsecutiveVote,
    StreamingWindowBuffer,
    VoteConfig,
    WindowSpec,
    make_snapshots,
    vote_sequence,
)


def _frames(n: int, channels: int = 16) -> np.ndarray:
    t = np.arange(n)[:, None]
    c = np.arange(channels)[None, :]
    return ((t + c) % 2).astype(np.uint8)


def test_first_snapshot_waits_for_100_frames_and_uses_training_padding() -> None:
    spec = WindowSpec()
    buffer = StreamingWindowBuffer(spec)
    frames = _frames(100)

    for frame in frames[:-1]:
        assert buffer.push(frame) is None
    snap = buffer.push(frames[-1])

    assert snap is not None
    assert (snap.window_id, snap.start_frame, snap.end_frame) == (0, 0, 100)
    assert snap.model_input.shape == (16, 128)
    assert np.all(snap.model_input[:, :14] == -1)
    assert np.all(snap.model_input[:, 114:] == -1)
    expected = frames.T.astype(np.int8) * 2 - 1
    np.testing.assert_array_equal(snap.model_input[:, 14:114], expected)


def test_next_snapshot_moves_10_frames_and_does_not_mutate_the_first() -> None:
    frames = _frames(110)
    buffer = StreamingWindowBuffer()
    first = None
    second = None
    for frame in frames:
        snap = buffer.push(frame)
        if snap is not None and first is None:
            first = snap
            first_copy = snap.model_input.copy()
        elif snap is not None:
            second = snap

    assert first is not None and second is not None
    assert (second.window_id, second.start_frame, second.end_frame) == (1, 10, 110)
    np.testing.assert_array_equal(first.model_input, first_copy)
    expected = frames[10:].T.astype(np.int8) * 2 - 1
    np.testing.assert_array_equal(second.model_input[:, 14:114], expected)


def test_offline_wrapper_has_the_same_100_then_10_frame_schedule() -> None:
    snaps = make_snapshots(_frames(131))
    assert [(s.start_frame, s.end_frame) for s in snaps] == [
        (0, 100),
        (10, 110),
        (20, 120),
        (30, 130),
    ]


def test_bad_frame_shape_or_value_is_rejected() -> None:
    buffer = StreamingWindowBuffer()
    with pytest.raises(ValueError, match="shape"):
        buffer.push(np.zeros(15, dtype=np.uint8))
    with pytest.raises(ValueError, match="0 or 1"):
        buffer.push(np.full(16, 2, dtype=np.uint8))


def test_three_adjacent_keyword_results_emit_once() -> None:
    steps = vote_sequence([10, 0, 0, 11, 0, 0, 0, 0])
    assert [s.detection for s in steps] == [None, None, None, None,
                                             None, None, 0, None]


def test_different_keyword_and_missing_window_break_the_streak() -> None:
    vote = ConsecutiveVote()
    seq = [(2, 0), (2, 1), (3, 2), (2, 3), (2, 5), (2, 6), (2, 7)]
    got = [vote.update(pred, wid).detection for pred, wid in seq]
    assert got == [None, None, None, None, None, None, 2]


def test_rearm_requires_both_quiet_and_cooldown() -> None:
    cfg = VoteConfig(required_consecutive=2, cooldown_windows=3)
    steps = vote_sequence([4, 4, 10, 4, 4, 4], config=cfg)
    assert [s.detection for s in steps] == [None, 4, None, None, None, 4]


def test_cooldown_alone_does_not_rearm_a_held_keyword() -> None:
    cfg = VoteConfig(required_consecutive=2, cooldown_windows=2)
    steps = vote_sequence([6, 6, 6, 6, 6, 6], config=cfg)
    assert [s.detection for s in steps] == [None, 6, None, None, None, None]


def test_window_ids_must_increase() -> None:
    vote = ConsecutiveVote()
    vote.update(0, 3)
    with pytest.raises(ValueError, match="increase strictly"):
        vote.update(0, 3)
