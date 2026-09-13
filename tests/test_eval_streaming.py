import numpy as np
import pytest
import torch

from experiments.eval_streaming import (
    collect_balanced_waveforms,
    select_balanced_indices,
    splice_clips,
    target_frames_in_window,
)
from experiments.streaming_window import WindowSnapshot, WindowSpec


def _snapshot(start: int, end: int) -> WindowSnapshot:
    return WindowSnapshot(
        window_id=0,
        start_frame=start,
        end_frame=end,
        model_input=np.empty((16, 128), dtype=np.int8),
    )


def test_balanced_selection_is_deterministic_and_capped():
    labels = np.repeat(np.arange(3), [5, 4, 2])
    first = select_balanced_indices(labels, 3, 3, seed=17)
    second = select_balanced_indices(labels, 3, 3, seed=17)

    assert first == second
    chosen = labels[first]
    assert np.bincount(chosen, minlength=3).tolist() == [3, 3, 2]
    assert len(first) == len(set(first))


@pytest.mark.parametrize("clips_per_class", [0, -1])
def test_balanced_selection_rejects_non_positive_cap(clips_per_class):
    with pytest.raises(ValueError, match="positive"):
        select_balanced_indices([0, 1], clips_per_class, 2, seed=0)


def test_waveform_reservoir_is_balanced_and_reproducible():
    loader = [
        (torch.arange(12, dtype=torch.float32).reshape(4, 3),
         torch.tensor([0, 0, 1, 1])),
        (torch.arange(12, 24, dtype=torch.float32).reshape(4, 3),
         torch.tensor([0, 1, 2, 2])),
    ]

    first = collect_balanced_waveforms(loader, 2, 3, seed=9)
    second = collect_balanced_waveforms(loader, 2, 3, seed=9)

    assert first[0] == second[0]
    torch.testing.assert_close(first[1], second[1])
    np.testing.assert_array_equal(first[2], [0, 0, 1, 1, 2, 2])
    assert first[1].shape == (6, 3)


def test_splice_clips_preserves_order_and_changes_to_time_major():
    spec = WindowSpec(n_channels=2, native_frames=3, hop_frames=1,
                      pad_left=0, pad_right=0)
    before = np.zeros((2, 3), dtype=np.uint8)
    target = np.ones((2, 3), dtype=np.uint8)
    after = np.tile(np.array([[0], [1]], dtype=np.uint8), (1, 3))

    stream = splice_clips(before, target, after, spec)

    assert stream.shape == (9, 2)
    np.testing.assert_array_equal(stream[:3], 0)
    np.testing.assert_array_equal(stream[3:6], 1)
    np.testing.assert_array_equal(stream[6:], [[0, 1]] * 3)


def test_splice_rejects_non_binary_data():
    clip = np.zeros((16, 100), dtype=np.uint8)
    bad = clip.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        splice_clips(clip, bad, clip)


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (0, 100, 0),
        (10, 110, 10),
        (50, 150, 50),
        (100, 200, 100),
        (150, 250, 50),
        (190, 290, 10),
        (200, 300, 0),
    ],
)
def test_target_frames_in_window(start, end, expected):
    assert target_frames_in_window(_snapshot(start, end)) == expected
