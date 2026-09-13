"""Reference model for continuous AFE windows and keyword decisions.

The deployed FPGA receives one 16-bit binary AFE frame every 10 ms.  It keeps
the latest 100 real frames, emits a snapshot every 10 frames, and pads that
snapshot with the same -1 values used during training.  This module is the
small, executable contract for the future ``kws_window`` and ``kws_vote`` RTL;
it deliberately has no dependency on PyTorch or on a trained checkpoint.

Frame indices use half-open intervals.  The first snapshot is [0, 100), and is
available when frame 99 has been captured.  The next is [10, 110), then
[20, 120), and so on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    """Geometry shared by training, the Python evaluator, and the RTL."""

    n_channels: int = 16
    native_frames: int = 100
    hop_frames: int = 10
    pad_left: int = 14
    pad_right: int = 14

    def __post_init__(self) -> None:
        for name in ("n_channels", "native_frames", "hop_frames"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pad_left < 0 or self.pad_right < 0:
            raise ValueError("padding must be non-negative")

    @property
    def model_frames(self) -> int:
        return self.pad_left + self.native_frames + self.pad_right


@dataclass(frozen=True)
class WindowSnapshot:
    """One immutable model input produced from the continuous frame stream."""

    window_id: int
    start_frame: int
    end_frame: int
    model_input: np.ndarray  # [channel, model_frames], values {-1, +1}


class StreamingWindowBuffer:
    """Hardware-shaped ring buffer that emits overlapping model snapshots."""

    def __init__(self, spec: WindowSpec = WindowSpec()) -> None:
        self.spec = spec
        self._ring = np.zeros(
            (spec.native_frames, spec.n_channels), dtype=np.uint8
        )
        self._write_index = 0
        self._frames_seen = 0
        self._next_emit = spec.native_frames
        self._next_window_id = 0

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def push(self, frame: Sequence[int]) -> Optional[WindowSnapshot]:
        """Capture one {0,1} frame and return a snapshot when one is due."""

        a = np.asarray(frame)
        if a.shape != (self.spec.n_channels,):
            raise ValueError(
                f"frame shape must be ({self.spec.n_channels},), got {a.shape}"
            )
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("AFE frames must contain only 0 or 1")

        self._ring[self._write_index] = a.astype(np.uint8, copy=False)
        self._write_index = (self._write_index + 1) % self.spec.native_frames
        self._frames_seen += 1

        if self._frames_seen != self._next_emit:
            return None

        # Once full, write_index points at the oldest frame.  Copying here is
        # intentional: a snapshot must not change while later frames overwrite
        # the ring.  RTL may realize the same contract with a frozen second RAM.
        chronological = np.concatenate(
            (self._ring[self._write_index :], self._ring[: self._write_index]),
            axis=0,
        )
        model_input = np.full(
            (self.spec.n_channels, self.spec.model_frames), -1, dtype=np.int8
        )
        real = chronological.T.astype(np.int8) * 2 - 1
        lo = self.spec.pad_left
        model_input[:, lo : lo + self.spec.native_frames] = real

        snapshot = WindowSnapshot(
            window_id=self._next_window_id,
            start_frame=self._frames_seen - self.spec.native_frames,
            end_frame=self._frames_seen,
            model_input=model_input,
        )
        self._next_window_id += 1
        self._next_emit += self.spec.hop_frames
        return snapshot


def make_snapshots(
    frames: np.ndarray, spec: WindowSpec = WindowSpec()
) -> List[WindowSnapshot]:
    """Offline convenience wrapper around :class:`StreamingWindowBuffer`.

    ``frames`` is time-major ``[n_frames, n_channels]`` in hardware {0,1}
    notation.  Returned tensors are channel-major and use model {-1,+1}
    notation.
    """

    a = np.asarray(frames)
    if a.ndim != 2 or a.shape[1] != spec.n_channels:
        raise ValueError(
            f"frames shape must be [time, {spec.n_channels}], got {a.shape}"
        )
    buffer = StreamingWindowBuffer(spec)
    out: List[WindowSnapshot] = []
    for frame in a:
        snapshot = buffer.push(frame)
        if snapshot is not None:
            out.append(snapshot)
    return out


@dataclass(frozen=True)
class VoteConfig:
    """Decision policy applied to the 10 Hz stream of model results."""

    required_consecutive: int = 3
    cooldown_windows: int = 10
    n_classes: int = 12
    quiet_classes: Tuple[int, ...] = (10, 11)

    def __post_init__(self) -> None:
        if self.required_consecutive <= 0:
            raise ValueError("required_consecutive must be positive")
        if self.cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        if self.n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if any(c < 0 or c >= self.n_classes for c in self.quiet_classes):
            raise ValueError("quiet class is outside n_classes")


@dataclass(frozen=True)
class VoteStep:
    """State after consuming one classified snapshot."""

    window_id: int
    prediction: int
    candidate: Optional[int]
    streak: int
    cooldown_remaining: int
    armed: bool
    detection: Optional[int]


class ConsecutiveVote:
    """Emit one event after N adjacent snapshots agree on a keyword.

    A missing window ID breaks the streak.  After a detection, both a quiet
    result (silence or unknown) and the configured cooldown must occur before
    another event can be emitted.  This prevents a long word from repeatedly
    firing while preserving an explicit re-arm condition.
    """

    def __init__(self, config: VoteConfig = VoteConfig()) -> None:
        self.config = config
        self._candidate: Optional[int] = None
        self._streak = 0
        self._last_window_id: Optional[int] = None
        self._cooldown_remaining = 0
        self._armed = True
        self._quiet_seen = True

    def update(self, prediction: int, window_id: int) -> VoteStep:
        if prediction < 0 or prediction >= self.config.n_classes:
            raise ValueError(f"prediction {prediction} is outside n_classes")
        if window_id < 0:
            raise ValueError("window_id must be non-negative")
        if self._last_window_id is not None and window_id <= self._last_window_id:
            raise ValueError("window_id must increase strictly")

        adjacent = (
            self._last_window_id is None
            or window_id == self._last_window_id + 1
        )
        if self._last_window_id is not None:
            elapsed = window_id - self._last_window_id
            self._cooldown_remaining = max(
                0, self._cooldown_remaining - elapsed
            )
        self._last_window_id = window_id

        if not adjacent:
            self._candidate = None
            self._streak = 0

        quiet = prediction in self.config.quiet_classes
        if not self._armed and quiet:
            self._quiet_seen = True
        if not self._armed and self._quiet_seen and self._cooldown_remaining == 0:
            self._armed = True

        detection: Optional[int] = None
        if quiet:
            self._candidate = None
            self._streak = 0
        elif not self._armed:
            self._candidate = None
            self._streak = 0
        else:
            if prediction == self._candidate and adjacent:
                self._streak += 1
            else:
                self._candidate = prediction
                self._streak = 1
            if self._streak >= self.config.required_consecutive:
                detection = prediction
                self._candidate = None
                self._streak = 0
                self._armed = False
                self._quiet_seen = False
                self._cooldown_remaining = self.config.cooldown_windows

        return VoteStep(
            window_id=window_id,
            prediction=prediction,
            candidate=self._candidate,
            streak=self._streak,
            cooldown_remaining=self._cooldown_remaining,
            armed=self._armed,
            detection=detection,
        )


def vote_sequence(
    predictions: Iterable[int],
    window_ids: Optional[Iterable[int]] = None,
    config: VoteConfig = VoteConfig(),
) -> List[VoteStep]:
    """Run the decision state machine over an offline prediction sequence."""

    pred = list(predictions)
    ids = list(range(len(pred))) if window_ids is None else list(window_ids)
    if len(pred) != len(ids):
        raise ValueError("predictions and window_ids must have the same length")
    vote = ConsecutiveVote(config)
    return [vote.update(p, i) for p, i in zip(pred, ids)]
