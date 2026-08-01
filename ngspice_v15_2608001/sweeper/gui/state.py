"""Explicit state shared by DC, AC, Detector, Transient, and PWL controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..constants import DEFAULT_TABLE
from ..models import Channel, ChannelResult, DetectorResult, TransientResult


@dataclass
class AppState:
    """Keep cross-tab domain state separate from Tk widget construction."""

    default_channels: list[Channel] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    last_results: list[ChannelResult] = field(default_factory=list)
    last_ac_results: list[ChannelResult] = field(default_factory=list)
    last_op_results: list[ChannelResult] = field(default_factory=list)
    last_transient_results: list[TransientResult] = field(default_factory=list)
    last_detector_result: DetectorResult | None = None
    detector_result_cache: dict[int, DetectorResult] = field(default_factory=dict)
    transient_pwl_files: list[Path] = field(default_factory=list)
    component_source_path: Path = DEFAULT_TABLE
    component_dirty: bool = False
    component_editor_channel: int | None = None
    pending_margin_retune: tuple[int, float] | None = None
    component_revision: int = 0
    component_last_change_source: str | None = None
    component_last_change_channel: int | None = None
    ac_magnitude_unit: str = "dB"
    ac_manual_y_limits: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {"dB": ("", ""), "Linear": ("", "")}
    )
    transient_result_lookup: dict[tuple[str, int], TransientResult] = field(
        default_factory=dict
    )
    transient_result_stimulus_paths: dict[str, Path] = field(
        default_factory=dict
    )
    # Per-run live-result state.  The displayed result is tracked so a job
    # finishing mid-run cannot redraw, and thereby reset, the plot the user is
    # currently inspecting.
    transient_displayed_result: TransientResult | None = None
    transient_live_rendered: bool = False
    transient_completed_jobs: int = 0
    transient_total_jobs: int = 0
    transient_manual_x_limits_ms: tuple[float, float] | None = None
    transient_manual_y_limits_mv: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    transient_y_modes: dict[str, str] = field(
        default_factory=lambda: {
            "v_in": "자동",
            "v_filt": "자동",
            "v_env + v_thr": "자동",
        }
    )
    detector_selected_channel: int | None = None
    detector_live_rendered: bool = False
    detector_completed_jobs: int = 0
    detector_total_jobs: int = 0
    detector_display_mode: str = "절대전압"
    detector_manual_x_limits_ms: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "절대전압": ("", ""),
            "AC 비교": ("", ""),
        }
    )
    detector_manual_y_limits_mv: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "절대전압": ("", ""),
            "AC 비교": ("", ""),
        }
    )


class SharedStateField:
    """Descriptor preserving legacy attribute access through ``AppState``."""

    def __init__(self, field_name: str) -> None:
        """Remember the dataclass field represented by this descriptor."""

        self.field_name = field_name

    @staticmethod
    def _state(instance: object) -> AppState:
        """Return existing state or create it for headless ``__new__`` tests."""

        state = instance.__dict__.get("state")
        if state is None:
            state = AppState()
            instance.__dict__["state"] = state
        return state

    def __get__(self, instance: object | None, owner: type[object]) -> Any:
        """Read a shared value while retaining class-level descriptor access."""

        if instance is None:
            return self
        return getattr(self._state(instance), self.field_name)

    def __set__(self, instance: object, value: object) -> None:
        """Write a shared value into the application's state object."""

        setattr(self._state(instance), self.field_name, value)
