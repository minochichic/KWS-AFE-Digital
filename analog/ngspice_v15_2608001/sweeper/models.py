"""Typed channel, settings, job, and simulation-result models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .constants import (
    AC_RAW_FILENAME,
    DEFAULT_DETECTOR_OPAMP,
    DETECTOR_FREQUENCY_DEFAULT_HZ,
    DETECTOR_GATE_DURATION_DEFAULT_S,
    DETECTOR_GATE_ON_DEFAULT_S,
    DETECTOR_INPUT_VPP_DEFAULT_V,
    DETECTOR_MAXIMUM_STEP_DEFAULT_S,
    DETECTOR_TIMESTEP_WARNING_POINTS_PER_CYCLE,
    DETECTOR_TOTAL_TIME_DEFAULT_S,
    OP_RAW_FILENAME,
    TRANSIENT_INPUT_VPP_DEFAULT_V,
    TRANSIENT_MAXIMUM_STEP_DEFAULT_S,
    TRANSIENT_OUTPUT_STEP_DEFAULT_S,
    VIN_NODE,
    V_ENV_NODE,
    V_FILT_NODE,
    V_THR_NODE,
)
from .values import resistance_spice_value, spice_value


@dataclass(frozen=True)
class Channel:
    """One CSV row normalized into the values used by the netlist generator."""

    ch: int
    f_c_hz: float
    q: float
    ra_kohm: float
    c_nf: float
    r1_kohm: float
    gain_db: float
    r2_kohm: float
    r4_kohm: float
    r5_kohm: float
    r6_kohm: float
    c3_nf: float
    r7_kohm: float
    r8_kohm: float
    vthr_v: float
    # Only the active detector op-amp (XU3) is selectable; the filter op-amps
    # stay on OPA379 so AC characteristics remain comparable across models.
    detector_opamp: str = DEFAULT_DETECTOR_OPAMP

    def spice_parameters(self) -> dict[str, str]:
        """Return the ten editable R/C values in ngspice engineering notation."""

        return {
            "RA": resistance_spice_value(self.ra_kohm),
            "R1": resistance_spice_value(self.r1_kohm),
            "R2": resistance_spice_value(self.r2_kohm),
            "R4": resistance_spice_value(self.r4_kohm),
            "R5": resistance_spice_value(self.r5_kohm),
            "R6": resistance_spice_value(self.r6_kohm),
            "R7": resistance_spice_value(self.r7_kohm),
            "R8": resistance_spice_value(self.r8_kohm),
            "C1": spice_value(self.c_nf, "n"),
            "C3": spice_value(self.c3_nf, "n"),
        }

@dataclass(frozen=True)
class SweepSettings:
    """Validated settings shared by the currently implemented AC and OP jobs."""

    ac_points_per_decade: int = 100
    ac_start_hz: float = 10.0
    ac_stop_hz: float = 20_000.0
    output_node: str = V_FILT_NODE
    pspice_compat: bool = True

    def validate(self, analyses: Sequence[str] = ("ac", "op")) -> None:
        """Reject invalid settings before any output directory or process is made."""

        selected = normalize_analyses(analyses)
        if not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", self.output_node):
            raise ValueError(f"잘못된 출력 노드: {self.output_node!r}")
        if "ac" in selected:
            if self.ac_points_per_decade < 1:
                raise ValueError("AC points/decade는 1 이상이어야 합니다.")
            if self.ac_start_hz <= 0 or self.ac_stop_hz <= self.ac_start_hz:
                raise ValueError("AC 주파수 범위가 잘못되었습니다.")

@dataclass(frozen=True)
class TransientSettings:
    """Validated transient time controls and saved analog circuit-node names."""

    output_step_s: float = TRANSIENT_OUTPUT_STEP_DEFAULT_S
    stop_time_s: float | None = None
    maximum_step_s: float = TRANSIENT_MAXIMUM_STEP_DEFAULT_S
    input_vpp_v: float = TRANSIENT_INPUT_VPP_DEFAULT_V
    vin_node: str = VIN_NODE
    vfilt_node: str = V_FILT_NODE
    venv_node: str = V_ENV_NODE
    vref_node: str = V_THR_NODE
    pspice_compat: bool = True
    # transient.csv repeats tran.raw exactly and is tens of megabytes per job,
    # so the GUI leaves it off.  The default stays true for existing callers.
    write_csv: bool = True

    @property
    def vthr_node(self) -> str:
        """Expose the canonical threshold-node name while preserving vref_node."""

        return self.vref_node

    def validate(self) -> None:
        """Reject invalid time values and node spellings."""

        if not math.isfinite(self.output_step_s) or self.output_step_s <= 0.0:
            raise ValueError("Transient output step은 0보다 커야 합니다.")
        if (
            self.stop_time_s is not None
            and (
                not math.isfinite(self.stop_time_s)
                or self.stop_time_s <= 0.0
            )
        ):
            raise ValueError("Transient stop time은 빈칸 또는 양수여야 합니다.")
        if not math.isfinite(self.maximum_step_s) or self.maximum_step_s <= 0.0:
            raise ValueError("Transient maximum step은 0보다 커야 합니다.")
        if not math.isfinite(self.input_vpp_v) or self.input_vpp_v <= 0.0:
            raise ValueError("Transient 입력 PWL Vpp는 0보다 커야 합니다.")
        for label, node in (
            ("v_in", self.vin_node),
            ("v_filt", self.vfilt_node),
            ("v_env", self.venv_node),
            ("v_thr", self.vthr_node),
        ):
            if not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", node):
                raise ValueError(f"{label} 노드명이 잘못되었습니다: {node!r}")

    def resolved_stop_time(self, pwl_end_s: float) -> float:
        """Use explicit stop time or the final 0 V point of the selected PWL."""

        self.validate()
        if not math.isfinite(pwl_end_s) or pwl_end_s <= 0.0:
            raise ValueError("PWL 종료 시각은 0보다 커야 합니다.")
        return self.stop_time_s if self.stop_time_s is not None else pwl_end_s


@dataclass(frozen=True)
class DetectorSettings:
    """Validated gated-sine controls for the independent detector workflow."""

    frequency_hz: float = DETECTOR_FREQUENCY_DEFAULT_HZ
    input_vpp_v: float = DETECTOR_INPUT_VPP_DEFAULT_V
    gate_on_s: float = DETECTOR_GATE_ON_DEFAULT_S
    gate_duration_s: float = DETECTOR_GATE_DURATION_DEFAULT_S
    total_time_s: float = DETECTOR_TOTAL_TIME_DEFAULT_S
    maximum_step_s: float = DETECTOR_MAXIMUM_STEP_DEFAULT_S
    vin_node: str = VIN_NODE
    vfilt_node: str = V_FILT_NODE
    venv_node: str = V_ENV_NODE
    vthr_node: str = V_THR_NODE
    pspice_compat: bool = True
    # detector.csv repeats detector.raw exactly; see TransientSettings.write_csv.
    write_csv: bool = True

    @property
    def gate_off_s(self) -> float:
        """Return the requested end of the gated sinusoidal interval."""

        return self.gate_on_s + self.gate_duration_s

    @property
    def points_per_cycle(self) -> float:
        """Return the nominal simulator samples available per sine period."""

        return 1.0 / (self.frequency_hz * self.maximum_step_s)

    def validate(self) -> None:
        """Reject invalid amplitudes, timing, frequency, and canonical nodes."""

        numeric = (
            self.frequency_hz,
            self.input_vpp_v,
            self.gate_on_s,
            self.gate_duration_s,
            self.total_time_s,
            self.maximum_step_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Detector 설정은 모두 유한한 숫자여야 합니다.")
        if self.frequency_hz <= 0.0:
            raise ValueError("정현파 주파수는 0 Hz보다 커야 합니다.")
        if self.input_vpp_v <= 0.0:
            raise ValueError("Detector 입력 Vpp는 0보다 커야 합니다.")
        if self.gate_on_s < 0.0:
            raise ValueError("Gate ON 시각은 0 이상이어야 합니다.")
        if self.gate_duration_s <= 0.0:
            raise ValueError("Gate 지속시간은 0보다 커야 합니다.")
        if self.gate_duration_s < 1.0 / self.frequency_hz:
            raise ValueError(
                "실제 /vin Vpp를 보장하려면 Gate 지속시간은 "
                "정현파 한 주기 이상이어야 합니다."
            )
        if self.total_time_s <= self.gate_off_s:
            raise ValueError("전체 해석시간은 Gate OFF 시각보다 커야 합니다.")
        if self.maximum_step_s <= 0.0:
            raise ValueError("Detector maximum timestep은 0보다 커야 합니다.")
        for label, node in (
            ("v_in", self.vin_node),
            ("v_filt", self.vfilt_node),
            ("v_env", self.venv_node),
            ("v_thr", self.vthr_node),
        ):
            if not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", node):
                raise ValueError(f"{label} 노드명이 잘못되었습니다: {node!r}")

    def timestep_warning(self) -> str | None:
        """Warn when fewer than 20 transient points represent one sine cycle.

        Twenty points per cycle is a common minimum transient-resolution rule.
        It limits the phase spacing to 18 degrees and reduces missed peaks.  It
        is a warning rather than a hard error because ngspice can still reduce
        its internal step below the requested maximum when the models demand it.
        """

        self.validate()
        threshold = DETECTOR_TIMESTEP_WARNING_POINTS_PER_CYCLE
        if self.points_per_cycle >= threshold:
            return None
        recommended_s = 1.0 / (self.frequency_hz * threshold)
        return (
            f"현재 maximum timestep은 주기당 {self.points_per_cycle:.3g}점입니다. "
            f"최소 {threshold} points/cycle 기준으로 "
            f"{recommended_s * 1e6:.6g} us 이하를 권장합니다."
        )

    @classmethod
    def quick_200ms_preset(cls) -> "DetectorSettings":
        """Return the editable 200 ms quick preset requested for the GUI."""

        return cls()

    @classmethod
    def paper_2020_preset(cls) -> "DetectorSettings":
        """Return the paper's literal 1.5 s gated-sine duration preset.

        The 2020 comparison paper states a 1.5 s sinusoidal duration.  A 100 ms
        pre-gate interval and a 200 ms post-gate interval are added here solely
        so DC baselines and falling crossings can be measured in this circuit.
        """

        return cls(
            gate_on_s=0.100,
            gate_duration_s=1.500,
            total_time_s=1.800,
        )


@dataclass(frozen=True)
class DetectorMeasurement:
    """Headroom and interpolated rise/fall crossings for one detector result."""

    vlow_v: float
    vhigh_v: float
    headroom_v: float
    rise_time_s: float | None
    fall_time_s: float | None
    rise_low_s: float | None
    rise_high_s: float | None
    fall_high_s: float | None
    fall_low_s: float | None
    rise_success: bool
    fall_success: bool
    rise_reason: str
    fall_reason: str

    @property
    def success(self) -> bool:
        """Return true only when both ordered crossing pairs were measured."""

        return self.rise_success and self.fall_success

    @property
    def failure_reason(self) -> str:
        """Join only the unavailable measurement explanations."""

        reasons = [
            reason
            for success, reason in (
                (self.rise_success, self.rise_reason),
                (self.fall_success, self.fall_reason),
            )
            if not success and reason
        ]
        return "; ".join(reasons)

@dataclass(frozen=True)
class AcMetrics:
    """Measured AC peak and -3 dB bandwidth values for one channel."""

    center_hz: float
    peak_db: float
    low_3db_hz: float | None
    high_3db_hz: float | None
    q: float | None
    status: str

@dataclass
class ChannelResult:
    """All normalized results and provenance paths produced for one channel."""

    channel: Channel
    run_dir: Path
    ac_rows: list[tuple[float, float, float]]
    op_voltages: dict[str, float]
    log_text: str
    ac_metrics: AcMetrics | None = None

@dataclass
class TransientResult:
    """One transient result containing only the four requested analog traces."""

    channel: Channel
    stimulus_path: Path
    run_dir: Path
    rows: list[tuple[float, float, float, float, float]]
    log_text: str


@dataclass
class DetectorResult:
    """One gated-sine detector result with component-revision provenance."""

    channel: Channel
    settings: DetectorSettings
    component_revision: int
    run_dir: Path
    rows: list[tuple[float, float, float, float, float]]
    log_text: str

    def is_stale(self, current_revision: int) -> bool:
        """Return whether components changed after this result was produced."""

        return self.component_revision != current_revision

@dataclass(frozen=True)
class AnalysisJobSpec:
    """Static file naming and display metadata for one analysis kind.

    AC and OP use this common per-channel job description.  Transient reuses
    the same process launcher but has a separate stimulus-by-channel runner.
    """

    key: str
    label: str
    raw_filename: str
    netlist_suffix: str
    log_stem: str

@dataclass(frozen=True)
class AnalysisJob:
    """Resolved per-channel paths passed to the generic ngspice job loop."""

    spec: AnalysisJobSpec
    netlist_path: Path
    raw_path: Path

ANALYSIS_JOB_SPECS = {
    "ac": AnalysisJobSpec("ac", "AC", AC_RAW_FILENAME, "ac", "ac"),
    "op": AnalysisJobSpec("op", "DC 동작점", OP_RAW_FILENAME, "op", "op"),
}

def normalize_analyses(analyses: Sequence[str] | str) -> tuple[str, ...]:
    """Normalize CLI/GUI aliases into an ordered tuple of supported job keys."""

    if isinstance(analyses, str):
        value = analyses.strip().lower()
        if value == "both":
            return ("ac", "op")
        analyses = (value,)
    normalized = tuple(
        dict.fromkeys(
            "op" if item.strip().lower() == "dc" else item.strip().lower()
            for item in analyses
        )
    )
    if not normalized or any(item not in {"ac", "op"} for item in normalized):
        raise ValueError("해석 종류는 AC, OP 또는 AC+OP 중 하나여야 합니다.")
    return normalized
