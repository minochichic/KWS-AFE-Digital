"""Unit parsing, formatting, scaling, and display-range helpers."""

from __future__ import annotations

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from .constants import (
    AUTO_RANGE_DATA_FRACTION,
    RESISTANCE_DECIMALS,
    VOLTAGE_DISPLAY_DECIMALS,
)


def spice_value(value: float, suffix: str) -> str:
    """Format a numeric component value with an ngspice scale suffix."""

    return f"{value:.12g}{suffix}"

def round_resistance_kohm(
    value: float,
    decimals: int = RESISTANCE_DECIMALS,
) -> float:
    """Round one positive resistance with decimal half-up semantics."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("저항값은 0보다 큰 유한한 값이어야 합니다.")
    if isinstance(decimals, bool) or int(decimals) != decimals or decimals < 0:
        raise ValueError("저항 소수 자릿수는 0 이상의 정수여야 합니다.")
    quantum = Decimal(1).scaleb(-int(decimals))
    rounded = float(
        Decimal(str(numeric)).quantize(quantum, rounding=ROUND_HALF_UP)
    )
    if rounded <= 0.0:
        raise ValueError(
            f"저항값은 소수점 {int(decimals)}자리 반올림 후 0보다 커야 합니다."
        )
    return rounded

def format_resistance_kohm(value: float) -> str:
    """Show every resistance with exactly two decimal places in kΩ."""

    rounded = round_resistance_kohm(value)
    return f"{rounded:.{RESISTANCE_DECIMALS}f}"

def resistance_spice_value(value: float) -> str:
    """Return a two-decimal kΩ value for generated netlists and overlays."""

    return f"{format_resistance_kohm(value)}k"

def parallel_resistance_kohm(
    first_kohm: float,
    second_kohm: float,
) -> float:
    """Calculate and two-decimal-round ``first || second`` in kΩ."""

    first = round_resistance_kohm(first_kohm)
    second = round_resistance_kohm(second_kohm)
    return round_resistance_kohm((first * second) / (first + second))

def format_millivolts(
    voltage_v: float,
    decimals: int = VOLTAGE_DISPLAY_DECIMALS,
) -> str:
    """Return a bare millivolt number for entry fields and result tables."""

    return f"{voltage_v * 1000.0:.{decimals}f}"

def format_voltage_mv(
    voltage_v: float,
    decimals: int = VOLTAGE_DISPLAY_DECIMALS,
) -> str:
    """Format an internal volt value for the schematic's millivolt display."""

    return f"{format_millivolts(voltage_v, decimals)} mV"

def voltage_margin_v(venv_dc_v: float, vthr_v: float) -> float:
    """Return DC threshold margin using ``Vmargin = Vthr - Venv_DC``."""

    if not math.isfinite(venv_dc_v) or not math.isfinite(vthr_v):
        raise ValueError("Venv_DC와 Vthr는 유한한 전압이어야 합니다.")
    return vthr_v - venv_dc_v

def format_margin_mv(
    voltage_v: float,
    decimals: int = VOLTAGE_DISPLAY_DECIMALS,
) -> str:
    """Format a signed ``Vthr - Venv_DC`` result in millivolts."""

    return f"{voltage_v * 1000.0:+.{decimals}f} mV"


def db_to_linear(magnitude_db: float) -> float:
    """Convert a voltage-ratio magnitude from dB without clipping small values."""

    value = float(magnitude_db)
    if math.isnan(value):
        raise ValueError("Magnitude dB는 NaN일 수 없습니다.")
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return math.inf
    return 10.0 ** (value / 20.0)


def linear_to_db(magnitude_linear: float) -> float:
    """Convert a nonnegative voltage-ratio magnitude to dB exactly."""

    value = float(magnitude_linear)
    if math.isnan(value) or value < 0.0:
        raise ValueError("Linear magnitude는 0 이상의 숫자여야 합니다.")
    if value == 0.0:
        return -math.inf
    return 20.0 * math.log10(value)


def linear_magnitude_auto_limits(
    values: Sequence[float],
    headroom_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return a zero-based linear-magnitude range with upper headroom."""

    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("Linear 자동 범위를 계산할 유효 데이터가 없습니다.")
    if any(value < 0.0 for value in finite):
        raise ValueError("Linear magnitude 데이터는 음수일 수 없습니다.")
    if not math.isfinite(headroom_fraction) or headroom_fraction <= 0.0:
        raise ValueError("Linear 자동 범위 여백은 0보다 커야 합니다.")
    maximum = max(finite)
    upper = maximum * (1.0 + headroom_fraction)
    if upper <= 0.0:
        upper = 1.0
    return 0.0, upper

def parse_time_seconds(text: str, label: str = "시간") -> float:
    """Parse seconds or a SPICE-style time suffix into positive seconds.

    Accepted examples are ``1e-5``, ``10u``, ``10us``, ``2.5ms``, and ``1s``.
    Keeping the displayed default as ``10u`` prevents the minus sign in
    ``1e-5`` from being overlooked while preserving the netlist's SI units.
    """

    match = re.fullmatch(
        r"\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
        r"\s*(s|ms|m|us|µs|μs|u|ns|n|ps|p)?\s*",
        text,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"{label}: 초 단위 숫자 또는 10u/5ms 같은 값을 입력하세요."
        )
    value = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    scale = {
        "": 1.0,
        "s": 1.0,
        "m": 1e-3,
        "ms": 1e-3,
        "u": 1e-6,
        "us": 1e-6,
        "µs": 1e-6,
        "μs": 1e-6,
        "n": 1e-9,
        "ns": 1e-9,
        "p": 1e-12,
        "ps": 1e-12,
    }[suffix]
    seconds = value * scale
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label}: 0보다 큰 유한한 값을 입력하세요.")
    return seconds

def transient_rows_to_ms_mv(
    rows: Sequence[tuple[float, float, float, float, float]],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Convert raw SI-unit rows into the transient plot's ms/mV series."""

    return (
        [row[0] * 1000.0 for row in rows],
        [row[1] * 1000.0 for row in rows],
        [row[2] * 1000.0 for row in rows],
        [row[3] * 1000.0 for row in rows],
        [row[4] * 1000.0 for row in rows],
    )

def data_occupancy_limits(
    values: Sequence[float],
    *,
    occupancy: float = AUTO_RANGE_DATA_FRACTION,
    logarithmic: bool = False,
) -> tuple[float, float]:
    """Return limits whose finite data span occupies ``occupancy`` of the axis.

    For a linear axis, equal padding is added above and below the data.  For a
    logarithmic axis the same operation is performed in log10 space, so the
    visual margin is symmetric.  A constant trace has no measurable span; a
    small scale-relative fallback window is used only for that degenerate case.
    """

    if not math.isfinite(occupancy) or not 0.0 < occupancy < 1.0:
        raise ValueError("자동 범위 데이터 점유율은 0과 1 사이여야 합니다.")
    finite = [
        float(value)
        for value in values
        if math.isfinite(value) and (not logarithmic or value > 0.0)
    ]
    if not finite:
        raise ValueError("자동 범위를 계산할 유효 데이터가 없습니다.")
    transformed = (
        [math.log10(value) for value in finite] if logarithmic else finite
    )
    data_low = min(transformed)
    data_high = max(transformed)
    data_span = data_high - data_low
    if data_span == 0.0:
        if logarithmic:
            fallback_span = math.log10(1.01)
        else:
            fallback_span = max(abs(data_low) * 1e-6, 1e-6)
        padded_low = data_low - fallback_span / 2.0
        padded_high = data_high + fallback_span / 2.0
    else:
        total_span = data_span / occupancy
        padding = (total_span - data_span) / 2.0
        padded_low = data_low - padding
        padded_high = data_high + padding
    if logarithmic:
        return 10.0**padded_low, 10.0**padded_high
    return padded_low, padded_high

def scale_pwl_pairs_to_vpp(
    pairs: Sequence[tuple[float, float]],
    target_vpp_v: float,
) -> list[tuple[float, float]]:
    """Scale a zero-DC PWL waveform to the requested peak-to-peak voltage.

    Multiplying every voltage by ``target/current`` preserves waveform shape,
    zero DC, and the explicit final 0 V tail.  A constant/silent PWL remains
    zero because a non-zero Vpp cannot be created without inventing a signal.
    """

    if not math.isfinite(target_vpp_v) or target_vpp_v <= 0.0:
        raise ValueError("입력 PWL Vpp는 0보다 큰 유한한 값이어야 합니다.")
    if len(pairs) < 2:
        raise ValueError("PWL Vpp 조정에는 최소 두 점이 필요합니다.")
    voltages = [float(voltage) for _time, voltage in pairs]
    if not all(math.isfinite(voltage) for voltage in voltages):
        raise ValueError("PWL 전압은 유한한 값이어야 합니다.")
    current_vpp = max(voltages) - min(voltages)
    if current_vpp <= 0.0:
        return [(float(time_s), 0.0) for time_s, _voltage in pairs]
    scale = target_vpp_v / current_vpp
    return [
        (float(time_s), float(voltage_v) * scale)
        for time_s, voltage_v in pairs
    ]

def recommended_maximum_step_s(
    highest_frequency_hz: float,
    points_per_cycle: int = 20,
) -> float:
    """Calculate the usual period/points upper bound for transient ``tmax``."""

    if (
        not math.isfinite(highest_frequency_hz)
        or highest_frequency_hz <= 0.0
        or points_per_cycle < 2
    ):
        raise ValueError("최고 주파수와 주기당 점 수가 잘못되었습니다.")
    return 1.0 / (highest_frequency_hz * points_per_cycle)
