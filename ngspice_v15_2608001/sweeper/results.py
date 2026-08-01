"""ngspice rawfile parsing, AC metrics, and normalized CSV output."""

from __future__ import annotations

import cmath
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .models import (
    AcMetrics,
    Channel,
    ChannelResult,
    DetectorMeasurement,
    DetectorSettings,
    TransientSettings,
)
from .values import db_to_linear


@dataclass(frozen=True)
class RawPlot:
    """Parsed metadata and samples from one ASCII ngspice rawfile.

    ``points`` holds ``float`` values for a real plot and ``complex`` values for
    a complex one.  Both types expose ``.real``, so every consumer that reads a
    real analysis keeps working while avoiding one boxed complex per sample.
    """

    plotname: str
    flags: frozenset[str]
    variables: tuple[str, ...]
    points: tuple[tuple[complex, ...], ...]

@dataclass(frozen=True)
class RawHeader:
    """Everything needed to interpret the value section that follows it."""

    plotname: str
    flags: frozenset[str]
    variables: tuple[str, ...]
    point_count: int

    @property
    def is_complex(self) -> bool:
        """Return whether each stored value is a real/imaginary token pair."""

        return "complex" in self.flags

    @property
    def variable_count(self) -> int:
        """Return how many vectors make up one data point."""

        return len(self.variables)

_RAW_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)
# One mebibyte keeps the transient token list small enough to stay cache and
# allocator friendly.  Larger blocks parse no faster but raise peak memory.
_RAW_VALUES_CHUNK_CHARS = 1 << 20

def _raw_numbers(text: str) -> list[float]:
    """Extract SPICE decimal/exponent tokens, including Fortran D exponents."""

    return [float(token.replace("D", "E").replace("d", "e")) for token in _RAW_NUMBER.findall(text)]

def _parse_raw_header(handle: object, path: Path) -> RawHeader:
    """Consume only the header lines and stop on the ``Values:`` marker."""

    headers: dict[str, str] = {}
    variables: list[str] = []
    in_variables = False
    found_values = False
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower == "variables:":
            in_variables = True
            continue
        if lower == "values:":
            found_values = True
            break
        if in_variables:
            parts = stripped.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if not found_values:
        raise ValueError(f"ASCII rawfile에 Values 섹션이 없습니다: {path}")
    try:
        variable_count = int(headers["no. variables"])
        point_count = int(headers["no. points"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"rawfile 헤더가 불완전합니다: {path}") from exc
    if len(variables) != variable_count:
        raise ValueError(
            f"rawfile 변수 수 불일치: 헤더 {variable_count}, 목록 {len(variables)}"
        )
    return RawHeader(
        plotname=headers.get("plotname", ""),
        flags=frozenset(headers.get("flags", "real").lower().split()),
        variables=tuple(variables),
        point_count=point_count,
    )

def _complex_token(token: str) -> complex:
    """Convert one ``real,imaginary`` rawfile token into a complex sample."""

    real_text, separator, imaginary_text = token.partition(",")
    if not separator:
        raise ValueError(f"복소 raw 값이 실수,허수 쌍이 아닙니다: {token!r}")
    return complex(float(real_text), float(imaginary_text))

def _fast_raw_points(
    handle: object,
    header: RawHeader,
    columns: Sequence[int] | None,
) -> list[tuple[float, ...]] | None:
    """Read the value section as fixed-width token records.

    Every data point is written as one index token followed by one token per
    vector, so the whole section can be split once per chunk instead of once per
    line.  ``None`` is returned whenever the observed layout does not match that
    contract, which lets the caller fall back to the tolerant line parser rather
    than risk emitting misaligned samples.
    """

    variable_count = header.variable_count
    stride = 1 + variable_count
    is_complex = header.is_complex
    # Offsets are measured from the point's index token, which is skipped.
    value_offsets = (
        tuple(1 + column for column in columns)
        if columns is not None
        else tuple(range(1, stride))
    )
    take_real = columns is not None
    points: list[tuple[float, ...]] = []
    carry: list[str] = []
    tail = ""
    while True:
        block = handle.read(_RAW_VALUES_CHUNK_CHARS)
        last = not block
        if last:
            block, tail = tail, ""
        else:
            block = tail + block
            cut = block.rfind("\n")
            if cut < 0:
                tail = block
                continue
            block, tail = block[:cut], block[cut + 1 :]
        tokens = block.split()
        if carry:
            tokens = carry + tokens
            carry = []
        complete = len(tokens) // stride
        # The loop body is chosen once per chunk instead of once per point
        # because this is the hot path for multi-hundred-megabyte transients.
        try:
            if is_complex:
                for position in range(complete):
                    base = position * stride
                    if not tokens[base].isdigit():
                        return None
                    values = [
                        _complex_token(tokens[base + offset])
                        for offset in value_offsets
                    ]
                    points.append(
                        tuple(value.real for value in values)
                        if take_real
                        else tuple(values)
                    )
            elif columns is None:
                for position in range(complete):
                    base = position * stride
                    if not tokens[base].isdigit():
                        return None
                    points.append(
                        tuple(map(float, tokens[base + 1 : base + stride]))
                    )
            else:
                for position in range(complete):
                    base = position * stride
                    if not tokens[base].isdigit():
                        return None
                    points.append(
                        tuple(
                            [
                                float(tokens[base + offset])
                                for offset in value_offsets
                            ]
                        )
                    )
        except ValueError:
            return None
        remainder = len(tokens) % stride
        carry = tokens[-remainder:] if remainder else []
        if last:
            break
    if carry or len(points) != header.point_count:
        return None
    return points

def _fallback_raw_points(
    path: Path,
    header: RawHeader,
) -> list[tuple[float, ...]]:
    """Re-read the value section line by line for non-standard layouts.

    This tolerant path also accepts Fortran ``D`` exponents and values split
    across lines in ways the fixed-width reader rejects.
    """

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    values_at = -1
    for index, line in enumerate(lines):
        if line.strip().lower() == "values:":
            values_at = index + 1
            break
    if values_at < 0:
        raise ValueError(f"ASCII rawfile에 Values 섹션이 없습니다: {path}")
    variable_count = header.variable_count
    is_complex = header.is_complex
    points: list[tuple[float, ...]] = []
    current: list[float] = []
    for line in lines[values_at:]:
        if not line.strip():
            continue
        payload = line
        if not current:
            point_match = re.match(r"^\s*(\d+)\s+(.*)$", line)
            if not point_match:
                continue
            payload = point_match.group(2)
        numbers = _raw_numbers(payload)
        if is_complex:
            if len(numbers) % 2:
                raise ValueError(f"복소 raw 값의 실수/허수 쌍이 깨졌습니다: {line!r}")
            current.extend(
                complex(numbers[i], numbers[i + 1])
                for i in range(0, len(numbers), 2)
            )
        else:
            current.extend(numbers)
        if len(current) == variable_count:
            points.append(tuple(current))
            current = []
        elif len(current) > variable_count:
            raise ValueError(f"rawfile 한 점의 변수 수가 너무 많습니다: {path}")
    if current:
        raise ValueError(f"rawfile 마지막 데이터 점이 불완전합니다: {path}")
    if len(points) != header.point_count:
        raise ValueError(
            f"rawfile 점 수 불일치: 헤더 {header.point_count}, 실제 {len(points)}"
        )
    return points

def read_ascii_rawfile(path: Path) -> RawPlot:
    """Read one ASCII ngspice rawfile created by command-line ``-r``."""

    if not path.is_file():
        raise FileNotFoundError(f"ngspice가 {path.name}을 만들지 않았습니다.")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = _parse_raw_header(handle, path)
        points = _fast_raw_points(handle, header, None)
    if points is None:
        points = _fallback_raw_points(path, header)
    return RawPlot(
        plotname=header.plotname,
        flags=header.flags,
        variables=header.variables,
        points=tuple(points),
    )

def _read_raw_columns(
    path: Path,
    requested_nodes: Sequence[str],
) -> list[tuple[float, ...]]:
    """Read only the requested vectors, skipping the all-vector intermediate."""

    if not path.is_file():
        raise FileNotFoundError(f"ngspice가 {path.name}을 만들지 않았습니다.")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = _parse_raw_header(handle, path)
        indices = tuple(
            _variable_index(header.variables, node) for node in requested_nodes
        )
        rows = _fast_raw_points(handle, header, indices)
    if rows is None:
        points = _fallback_raw_points(path, header)
        rows = [tuple(point[index].real for index in indices) for point in points]
    return rows

def _canonical_vector_name(name: str) -> str:
    """Normalize ``v(/node)`` and ``/node`` spellings for node lookup."""

    value = name.strip().lower()
    if value.startswith("v(") and value.endswith(")"):
        value = value[2:-1]
    return value.lstrip("/")

def _variable_index(variables: Sequence[str], requested_node: str) -> int:
    """Locate one requested voltage vector in a rawfile's variable list."""

    wanted = _canonical_vector_name(requested_node)
    for index, name in enumerate(variables):
        if _canonical_vector_name(name) == wanted:
            return index
    raise ValueError(
        f"rawfile에 V({requested_node})가 없습니다. 저장된 변수: "
        + ", ".join(variables)
    )

def _vector_index(plot: RawPlot, requested_node: str) -> int:
    """Locate one requested voltage vector in a parsed raw plot."""

    return _variable_index(plot.variables, requested_node)

def read_ac_raw(path: Path, output_node: str) -> list[tuple[float, float, float]]:
    """Convert a complex AC rawfile vector to frequency, dB, and phase rows."""

    plot = read_ascii_rawfile(path)
    output_index = _vector_index(plot, output_node)
    rows: list[tuple[float, float, float]] = []
    for point in plot.points:
        frequency = point[0].real
        output = point[output_index]
        magnitude = abs(output)
        magnitude_db = 20.0 * math.log10(magnitude) if magnitude > 0.0 else -math.inf
        rows.append((frequency, magnitude_db, math.degrees(cmath.phase(output))))
    return rows

def _quadratic_peak(
    left: tuple[float, float],
    middle: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    """Estimate a peak by fitting three dB samples against log10(frequency)."""
    x1, y1 = math.log10(left[0]), left[1]
    x2, y2 = math.log10(middle[0]), middle[1]
    x3, y3 = math.log10(right[0]), right[1]
    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denominator == 0.0:
        return middle
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
    b = (
        x3 * x3 * (y1 - y2)
        + x2 * x2 * (y3 - y1)
        + x1 * x1 * (y2 - y3)
    ) / denominator
    if not math.isfinite(a) or not math.isfinite(b) or a >= 0.0:
        return middle
    peak_x = -b / (2.0 * a)
    if not (min(x1, x3) <= peak_x <= max(x1, x3)):
        return middle
    c = y1 - a * x1 * x1 - b * x1
    peak_db = a * peak_x * peak_x + b * peak_x + c
    return 10.0**peak_x, peak_db

def _log_frequency_crossing(
    first: tuple[float, float],
    second: tuple[float, float],
    target_db: float,
) -> float | None:
    """Interpolate a target dB crossing linearly in log10(frequency)."""

    f1, y1 = first
    f2, y2 = second
    if f1 <= 0.0 or f2 <= 0.0 or y1 == y2:
        return None
    fraction = (target_db - y1) / (y2 - y1)
    if not 0.0 <= fraction <= 1.0:
        return None
    return 10.0 ** (math.log10(f1) + fraction * (math.log10(f2) - math.log10(f1)))

def calculate_ac_metrics(
    rows: Sequence[tuple[float, float, float]],
) -> AcMetrics | None:
    """Calculate peak center frequency and -3.0103 dB bandwidth Q."""
    valid = sorted(
        (
            (frequency, magnitude_db)
            for frequency, magnitude_db, _phase in rows
            if frequency > 0.0 and math.isfinite(frequency) and math.isfinite(magnitude_db)
        ),
        key=lambda item: item[0],
    )
    if len(valid) < 3:
        return None
    peak_index = max(range(len(valid)), key=lambda index: valid[index][1])
    sample_peak = valid[peak_index]
    if 0 < peak_index < len(valid) - 1:
        center_hz, peak_db = _quadratic_peak(
            valid[peak_index - 1], sample_peak, valid[peak_index + 1]
        )
    else:
        center_hz, peak_db = sample_peak
    target_db = peak_db - 10.0 * math.log10(2.0)

    low_hz: float | None = None
    for index in range(peak_index - 1, -1, -1):
        first, second = valid[index], valid[index + 1]
        if (first[1] - target_db) * (second[1] - target_db) <= 0.0:
            low_hz = _log_frequency_crossing(first, second, target_db)
            if low_hz is not None:
                break

    high_hz: float | None = None
    for index in range(peak_index, len(valid) - 1):
        first, second = valid[index], valid[index + 1]
        if (first[1] - target_db) * (second[1] - target_db) <= 0.0:
            high_hz = _log_frequency_crossing(first, second, target_db)
            if high_hz is not None:
                break

    if peak_index in {0, len(valid) - 1}:
        status = "피크가 스윕 경계에 있음"
    elif low_hz is None or high_hz is None:
        status = "−3 dB 교차점 측정 범위 부족"
    elif high_hz <= low_hz:
        status = "−3 dB 대역폭 계산 불가"
    else:
        q = center_hz / (high_hz - low_hz)
        return AcMetrics(center_hz, peak_db, low_hz, high_hz, q, "정상")
    return AcMetrics(center_hz, peak_db, low_hz, high_hz, None, status)

def read_op_raw(path: Path) -> dict[str, float]:
    """Return every voltage vector from a one-point operating-point rawfile."""
    plot = read_ascii_rawfile(path)
    if not plot.points:
        raise ValueError(f"동작점 rawfile에 데이터 점이 없습니다: {path}")
    point = plot.points[0]
    voltages: dict[str, float] = {"GND": 0.0}
    for index, vector in enumerate(plot.variables):
        match = re.fullmatch(r"v\((.+)\)", vector.strip(), re.I)
        if match:
            voltages[match.group(1)] = point[index].real
    if len(voltages) == 1:
        raise ValueError(
            f"동작점 rawfile에 전압 벡터가 없습니다. 저장된 변수: "
            + ", ".join(plot.variables)
        )
    return voltages

def _read_analog_rows(
    path: Path,
    nodes: Sequence[str],
    label: str,
) -> list[tuple[float, float, float, float, float]]:
    """Read one time-ordered analog table and reject unusable simulator data."""

    rows = _read_raw_columns(path, nodes)
    previous_time = -math.inf
    for values in rows:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{label} rawfile에 유한하지 않은 값이 있습니다: {path}"
            )
        time_s = values[0]
        if time_s < previous_time:
            raise ValueError(f"{label} rawfile 시간이 역순입니다: {path}")
        previous_time = time_s
    if len(rows) < 2:
        raise ValueError(f"{label} rawfile 데이터 점이 부족합니다: {path}")
    return rows


def read_transient_raw(
    path: Path,
    settings: TransientSettings,
) -> list[tuple[float, float, float, float, float]]:
    """Read time and the four analog voltages used by the transient viewer."""

    settings.validate()
    return _read_analog_rows(
        path,
        (
            "time",
            settings.vin_node,
            settings.vfilt_node,
            settings.venv_node,
            settings.vthr_node,
        ),
        "Transient",
    )


def read_detector_raw(
    path: Path,
    settings: DetectorSettings,
) -> list[tuple[float, float, float, float, float]]:
    """Read time plus /vin, /v_filt, /v_env, and /v_thr for Detector."""

    settings.validate()
    return _read_analog_rows(
        path,
        (
            "time",
            settings.vin_node,
            settings.vfilt_node,
            settings.venv_node,
            settings.vthr_node,
        ),
        "Detector",
    )


def pre_gate_baselines(
    rows: Sequence[tuple[float, float, float, float, float]],
    gate_on_s: float,
) -> tuple[float, float, float, float]:
    """Average each analog trace over samples strictly before Gate ON."""

    if not math.isfinite(gate_on_s) or gate_on_s <= 0.0:
        raise ValueError("AC 비교에는 0보다 큰 Gate ON 이전 구간이 필요합니다.")
    pre_rows = [row for row in rows if row[0] < gate_on_s]
    if not pre_rows:
        raise ValueError("Gate ON 이전 DC baseline 샘플이 없습니다.")
    return tuple(
        sum(row[index] for row in pre_rows) / len(pre_rows)
        for index in range(1, 5)
    )


def detector_ac_comparison_rows(
    rows: Sequence[tuple[float, float, float, float, float]],
    gate_on_s: float,
) -> tuple[
    list[tuple[float, float, float, float, float]],
    tuple[float, float, float, float],
]:
    """Remove pre-gate DC values without amplitude or time normalization.

    /vin, /v_filt, and /v_env each use their own pre-gate mean.  /v_thr is
    referenced to the /v_env pre-gate mean so its comparator threshold meaning
    is retained in AC comparison mode.
    """

    baselines = pre_gate_baselines(rows, gate_on_s)
    vin_pre, vfilt_pre, venv_pre, _vthr_pre = baselines
    converted = [
        (
            row[0],
            row[1] - vin_pre,
            row[2] - vfilt_pre,
            row[3] - venv_pre,
            row[4] - venv_pre,
        )
        for row in rows
    ]
    return converted, baselines


def default_detector_levels(
    rows: Sequence[tuple[float, float, float, float, float]],
    settings: DetectorSettings,
) -> tuple[float, float]:
    """Return editable 10 and 90 percent levels from the observed v_env swing."""

    settings.validate()
    _vin_pre, _vfilt_pre, venv_pre, _vthr_pre = pre_gate_baselines(
        rows,
        settings.gate_on_s,
    )
    gated = [
        row[3]
        for row in rows
        if settings.gate_on_s <= row[0] <= settings.gate_off_s
    ]
    if not gated:
        raise ValueError("Gate ON 구간의 v_env 샘플이 없습니다.")
    swing_v = max(gated) - venv_pre
    if not math.isfinite(swing_v) or swing_v <= 0.0:
        raise ValueError("v_env가 Gate 이전 baseline보다 상승하지 않았습니다.")
    return venv_pre + 0.1 * swing_v, venv_pre + 0.9 * swing_v


def _interpolated_crossing(
    first: tuple[float, float],
    second: tuple[float, float],
    level_v: float,
    direction: str,
) -> float | None:
    """Return a linearly interpolated up/down crossing between two samples."""

    t1, y1 = first
    t2, y2 = second
    if t2 <= t1 or y2 == y1:
        return None
    if direction == "up":
        crossed = y2 > y1 and y1 <= level_v <= y2
    elif direction == "down":
        crossed = y2 < y1 and y2 <= level_v <= y1
    else:
        raise ValueError(f"지원하지 않는 교차 방향: {direction!r}")
    if not crossed:
        return None
    fraction = (level_v - y1) / (y2 - y1)
    return t1 + fraction * (t2 - t1)


def _series_from_time(
    rows: Sequence[tuple[float, float, float, float, float]],
    start_s: float,
    value_index: int,
) -> list[tuple[float, float]]:
    """Create a series beginning exactly at start_s by linear interpolation."""

    if not rows:
        return []
    series: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        if row[0] < start_s:
            continue
        if row[0] == start_s or index == 0:
            series.append((row[0], row[value_index]))
        else:
            previous = rows[index - 1]
            span = row[0] - previous[0]
            if span <= 0.0:
                return []
            fraction = (start_s - previous[0]) / span
            start_value = previous[value_index] + fraction * (
                row[value_index] - previous[value_index]
            )
            series.append((start_s, start_value))
            series.append((row[0], row[value_index]))
        series.extend((item[0], item[value_index]) for item in rows[index + 1 :])
        break
    return series


def _ordered_crossing_pair(
    series: Sequence[tuple[float, float]],
    first_level_v: float,
    second_level_v: float,
    direction: str,
) -> tuple[float | None, float | None, str]:
    """Find the first ordered crossing pair and reject ripple-broken attempts."""

    first_time: float | None = None
    for left, right in zip(series, series[1:]):
        if first_time is None:
            first_time = _interpolated_crossing(
                left,
                right,
                first_level_v,
                direction,
            )
        if first_time is not None:
            second_time = _interpolated_crossing(
                left,
                right,
                second_level_v,
                direction,
            )
            if second_time is not None and second_time >= first_time:
                return first_time, second_time, ""
            if direction == "up" and right[1] < first_level_v:
                first_time = None
            if direction == "down" and right[1] > first_level_v:
                first_time = None
    first_name = "Vlow" if direction == "up" else "Vhigh"
    second_name = "Vhigh" if direction == "up" else "Vlow"
    if first_time is None:
        return None, None, f"{first_name} {direction} 교차점을 찾지 못했습니다."
    return first_time, None, f"{second_name} {direction} 교차점을 찾지 못했습니다."


def measure_detector_response(
    rows: Sequence[tuple[float, float, float, float, float]],
    settings: DetectorSettings,
    vlow_v: float,
    vhigh_v: float,
) -> DetectorMeasurement:
    """Measure headroom and first valid interpolated rise/fall crossing pairs."""

    settings.validate()
    low = float(vlow_v)
    high = float(vhigh_v)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Vlow와 Vhigh는 유한한 전압이어야 합니다.")
    if high <= low:
        raise ValueError("Vhigh는 Vlow보다 커야 합니다.")
    rise_series = _series_from_time(rows, settings.gate_on_s, 3)
    fall_series = _series_from_time(rows, settings.gate_off_s, 3)
    rise_low, rise_high, rise_reason = _ordered_crossing_pair(
        rise_series,
        low,
        high,
        "up",
    )
    fall_high, fall_low, fall_reason = _ordered_crossing_pair(
        fall_series,
        high,
        low,
        "down",
    )
    rise_success = rise_low is not None and rise_high is not None
    fall_success = fall_high is not None and fall_low is not None
    return DetectorMeasurement(
        vlow_v=low,
        vhigh_v=high,
        headroom_v=high - low,
        rise_time_s=(rise_high - rise_low) if rise_success else None,
        fall_time_s=(fall_low - fall_high) if fall_success else None,
        rise_low_s=rise_low,
        rise_high_s=rise_high,
        fall_high_s=fall_high,
        fall_low_s=fall_low,
        rise_success=rise_success,
        fall_success=fall_success,
        rise_reason=rise_reason,
        fall_reason=fall_reason,
    )

def write_normalized_csv(
    path: Path, headers: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    """Write one normalized result table using a stable UTF-8 CSV format."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

def write_ac_metrics_summary(
    path: Path,
    results: Sequence[ChannelResult],
) -> None:
    """Write a session-level AC metrics table for every completed AC channel."""

    rows = []
    for result in sorted(results, key=lambda item: item.channel.ch):
        metric = result.ac_metrics
        if not result.ac_rows:
            continue
        rows.append(
            (
                result.channel.ch,
                metric.center_hz if metric else "",
                metric.peak_db if metric else "",
                db_to_linear(metric.peak_db) if metric else "",
                metric.q if metric and metric.q is not None else "",
                metric.status if metric else "데이터 부족",
            )
        )
    if rows:
        write_normalized_csv(
            path,
            ("ch", "center_hz", "gain_db", "gain_linear", "q", "status"),
            rows,
        )

def format_metric_copy_line(
    channel: Channel,
    metric: AcMetrics | None,
    magnitude_unit: str = "dB",
) -> str:
    """Build one AC summary line using the currently displayed gain unit."""

    if metric is None:
        return f"ch{channel.ch:02d}\tf0=—\tgain=—\tQ=—"
    if magnitude_unit == "dB":
        gain_text = f"{metric.peak_db:.7g} dB"
    elif magnitude_unit == "Linear":
        gain_text = f"{db_to_linear(metric.peak_db):.7g} Linear"
    else:
        raise ValueError(f"지원하지 않는 Magnitude 단위: {magnitude_unit!r}")
    q_text = "—" if metric.q is None else f"{metric.q:.7g}"
    return (
        f"ch{channel.ch:02d}\t"
        f"f0={metric.center_hz:.9g} Hz\t"
        f"gain={gain_text}\t"
        f"Q={q_text}"
    )
