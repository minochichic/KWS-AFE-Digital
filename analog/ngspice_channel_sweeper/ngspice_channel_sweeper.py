#!/usr/bin/env python3
"""16-channel ngspice AC/OP/transient simulator with a Tk GUI.

The component table stays outside the circuit.  For every selected channel this
program creates a parameter include file, injects it into the common netlist,
runs AC and DC operating point as separate ngspice rawfile jobs, normalizes
them to CSV, and presents interactive plots and annotated schematics.  A
separate WAV-to-PWL workflow prepares 0 V-DC, 10 mVpp speech stimuli, and the
transient workflow can rescale their Vpp at run time without changing the
original PWL files or the AC/OP contract.
"""

from __future__ import annotations

import argparse
import bisect
import cmath
import csv
import math
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ngspice_runner import resolve_ngspice, run_ngspice
from wav_pwl import (
    convert_wav_tree,
    discover_pwl_files,
    discover_wav_files,
    read_pwl_data,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_TABLE = APP_DIR / "channel_components.csv"
DEFAULT_NETLIST = APP_DIR / "netlist_template.cir"
DEFAULT_RESULTS = APP_DIR / "results"
DEFAULT_SCHEMATIC = APP_DIR / "circuit_template.png"
DEFAULT_COMPONENT_VERSIONS = APP_DIR / "component_versions"
AC_RAW_FILENAME = "ac.raw"
OP_RAW_FILENAME = "op.raw"
TRAN_RAW_FILENAME = "tran.raw"
TRANSIENT_OUTPUT_STEP_DEFAULT_S = 10e-6
TRANSIENT_MAXIMUM_STEP_DEFAULT_S = 5e-6
TRANSIENT_INPUT_VPP_DEFAULT_V = 0.010
AUTO_RANGE_DATA_FRACTION = 0.90
DIVIDER_SUPPLY_V = 1.8
DIVIDER_NOMINAL_TOTAL_KOHM = 1000.0
RESISTANCE_DECIMALS = 2
DIVIDER_RESISTANCE_DECIMALS = RESISTANCE_DECIMALS
OP_DEFAULT_SCHEMATIC_FRACTION = 0.82

PARAM_NAMES = ("RA", "R1", "R2", "R4", "R5", "R6", "R7", "R8", "C1", "C3")
CHANNEL_CSV_COLUMNS = (
    "ch",
    "f_c_hz",
    "Q",
    "RA_kohm",
    "C_nF",
    "R1_kohm",
    "gain_dB",
    "R2_kohm",
    "R4_kohm",
    "R5_kohm",
    "R6_kohm",
    "C3_nF",
    "R7_kohm",
    "R8_kohm",
    "Vthr_V",
)
COMPONENT_EDIT_FIELDS = (
    ("RA", "RA1, RA2 [kΩ]", "ra_kohm"),
    ("R1", "R1 [kΩ]", "r1_kohm"),
    ("R2", "R2, R3 [kΩ]", "r2_kohm"),
    ("R4", "R4 [kΩ]", "r4_kohm"),
    ("R5", "R5 [kΩ]", "r5_kohm"),
    ("R6", "R6 [kΩ]", "r6_kohm"),
    ("R7", "R7 [kΩ]", "r7_kohm"),
    ("R8", "R8 [kΩ]", "r8_kohm"),
    ("C1", "C1, C2 [nF]", "c_nf"),
    ("C3", "C3 [nF]", "c3_nf"),
)
RESISTANCE_EDIT_KEYS = frozenset(
    {"RA", "R1", "R2", "R4", "R5", "R6", "R7", "R8"}
)
REQUIRED_COLUMNS = {
    "ch",
    "f_c_hz",
    "Q",
    "RA_kohm",
    "C_nF",
    "R1_kohm",
    "R7_kohm",
    "R8_kohm",
}
DEFAULT_COMMON = {
    "R2_kohm": 100.0,
    "R4_kohm": 10.0,
    "R5_kohm": 47.0,
    "R6_kohm": 8.25,
    "C3_nF": 100.0,
}


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
    output_node: str = "/v_filt_out"
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
    vin_node: str = "/vin"
    vfilt_node: str = "/v_filt_out"
    venv_node: str = "/v_detect_out"
    vref_node: str = "/v_ref"
    pspice_compat: bool = True

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
            ("Vin", self.vin_node),
            ("Vfilt_out", self.vfilt_node),
            ("Venv_out", self.venv_node),
            ("Vref", self.vref_node),
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


def parse_float(row: dict[str, str], name: str, default: float | None = None) -> float:
    """Read one finite floating-point CSV field, optionally using a default."""

    raw = row.get(name, "")
    if raw is None or not raw.strip():
        if default is None:
            raise ValueError(f"필수 열 {name!r}의 값이 비었습니다.")
        return default
    value = float(raw.strip())
    if not math.isfinite(value):
        raise ValueError(f"{name!r} 값은 유한한 숫자여야 합니다.")
    return value


def divider_vref_v(
    r7_kohm: float,
    r8_kohm: float,
    supply_v: float = DIVIDER_SUPPLY_V,
) -> float:
    """Calculate Vref for the R7-top/R8-bottom divider in the template.

    The netlist connects R7 from 1.8 V to ``/v_ref`` and R8 from ``/v_ref`` to
    ground, so the unloaded divider equation is
    ``Vref = supply * R8 / (R7 + R8)``.
    """

    values = (float(r7_kohm), float(r8_kohm), float(supply_v))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("R7, R8, 분압 전원은 유한한 값이어야 합니다.")
    if r7_kohm <= 0.0 or r8_kohm <= 0.0 or supply_v <= 0.0:
        raise ValueError("R7, R8, 분압 전원은 0보다 커야 합니다.")
    return supply_v * r8_kohm / (r7_kohm + r8_kohm)


def divider_resistors_for_vref(
    vref_v: float,
    nominal_total_kohm: float = DIVIDER_NOMINAL_TOTAL_KOHM,
    supply_v: float = DIVIDER_SUPPLY_V,
    decimals: int = DIVIDER_RESISTANCE_DECIMALS,
) -> tuple[float, float]:
    """Return independently rounded R7/R8 values for a requested Vref.

    ``nominal_total_kohm`` preserves the divider's approximate loading/current
    scale.  It is only used to form the ideal pair; each resistor is then
    rounded independently, so no post-correction forces their sum to the
    nominal value.
    """

    values = (
        float(vref_v),
        float(nominal_total_kohm),
        float(supply_v),
        float(decimals),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "Vref, 기준 총 저항, 분압 전원, 소수 자릿수는 유한해야 합니다."
        )
    if nominal_total_kohm <= 0.0 or supply_v <= 0.0:
        raise ValueError("기준 총 저항과 분압 전원은 0보다 커야 합니다.")
    if isinstance(decimals, bool) or int(decimals) != decimals or decimals < 0:
        raise ValueError("저항 소수 자릿수는 0 이상의 정수여야 합니다.")
    if not 0.0 < vref_v < supply_v:
        raise ValueError(
            f"계산 Vref는 0보다 크고 {supply_v * 1000:g} mV보다 "
            "작아야 합니다."
        )
    r8_ideal_kohm = nominal_total_kohm * vref_v / supply_v
    r7_ideal_kohm = nominal_total_kohm - r8_ideal_kohm
    quantum = Decimal(1).scaleb(-int(decimals))
    r7_kohm = float(
        Decimal(str(r7_ideal_kohm)).quantize(quantum, rounding=ROUND_HALF_UP)
    )
    r8_kohm = float(
        Decimal(str(r8_ideal_kohm)).quantize(quantum, rounding=ROUND_HALF_UP)
    )
    if r7_kohm <= 0.0 or r8_kohm <= 0.0:
        raise ValueError(
            "요청한 Vref는 지정한 저항 분해능에서 양수 R7/R8로 만들 수 없습니다."
        )
    return r7_kohm, r8_kohm


def vref_from_margin_v(
    vdet_v: float,
    margin_v: float,
    supply_v: float = DIVIDER_SUPPLY_V,
) -> float:
    """Derive Vref from ``Vmargin = Vref - Vdet`` and validate the rail range.

    A positive margin is the detector's remaining headroom to the comparator
    threshold at the DC operating point.  The detector must rise by this
    amount before ``Vdet > Vref`` becomes true.
    """

    values = (float(vdet_v), float(margin_v), float(supply_v))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Vdet, Vmargin, 분압 전원은 유한한 값이어야 합니다.")
    if supply_v <= 0.0:
        raise ValueError("분압 전원은 0보다 커야 합니다.")
    if margin_v <= 0.0:
        raise ValueError(
            "목표 Vmargin은 Vref가 Vdet보다 높도록 0보다 커야 합니다."
        )
    vref_v = vdet_v + margin_v
    if not 0.0 < vref_v < supply_v:
        raise ValueError(
            "Vref = Vdet + Vmargin 결과는 "
            f"0보다 크고 {supply_v * 1000:g} mV보다 작아야 합니다."
        )
    return vref_v


def load_channels(path: Path) -> list[Channel]:
    """Load, validate, and channel-sort a component CSV without modifying it."""

    if not path.is_file():
        raise FileNotFoundError(f"소자 테이블을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError("소자 테이블 필수 열 누락: " + ", ".join(sorted(missing)))
        channels: list[Channel] = []
        seen: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                ch = int(row["ch"].strip())
                if ch in seen:
                    raise ValueError(f"중복 채널 {ch}")
                seen.add(ch)
                r7 = round_resistance_kohm(parse_float(row, "R7_kohm"))
                r8 = round_resistance_kohm(parse_float(row, "R8_kohm"))
                divider = divider_vref_v(r7, r8)
                channel = Channel(
                    ch=ch,
                    f_c_hz=parse_float(row, "f_c_hz"),
                    q=parse_float(row, "Q"),
                    ra_kohm=round_resistance_kohm(
                        parse_float(row, "RA_kohm")
                    ),
                    c_nf=parse_float(row, "C_nF"),
                    r1_kohm=round_resistance_kohm(
                        parse_float(row, "R1_kohm")
                    ),
                    gain_db=parse_float(row, "gain_dB", 0.0),
                    r2_kohm=round_resistance_kohm(
                        parse_float(
                            row, "R2_kohm", DEFAULT_COMMON["R2_kohm"]
                        )
                    ),
                    r4_kohm=round_resistance_kohm(
                        parse_float(
                            row, "R4_kohm", DEFAULT_COMMON["R4_kohm"]
                        )
                    ),
                    r5_kohm=round_resistance_kohm(
                        parse_float(
                            row, "R5_kohm", DEFAULT_COMMON["R5_kohm"]
                        )
                    ),
                    r6_kohm=round_resistance_kohm(
                        parse_float(
                            row, "R6_kohm", DEFAULT_COMMON["R6_kohm"]
                        )
                    ),
                    c3_nf=parse_float(row, "C3_nF", DEFAULT_COMMON["C3_nF"]),
                    r7_kohm=r7,
                    r8_kohm=r8,
                    vthr_v=parse_float(row, "Vthr_V", divider),
                )
                if abs(channel.vthr_v - divider) > 0.001:
                    raise ValueError(
                        f"ch{ch}: CSV Vref={channel.vthr_v:.4f} V와 "
                        f"R7/R8 분압값={divider:.4f} V가 1 mV 넘게 다릅니다."
                    )
                channels.append(channel)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path.name} {line_number}행: {exc}") from exc
    if not channels:
        raise ValueError("소자 테이블에 채널이 없습니다.")
    return sorted(channels, key=lambda item: item.ch)


def channel_to_csv_row(channel: Channel) -> dict[str, object]:
    """Convert one normalized channel back to the canonical version-CSV schema."""

    return {
        "ch": channel.ch,
        "f_c_hz": channel.f_c_hz,
        "Q": channel.q,
        "RA_kohm": format_resistance_kohm(channel.ra_kohm),
        "C_nF": channel.c_nf,
        "R1_kohm": format_resistance_kohm(channel.r1_kohm),
        "gain_dB": channel.gain_db,
        "R2_kohm": format_resistance_kohm(channel.r2_kohm),
        "R4_kohm": format_resistance_kohm(channel.r4_kohm),
        "R5_kohm": format_resistance_kohm(channel.r5_kohm),
        "R6_kohm": format_resistance_kohm(channel.r6_kohm),
        "C3_nF": channel.c3_nf,
        "R7_kohm": format_resistance_kohm(channel.r7_kohm),
        "R8_kohm": format_resistance_kohm(channel.r8_kohm),
        "Vthr_V": channel.vthr_v,
    }


def write_channels_csv(path: Path, channels: Sequence[Channel]) -> None:
    """Write a complete component snapshot while protecting the factory CSV.

    ``channel_components.csv`` is the immutable factory source.  Every user
    save and every simulation snapshot must target a different path.
    """

    target = path.expanduser().resolve()
    if target == DEFAULT_TABLE.resolve():
        raise PermissionError(
            "channel_components.csv는 읽기 전용 기본값이므로 덮어쓸 수 없습니다."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHANNEL_CSV_COLUMNS)
        writer.writeheader()
        for channel in sorted(channels, key=lambda item: item.ch):
            writer.writerow(channel_to_csv_row(channel))


def next_component_version_path(
    version_dir: Path = DEFAULT_COMPONENT_VERSIONS,
    timestamp: datetime | None = None,
) -> Path:
    """Return a collision-free timestamped filename for a component version."""

    moment = timestamp or datetime.now()
    stem = f"channel_components_v{moment.strftime('%Y%m%d_%H%M%S')}"
    directory = version_dir.expanduser().resolve()
    candidate = directory / f"{stem}.csv"
    serial = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{serial:02d}.csv"
        serial += 1
    return candidate


def save_component_version(
    channels: Sequence[Channel],
    version_dir: Path = DEFAULT_COMPONENT_VERSIONS,
) -> Path:
    """Persist all applied channels as a new version; never overwrite a version."""

    target = next_component_version_path(version_dir)
    write_channels_csv(target, channels)
    return target


def apply_component_updates(
    channel: Channel,
    values: Mapping[str, str],
    requested_margin_mv: str | None = None,
    reference_vdet_v: float | None = None,
    divider_nominal_total_kohm: float | None = None,
) -> Channel:
    """Validate editor text and return a new immutable channel instance.

    Normally R7/R8 determine ``vthr_v``.  When Vmargin is the most recently
    changed divider control, ``Vref = Vdet + Vmargin`` is converted into an
    independently rounded 0.01 kΩ R7/R8 pair.  The explicitly supplied
    nominal divider total, or the channel's previous total when omitted, keeps
    the loading scale stable without forcing the rounded pair to an exact sum.
    """

    updates: dict[str, float] = {}
    for key, label, attribute in COMPONENT_EDIT_FIELDS:
        raw = values.get(key, "").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{label}: 숫자를 입력하세요.") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}: 0보다 큰 유한한 값을 입력하세요.")
        updates[attribute] = (
            round_resistance_kohm(value)
            if key in RESISTANCE_EDIT_KEYS
            else value
        )
    if requested_margin_mv is not None:
        if reference_vdet_v is None:
            raise ValueError(
                "Vmargin으로 Vref를 계산하려면 최근 DC Vdet 결과가 필요합니다."
            )
        try:
            margin_v = float(requested_margin_mv.strip()) / 1000.0
        except ValueError as exc:
            raise ValueError("목표 Vmargin: mV 단위 숫자를 입력하세요.") from exc
        vref_v = vref_from_margin_v(reference_vdet_v, margin_v)
        nominal_total_kohm = (
            channel.r7_kohm + channel.r8_kohm
            if divider_nominal_total_kohm is None
            else float(divider_nominal_total_kohm)
        )
        r7_kohm, r8_kohm = divider_resistors_for_vref(
            vref_v,
            nominal_total_kohm=nominal_total_kohm,
        )
        updates["r7_kohm"] = r7_kohm
        updates["r8_kohm"] = r8_kohm
    updated = replace(channel, **updates)
    threshold = divider_vref_v(updated.r7_kohm, updated.r8_kohm)
    return replace(updated, vthr_v=threshold)


def format_voltage_mv(voltage_v: float, digits: int = 7) -> str:
    """Format an internal volt value for the schematic's millivolt display."""

    return f"{voltage_v * 1000.0:.{digits}g} mV"


def voltage_margin_v(vdet_v: float, vref_v: float) -> float:
    """Return threshold headroom using ``V_margin = Vref - V_det``."""

    if not math.isfinite(vdet_v) or not math.isfinite(vref_v):
        raise ValueError("V_det와 Vref는 유한한 전압이어야 합니다.")
    return vref_v - vdet_v


def format_margin_mv(voltage_v: float, digits: int = 7) -> str:
    """Format a signed ``Vref - Vdet`` result in millivolts."""

    return f"{voltage_v * 1000.0:+.{digits}g} mV"


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


def format_metric_copy_line(
    channel: Channel,
    metric: AcMetrics | None,
) -> str:
    """Build one tab-separated, spreadsheet-friendly AC summary line."""

    if metric is None:
        return f"ch{channel.ch:02d}\tf0=—\tgain=—\tQ=—"
    q_text = "—" if metric.q is None else f"{metric.q:.7g}"
    return (
        f"ch{channel.ch:02d}\t"
        f"f0={metric.center_hz:.9g} Hz\t"
        f"gain={metric.peak_db:.7g} dB\t"
        f"Q={q_text}"
    )


def make_parameter_include(channel: Channel) -> str:
    """Render the per-channel ``.param`` include consumed by every analysis."""

    p = channel.spice_parameters()
    return "\n".join(
        [
            f"* Auto-generated parameters for channel {channel.ch}",
            f"* fc={channel.f_c_hz:.12g}Hz Q={channel.q:.12g} "
            f"gain={channel.gain_db:.12g}dB Vref={channel.vthr_v:.6f}V",
            ".param " + " ".join(f"{name}={p[name]}" for name in PARAM_NAMES),
            "",
        ]
    )


def strip_generated_sections(lines: Sequence[str]) -> list[str]:
    """Remove prior generated analyses/control blocks before adding a clean job."""

    clean: list[str] = []
    in_control = False
    analysis = re.compile(
        r"^\s*\.(?:ac|dc|tran|op|tf|noise|pz|disto|save)\b",
        re.I,
    )
    param_assignment = re.compile(
        r"(?i)(?:^|\s)(?:" + "|".join(map(re.escape, PARAM_NAMES)) + r")\s*=\s*[^\s]+"
    )
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\.control\b", stripped, re.I):
            in_control = True
            continue
        if in_control:
            if re.match(r"^\.endc\b", stripped, re.I):
                in_control = False
            continue
        if re.match(r"^\.end\b", stripped, re.I):
            continue
        if analysis.match(line):
            continue
        if re.match(r"^\s*\.param\b", line, re.I):
            remainder = param_assignment.sub(" ", line)
            remainder = re.sub(r"\s+", " ", remainder).strip()
            if remainder.lower() == ".param":
                continue
            line = remainder
        clean.append(line.rstrip())
    return clean


def spice_path(path: Path) -> str:
    """Convert a local include path to ngspice's portable forward-slash form."""

    return path.resolve().as_posix().replace('"', r"\"")


def analysis_directives(
    settings: SweepSettings,
    analysis: str,
) -> list[str]:
    """Return only the save/analysis statements for one AC or OP job.

    Transient has its own settings and ``transient_directives()`` so time
    controls cannot accidentally leak into the AC/OP execution path.
    """

    if analysis == "ac":
        return [
            f".save v({settings.output_node})",
            (
                f".ac dec {settings.ac_points_per_decade} "
                f"{settings.ac_start_hz:.12g} {settings.ac_stop_hz:.12g}"
            ),
        ]
    if analysis == "op":
        return [".save all", ".op"]
    raise ValueError(f"지원하지 않는 해석 종류: {analysis!r}")


def make_analysis_netlist(
    template_text: str,
    include_path: Path,
    settings: SweepSettings,
    analysis: str,
) -> str:
    """Build one standalone AC or OP netlist from the shared circuit template."""

    analysis = analysis.lower()
    if analysis == "dc":
        analysis = "op"
    if analysis not in {"ac", "op"}:
        raise ValueError(f"지원하지 않는 해석 종류: {analysis!r}")
    settings.validate((analysis,))
    lines = strip_generated_sections(template_text.splitlines())
    insert_at = 1
    for index, line in enumerate(lines):
        if re.match(r"^\s*\.include\b", line, re.I):
            insert_at = index + 1
    lines.insert(insert_at, f'.include "{spice_path(include_path)}"')
    lines.append("")
    lines.extend(analysis_directives(settings, analysis))
    lines.extend([".end", ""])
    return "\n".join(lines)


def build_analysis_jobs(
    run_dir: Path,
    channel_number: int,
    analyses: Sequence[str],
) -> tuple[AnalysisJob, ...]:
    """Resolve generic per-analysis netlist/raw paths for one channel."""

    jobs: list[AnalysisJob] = []
    for key in normalize_analyses(analyses):
        spec = ANALYSIS_JOB_SPECS[key]
        jobs.append(
            AnalysisJob(
                spec=spec,
                netlist_path=(
                    run_dir
                    / f"channel_{channel_number:02d}_{spec.netlist_suffix}.cir"
                ),
                raw_path=run_dir / spec.raw_filename,
            )
        )
    return tuple(jobs)


def find_input_voltage_source(
    template_text: str,
    input_node: str = "/vin",
) -> tuple[str, str, str]:
    """Find the existing independent source connected to the input node.

    The source may be named ``V4``, ``Vsin``, or another valid SPICE name.  Its
    original terminal order is preserved so the PWL waveform has the same sign
    as the source that it replaces.
    """

    wanted = input_node.strip().casefold()
    for line in template_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".", "+")):
            continue
        parts = stripped.split()
        if (
            len(parts) >= 3
            and parts[0].casefold().startswith("v")
            and wanted in {parts[1].casefold(), parts[2].casefold()}
        ):
            return parts[0], parts[1], parts[2]
    raise ValueError(
        f"네트리스트에서 {input_node!r}에 연결된 독립 전압원을 찾지 못했습니다."
    )


def remove_input_voltage_source(
    lines: Sequence[str],
    source_name: str,
) -> list[str]:
    """Remove one voltage-source statement and all of its continuation lines."""

    clean: list[str] = []
    removing_continuations = False
    for line in lines:
        stripped = line.strip()
        if removing_continuations and stripped.startswith("+"):
            continue
        removing_continuations = False
        parts = stripped.split()
        if parts and parts[0].casefold() == source_name.casefold():
            removing_continuations = True
            continue
        clean.append(line)
    return clean


def make_pwl_source_include(
    source_name: str,
    positive_node: str,
    negative_node: str,
    pairs: Sequence[tuple[float, float]],
    source_path: Path | None = None,
) -> str:
    """Render validated PWL pairs as one independent voltage-source include."""

    if len(pairs) < 2:
        raise ValueError("PWL 전압원에는 최소 두 점이 필요합니다.")
    previous_time = -math.inf
    for time_s, voltage_v in pairs:
        if (
            not math.isfinite(time_s)
            or not math.isfinite(voltage_v)
            or time_s < 0.0
            or time_s <= previous_time
        ):
            raise ValueError("PWL 시간은 0 이상이며 엄격히 증가해야 합니다.")
        previous_time = time_s
    lines = [
        "* Auto-generated transient stimulus include",
        f"* source_data={source_path.resolve() if source_path else '<memory>'}",
        (
            f"{source_name} {positive_node} {negative_node} "
            f"PWL({pairs[0][0]:.12g} {pairs[0][1]:.12g}"
        ),
    ]
    for time_s, voltage_v in pairs[1:-1]:
        lines.append(f"+ {time_s:.12g} {voltage_v:.12g}")
    last_time, last_voltage = pairs[-1]
    lines.append(f"+ {last_time:.12g} {last_voltage:.12g})")
    lines.append("")
    return "\n".join(lines)


def transient_directives(
    settings: TransientSettings,
    stop_time_s: float,
) -> list[str]:
    """Return saved-node and ``.tran`` statements for one transient job."""

    settings.validate()
    if not math.isfinite(stop_time_s) or stop_time_s <= 0.0:
        raise ValueError("Transient stop time은 0보다 커야 합니다.")
    nodes = (
        settings.vin_node,
        settings.vfilt_node,
        settings.venv_node,
        settings.vref_node,
    )
    return [
        ".save " + " ".join(f"v({node})" for node in nodes),
        (
            f".tran {settings.output_step_s:.12g} {stop_time_s:.12g} "
            f"0 {settings.maximum_step_s:.12g}"
        ),
    ]


def make_transient_netlist(
    template_text: str,
    parameter_include_path: Path,
    stimulus_include_path: Path,
    settings: TransientSettings,
    stop_time_s: float,
) -> str:
    """Build a transient-only netlist that replaces the existing input source."""

    settings.validate()
    source_name, _positive_node, _negative_node = find_input_voltage_source(
        template_text, settings.vin_node
    )
    lines = strip_generated_sections(template_text.splitlines())
    lines = remove_input_voltage_source(lines, source_name)
    insert_at = 1
    for index, line in enumerate(lines):
        if re.match(r"^\s*\.include\b", line, re.I):
            insert_at = index + 1
    lines.insert(
        insert_at,
        f'.include "{spice_path(parameter_include_path)}"',
    )
    lines.insert(
        insert_at + 1,
        f'.include "{spice_path(stimulus_include_path)}"',
    )
    lines.append("")
    lines.extend(transient_directives(settings, stop_time_s))
    lines.extend([".end", ""])
    return "\n".join(lines)


@dataclass(frozen=True)
class RawPlot:
    """Parsed metadata and complex samples from one ASCII ngspice rawfile."""

    plotname: str
    flags: frozenset[str]
    variables: tuple[str, ...]
    points: tuple[tuple[complex, ...], ...]


_RAW_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)


def _raw_numbers(text: str) -> list[float]:
    """Extract SPICE decimal/exponent tokens, including Fortran D exponents."""

    return [float(token.replace("D", "E").replace("d", "e")) for token in _RAW_NUMBER.findall(text)]


def read_ascii_rawfile(path: Path) -> RawPlot:
    """Read one ASCII ngspice rawfile created by command-line ``-r``."""
    if not path.is_file():
        raise FileNotFoundError(f"ngspice가 {path.name}을 만들지 않았습니다.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers: dict[str, str] = {}
    variables: list[str] = []
    values_at = -1
    in_variables = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower == "variables:":
            in_variables = True
            continue
        if lower == "values:":
            values_at = index + 1
            in_variables = False
            break
        if in_variables:
            parts = stripped.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if values_at < 0:
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
    flags = frozenset(headers.get("flags", "real").lower().split())
    is_complex = "complex" in flags
    points: list[tuple[complex, ...]] = []
    current: list[complex] = []
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
            current.extend(complex(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2))
        else:
            current.extend(complex(value, 0.0) for value in numbers)
        if len(current) == variable_count:
            points.append(tuple(current))
            current = []
        elif len(current) > variable_count:
            raise ValueError(f"rawfile 한 점의 변수 수가 너무 많습니다: {path}")
    if current:
        raise ValueError(f"rawfile 마지막 데이터 점이 불완전합니다: {path}")
    if len(points) != point_count:
        raise ValueError(
            f"rawfile 점 수 불일치: 헤더 {point_count}, 실제 {len(points)}"
        )
    return RawPlot(
        plotname=headers.get("plotname", ""),
        flags=flags,
        variables=tuple(variables),
        points=tuple(points),
    )


def _canonical_vector_name(name: str) -> str:
    """Normalize ``v(/node)`` and ``/node`` spellings for node lookup."""

    value = name.strip().lower()
    if value.startswith("v(") and value.endswith(")"):
        value = value[2:-1]
    return value.lstrip("/")


def _vector_index(plot: RawPlot, requested_node: str) -> int:
    """Locate one requested voltage vector in a parsed raw plot."""

    wanted = _canonical_vector_name(requested_node)
    for index, name in enumerate(plot.variables):
        if _canonical_vector_name(name) == wanted:
            return index
    raise ValueError(
        f"rawfile에 V({requested_node})가 없습니다. 저장된 변수: "
        + ", ".join(plot.variables)
    )


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


def read_transient_raw(
    path: Path,
    settings: TransientSettings,
) -> list[tuple[float, float, float, float, float]]:
    """Read time and the four analog voltages used by the transient viewer."""

    settings.validate()
    plot = read_ascii_rawfile(path)
    indices = (
        _vector_index(plot, "time"),
        _vector_index(plot, settings.vin_node),
        _vector_index(plot, settings.vfilt_node),
        _vector_index(plot, settings.venv_node),
        _vector_index(plot, settings.vref_node),
    )
    rows: list[tuple[float, float, float, float, float]] = []
    previous_time = -math.inf
    for point in plot.points:
        values = tuple(point[index].real for index in indices)
        time_s = values[0]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Transient rawfile에 유한하지 않은 값이 있습니다: {path}"
            )
        if time_s < previous_time:
            raise ValueError(f"Transient rawfile 시간이 역순입니다: {path}")
        previous_time = time_s
        rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"Transient rawfile 데이터 점이 부족합니다: {path}")
    return rows


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
                metric.q if metric and metric.q is not None else "",
                metric.status if metric else "데이터 부족",
            )
        )
    if rows:
        write_normalized_csv(
            path,
            ("ch", "center_hz", "gain_db", "q", "status"),
            rows,
        )


def find_ngspice(configured: str = "") -> str:
    """Backward-compatible wrapper around the standalone launcher."""
    return resolve_ngspice(configured)


class Simulator:
    """Run registered ngspice jobs sequentially with cancellation and logging."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        """Create an idle simulator that reports progress through ``log``."""

        self.log = log
        self.stop_event = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Request cancellation and terminate the currently active subprocess."""

        self.stop_event.set()
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            process.terminate()

    def run(
        self,
        channels: Sequence[Channel],
        template_path: Path,
        output_root: Path,
        ngspice: str,
        settings: SweepSettings,
        analyses: Sequence[str] | str = ("ac", "op"),
    ) -> list[ChannelResult]:
        """Execute selected analyses and return normalized per-channel results."""

        selected_analyses = normalize_analyses(analyses)
        settings.validate(selected_analyses)
        template_text = template_path.read_text(encoding="utf-8-sig")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_root / f"run_{stamp}"
        session_dir.mkdir(parents=True, exist_ok=False)
        write_channels_csv(session_dir / "applied_components.csv", channels)
        session_launcher_log = session_dir / "launcher.log"
        session_launcher_log.write_text(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"SESSION START channels={','.join(str(item.ch) for item in channels)} "
            f"analyses={','.join(selected_analyses)}\n",
            encoding="utf-8",
        )
        results: list[ChannelResult] = []
        for position, channel in enumerate(channels, start=1):
            if self.stop_event.is_set():
                break
            self.log(
                f"[{position}/{len(channels)}] ch{channel.ch:02d} "
                f"({channel.f_c_hz:g} Hz) 실행"
            )
            run_dir = session_dir / f"ch{channel.ch:02d}"
            run_dir.mkdir()
            inc_path = run_dir / f"channel_{channel.ch:02d}_params.inc"
            jobs = build_analysis_jobs(
                run_dir, channel.ch, selected_analyses
            )
            job_by_key = {job.spec.key: job for job in jobs}
            init_path = run_dir / ".spiceinit"
            inc_path.write_text(make_parameter_include(channel), encoding="utf-8")
            init_lines = [
                "* Auto-generated ngspice initialization for this simulation",
                "set filetype=ascii",
            ]
            if settings.pspice_compat:
                # This must be active before ngspice parses .include files.
                # "ps" translates included PSpice models and "ki" handles
                # KiCad vector names beginning with '/'.
                init_lines.append("set ngbehavior=pski")
            init_path.write_text("\n".join(init_lines) + "\n", encoding="utf-8")
            for job in jobs:
                job.netlist_path.write_text(
                    make_analysis_netlist(
                        template_text, inc_path, settings, job.spec.key
                    ),
                    encoding="utf-8",
                )
            self.log(f"실행 파일: {ngspice}")

            def remember_process(process: subprocess.Popen[str] | None) -> None:
                """Expose only the currently running child for cancellation."""

                with self._lock:
                    self._process = process

            output_sections: list[str] = []
            for job in jobs:
                if job.raw_path.exists():
                    job.raw_path.unlink()
                self.log(f"  {job.spec.label} 해석 실행")
                launch = run_ngspice(
                    executable=ngspice,
                    netlist=job.netlist_path,
                    run_dir=run_dir,
                    init_dir=run_dir,
                    stop_event=self.stop_event,
                    on_process=remember_process,
                    session_log=session_launcher_log,
                    rawfile=job.raw_path,
                    log_stem=job.spec.log_stem,
                )
                ngspice_output = (
                    launch.ngspice_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if launch.ngspice_log.is_file()
                    else ""
                )
                launcher_output = (
                    launch.launcher_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if launch.launcher_log.is_file()
                    else ""
                )
                job_text = (
                    f"[{job.spec.label} ngspice]\n{ngspice_output.rstrip()}\n\n"
                    f"[{job.spec.label} launcher]\n{launcher_output.rstrip()}"
                ).strip()
                output_sections.append(job_text)
                if self.stop_event.is_set():
                    self.log("사용자가 실행을 중지했습니다.")
                    return results
                if launch.returncode != 0 or not job.raw_path.is_file():
                    tail = "\n".join(job_text.splitlines()[-40:])
                    hint = ""
                    if re.search(r"no such function\s+['\"]?if", job_text, re.I):
                        hint = (
                            "\n\nPSpice 호환 초기화가 적용되지 않았습니다. "
                            f"{init_path}에 'set ngbehavior=pski'가 있는지 확인하세요."
                        )
                    failure = (
                        f"종료 코드 {launch.returncode}"
                        if launch.returncode != 0
                        else f"종료 코드는 0이지만 {job.raw_path.name} 없음"
                    )
                    raise RuntimeError(
                        f"ch{channel.ch:02d} {job.spec.label}: "
                        f"ngspice {failure}\n\n"
                        f"실행 로그 마지막 부분:\n{tail}{hint}\n\n"
                        f"전체 진단: {launch.ngspice_log} 및 "
                        f"{launch.launcher_log}"
                    )
            stdout = "\n\n".join(output_sections) + "\n"
            ac_rows: list[tuple[float, float, float]] = []
            op_voltages: dict[str, float] = {}
            ac_metrics: AcMetrics | None = None
            if "ac" in selected_analyses:
                ac_rows = read_ac_raw(
                    job_by_key["ac"].raw_path, settings.output_node
                )
                ac_metrics = calculate_ac_metrics(ac_rows)
                write_normalized_csv(
                    run_dir / "ac.csv",
                    ("frequency_hz", "magnitude_db", "phase_deg"),
                    ac_rows,
                )
                if ac_metrics is not None:
                    write_normalized_csv(
                        run_dir / "ac_metrics.csv",
                        (
                            "center_hz",
                            "peak_db",
                            "low_3db_hz",
                            "high_3db_hz",
                            "q",
                            "status",
                        ),
                        (
                            (
                                ac_metrics.center_hz,
                                ac_metrics.peak_db,
                                ac_metrics.low_3db_hz or "",
                                ac_metrics.high_3db_hz or "",
                                ac_metrics.q or "",
                                ac_metrics.status,
                            ),
                        ),
                    )
            if "op" in selected_analyses:
                op_voltages = read_op_raw(job_by_key["op"].raw_path)
                write_normalized_csv(
                    run_dir / "operating_point.csv",
                    ("node", "voltage_v", "voltage_mV"),
                    sorted(
                        (
                            (node, voltage, voltage * 1000.0)
                            for node, voltage in op_voltages.items()
                        ),
                        key=lambda item: _canonical_vector_name(item[0]),
                    ),
                )
            results.append(
                ChannelResult(
                    channel, run_dir, ac_rows, op_voltages, stdout, ac_metrics
                )
            )
            point_summary = ", ".join(
                part
                for part in (
                    f"AC {len(ac_rows)}점" if ac_rows else "",
                    f"OP {len(op_voltages)}개 노드" if op_voltages else "",
                )
                if part
            )
            self.log(
                f"ch{channel.ch:02d} 완료: {point_summary}"
            )
        write_ac_metrics_summary(session_dir / "ac_metrics_summary.csv", results)
        self.log(f"결과 폴더: {session_dir}")
        return results


class TransientSimulator:
    """Run PWL speech stimuli through selected channels as isolated jobs."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        """Create an idle transient runner with cancellation support."""

        self.log = log
        self.stop_event = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Request cancellation and terminate the current ngspice process."""

        self.stop_event.set()
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            process.terminate()

    def run(
        self,
        channels: Sequence[Channel],
        pwl_files: Sequence[Path],
        template_path: Path,
        output_root: Path,
        ngspice: str,
        settings: TransientSettings,
    ) -> list[TransientResult]:
        """Execute every selected stimulus/channel pair and normalize results."""

        settings.validate()
        if not channels:
            raise ValueError("Transient 실행 채널을 하나 이상 선택하세요.")
        if not pwl_files:
            raise ValueError("Transient PWL 파일을 하나 이상 선택하세요.")
        template = template_path.expanduser().resolve()
        if not template.is_file():
            raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {template}")
        template_text = template.read_text(encoding="utf-8-sig")
        source_name, positive_node, negative_node = find_input_voltage_source(
            template_text, settings.vin_node
        )
        session_dir = output_root.expanduser().resolve() / (
            "tran_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        write_channels_csv(session_dir / "applied_components.csv", channels)
        session_log = session_dir / "launcher.log"
        session_log.write_text(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"TRANSIENT SESSION START channels="
            f"{','.join(str(channel.ch) for channel in channels)} "
            f"stimuli={len(pwl_files)} "
            f"input_vpp_v={settings.input_vpp_v:.12g}\n",
            encoding="utf-8",
        )
        results: list[TransientResult] = []
        summary_rows: list[tuple[object, ...]] = []
        total_jobs = len(channels) * len(pwl_files)
        job_position = 0
        for stimulus_index, pwl_path in enumerate(pwl_files, start=1):
            if self.stop_event.is_set():
                break
            stimulus = pwl_path.expanduser().resolve()
            source_pairs = read_pwl_data(stimulus)
            pairs = scale_pwl_pairs_to_vpp(
                source_pairs,
                settings.input_vpp_v,
            )
            stop_time_s = settings.resolved_stop_time(pairs[-1][0])
            safe_stem = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                f"{stimulus.parent.name}_{stimulus.stem}",
            ).strip("._")[:100] or f"stimulus_{stimulus_index:04d}"
            stimulus_dir = session_dir / (
                f"stim_{stimulus_index:04d}_{safe_stem}"
            )
            stimulus_dir.mkdir()
            stimulus_include = stimulus_dir / "stimulus.pwl.inc"
            stimulus_include.write_text(
                make_pwl_source_include(
                    source_name,
                    positive_node,
                    negative_node,
                    pairs,
                    stimulus,
                ),
                encoding="ascii",
            )
            for channel in channels:
                if self.stop_event.is_set():
                    break
                job_position += 1
                self.log(
                    f"[{job_position}/{total_jobs}] "
                    f"{stimulus.parent.name}/{stimulus.name} · "
                    f"ch{channel.ch:02d} transient 실행"
                )
                run_dir = stimulus_dir / f"ch{channel.ch:02d}"
                run_dir.mkdir()
                parameter_include = (
                    run_dir / f"channel_{channel.ch:02d}_params.inc"
                )
                parameter_include.write_text(
                    make_parameter_include(channel),
                    encoding="utf-8",
                )
                init_path = run_dir / ".spiceinit"
                init_lines = [
                    "* Auto-generated ngspice initialization for transient",
                    "set filetype=ascii",
                ]
                if settings.pspice_compat:
                    init_lines.append("set ngbehavior=pski")
                init_path.write_text(
                    "\n".join(init_lines) + "\n",
                    encoding="utf-8",
                )
                netlist_path = (
                    run_dir / f"channel_{channel.ch:02d}_tran.cir"
                )
                netlist_path.write_text(
                    make_transient_netlist(
                        template_text,
                        parameter_include,
                        stimulus_include,
                        settings,
                        stop_time_s,
                    ),
                    encoding="utf-8",
                )
                raw_path = run_dir / TRAN_RAW_FILENAME
                if raw_path.exists():
                    raw_path.unlink()

                def remember_process(
                    process: subprocess.Popen[str] | None,
                ) -> None:
                    """Expose only the current transient child for cancellation."""

                    with self._lock:
                        self._process = process

                launch = run_ngspice(
                    executable=ngspice,
                    netlist=netlist_path,
                    run_dir=run_dir,
                    init_dir=run_dir,
                    stop_event=self.stop_event,
                    on_process=remember_process,
                    session_log=session_log,
                    rawfile=raw_path,
                    log_stem="tran",
                )
                ngspice_output = (
                    launch.ngspice_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if launch.ngspice_log.is_file()
                    else ""
                )
                launcher_output = (
                    launch.launcher_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if launch.launcher_log.is_file()
                    else ""
                )
                log_text = (
                    f"[Transient ngspice]\n{ngspice_output.rstrip()}\n\n"
                    f"[Transient launcher]\n{launcher_output.rstrip()}"
                ).strip()
                if self.stop_event.is_set():
                    self.log("사용자가 transient 실행을 중지했습니다.")
                    return results
                if launch.returncode != 0 or not raw_path.is_file():
                    tail = "\n".join(log_text.splitlines()[-40:])
                    failure = (
                        f"종료 코드 {launch.returncode}"
                        if launch.returncode != 0
                        else f"종료 코드는 0이지만 {raw_path.name} 없음"
                    )
                    raise RuntimeError(
                        f"{stimulus.name} ch{channel.ch:02d}: "
                        f"ngspice {failure}\n\n"
                        f"실행 로그 마지막 부분:\n{tail}\n\n"
                        f"전체 진단: {launch.ngspice_log} 및 "
                        f"{launch.launcher_log}"
                    )
                rows = read_transient_raw(raw_path, settings)
                write_normalized_csv(
                    run_dir / "transient.csv",
                    (
                        "time_s",
                        "vin_v",
                        "vfilt_out_v",
                        "venv_out_v",
                        "vref_v",
                    ),
                    rows,
                )
                results.append(
                    TransientResult(
                        channel=channel,
                        stimulus_path=stimulus,
                        run_dir=run_dir,
                        rows=rows,
                        log_text=log_text,
                    )
                )
                summary_rows.append(
                    (
                        stimulus.parent.name,
                        stimulus.name,
                        channel.ch,
                        len(rows),
                        settings.input_vpp_v,
                        stop_time_s,
                        run_dir,
                    )
                )
                self.log(f"완료: {len(rows)}점")
        if summary_rows:
            write_normalized_csv(
                session_dir / "transient_summary.csv",
                (
                    "word",
                    "stimulus",
                    "ch",
                    "points",
                    "input_vpp_v",
                    "stop_time_s",
                    "run_dir",
                ),
                summary_rows,
            )
        self.log(f"Transient 결과 폴더: {session_dir}")
        return results


def validate_generation(table: Path, netlist: Path, output: Path) -> None:
    """Generate every AC/OP netlist without starting ngspice."""

    channels = load_channels(table)
    template = netlist.read_text(encoding="utf-8-sig")
    validation_dir = output / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    settings = SweepSettings()
    for channel in channels:
        inc = validation_dir / f"channel_{channel.ch:02d}_params.inc"
        inc.write_text(make_parameter_include(channel), encoding="utf-8")
        for analysis in ("ac", "op"):
            cir = validation_dir / f"channel_{channel.ch:02d}_{analysis}.cir"
            cir.write_text(
                make_analysis_netlist(template, inc, settings, analysis),
                encoding="utf-8",
            )
    print(f"검증 완료: {len(channels)}개 채널, 생성 위치 {validation_dir}")


def parse_channel_selection(value: str, available: Sequence[Channel]) -> list[Channel]:
    """Resolve CLI channel text such as ``0,3,7`` or ``all``."""

    if value.strip().lower() == "all":
        return list(available)
    requested: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if token:
            requested.add(int(token))
    selected = [channel for channel in available if channel.ch in requested]
    missing = requested - {channel.ch for channel in selected}
    if missing:
        raise ValueError("테이블에 없는 채널: " + ", ".join(map(str, sorted(missing))))
    if not selected:
        raise ValueError("실행할 채널을 하나 이상 지정하세요.")
    return selected


class InteractivePlotCursor:
    """Button-armed line-following inspector for one Matplotlib axes.

    Ordinary mouse movement remains inert.  After ``arm()`` a temporary
    vertical guide, point marker, and value label follow the closest plotted
    sample.  A left click converts that preview into one persistent cursor;
    repeated arming therefore leaves multiple independent markers.

    The temporary artists are animated and use Matplotlib blitting when the
    backend supports it.  This avoids redrawing hundreds of thousands of
    transient samples for every mouse-motion event.
    """

    def __init__(
        self,
        canvas: object,
        axes: object,
        lines: Sequence[object],
        *,
        x_name: str = "x",
        x_unit: str = "",
        y_name: str = "y",
        y_unit: str = "",
        interaction_blocker: Callable[[], bool] | None = None,
    ) -> None:
        """Attach reusable preview and fixed-cursor artists to one axes."""

        self.canvas = canvas
        self.axes = axes
        self.lines = tuple(lines)
        self.x_name = x_name
        self.x_unit = x_unit
        self.y_name = y_name
        self.y_unit = y_unit
        self.interaction_blocker = interaction_blocker
        self.armed = False
        self._disposed = False
        self._background = None
        self.artists: list[tuple[object, object, object]] = []
        self.preview_vertical = axes.axvline(
            0.0,
            color="#ef6c00",
            linewidth=1.0,
            linestyle="--",
            alpha=0.95,
            visible=False,
            animated=True,
            zorder=20,
        )
        self.preview_marker = axes.plot(
            [],
            [],
            marker="o",
            markersize=5,
            markerfacecolor="#fff3e0",
            markeredgecolor="#ef6c00",
            linestyle="none",
            visible=False,
            animated=True,
            zorder=21,
        )[0]
        self.preview_annotation = axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff3e0", "ec": "#ef6c00"},
            arrowprops={"arrowstyle": "->", "color": "#ef6c00"},
            fontsize=9,
            fontfamily="DejaVu Sans",
            visible=False,
            animated=True,
            zorder=22,
        )
        self._preview_artists = (
            self.preview_vertical,
            self.preview_marker,
            self.preview_annotation,
        )
        self._connection_ids = (
            canvas.mpl_connect("button_press_event", self._on_click),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("figure_leave_event", self._on_leave),
            canvas.mpl_connect("draw_event", self._on_draw),
        )

    def arm(self) -> None:
        """Enable line-following preview until the next left click."""

        if self._disposed:
            return
        self.armed = True

    def cancel(self) -> None:
        """Cancel preview mode without changing existing fixed cursors."""

        self.armed = False
        self._hide_preview()

    def _toolbar_busy(self) -> bool:
        """Return whether another interaction mode currently owns the mouse."""

        toolbar = getattr(self.canvas, "toolbar", None)
        blocked = (
            self.interaction_blocker is not None
            and bool(self.interaction_blocker())
        )
        return bool(getattr(toolbar, "mode", "")) or blocked

    def _nearest(self, event: object) -> tuple[object, float, float] | None:
        """Find the plotted sample nearest to the event in display coordinates."""

        event_pixel_x = getattr(event, "x", None)
        event_pixel_y = getattr(event, "y", None)
        if event_pixel_x is None or event_pixel_y is None:
            return None
        try:
            event_pixel_x = float(event_pixel_x)
            event_pixel_y = float(event_pixel_y)
        except (TypeError, ValueError, OverflowError):
            return None
        best: tuple[float, object, float, float] | None = None
        for line in self.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            count = min(len(x_data), len(y_data))
            if count == 0:
                continue
            indices: Iterable[int] = range(count)
            event_x = getattr(event, "xdata", None)
            if (
                count > 32
                and event_x is not None
                and math.isfinite(float(event_x))
            ):
                try:
                    first_x = float(x_data[0])
                    last_x = float(x_data[count - 1])
                    if first_x <= last_x:
                        position = bisect.bisect_left(
                            x_data, float(event_x), 0, count
                        )
                        indices = range(
                            max(0, position - 4),
                            min(count, position + 5),
                        )
                except (TypeError, ValueError, OverflowError):
                    indices = range(count)
            for index in indices:
                x_value = x_data[index]
                y_value = y_data[index]
                try:
                    x_float = float(x_value)
                    y_float = float(y_value)
                    if not math.isfinite(x_float) or not math.isfinite(y_float):
                        continue
                    pixel_x, pixel_y = self.axes.transData.transform(
                        (x_float, y_float)
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
                distance = (
                    (pixel_x - event_pixel_x) ** 2
                    + (pixel_y - event_pixel_y) ** 2
                )
                if best is None or distance < best[0]:
                    best = (distance, line, x_float, y_float)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _annotation_text(
        self,
        line: object,
        x_value: float,
        y_value: float,
        state: str,
    ) -> str:
        """Format one preview or fixed-cursor label with configured units."""

        label = line.get_label()
        if not label or label.startswith("_"):
            label = "data"
        x_suffix = f" {self.x_unit}" if self.x_unit else ""
        y_suffix = f" {self.y_unit}" if self.y_unit else ""
        return (
            f"{label}\n"
            f"{self.x_name} = {x_value:.7g}{x_suffix}\n"
            f"{self.y_name} = {y_value:.7g}{y_suffix}\n"
            f"[{state}]"
        )

    def _preview_is_visible(self) -> bool:
        """Return whether the temporary guide is currently on the axes."""

        return bool(self.preview_vertical.get_visible())

    def _redraw_preview(self) -> None:
        """Blit only animated preview artists, falling back to an idle redraw."""

        if self._disposed:
            return
        if getattr(self.canvas, "supports_blit", False) and self._background is not None:
            try:
                self.canvas.restore_region(self._background)
                for artist in self._preview_artists:
                    if artist.get_visible():
                        self.axes.draw_artist(artist)
                self.canvas.blit(self.axes.bbox)
                return
            except (AttributeError, RuntimeError, ValueError):
                self._background = None
        self.canvas.draw_idle()

    def _show_preview(
        self,
        line: object,
        x_value: float,
        y_value: float,
    ) -> None:
        """Move the temporary vertical guide and value marker to one sample."""

        self.preview_vertical.set_xdata((x_value, x_value))
        self.preview_marker.set_data((x_value,), (y_value,))
        self.preview_annotation.xy = (x_value, y_value)
        self.preview_annotation.set_text(
            self._annotation_text(line, x_value, y_value, "CLICK TO LOCK")
        )
        for artist in self._preview_artists:
            artist.set_visible(True)
        self._redraw_preview()

    def _hide_preview(self, *, redraw: bool = True) -> None:
        """Hide the temporary guide and optionally repaint its previous area."""

        was_visible = self._preview_is_visible()
        for artist in self._preview_artists:
            artist.set_visible(False)
        if was_visible and redraw:
            self._redraw_preview()

    def _on_draw(self, _event: object) -> None:
        """Refresh the clean axes background used for fast preview blitting."""

        if self._disposed or not getattr(self.canvas, "supports_blit", False):
            return
        try:
            self._background = self.canvas.copy_from_bbox(self.axes.bbox)
            if self._preview_is_visible():
                for artist in self._preview_artists:
                    self.axes.draw_artist(artist)
                self.canvas.blit(self.axes.bbox)
        except (AttributeError, RuntimeError, ValueError):
            self._background = None

    def _on_motion(self, event: object) -> None:
        """Follow the nearest line only while the user has armed the cursor."""

        if not self.armed or self._disposed:
            return
        if self._toolbar_busy() or getattr(event, "inaxes", None) is not self.axes:
            self._hide_preview()
            return
        nearest = self._nearest(event)
        if nearest is None:
            self._hide_preview()
            return
        self._show_preview(*nearest)

    def _on_leave(self, _event: object) -> None:
        """Remove the temporary guide when the pointer leaves the figure."""

        if self.armed:
            self._hide_preview()

    def _show(self, line: object, x_value: float, y_value: float) -> None:
        """Create one fixed vertical guide, point marker, and annotation."""

        vertical = self.axes.axvline(
            x_value,
            color="#d32f2f",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
        )
        marker = self.axes.plot(
            (x_value,),
            (y_value,),
            marker="o",
            markersize=5,
            markerfacecolor="#ffebee",
            markeredgecolor="#d32f2f",
            linestyle="none",
        )[0]
        annotation = self.axes.annotate(
            "",
            xy=(x_value, y_value),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff9c4", "ec": "#555"},
            arrowprops={"arrowstyle": "->", "color": "#555"},
            fontsize=9,
            fontfamily="DejaVu Sans",
        )
        annotation.set_text(
            self._annotation_text(line, x_value, y_value, "LOCKED")
        )
        self.artists.append((vertical, marker, annotation))
        self.canvas.draw_idle()

    def remove_last(self) -> None:
        """Remove the most recently added fixed cursor from this axes."""

        if not self.artists:
            return
        for artist in self.artists.pop():
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Cancel placement and delete every fixed cursor on this axes."""

        self.cancel()
        while self.artists:
            for artist in self.artists.pop():
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
        self.canvas.draw_idle()

    def _on_click(self, event: object) -> None:
        """Lock the line-following preview after one explicitly armed click."""

        if (
            not self.armed
            or self._disposed
            or getattr(event, "button", None) != 1
        ):
            return
        self.armed = False
        self._hide_preview(redraw=False)
        if (
            self._toolbar_busy()
            or getattr(event, "inaxes", None) is not self.axes
        ):
            self._redraw_preview()
            return
        nearest = self._nearest(event)
        if nearest is not None:
            self._show(*nearest)
        else:
            self._redraw_preview()

    def dispose(self) -> None:
        """Disconnect callbacks and remove artists before their widget is rebuilt."""

        if self._disposed:
            return
        self._disposed = True
        self.armed = False
        for connection_id in self._connection_ids:
            try:
                self.canvas.mpl_disconnect(connection_id)
            except (AttributeError, RuntimeError):
                pass
        for artist_group in (*self.artists, self._preview_artists):
            for artist in artist_group:
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
        self.artists.clear()
        self._background = None


class MetricSelectionOverlay:
    """Persistent peak cursors controlled by the 16-channel metrics list."""

    def __init__(
        self,
        canvas: object,
        axes: object,
        entries: Mapping[int, tuple[object, AcMetrics, float]],
    ) -> None:
        """Keep the plotted line, metric, and displayed peak Y for each channel."""

        self.canvas = canvas
        self.axes = axes
        self.entries = dict(entries)
        self.artists: list[object] = []

    def clear(self, redraw: bool = True) -> None:
        """Remove every list-controlled cursor and optional redraw the canvas."""

        for artist in self.artists:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.artists.clear()
        if redraw:
            self.canvas.draw_idle()

    def show_channels(self, channel_numbers: Sequence[int]) -> None:
        """Draw one color-matched f0/gain/Q cursor for each selected list row."""

        self.clear(redraw=False)
        visible = [
            (channel, self.entries[channel])
            for channel in channel_numbers
            if channel in self.entries
        ]
        for index, (channel, (line, metric, marker_y)) in enumerate(visible):
            color = line.get_color()
            vertical = self.axes.axvline(
                metric.center_hz,
                color=color,
                linewidth=1.2,
                linestyle=":",
                alpha=0.9,
            )
            point = self.axes.plot(
                [metric.center_hz],
                [marker_y],
                marker="o",
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=2,
                color=color,
                label="_nolegend_",
                zorder=6,
            )[0]
            q_text = "—" if metric.q is None else f"{metric.q:.5g}"
            annotation = self.axes.annotate(
                (
                    f"ch{channel:02d}\n"
                    f"f0={metric.center_hz:.7g} Hz\n"
                    f"gain={metric.peak_db:.5g} dB\n"
                    f"Q={q_text}"
                ),
                xy=(metric.center_hz, marker_y),
                xytext=(10, 12 + 42 * (index % 4)),
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontfamily="DejaVu Sans",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "fc": "white",
                    "ec": color,
                    "alpha": 0.92,
                },
                arrowprops={"arrowstyle": "->", "color": color},
                zorder=7,
            )
            self.artists.extend((vertical, point, annotation))
        self.canvas.draw_idle()


class SweeperGUI:
    """Tk application that connects editable channel values to the simulator."""

    def __init__(self, root: object) -> None:
        """Initialize application state, build widgets, and load factory values."""

        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("ngspice 16채널 AC / OP / Transient")
        self.root.geometry("1600x1000")
        self.root.minsize(1180, 780)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.simulator: Simulator | None = None
        self.transient_simulator: TransientSimulator | None = None
        self.worker: threading.Thread | None = None
        self.transient_worker: threading.Thread | None = None
        self.conversion_worker: threading.Thread | None = None
        self.conversion_stop_event = threading.Event()
        self.default_channels: list[Channel] = []
        self.channels: list[Channel] = []
        self.last_results: list[ChannelResult] = []
        self.last_ac_results: list[ChannelResult] = []
        self.last_op_results: list[ChannelResult] = []
        self.last_transient_results: list[TransientResult] = []
        self.transient_pwl_files: list[Path] = []
        self.component_source_path = DEFAULT_TABLE
        self.component_dirty = False
        self.component_editor_channel: int | None = None
        self.component_edit_vars: dict[str, object] = {}
        self.component_auto_update_suspended = False
        self.component_divider_edit_source = "resistors"
        self.component_divider_nominal_total_kohm = (
            DIVIDER_NOMINAL_TOTAL_KOHM
        )
        self.component_vdet_v: float | None = None
        self.pending_margin_retune: tuple[int, float] | None = None
        self.metric_selection_overlay: MetricSelectionOverlay | None = None
        self.metric_table = None
        self.selected_metric_channels: set[int] = set()
        self.metric_copy_rows: dict[str, str] = {}
        self.ac_plot_cursors: dict[str, list[InteractivePlotCursor]] = {
            "magnitude": [],
            "phase": [],
        }

        try:
            default_ngspice = resolve_ngspice("")
        except FileNotFoundError:
            default_ngspice = ""
        self.ngspice_var = tk.StringVar(value=default_ngspice)
        self.netlist_var = tk.StringVar(value=str(DEFAULT_NETLIST))
        self.table_var = tk.StringVar(value=str(DEFAULT_TABLE))
        self.component_source_label_var = tk.StringVar(
            value="기본값 (읽기 전용)"
        )
        self.component_vref_var = tk.StringVar(value="")
        self.component_margin_var = tk.StringVar(value="")
        self.component_actual_margin_var = tk.StringVar(value="—")
        self.output_var = tk.StringVar(value=str(DEFAULT_RESULTS))
        self.ac_points_var = tk.StringVar(value="100")
        self.ac_start_var = tk.StringVar(value="10")
        self.ac_stop_var = tk.StringVar(value="20000")
        self.output_node_var = tk.StringVar(value="/v_filt_out")
        self.pspice_compat_var = tk.BooleanVar(value=True)
        self.run_ac_var = tk.BooleanVar(value=True)
        self.run_op_var = tk.BooleanVar(value=True)
        self.mag_x_scale_var = tk.StringVar(value="Decade")
        self.mag_y_scale_var = tk.StringVar(value="Linear (dB)")
        self.mag_x_min_var = tk.StringVar(value="")
        self.mag_x_max_var = tk.StringVar(value="")
        self.mag_y_min_var = tk.StringVar(value="")
        self.mag_y_max_var = tk.StringVar(value="")
        self.phase_x_scale_var = tk.StringVar(value="Decade")
        self.phase_y_scale_var = tk.StringVar(value="Linear (deg)")
        self.selected_op_channel_var = tk.StringVar(value="")
        self.schematic_zoom_var = tk.DoubleVar(value=1.0)
        self.transient_plot_cursors: list[InteractivePlotCursor] = []
        self.schematic_photo = None
        self.schematic_canvas = None
        self.schematic_result: ChannelResult | None = None
        self.op_paned = None
        self.op_pane_fraction: float | None = OP_DEFAULT_SCHEMATIC_FRACTION
        self.op_canvas_view: tuple[float, float] | None = None
        self.op_detail_notebook = None
        self.op_detail_tabs: dict[str, object] = {}
        self.op_detail_selected_tab = "node"
        self.op_pending_view_state: tuple[
            str,
            float | None,
            tuple[float, float] | None,
        ] | None = None
        self.status_var = tk.StringVar(value="소자 테이블을 불러오는 중...")
        default_dataset = (
            r"C:\Users\duyou\Desktop\Personal\10_Study\10_SemiContest"
            r"\10_AI_Project\speech_commands_for_sim"
        )
        self.pwl_input_var = tk.StringVar(value=default_dataset)
        self.pwl_output_var = tk.StringVar(value=default_dataset + r"\pwl")
        self.pwl_overwrite_var = tk.BooleanVar(value=True)
        self.pwl_status_var = tk.StringVar(
            value="WAV 루트와 PWL 출력 폴더를 선택하세요."
        )
        self.transient_folder_var = tk.StringVar(
            value=default_dataset + r"\pwl"
        )
        self.transient_output_step_var = tk.StringVar(value="10u")
        self.transient_stop_time_var = tk.StringVar(value="")
        self.transient_max_step_var = tk.StringVar(value="5u")
        self.transient_input_vpp_var = tk.StringVar(value="10")
        self.transient_result_var = tk.StringVar(value="")
        self.transient_result_channel_var = tk.IntVar(value=-1)
        self.transient_x_min_var = tk.StringVar(value="")
        self.transient_x_max_var = tk.StringVar(value="")
        self.transient_x_mode_var = tk.StringVar(value="자동")
        self.transient_y_target_var = tk.StringVar(value="Vin")
        self.transient_y_min_var = tk.StringVar(value="")
        self.transient_y_max_var = tk.StringVar(value="")
        self.transient_y_mode_var = tk.StringVar(value="자동")
        self.transient_y_modes = {
            "Vin": "자동",
            "Vfilt_out": "자동",
            "Venv_out + Vref": "자동",
        }
        self.transient_range_status_var = tk.StringVar(
            value="X는 세 그래프 공통 · Y는 선택 그래프별로 독립 유지"
        )
        self.transient_status_var = tk.StringVar(
            value="PWL 폴더를 읽고 파일과 채널을 선택하세요."
        )
        self.transient_component_state_var = tk.StringVar(
            value="현재 소자값 상태를 확인하는 중..."
        )
        self.transient_result_lookup: dict[
            tuple[str, int], TransientResult
        ] = {}
        self.transient_result_stimulus_paths: dict[str, Path] = {}
        self.transient_result_channel_buttons: dict[int, object] = {}
        self.transient_axes: dict[str, object] = {}
        self.transient_default_x_limits_ms: tuple[float, float] | None = None
        self.transient_default_y_limits_mv: dict[
            str, tuple[float, float]
        ] = {}
        self.transient_manual_x_limits_ms: tuple[float, float] | None = None
        self.transient_manual_y_limits_mv: dict[
            str, tuple[float, float]
        ] = {}
        self.transient_x_min_entry = None
        self.transient_x_max_entry = None
        self.transient_y_min_entry = None
        self.transient_y_max_entry = None
        self.transient_canvas = None
        self.transient_range_sync_pending = False

        self._build()
        self._reload_channels(DEFAULT_TABLE)
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        """Construct the four independent simulation workflow tabs."""

        from tkinter import ttk

        self.workflow_notebook = ttk.Notebook(self.root)
        self.workflow_notebook.pack(fill="both", expand=True)
        self.dc_page = ttk.Frame(self.workflow_notebook)
        self.ac_page = ttk.Frame(self.workflow_notebook)
        self.transient_page = ttk.Frame(self.workflow_notebook)
        self.converter_page = ttk.Frame(self.workflow_notebook)
        self.workflow_notebook.add(self.dc_page, text="DC 동작점")
        self.workflow_notebook.add(self.ac_page, text="AC 시뮬")
        self.workflow_notebook.add(self.transient_page, text="Transient 시뮬")
        self.workflow_notebook.add(self.converter_page, text="WAV → PWL 시뮬")
        self.workflow_notebook.bind(
            "<<NotebookTabChanged>>",
            self._on_workflow_tab_changed,
        )

        self._build_dc_tab()
        self._build_ac_tab()
        self._build_transient_tab()
        self._build_converter_tab()
        self._show_empty_results()
        self.workflow_notebook.select(self.dc_page)
        self._update_run_button()

    def _on_workflow_tab_changed(self, _event: object = None) -> None:
        """Redraw a retained transient canvas after its hidden tab becomes visible."""

        try:
            is_transient = (
                self.workflow_notebook.select()
                == str(self.transient_page)
            )
        except Exception:
            return
        if not is_transient:
            return
        self._refresh_transient_component_state()
        if self.transient_canvas is None:
            return

        def redraw() -> None:
            """Refresh pixels only; preserve result selection and manual limits."""

            canvas = self.transient_canvas
            if canvas is None:
                return
            try:
                canvas.get_tk_widget().update_idletasks()
                canvas.draw_idle()
            except Exception:
                pass

        self.root.after_idle(redraw)

    def _build_file_settings(self, parent: object) -> object:
        """Build one view of the shared paths and component-version controls."""

        files = self.ttk.LabelFrame(parent, text="파일 설정", padding=8)
        self._path_row(files, 0, "ngspice", self.ngspice_var, self._browse_ngspice)
        self._path_row(files, 1, "네트리스트", self.netlist_var, self._browse_netlist)
        self._path_row(
            files,
            2,
            "적용 CSV",
            self.table_var,
            self._reset_component_defaults,
            entry_state="readonly",
            button_text="기본값",
        )
        version_controls = self.ttk.Frame(files)
        version_controls.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(4, 2)
        )
        self.ttk.Label(version_controls, text="CSV 버전", width=11).pack(
            side="left"
        )
        self.ttk.Label(
            version_controls,
            textvariable=self.component_source_label_var,
        ).pack(side="left", fill="x", expand=True)
        self.ttk.Button(
            version_controls,
            text="새 버전 저장",
            command=self._save_component_version_from_gui,
        ).pack(side="right")
        self.ttk.Button(
            version_controls,
            text="버전 불러오기",
            command=self._browse_table,
        ).pack(side="right", padx=(4, 5))
        self._path_row(
            files, 4, "결과 폴더", self.output_var, self._browse_output
        )
        files.columnconfigure(1, weight=1)
        return files

    def _build_channel_selector(
        self,
        parent: object,
        attribute_name: str,
        select_all_command: Callable[[], None],
        clear_command: Callable[[], None],
        *,
        title: str = "채널",
    ) -> object:
        """Build a reusable multi-channel selector and store its listbox."""

        frame = self.ttk.LabelFrame(parent, text=title, padding=8)
        listbox = self.tk.Listbox(
            frame,
            selectmode="extended",
            exportselection=False,
            height=7,
            width=25,
        )
        listbox.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(
            frame, orient="vertical", command=listbox.yview
        )
        scrollbar.grid(row=0, column=3, sticky="ns")
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_run_button(),
        )
        self.ttk.Button(
            frame, text="전체 선택", command=select_all_command
        ).grid(row=1, column=0, pady=(5, 0))
        self.ttk.Button(frame, text="해제", command=clear_command).grid(
            row=1, column=1, pady=(5, 0)
        )
        self.ttk.Button(
            frame, text="CSV 다시 읽기", command=self._reload_channels
        ).grid(row=1, column=2, pady=(5, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        setattr(self, attribute_name, listbox)
        return frame

    def _build_dc_tab(self) -> None:
        """Build DC operating-point controls and its full-width schematic view."""

        outer = self.ttk.Frame(self.dc_page, padding=10)
        outer.pack(fill="both", expand=True)
        settings_row = self.ttk.Frame(outer)
        settings_row.pack(fill="x")
        settings_row.columnconfigure(0, weight=5)
        settings_row.columnconfigure(1, weight=2)
        settings_row.columnconfigure(2, weight=2)
        files = self._build_file_settings(settings_row)
        files.grid(row=0, column=0, sticky="nsew")
        dc_settings = self.ttk.LabelFrame(
            settings_row, text="DC 동작점 설정", padding=8
        )
        dc_settings.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.ttk.Checkbutton(
            dc_settings,
            text="TI PSpice + KiCad 호환 모드\n(ngbehavior=pski)",
            variable=self.pspice_compat_var,
        ).pack(anchor="w")
        self.ttk.Label(
            dc_settings,
            text=(
                ".save all + .op를 실행합니다.\n"
                "소자 편집의 변경값 적용도 현재 채널의\n"
                "DC 동작점을 자동 재실행합니다."
            ),
            justify="left",
            foreground="#164a7b",
        ).pack(anchor="w", pady=(8, 0))
        channels_frame = self._build_channel_selector(
            settings_row,
            "channel_list",
            self._select_all,
            self._clear_selection,
            title="DC 채널",
        )
        channels_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=8)
        self.op_run_button = self.ttk.Button(
            action,
            text="DC 동작점 실행",
            command=lambda: self._start_run(("op",), self.channel_list),
        )
        self.op_run_button.pack(side="left")
        self.op_stop_button = self.ttk.Button(
            action, text="중지", command=self._stop_run, state="disabled"
        )
        self.op_stop_button.pack(side="left", padx=(6, 0))
        self.ttk.Button(
            action, text="결과 폴더 열기", command=self._open_results
        ).pack(side="left", padx=(6, 0))
        self.op_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=180
        )
        self.op_progress.pack(side="left", padx=12)
        self.ttk.Label(action, textvariable=self.status_var).pack(
            side="left", fill="x"
        )
        self.op_frame = self.ttk.Frame(outer)
        self.op_frame.pack(fill="both", expand=True)

    def _build_ac_tab(self) -> None:
        """Build AC controls plus magnitude, phase, and log result tabs."""

        outer = self.ttk.Frame(self.ac_page, padding=10)
        outer.pack(fill="both", expand=True)
        settings_row = self.ttk.Frame(outer)
        settings_row.pack(fill="x")
        settings_row.columnconfigure(0, weight=5)
        settings_row.columnconfigure(1, weight=3)
        settings_row.columnconfigure(2, weight=2)
        files = self._build_file_settings(settings_row)
        files.grid(row=0, column=0, sticky="nsew")

        sweep = self.ttk.LabelFrame(settings_row, text="AC 해석 설정", padding=8)
        sweep.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        fields = (
            ("AC points/dec", self.ac_points_var),
            ("AC start [Hz]", self.ac_start_var),
            ("AC stop [Hz]", self.ac_stop_var),
            ("AC 출력 노드", self.output_node_var),
        )
        for row, (label, variable) in enumerate(fields):
            self.ttk.Label(sweep, text=label).grid(
                row=row, column=0, sticky="e", padx=(4, 3), pady=3
            )
            self.ttk.Entry(sweep, textvariable=variable, width=16).grid(
                row=row, column=1, columnspan=2, sticky="ew", padx=(0, 7), pady=3
            )
        sweep.columnconfigure(1, weight=1)
        self.ttk.Checkbutton(
            sweep,
            text="TI PSpice + KiCad 호환 모드 (ngbehavior=pski)",
            variable=self.pspice_compat_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(5, 0))
        channels_frame = self._build_channel_selector(
            settings_row,
            "ac_channel_list",
            self._select_all_ac,
            self._clear_ac_selection,
            title="AC 채널",
        )
        channels_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=8)
        self.ac_run_button = self.ttk.Button(
            action,
            text="AC 시뮬 실행",
            command=lambda: self._start_run(("ac",), self.ac_channel_list),
        )
        self.ac_run_button.pack(side="left")
        self.ac_stop_button = self.ttk.Button(
            action, text="중지", command=self._stop_run, state="disabled"
        )
        self.ac_stop_button.pack(side="left", padx=(6, 0))
        self.ttk.Button(
            action, text="결과 폴더 열기", command=self._open_results
        ).pack(side="left", padx=(6, 0))
        self.ac_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=180
        )
        self.ac_progress.pack(side="left", padx=12)
        self.ttk.Label(action, textvariable=self.status_var).pack(
            side="left", fill="x"
        )

        self.ac_result_notebook = self.ttk.Notebook(outer)
        self.ac_result_notebook.pack(fill="both", expand=True)
        self.mag_frame = self.ttk.Frame(self.ac_result_notebook)
        self.phase_frame = self.ttk.Frame(self.ac_result_notebook)
        self.log_frame = self.ttk.Frame(self.ac_result_notebook)
        self.ac_result_notebook.add(self.mag_frame, text="AC Magnitude")
        self.ac_result_notebook.add(self.phase_frame, text="AC Phase")
        self.ac_result_notebook.add(self.log_frame, text="실행 로그")
        self.log_text = self.tk.Text(self.log_frame, wrap="none", height=15)
        log_y = self.ttk.Scrollbar(
            self.log_frame, orient="vertical", command=self.log_text.yview
        )
        log_x = self.ttk.Scrollbar(
            self.log_frame, orient="horizontal", command=self.log_text.xview
        )
        self.log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")
        self.log_frame.rowconfigure(0, weight=1)
        self.log_frame.columnconfigure(0, weight=1)
        # Compatibility aliases retained for extension code written against v8.
        self.run_button = self.ac_run_button
        self.stop_button = self.ac_stop_button
        self.progress = self.ac_progress
        self.notebook = self.ac_result_notebook

    def _path_row(
        self,
        parent: object,
        row: int,
        label: str,
        variable: object,
        command: Callable[[], None],
        entry_state: str = "normal",
        button_text: str = "찾기...",
    ) -> None:
        """Add a consistent label/entry/action row to the file-settings panel."""

        self.ttk.Label(parent, text=label, width=11).grid(
            row=row, column=0, sticky="e", padx=(0, 5), pady=2
        )
        self.ttk.Entry(
            parent, textvariable=variable, state=entry_state
        ).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        self.ttk.Button(parent, text=button_text, command=command).grid(
            row=row, column=2, padx=(5, 0), pady=2
        )

    def _build_converter_tab(self) -> None:
        """Build the WAV-tree scanner and 0 V-DC/10 mVpp conversion page."""

        outer = self.ttk.Frame(self.converter_page, padding=12)
        outer.pack(fill="both", expand=True)
        settings = self.ttk.LabelFrame(
            outer, text="WAV → ngspice PWL 변환", padding=10
        )
        settings.pack(fill="x")
        self._path_row(
            settings,
            0,
            "WAV 루트",
            self.pwl_input_var,
            self._browse_pwl_input,
        )
        self._path_row(
            settings,
            1,
            "PWL 출력",
            self.pwl_output_var,
            self._browse_pwl_output,
        )
        settings.columnconfigure(1, weight=1)
        options = self.ttk.Frame(settings)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(7, 2))
        self.ttk.Checkbutton(
            options,
            text="기존 PWL 덮어쓰기",
            variable=self.pwl_overwrite_var,
        ).pack(side="left")
        self.ttk.Button(
            options,
            text="WAV 폴더 스캔",
            command=self._scan_wav_tree,
        ).pack(side="left", padx=(10, 0))
        self.pwl_convert_button = self.ttk.Button(
            options,
            text="변환 시작",
            command=self._start_pwl_conversion,
        )
        self.pwl_convert_button.pack(side="left", padx=(5, 0))
        self.pwl_stop_button = self.ttk.Button(
            options,
            text="중지",
            command=self._stop_pwl_conversion,
            state="disabled",
        )
        self.pwl_stop_button.pack(side="left", padx=(5, 0))
        self.pwl_progress = self.ttk.Progressbar(
            options, mode="determinate", length=220
        )
        self.pwl_progress.pack(side="left", padx=(12, 0))
        self.ttk.Label(
            settings,
            text=(
                "파일별 처리: y=(x−mean)·0.010/(max−min) [V] · "
                "DC=0 V · peak-to-peak=10 mV. 마지막 원 샘플 다음 시각에 "
                "0 V를 추가해 더 긴 .tran에서도 입력을 무음으로 유지합니다."
            ),
            justify="left",
            foreground="#164a7b",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(7, 2))

        body = self.ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 0))
        words_host = self.ttk.LabelFrame(
            body, text="발견된 단어 폴더", padding=6
        )
        log_host = self.ttk.LabelFrame(body, text="변환 로그", padding=6)
        body.add(words_host, weight=1)
        body.add(log_host, weight=3)
        self.pwl_scan_table = self.ttk.Treeview(
            words_host,
            columns=("word", "count"),
            show="headings",
            height=18,
        )
        self.pwl_scan_table.heading("word", text="단어")
        self.pwl_scan_table.heading("count", text="WAV 수")
        self.pwl_scan_table.column("word", width=180, stretch=True)
        self.pwl_scan_table.column("count", width=80, anchor="e")
        scan_scroll = self.ttk.Scrollbar(
            words_host,
            orient="vertical",
            command=self.pwl_scan_table.yview,
        )
        self.pwl_scan_table.configure(yscrollcommand=scan_scroll.set)
        self.pwl_scan_table.grid(row=0, column=0, sticky="nsew")
        scan_scroll.grid(row=0, column=1, sticky="ns")
        words_host.rowconfigure(0, weight=1)
        words_host.columnconfigure(0, weight=1)
        self.pwl_log_text = self.tk.Text(log_host, wrap="none", height=18)
        pwl_log_y = self.ttk.Scrollbar(
            log_host, orient="vertical", command=self.pwl_log_text.yview
        )
        pwl_log_x = self.ttk.Scrollbar(
            log_host, orient="horizontal", command=self.pwl_log_text.xview
        )
        self.pwl_log_text.configure(
            yscrollcommand=pwl_log_y.set,
            xscrollcommand=pwl_log_x.set,
        )
        self.pwl_log_text.grid(row=0, column=0, sticky="nsew")
        pwl_log_y.grid(row=0, column=1, sticky="ns")
        pwl_log_x.grid(row=1, column=0, sticky="ew")
        log_host.rowconfigure(0, weight=1)
        log_host.columnconfigure(0, weight=1)
        self.ttk.Label(
            outer, textvariable=self.pwl_status_var
        ).pack(fill="x", pady=(7, 0))

    def _build_transient_tab(self) -> None:
        """Build transient input selection, timing controls, and result panes."""

        outer = self.ttk.Frame(self.transient_page, padding=10)
        outer.pack(fill="both", expand=True)
        controls = self.ttk.Frame(outer)
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=5)
        controls.columnconfigure(1, weight=2)
        controls.columnconfigure(2, weight=2)

        files = self.ttk.LabelFrame(controls, text="PWL 입력 파일", padding=7)
        files.grid(row=0, column=0, sticky="nsew")
        path_row = self.ttk.Frame(files)
        path_row.pack(fill="x")
        self.ttk.Entry(
            path_row, textvariable=self.transient_folder_var
        ).pack(side="left", fill="x", expand=True)
        self.ttk.Button(
            path_row,
            text="폴더...",
            command=self._browse_transient_folder,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            path_row,
            text="다시 읽기",
            command=self._refresh_transient_pwl_files,
        ).pack(side="left", padx=(5, 0))
        file_list_host = self.ttk.Frame(files)
        file_list_host.pack(fill="both", expand=True, pady=(5, 0))
        self.transient_file_list = self.tk.Listbox(
            file_list_host,
            selectmode="extended",
            exportselection=False,
            height=4,
        )
        file_scroll = self.ttk.Scrollbar(
            file_list_host,
            orient="vertical",
            command=self.transient_file_list.yview,
        )
        self.transient_file_list.configure(yscrollcommand=file_scroll.set)
        self.transient_file_list.pack(side="left", fill="both", expand=True)
        file_scroll.pack(side="right", fill="y")
        file_buttons = self.ttk.Frame(files)
        file_buttons.pack(fill="x", pady=(5, 0))
        self.ttk.Button(
            file_buttons,
            text="전체 선택",
            command=self._select_all_transient_files,
        ).pack(side="left")
        self.ttk.Button(
            file_buttons,
            text="해제",
            command=self._clear_transient_files,
        ).pack(side="left", padx=(5, 0))
        self.transient_file_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_transient_run_button(),
        )

        timing = self.ttk.LabelFrame(controls, text="Transient 설정", padding=7)
        timing.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        timing_fields = (
            ("입력 PWL Vpp [mVpp]", self.transient_input_vpp_var),
            ("output step tstep [s]", self.transient_output_step_var),
            ("stop time [s]", self.transient_stop_time_var),
            ("maximum step tmax [s]", self.transient_max_step_var),
        )
        for row, (label, variable) in enumerate(timing_fields):
            self.ttk.Label(timing, text=label).grid(
                row=row, column=0, sticky="e", padx=(0, 5), pady=2
            )
            self.ttk.Entry(timing, textvariable=variable, width=13).grid(
                row=row, column=1, sticky="ew", pady=2
            )
        timing.columnconfigure(1, weight=1)
        self.ttk.Label(
            timing,
            text=(
                "입력 Vpp 기본값: 10 mVpp\n"
                "10u = 10 µs, 5u = 5 µs\n"
                "stop time 빈칸: PWL 마지막 0 V 시각"
            ),
            justify="left",
            foreground="#164a7b",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.ttk.Button(
            timing,
            text="step 기본값 근거",
            command=self._show_transient_setting_basis,
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        channels = self.ttk.LabelFrame(controls, text="Transient 채널", padding=7)
        channels.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        channel_host = self.ttk.Frame(channels)
        channel_host.pack(fill="both", expand=True)
        self.transient_channel_list = self.tk.Listbox(
            channel_host,
            selectmode="extended",
            exportselection=False,
            height=5,
            width=22,
        )
        transient_channel_scroll = self.ttk.Scrollbar(
            channel_host,
            orient="vertical",
            command=self.transient_channel_list.yview,
        )
        self.transient_channel_list.configure(
            yscrollcommand=transient_channel_scroll.set
        )
        self.transient_channel_list.pack(side="left", fill="both", expand=True)
        transient_channel_scroll.pack(side="right", fill="y")
        channel_buttons = self.ttk.Frame(channels)
        channel_buttons.pack(fill="x", pady=(5, 0))
        self.ttk.Button(
            channel_buttons,
            text="전체 선택",
            command=self._select_all_transient_channels,
        ).pack(side="left")
        self.ttk.Button(
            channel_buttons,
            text="해제",
            command=self._clear_transient_channels,
        ).pack(side="left", padx=(5, 0))
        self.transient_channel_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_transient_run_button(),
        )

        self.ttk.Label(
            outer,
            textvariable=self.transient_component_state_var,
            foreground="#9a5500",
        ).pack(fill="x", pady=(7, 0))

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=(7, 6))
        self.transient_run_button = self.ttk.Button(
            action,
            text="Transient 실행",
            command=self._start_transient_run,
        )
        self.transient_run_button.pack(side="left")
        self.transient_stop_button = self.ttk.Button(
            action,
            text="중지",
            command=self._stop_transient_run,
            state="disabled",
        )
        self.transient_stop_button.pack(side="left", padx=(5, 0))
        self.transient_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=170
        )
        self.transient_progress.pack(side="left", padx=10)
        self.ttk.Label(action, text="결과 PWL").pack(side="left")
        self.transient_result_combo = self.ttk.Combobox(
            action,
            textvariable=self.transient_result_var,
            state="readonly",
            width=32,
        )
        self.transient_result_combo.pack(side="left", padx=(5, 8))
        self.transient_result_combo.bind(
            "<<ComboboxSelected>>",
            self._on_transient_result_changed,
        )
        self.ttk.Label(action, text="채널 ch").pack(side="left")
        result_channels = self.ttk.Frame(action)
        result_channels.pack(side="left", fill="x", expand=True)
        for channel_number in range(16):
            button = self.ttk.Radiobutton(
                result_channels,
                text=f"{channel_number:02d}",
                value=channel_number,
                variable=self.transient_result_channel_var,
                command=self._on_transient_result_channel_changed,
                state="disabled",
                width=3,
            )
            button.pack(side="left", padx=(1, 0))
            self.transient_result_channel_buttons[channel_number] = button
        self.ttk.Label(
            outer, textvariable=self.transient_status_var
        ).pack(fill="x", pady=(0, 5))

        self.transient_notebook = self.ttk.Notebook(outer)
        self.transient_notebook.pack(fill="both", expand=True)
        self.transient_plot_frame = self.ttk.Frame(self.transient_notebook)
        self.transient_log_frame = self.ttk.Frame(self.transient_notebook)
        self.transient_notebook.add(
            self.transient_plot_frame, text="Transient 결과"
        )
        self.transient_notebook.add(
            self.transient_log_frame, text="Transient 로그"
        )
        self.ttk.Label(
            self.transient_plot_frame,
            text=(
                "Transient 실행 후 Vin, Vfilt_out, "
                "Venv_out + Vref의 세 아날로그 그래프가 표시됩니다."
            ),
        ).pack(expand=True)
        self.transient_log_text = self.tk.Text(
            self.transient_log_frame, wrap="none", height=12
        )
        transient_log_y = self.ttk.Scrollbar(
            self.transient_log_frame,
            orient="vertical",
            command=self.transient_log_text.yview,
        )
        transient_log_x = self.ttk.Scrollbar(
            self.transient_log_frame,
            orient="horizontal",
            command=self.transient_log_text.xview,
        )
        self.transient_log_text.configure(
            yscrollcommand=transient_log_y.set,
            xscrollcommand=transient_log_x.set,
        )
        self.transient_log_text.grid(row=0, column=0, sticky="nsew")
        transient_log_y.grid(row=0, column=1, sticky="ns")
        transient_log_x.grid(row=1, column=0, sticky="ew")
        self.transient_log_frame.rowconfigure(0, weight=1)
        self.transient_log_frame.columnconfigure(0, weight=1)
        self._update_transient_run_button()

    def _browse_ngspice(self) -> None:
        """Let the user choose the Windows ngspice executable."""

        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="ngspice 실행 파일 선택",
            filetypes=(("실행 파일", "*.exe"), ("모든 파일", "*.*")),
        )
        if path:
            self.ngspice_var.set(path)

    def _browse_netlist(self) -> None:
        """Let the user choose the shared circuit netlist template."""

        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="공통 네트리스트 선택",
            filetypes=(("SPICE netlist", "*.cir *.sp *.spice *.net *.txt"), ("모든 파일", "*.*")),
        )
        if path:
            self.netlist_var.set(path)

    def _browse_table(self) -> None:
        """Load a previously saved component version without changing defaults."""

        from tkinter import filedialog, messagebox

        if (
            self.component_dirty or self._component_editor_has_changes()
        ) and not messagebox.askyesno(
            "버전 불러오기",
            "저장하지 않은 소자값 변경을 버리고 다른 버전을 불러올까요?",
        ):
            return

        path = filedialog.askopenfilename(
            title="저장된 채널 소자 버전 불러오기",
            initialdir=str(DEFAULT_COMPONENT_VERSIONS),
            filetypes=(("CSV", "*.csv"), ("모든 파일", "*.*")),
        )
        if path:
            self._reload_channels(Path(path))

    def _browse_output(self) -> None:
        """Let the user choose the parent directory for simulation sessions."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="결과 폴더 선택")
        if path:
            self.output_var.set(path)

    def _browse_pwl_input(self) -> None:
        """Choose the Speech Commands subset root and derive its PWL folder."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="WAV 데이터셋 루트 선택")
        if path:
            self.pwl_input_var.set(path)
            self.pwl_output_var.set(str(Path(path) / "pwl"))
            self._scan_wav_tree()

    def _browse_pwl_output(self) -> None:
        """Choose the parent that will contain mirrored word/PWL folders."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="PWL 출력 폴더 선택")
        if path:
            self.pwl_output_var.set(path)

    def _scan_wav_tree(self) -> None:
        """List per-word WAV counts without converting any file."""

        from tkinter import messagebox

        try:
            source_root = Path(self.pwl_input_var.get()).expanduser().resolve()
            target_root = Path(self.pwl_output_var.get()).expanduser().resolve()
            files = discover_wav_files(source_root, target_root)
            counts: dict[str, int] = {}
            for source in files:
                relative = source.relative_to(source_root)
                word = relative.parts[0] if len(relative.parts) > 1 else source_root.name
                counts[word] = counts.get(word, 0) + 1
            for item in self.pwl_scan_table.get_children():
                self.pwl_scan_table.delete(item)
            for word, count in sorted(
                counts.items(), key=lambda item: item[0].casefold()
            ):
                self.pwl_scan_table.insert(
                    "", "end", values=(word, count)
                )
            self.pwl_progress.configure(maximum=max(1, len(files)), value=0)
            self.pwl_status_var.set(
                f"{len(counts)}개 단어 폴더 · WAV {len(files)}개 발견"
            )
        except Exception as exc:
            self.pwl_status_var.set("WAV 스캔 실패")
            messagebox.showerror("WAV 폴더 스캔 오류", str(exc))

    def _start_pwl_conversion(self) -> None:
        """Convert the scanned WAV tree on a worker thread."""

        from tkinter import messagebox

        if self.conversion_worker and self.conversion_worker.is_alive():
            return
        try:
            source_root = Path(self.pwl_input_var.get()).expanduser().resolve()
            target_root = Path(self.pwl_output_var.get()).expanduser().resolve()
            wav_files = discover_wav_files(source_root, target_root)
            if not wav_files:
                raise FileNotFoundError(f"WAV 파일이 없습니다: {source_root}")
            target_root.mkdir(parents=True, exist_ok=True)
            overwrite = bool(self.pwl_overwrite_var.get())
        except Exception as exc:
            messagebox.showerror("PWL 변환 설정 오류", str(exc))
            return
        self.conversion_stop_event.clear()
        self.pwl_log_text.delete("1.0", "end")
        self.pwl_progress.configure(maximum=len(wav_files), value=0)
        self.pwl_convert_button.configure(state="disabled")
        self.pwl_stop_button.configure(state="normal")
        self.pwl_status_var.set(f"WAV {len(wav_files)}개 변환 중...")

        def progress(
            position: int,
            total: int,
            source: Path,
            target: Path,
        ) -> None:
            """Report one file before conversion and honor a stop request."""

            if self.conversion_stop_event.is_set():
                raise InterruptedError("사용자가 PWL 변환을 중지했습니다.")
            self.events.put(
                (
                    "pwl_progress",
                    (position, total, source, target),
                )
            )

        def worker() -> None:
            """Run WAV decoding and disk writes outside Tk's event loop."""

            try:
                results = convert_wav_tree(
                    source_root,
                    target_root,
                    target_vpp=0.010,
                    overwrite=overwrite,
                    progress=progress,
                )
                self.events.put(("pwl_done", results))
            except InterruptedError as exc:
                self.events.put(("pwl_stopped", exc))
            except Exception as exc:
                self.events.put(("pwl_error", exc))

        self.conversion_worker = threading.Thread(target=worker, daemon=True)
        self.conversion_worker.start()

    def _stop_pwl_conversion(self) -> None:
        """Stop before the next WAV file is converted."""

        self.conversion_stop_event.set()
        self.pwl_status_var.set("PWL 변환 중지 요청 중...")

    def _finish_pwl_controls(self) -> None:
        """Restore converter buttons after success, stop, or error."""

        self.conversion_worker = None
        self.pwl_convert_button.configure(state="normal")
        self.pwl_stop_button.configure(state="disabled")

    def _browse_transient_folder(self) -> None:
        """Choose a converted PWL folder and populate its recursive file list."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="변환된 PWL 폴더 선택")
        if path:
            self.transient_folder_var.set(path)
            self._refresh_transient_pwl_files()

    def _refresh_transient_pwl_files(self) -> None:
        """Read every PWL file under the selected transient input folder."""

        from tkinter import messagebox

        try:
            root = Path(self.transient_folder_var.get()).expanduser().resolve()
            files = discover_pwl_files(root)
            if not files:
                raise FileNotFoundError(f"PWL 파일이 없습니다: {root}")
            self.transient_pwl_files = files
            self.transient_file_list.delete(0, "end")
            for path in files:
                self.transient_file_list.insert(
                    "end", path.relative_to(root).as_posix()
                )
            self.transient_file_list.selection_set(0)
            self.transient_status_var.set(
                f"PWL {len(files)}개 발견 · 실행할 파일을 선택하세요."
            )
            self._update_transient_run_button()
        except Exception as exc:
            self.transient_pwl_files = []
            self.transient_file_list.delete(0, "end")
            self.transient_status_var.set("PWL 폴더 읽기 실패")
            self._update_transient_run_button()
            messagebox.showerror("PWL 폴더 오류", str(exc))

    def _populate_transient_channels(self) -> None:
        """Mirror the current component-version channels into transient controls."""

        if not hasattr(self, "transient_channel_list"):
            return
        previous = {
            self.channels[index].ch
            for index in self.transient_channel_list.curselection()
            if 0 <= index < len(self.channels)
        }
        self.transient_channel_list.delete(0, "end")
        restored = False
        for index, channel in enumerate(self.channels):
            self.transient_channel_list.insert(
                "end", f"ch{channel.ch:02d} · fc={channel.f_c_hz:g} Hz"
            )
            if channel.ch in previous:
                self.transient_channel_list.selection_set(index)
                restored = True
        if self.channels and not restored:
            self.transient_channel_list.selection_set(0)
        self._update_transient_run_button()

    def _select_all_transient_files(self) -> None:
        """Select every discovered PWL file and refresh the estimated job count."""

        self.transient_file_list.selection_set(0, "end")
        self._update_transient_run_button()

    def _clear_transient_files(self) -> None:
        """Clear PWL selection and refresh the transient run button."""

        self.transient_file_list.selection_clear(0, "end")
        self._update_transient_run_button()

    def _select_all_transient_channels(self) -> None:
        """Select all loaded channels for transient simulation."""

        self.transient_channel_list.selection_set(0, "end")
        self._update_transient_run_button()

    def _clear_transient_channels(self) -> None:
        """Clear transient channel selection and refresh the run button."""

        self.transient_channel_list.selection_clear(0, "end")
        self._update_transient_run_button()

    def _update_transient_run_button(self) -> None:
        """Show job count and prevent an empty or concurrent transient run."""

        if not hasattr(self, "transient_run_button"):
            return
        file_count = len(self.transient_file_list.curselection())
        channel_count = len(self.transient_channel_list.curselection())
        jobs = file_count * channel_count
        running = bool(
            self.transient_worker and self.transient_worker.is_alive()
        ) or bool(self.worker and self.worker.is_alive())
        self.transient_run_button.configure(
            text=(
                f"Transient 실행 ({jobs} jobs)"
                if jobs
                else "PWL/채널 선택 필요"
            ),
            state="disabled" if running or jobs == 0 else "normal",
        )

    def _show_transient_setting_basis(self) -> None:
        """Explain the circuit-specific transient time-step defaults."""

        from tkinter import messagebox

        highest_channel_hz = max(
            (channel.f_c_hz for channel in self.channels),
            default=6761.0,
        )
        channel_limit_us = (
            recommended_maximum_step_s(highest_channel_hz) * 1e6
        )
        speech_limit_us = recommended_maximum_step_s(8000.0) * 1e6
        messagebox.showinfo(
            "Transient 기본값 근거",
            (
                "output step tstep = 10 µs\n"
                "저장/표시용 제안 간격입니다. 1초당 명목 100,000간격이며 "
                "원래의 1e-5 s와 같은 값입니다.\n\n"
                "maximum step tmax = 5 µs\n"
                "일반적인 20 points/cycle 기준은 "
                f"최고 채널 {highest_channel_hz:.0f} Hz에서 "
                f"{channel_limit_us:.2f} µs 이하, 16 kHz WAV의 Nyquist "
                f"8 kHz에서 {speech_limit_us:.2f} µs 이하입니다. "
                "5 µs는 두 기준보다 작게 잡은 보수적인 기본값입니다.\n\n"
                "입력 PWL Vpp = 10 mVpp\n"
                "기존 PWL 파일을 수정하지 않고 실행용 사본의 전압값만 "
                "target/current 비율로 스케일합니다."
            ),
        )

    def _transient_settings(self) -> TransientSettings:
        """Parse GUI seconds/volts into a validated transient settings object."""

        stop_text = self.transient_stop_time_var.get().strip()
        try:
            input_vpp_mv = float(self.transient_input_vpp_var.get())
        except ValueError as exc:
            raise ValueError(
                "입력 PWL Vpp에는 mVpp 단위 숫자를 입력하세요."
            ) from exc
        if not math.isfinite(input_vpp_mv) or input_vpp_mv <= 0.0:
            raise ValueError("입력 PWL Vpp는 0보다 큰 유한한 값이어야 합니다.")
        settings = TransientSettings(
            output_step_s=parse_time_seconds(
                self.transient_output_step_var.get(),
                "Transient output step",
            ),
            stop_time_s=(
                parse_time_seconds(stop_text, "Transient stop time")
                if stop_text
                else None
            ),
            maximum_step_s=parse_time_seconds(
                self.transient_max_step_var.get(),
                "Transient maximum step",
            ),
            input_vpp_v=input_vpp_mv / 1000.0,
            pspice_compat=self.pspice_compat_var.get(),
        )
        settings.validate()
        return settings

    def _start_transient_run(self) -> None:
        """Commit component edits and launch selected transient jobs."""

        from tkinter import messagebox

        if self.transient_worker and self.transient_worker.is_alive():
            return
        try:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("AC/OP 실행이 끝난 뒤 transient를 시작하세요.")
            self._commit_component_editor()
            file_indices = list(self.transient_file_list.curselection())
            channel_indices = list(self.transient_channel_list.curselection())
            if not file_indices or not channel_indices:
                raise ValueError("PWL 파일과 채널을 하나 이상 선택하세요.")
            stimuli = [self.transient_pwl_files[index] for index in file_indices]
            channels = [self.channels[index] for index in channel_indices]
            settings = self._transient_settings()
            ngspice = find_ngspice(self.ngspice_var.get())
            netlist = Path(self.netlist_var.get()).expanduser().resolve()
            output = Path(self.output_var.get()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            if not netlist.is_file():
                raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {netlist}")
        except Exception as exc:
            messagebox.showerror("Transient 실행 설정 오류", str(exc))
            return
        self.transient_log_text.delete("1.0", "end")
        self.transient_simulator = TransientSimulator(
            lambda text: self.events.put(("tran_log", text))
        )
        self.transient_run_button.configure(state="disabled")
        self.transient_stop_button.configure(state="normal")
        self.op_run_button.configure(state="disabled")
        self.ac_run_button.configure(state="disabled")
        self.transient_progress.start(12)
        jobs = len(stimuli) * len(channels)
        self.transient_status_var.set(f"Transient {jobs} jobs 실행 중...")

        def worker() -> None:
            """Run blocking transient jobs outside Tk's event loop."""

            try:
                results = self.transient_simulator.run(
                    channels,
                    stimuli,
                    netlist,
                    output,
                    ngspice,
                    settings,
                )
                self.events.put(("tran_done", results))
            except Exception as exc:
                self.events.put(("tran_error", exc))

        self.transient_worker = threading.Thread(target=worker, daemon=True)
        self.transient_worker.start()

    def _stop_transient_run(self) -> None:
        """Forward a stop request to the active transient runner."""

        if self.transient_simulator:
            self.transient_simulator.stop()
            self.transient_status_var.set("Transient 중지 요청 중...")

    def _finish_transient_controls(self) -> None:
        """Restore transient controls after success, stop, or failure."""

        self.transient_progress.stop()
        self.transient_worker = None
        self.transient_stop_button.configure(state="disabled")
        self._update_run_button()
        self._update_transient_run_button()

    def _on_transient_result_changed(self, _event: object) -> None:
        """Enable completed channels for the selected PWL and show one result."""

        self._refresh_transient_result_channel_buttons()

    def _on_transient_result_channel_changed(self) -> None:
        """Render the completed channel clicked in the 16-channel result row."""

        key = (
            self.transient_result_var.get(),
            int(self.transient_result_channel_var.get()),
        )
        result = self.transient_result_lookup.get(key)
        if result is not None:
            self._render_transient_result(result)
        self._refresh_transient_component_state()

    def _refresh_transient_result_channel_buttons(
        self,
        preferred_channel: int | None = None,
    ) -> None:
        """Enable only channels completed for the selected result PWL."""

        stimulus_label = self.transient_result_var.get()
        available = sorted(
            channel_number
            for label, channel_number in self.transient_result_lookup
            if label == stimulus_label
        )
        for channel_number, button in (
            self.transient_result_channel_buttons.items()
        ):
            button.configure(
                state="normal" if channel_number in available else "disabled"
            )
        if not available:
            self.transient_result_channel_var.set(-1)
            self._refresh_transient_component_state()
            return
        current = int(self.transient_result_channel_var.get())
        if current not in available:
            current = (
                preferred_channel
                if preferred_channel in available
                else available[0]
            )
            self.transient_result_channel_var.set(current)
        self._on_transient_result_channel_changed()

    def _transient_stimulus_label(self, path: Path) -> str:
        """Return a stable PWL dropdown label, relative to its selected root."""

        candidate = path.expanduser().resolve()
        try:
            root = Path(
                self.transient_folder_var.get()
            ).expanduser().resolve()
            return candidate.relative_to(root).as_posix()
        except (OSError, ValueError):
            return f"{candidate.parent.name}/{candidate.name}"

    def _install_transient_results(
        self,
        results: Sequence[TransientResult],
    ) -> None:
        """Populate the PWL dropdown and 16-channel enabled result options."""

        previous_label = self.transient_result_var.get()
        previous_channel = int(self.transient_result_channel_var.get())
        lookup: dict[tuple[str, int], TransientResult] = {}
        path_by_label: dict[str, Path] = {}
        ordered_labels: list[str] = []
        for index, result in enumerate(results, start=1):
            base = self._transient_stimulus_label(result.stimulus_path)
            label = base
            while (
                label in path_by_label
                and path_by_label[label] != result.stimulus_path
            ):
                label = f"{base} [{index}]"
            if label not in path_by_label:
                path_by_label[label] = result.stimulus_path
                ordered_labels.append(label)
            lookup[(label, result.channel.ch)] = result
        self.transient_result_lookup = lookup
        self.transient_result_stimulus_paths = path_by_label
        labels = tuple(ordered_labels)
        self.transient_result_combo.configure(values=labels)
        if not labels:
            self.transient_result_var.set("")
            self.transient_result_channel_var.set(-1)
            self._refresh_transient_result_channel_buttons()
            return
        selected_label = (
            previous_label if previous_label in path_by_label else labels[0]
        )
        self.transient_result_var.set(selected_label)
        self._refresh_transient_result_channel_buttons(previous_channel)

    def _refresh_transient_component_state(self) -> None:
        """Explain changed components and whether the visible result is stale."""

        if not hasattr(self, "transient_component_state_var"):
            return
        default_by_channel = {
            channel.ch: channel for channel in self.default_channels
        }
        changed_channels = [
            channel.ch
            for channel in self.channels
            if default_by_channel.get(channel.ch) != channel
        ]
        if changed_channels:
            channel_text = ", ".join(
                f"ch{channel_number:02d}"
                for channel_number in changed_channels
            )
            text = (
                f"현재 소자값: 기본 CSV 대비 변경됨 ({channel_text}) · "
                "다음 Transient 실행에 자동 적용"
            )
        else:
            text = "현재 소자값: 기본 CSV와 동일"
        try:
            editor_has_changes = self._component_editor_has_changes()
        except Exception:
            editor_has_changes = False
        if editor_has_changes:
            text += (
                " · DC 편집칸에 아직 적용하지 않은 변경 있음"
                " (Transient 실행 시 자동 적용)"
            )

        result = self.transient_result_lookup.get(
            (
                self.transient_result_var.get(),
                int(self.transient_result_channel_var.get()),
            )
        )
        if result is not None:
            try:
                current = self._channel_by_number(result.channel.ch)
            except ValueError:
                current = None
            if current == result.channel:
                text += (
                    f" · 표시 중 ch{result.channel.ch:02d} 결과에 "
                    "현재 소자값 반영됨"
                )
            else:
                text += (
                    f" · 표시 중 ch{result.channel.ch:02d} 결과는 변경 전 값"
                    " (재실행 필요)"
                )
        self.transient_component_state_var.set(text)

    def _build_transient_range_controls(self, parent: object) -> None:
        """Build a compact one-line X/Y range editor plus one status line."""

        ranges = self.ttk.LabelFrame(
            parent,
            text="표시 범위",
            padding=(6, 3),
        )
        ranges.pack(anchor="w", padx=6, pady=(0, 3))
        self.ttk.Label(ranges, text="X [ms]").grid(
            row=0, column=0, sticky="e", padx=(0, 4)
        )
        x_mode = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_x_mode_var,
            values=("자동", "수동"),
            state="readonly",
            width=5,
        )
        x_mode.grid(row=0, column=1)
        x_mode.bind(
            "<<ComboboxSelected>>",
            self._on_transient_x_mode_changed,
        )
        self.ttk.Label(ranges, text="min").grid(
            row=0, column=2, sticky="e", padx=(7, 3)
        )
        self.transient_x_min_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_x_min_var, width=9
        )
        self.transient_x_min_entry.grid(row=0, column=3)
        self.ttk.Label(ranges, text="–").grid(row=0, column=4, padx=3)
        self.transient_x_max_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_x_max_var, width=9
        )
        self.transient_x_max_entry.grid(row=0, column=5)
        for entry in (
            self.transient_x_min_entry,
            self.transient_x_max_entry,
        ):
            entry.bind(
                "<Return>",
                lambda _event: self._apply_transient_x_range(),
            )
        self.ttk.Button(
            ranges,
            text="적용",
            command=self._apply_transient_x_range,
        ).grid(row=0, column=6, padx=(5, 0))
        self.ttk.Separator(
            ranges,
            orient="vertical",
        ).grid(row=0, column=7, sticky="ns", padx=9)

        self.ttk.Label(ranges, text="Y [mV]").grid(
            row=0, column=8, sticky="e", padx=(0, 4)
        )
        y_target = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_y_target_var,
            values=("Vin", "Vfilt_out", "Venv_out + Vref"),
            state="readonly",
            width=15,
        )
        y_target.grid(row=0, column=9)
        y_target.bind(
            "<<ComboboxSelected>>",
            self._on_transient_y_target_changed,
        )
        y_mode = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_y_mode_var,
            values=("자동", "수동"),
            state="readonly",
            width=5,
        )
        y_mode.grid(row=0, column=10, padx=(5, 0))
        y_mode.bind(
            "<<ComboboxSelected>>",
            self._on_transient_y_mode_changed,
        )
        self.ttk.Label(ranges, text="min").grid(
            row=0, column=11, sticky="e", padx=(7, 3)
        )
        self.transient_y_min_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_y_min_var, width=9
        )
        self.transient_y_min_entry.grid(row=0, column=12)
        self.ttk.Label(ranges, text="–").grid(row=0, column=13, padx=3)
        self.transient_y_max_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_y_max_var, width=9
        )
        self.transient_y_max_entry.grid(row=0, column=14)
        for entry in (
            self.transient_y_min_entry,
            self.transient_y_max_entry,
        ):
            entry.bind(
                "<Return>",
                lambda _event: self._apply_transient_y_range(),
            )
        self.ttk.Button(
            ranges,
            text="적용",
            command=self._apply_transient_y_range,
        ).grid(row=0, column=15, padx=(5, 0))
        self.ttk.Label(
            ranges,
            textvariable=self.transient_range_status_var,
            foreground="#164a7b",
        ).grid(
            row=1,
            column=0,
            columnspan=16,
            sticky="w",
            pady=(3, 0),
        )

    def _render_transient_result(self, result: TransientResult) -> None:
        """Render large ms/mV plots with compact numeric range controls."""

        from matplotlib.figure import Figure

        self._dispose_cursor_group(self.transient_plot_cursors)
        self.transient_plot_cursors.clear()
        self.transient_axes.clear()
        self.transient_canvas = None
        self._clear_frame(self.transient_plot_frame)
        header = self.ttk.Frame(self.transient_plot_frame, padding=(8, 4))
        header.pack(fill="x")
        self.ttk.Label(
            header,
            text=(
                f"{result.stimulus_path.parent.name}/{result.stimulus_path.name} · "
                f"ch{result.channel.ch:02d} · {len(result.rows)} points"
            ),
        ).pack(side="left")
        self._build_transient_range_controls(self.transient_plot_frame)

        plot_host = self.ttk.Frame(self.transient_plot_frame)
        plot_host.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        figure = Figure(
            figsize=(14.0, 9.2), dpi=100, constrained_layout=True
        )
        axes = figure.subplots(3, 1, sharex=True)
        times_ms, vin_mv, vfilt_mv, venv_mv, vref_mv = (
            transient_rows_to_ms_mv(result.rows)
        )
        line_vin = axes[0].plot(
            times_ms, vin_mv, color="#1565c0", label="Vin"
        )[0]
        line_vfilt = axes[1].plot(
            times_ms, vfilt_mv, color="#00897b", label="Vfilt_out"
        )[0]
        line_venv = axes[2].plot(
            times_ms, venv_mv, color="#1565c0", label="Venv_out"
        )[0]
        line_vref = axes[2].plot(
            times_ms, vref_mv, color="#2e7d32", label="Vref"
        )[0]
        axes[0].set_ylabel("Vin [mV]")
        axes[1].set_ylabel("Vfilt_out [mV]")
        axes[2].set_ylabel("Venv_out / Vref [mV]")
        axes[2].set_xlabel("Time [ms]")
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        axes[2].legend(loc="best")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.28)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)

        x_low = min(times_ms)
        x_high = max(times_ms)
        if x_high <= x_low:
            x_high = x_low + 1.0
        self.transient_default_x_limits_ms = (x_low, x_high)
        self.transient_default_y_limits_mv = {
            "Vin": data_occupancy_limits(vin_mv),
            "Vfilt_out": data_occupancy_limits(vfilt_mv),
            "Venv_out + Vref": data_occupancy_limits(
                [*venv_mv, *vref_mv]
            ),
        }
        self.transient_axes = {
            "Vin": axes[0],
            "Vfilt_out": axes[1],
            "Venv_out + Vref": axes[2],
        }
        use_manual_x = (
            self.transient_x_mode_var.get() == "수동"
            and self.transient_manual_x_limits_ms is not None
        )
        axes[0].set_xlim(
            *(
                self.transient_manual_x_limits_ms
                if use_manual_x
                else self.transient_default_x_limits_ms
            )
        )
        for name, axis in self.transient_axes.items():
            use_manual_y = (
                self.transient_y_modes.get(name) == "수동"
                and name in self.transient_manual_y_limits_mv
            )
            axis.set_ylim(
                *(
                    self.transient_manual_y_limits_mv[name]
                    if use_manual_y
                    else self.transient_default_y_limits_mv[name]
                )
            )
        if self.transient_y_target_var.get() not in self.transient_axes:
            self.transient_y_target_var.set("Vin")
        self.transient_y_mode_var.set(
            self.transient_y_modes.get(
                self.transient_y_target_var.get(),
                "자동",
            )
        )

        self.transient_canvas = self._embed_multi_axes_figure(
            plot_host,
            figure,
            (
                (axes[0], (line_vin,)),
                (axes[1], (line_vfilt,)),
                (axes[2], (line_venv, line_vref)),
            ),
        )
        for axis in axes:
            axis.callbacks.connect(
                "xlim_changed",
                self._schedule_transient_range_sync,
            )
            axis.callbacks.connect(
                "ylim_changed",
                self._schedule_transient_range_sync,
            )
        self._sync_transient_range_fields()
        self._update_transient_range_entry_states()

    @staticmethod
    def _validated_axis_pair(
        minimum_text: str,
        maximum_text: str,
        label: str,
    ) -> tuple[float, float]:
        """Parse a finite, increasing min/max pair from transient controls."""

        try:
            minimum = float(minimum_text)
            maximum = float(maximum_text)
        except ValueError as exc:
            raise ValueError(f"{label}: 최소/최대값에 숫자를 입력하세요.") from exc
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"{label}: 유한한 값을 입력하세요.")
        if minimum >= maximum:
            raise ValueError(f"{label}: 최소값은 최대값보다 작아야 합니다.")
        return minimum, maximum

    def _apply_transient_x_range(self) -> None:
        """Apply only the shared X-axis auto/manual mode and limits."""

        from tkinter import messagebox

        if (
            not self.transient_axes
            or self.transient_canvas is None
            or self.transient_default_x_limits_ms is None
        ):
            return
        mode = self.transient_x_mode_var.get()
        try:
            if mode == "자동":
                self.transient_manual_x_limits_ms = None
                x_limits = self.transient_default_x_limits_ms
            elif mode == "수동":
                x_limits = self._validated_axis_pair(
                    self.transient_x_min_var.get(),
                    self.transient_x_max_var.get(),
                    "X 범위 [ms]",
                )
                self.transient_manual_x_limits_ms = x_limits
            else:
                raise ValueError("X축 모드는 자동 또는 수동이어야 합니다.")
        except ValueError as exc:
            messagebox.showerror("Transient 표시 범위 오류", str(exc))
            return
        next(iter(self.transient_axes.values())).set_xlim(*x_limits)
        self._sync_transient_range_fields(update_x=True)
        self._update_transient_range_entry_states()
        self.transient_canvas.draw_idle()
        self.transient_range_status_var.set(
            f"X축 {mode}: {x_limits[0]:.7g}–{x_limits[1]:.7g} ms"
        )

    def _apply_transient_y_range(self) -> None:
        """Apply only the selected graph's Y-axis auto/manual mode and limits."""

        from tkinter import messagebox

        if not self.transient_axes or self.transient_canvas is None:
            return
        target = self.transient_y_target_var.get()
        mode = self.transient_y_mode_var.get()
        try:
            axis = self.transient_axes[target]
            if mode == "자동":
                self.transient_manual_y_limits_mv.pop(target, None)
                y_limits = self.transient_default_y_limits_mv[target]
            elif mode == "수동":
                y_limits = self._validated_axis_pair(
                    self.transient_y_min_var.get(),
                    self.transient_y_max_var.get(),
                    "Y 범위 [mV]",
                )
                self.transient_manual_y_limits_mv[target] = y_limits
            else:
                raise ValueError("Y축 모드는 자동 또는 수동이어야 합니다.")
        except (KeyError, ValueError) as exc:
            messagebox.showerror("Transient 표시 범위 오류", str(exc))
            return
        self.transient_y_modes[target] = mode
        axis.set_ylim(*y_limits)
        self._sync_transient_range_fields(update_x=False)
        self._update_transient_range_entry_states()
        self.transient_canvas.draw_idle()
        self.transient_range_status_var.set(
            f"{target} Y축 {mode}: "
            f"{y_limits[0]:.7g}–{y_limits[1]:.7g} mV"
        )

    def _on_transient_x_mode_changed(self, _event: object = None) -> None:
        """Enable X entries for manual mode and apply auto mode immediately."""

        self._update_transient_range_entry_states()
        if self.transient_x_mode_var.get() == "자동":
            self._apply_transient_x_range()

    def _on_transient_y_mode_changed(self, _event: object = None) -> None:
        """Remember the selected graph's Y mode and apply auto immediately."""

        target = self.transient_y_target_var.get()
        self.transient_y_modes[target] = self.transient_y_mode_var.get()
        self._update_transient_range_entry_states()
        if self.transient_y_mode_var.get() == "자동":
            self._apply_transient_y_range()

    def _on_transient_y_target_changed(self, _event: object = None) -> None:
        """Load the selected graph's independent Y mode and current limits."""

        self.transient_y_mode_var.set(
            self.transient_y_modes.get(
                self.transient_y_target_var.get(),
                "자동",
            )
        )
        self._sync_transient_range_fields(update_x=False)
        self._update_transient_range_entry_states()

    def _update_transient_range_entry_states(self) -> None:
        """Enable numeric entries only for axes currently in manual mode."""

        x_state = (
            "normal"
            if self.transient_x_mode_var.get() == "수동"
            else "disabled"
        )
        y_state = (
            "normal"
            if self.transient_y_mode_var.get() == "수동"
            else "disabled"
        )
        for entry in (
            self.transient_x_min_entry,
            self.transient_x_max_entry,
        ):
            if entry is not None:
                entry.configure(state=x_state)
        for entry in (
            self.transient_y_min_entry,
            self.transient_y_max_entry,
        ):
            if entry is not None:
                entry.configure(state=y_state)

    def _sync_transient_range_fields(
        self,
        _event: object = None,
        *,
        update_x: bool = True,
    ) -> None:
        """Mirror current Matplotlib limits into the ms/mV entry fields."""

        if not self.transient_axes:
            return
        if update_x:
            x_min, x_max = next(iter(self.transient_axes.values())).get_xlim()
            self.transient_x_min_var.set(f"{x_min:.9g}")
            self.transient_x_max_var.set(f"{x_max:.9g}")
        target = self.transient_y_target_var.get()
        axis = self.transient_axes.get(target)
        if axis is not None:
            y_min, y_max = axis.get_ylim()
            self.transient_y_min_var.set(f"{y_min:.9g}")
            self.transient_y_max_var.set(f"{y_max:.9g}")

    def _schedule_transient_range_sync(self, _axis: object) -> None:
        """Coalesce repeated shared-axis callbacks into one idle field update."""

        if self.transient_range_sync_pending:
            return
        self.transient_range_sync_pending = True

        def sync_once() -> None:
            """Clear the pending flag and mirror the latest displayed limits."""

            self.transient_range_sync_pending = False
            self._sync_transient_range_fields()

        self.root.after_idle(sync_once)

    def _embed_multi_axes_figure(
        self,
        parent: object,
        figure: object,
        axes_lines: Sequence[tuple[object, Sequence[object]]],
    ) -> object:
        """Embed one figure with button-armed line-following cursors per axes."""

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as exc:
            raise RuntimeError(
                "그래프를 표시하려면 matplotlib가 필요합니다."
            ) from exc
        canvas = FigureCanvasTkAgg(figure, master=parent)
        new_cursors: list[InteractivePlotCursor] = []
        for axes, lines in axes_lines:
            cursor = InteractivePlotCursor(
                canvas,
                axes,
                lines,
                x_name="t",
                x_unit="ms",
                y_name="V",
                y_unit="mV",
            )
            new_cursors.append(cursor)
            self.transient_plot_cursors.append(cursor)
        cursor_controls = self.ttk.Frame(parent)
        cursor_controls.pack(side="bottom", fill="x", pady=(2, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 추가",
            command=lambda: self._arm_plot_cursors(new_cursors),
        ).pack(side="left")
        self.ttk.Button(
            cursor_controls,
            text="커서 전체 삭제",
            command=lambda: self._clear_plot_cursors(new_cursors),
        ).pack(side="left", padx=(5, 0))
        self.ttk.Label(
            cursor_controls,
            text=(
                "커서 추가 → 선을 따라 이동 → 세로 보조선 위치를 클릭해 고정 "
                "· 반복 추가 가능"
            ),
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        return canvas

    @staticmethod
    def _arm_plot_cursors(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Arm one placement on the next left click across a cursor group."""

        for cursor in cursors:
            cursor.arm()

    @staticmethod
    def _clear_plot_cursors(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Remove all fixed cursors from every axes in a cursor group."""

        for cursor in cursors:
            cursor.clear()

    @staticmethod
    def _dispose_cursor_group(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Disconnect all callbacks before destroying a plot's Tk widgets."""

        for cursor in tuple(cursors):
            cursor.dispose()

    def _select_all(self) -> None:
        """Select every loaded channel in the DC run list."""

        self.channel_list.selection_set(0, "end")

    def _clear_selection(self) -> None:
        """Clear the DC run-list channel selection."""

        self.channel_list.selection_clear(0, "end")

    def _select_all_ac(self) -> None:
        """Select every loaded channel in the AC run list."""

        self.ac_channel_list.selection_set(0, "end")

    def _clear_ac_selection(self) -> None:
        """Clear the AC run-list channel selection."""

        self.ac_channel_list.selection_clear(0, "end")

    def _reset_component_defaults(self) -> None:
        """Restore the immutable factory CSV into the in-memory applied set."""

        from tkinter import messagebox

        if (
            self.component_dirty or self._component_editor_has_changes()
        ) and not messagebox.askyesno(
            "기본값 복원",
            "저장하지 않은 소자값 변경을 버리고 기본값을 다시 불러올까요?",
        ):
            return
        self._reload_channels(DEFAULT_TABLE)

    def _save_component_version_from_gui(self) -> None:
        """Commit the visible editor and save all 16 channels to a new CSV."""

        from tkinter import messagebox

        try:
            self._commit_component_editor()
            target = save_component_version(self.channels)
            self.component_source_path = target
            self.table_var.set(str(target))
            self.component_source_label_var.set(target.name)
            self.component_dirty = False
            self.status_var.set(f"소자값 버전 저장됨: {target.name}")
            messagebox.showinfo(
                "소자값 버전 저장",
                "기본 channel_components.csv는 변경하지 않았습니다.\n\n"
                f"새 버전:\n{target}",
            )
        except Exception as exc:
            messagebox.showerror("소자값 버전 저장 오류", str(exc))

    def _reload_channels(self, path: Path | None = None) -> None:
        """Install a factory/version CSV as the current in-memory component set."""

        from tkinter import messagebox

        try:
            source = (path or self.component_source_path).expanduser().resolve()
            loaded = load_channels(source)
            if source == DEFAULT_TABLE.resolve():
                self.default_channels = list(loaded)
            elif self.default_channels:
                expected = {channel.ch for channel in self.default_channels}
                actual = {channel.ch for channel in loaded}
                if actual != expected:
                    raise ValueError(
                        "버전 CSV의 채널 집합이 기본 CSV와 다릅니다. "
                        f"기본={sorted(expected)}, 버전={sorted(actual)}"
                    )
            previous_by_list: dict[str, set[int]] = {}
            for attribute in ("channel_list", "ac_channel_list"):
                listbox = getattr(self, attribute, None)
                previous_by_list[attribute] = (
                    {
                        self.channels[index].ch
                        for index in listbox.curselection()
                        if 0 <= index < len(self.channels)
                    }
                    if listbox is not None
                    else set()
                )
            self.channels = loaded
            self.component_source_path = source
            self.table_var.set(str(source))
            self.component_source_label_var.set(
                "기본값 (읽기 전용)"
                if source == DEFAULT_TABLE.resolve()
                else source.name
            )
            self.component_dirty = False
            self.component_editor_channel = None
            self.component_edit_vars = {}
            self.component_divider_edit_source = "resistors"
            self.component_divider_nominal_total_kohm = (
                DIVIDER_NOMINAL_TOTAL_KOHM
            )
            self.component_vdet_v = None
            self.pending_margin_retune = None
            self.op_pending_view_state = None
            for attribute in ("channel_list", "ac_channel_list"):
                listbox = getattr(self, attribute, None)
                if listbox is None:
                    continue
                listbox.delete(0, "end")
                for channel in self.channels:
                    listbox.insert(
                        "end", f"ch{channel.ch:02d}   fc={channel.f_c_hz:g} Hz"
                    )
                restored = False
                for index, channel in enumerate(self.channels):
                    if channel.ch in previous_by_list[attribute]:
                        listbox.selection_set(index)
                        restored = True
                if self.channels and not restored:
                    listbox.selection_set(0)
            self.status_var.set(
                f"{len(self.channels)}개 채널 준비됨 / "
                f"{self.component_source_label_var.get()}"
            )
            self._populate_transient_channels()
            self._refresh_transient_component_state()
            self._update_run_button()
            if self.last_op_results:
                self._refresh_operating_point_tab()
        except Exception as exc:
            messagebox.showerror("소자 테이블 오류", str(exc))
            self.status_var.set("소자 테이블 오류")

    def _selected_analyses(self) -> tuple[str, ...]:
        """Translate the two run checkboxes into analysis job keys."""

        selected: list[str] = []
        if self.run_ac_var.get():
            selected.append("ac")
        if self.run_op_var.get():
            selected.append("op")
        if not selected:
            raise ValueError("AC Sweep 또는 DC 동작점을 하나 이상 체크하세요.")
        return tuple(selected)

    def _update_run_button(self) -> None:
        """Enable the separate DC and AC buttons when their channel lists are ready."""

        running = bool(self.worker and self.worker.is_alive()) or bool(
            self.transient_worker and self.transient_worker.is_alive()
        )
        if hasattr(self, "op_run_button"):
            has_dc = bool(self.channel_list.curselection())
            self.op_run_button.configure(
                state="disabled" if running or not has_dc else "normal"
            )
        if hasattr(self, "ac_run_button"):
            has_ac = bool(self.ac_channel_list.curselection())
            self.ac_run_button.configure(
                state="disabled" if running or not has_ac else "normal"
            )

    def _settings(self, analyses: Sequence[str]) -> SweepSettings:
        """Parse current GUI text into a validated settings value object."""

        selected = normalize_analyses(analyses)
        defaults = SweepSettings()
        return SweepSettings(
            ac_points_per_decade=(
                int(self.ac_points_var.get())
                if "ac" in selected
                else defaults.ac_points_per_decade
            ),
            ac_start_hz=(
                float(self.ac_start_var.get())
                if "ac" in selected
                else defaults.ac_start_hz
            ),
            ac_stop_hz=(
                float(self.ac_stop_var.get())
                if "ac" in selected
                else defaults.ac_stop_hz
            ),
            output_node=self.output_node_var.get().strip(),
            pspice_compat=self.pspice_compat_var.get(),
        )

    def _start_run(
        self,
        analyses: Sequence[str] | None = None,
        channel_list: object | None = None,
        selected_channels: Sequence[Channel] | None = None,
    ) -> None:
        """Commit edited components and launch one DC or AC worker job set."""

        from tkinter import messagebox

        if self.worker and self.worker.is_alive():
            return
        try:
            if self.transient_worker and self.transient_worker.is_alive():
                raise RuntimeError("Transient 실행이 끝난 뒤 AC/OP를 시작하세요.")
            self._commit_component_editor()
            requested_analyses = normalize_analyses(
                analyses if analyses is not None else self._selected_analyses()
            )
            if selected_channels is not None:
                selected = list(selected_channels)
            else:
                active_list = channel_list or (
                    self.ac_channel_list
                    if requested_analyses == ("ac",)
                    else self.channel_list
                )
                indices = list(active_list.curselection())
                if not indices:
                    raise ValueError("실행할 채널을 하나 이상 선택하세요.")
                selected = [self.channels[index] for index in indices]
            if not selected:
                raise ValueError("실행할 채널을 하나 이상 선택하세요.")
            ngspice = find_ngspice(self.ngspice_var.get())
            settings = self._settings(requested_analyses)
            settings.validate(requested_analyses)
            netlist = Path(self.netlist_var.get()).expanduser().resolve()
            if not netlist.is_file():
                raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {netlist}")
            output = Path(self.output_var.get()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("실행 설정 오류", str(exc))
            return
        self.log_text.delete("1.0", "end")
        self.simulator = Simulator(
            lambda text: self.events.put(("log", text))
        )
        self.op_run_button.configure(state="disabled")
        self.ac_run_button.configure(state="disabled")
        self.transient_run_button.configure(state="disabled")
        self.op_stop_button.configure(state="normal")
        self.ac_stop_button.configure(state="normal")
        if "op" in requested_analyses:
            self.op_progress.start(12)
        if "ac" in requested_analyses:
            self.ac_progress.start(12)
        analysis_label = "DC 동작점" if requested_analyses == ("op",) else "AC"
        self.status_var.set(
            f"{analysis_label} · {len(selected)}개 채널 실행 중..."
        )

        def worker() -> None:
            """Run blocking simulation work outside Tk's event thread."""

            try:
                results = self.simulator.run(
                    selected,
                    netlist,
                    output,
                    ngspice,
                    settings,
                    requested_analyses,
                )
                self.events.put(("done", (requested_analyses, results)))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop_run(self) -> None:
        """Forward the stop request to the active simulator."""

        if self.simulator:
            self.simulator.stop()
            self.status_var.set("중지 요청 중...")

    def _poll_events(self) -> None:
        """Drain worker events on Tk's main thread and update the GUI safely."""

        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log_text.insert("end", str(payload) + "\n")
                    self.log_text.see("end")
                elif kind == "done":
                    requested_analyses, completed = payload  # type: ignore[misc]
                    completed_results = list(completed)
                    retune_channel = None
                    if "op" in requested_analyses and completed_results:
                        try:
                            retune_channel = self._prepare_margin_retune(
                                completed_results
                            )
                        except Exception as exc:
                            self.pending_margin_retune = None
                            messagebox.showerror(
                                "Vmargin 자동 보정 오류",
                                "첫 DC 결과는 표시하지만 새 Vdet 기준 "
                                f"R7/R8 보정은 적용하지 못했습니다.\n\n{exc}",
                            )
                    if retune_channel is not None:
                        self._finish_controls()
                        self._start_run(
                            ("op",),
                            selected_channels=(retune_channel,),
                        )
                        self.status_var.set(
                            f"ch{retune_channel.ch:02d} 새 Vdet 기준 "
                            "Vmargin 보정 DC 재실행 중..."
                        )
                        continue
                    self.last_results = completed_results
                    if "ac" in requested_analyses:
                        self.last_ac_results = [
                            result for result in completed_results if result.ac_rows
                        ]
                    if "op" in requested_analyses:
                        self.last_op_results = [
                            result
                            for result in completed_results
                            if result.op_voltages
                        ]
                    self._finish_controls()
                    if completed_results:
                        try:
                            has_ac = "ac" in requested_analyses
                            has_op = "op" in requested_analyses
                            self._render_results(
                                completed_results,
                                analyses=requested_analyses,
                            )
                            self.status_var.set(
                                f"{len(completed_results)}개 채널 완료"
                                + (" / AC" if has_ac else "")
                                + (" / OP" if has_op else "")
                            )
                        except Exception as exc:
                            self.status_var.set("해석 완료 / 그래프 표시 실패")
                            self.log_text.insert("end", f"\n그래프 오류: {exc}\n")
                            messagebox.showwarning(
                                "그래프 표시 오류",
                                f"ngspice 해석과 CSV 저장은 완료됐지만 "
                                f"그래프를 표시하지 못했습니다.\n\n{exc}",
                            )
                    else:
                        self.pending_margin_retune = None
                        self.op_pending_view_state = None
                        self.status_var.set("실행 중지됨")
                elif kind == "error":
                    self.pending_margin_retune = None
                    self.op_pending_view_state = None
                    self._finish_controls()
                    self.status_var.set("실행 실패")
                    self.log_text.insert("end", f"\nERROR: {payload}\n")
                    self.log_text.see("end")
                    messagebox.showerror("ngspice 실행 오류", str(payload))
                elif kind == "pwl_progress":
                    position, total, source, target = payload  # type: ignore[misc]
                    self.pwl_progress.configure(maximum=total, value=position - 1)
                    self.pwl_log_text.insert(
                        "end",
                        f"[{position}/{total}] {source} -> {target}\n",
                    )
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set(
                        f"PWL 변환 중: {position}/{total} · {source.name}"
                    )
                elif kind == "pwl_done":
                    results = payload  # type: ignore[assignment]
                    self._finish_pwl_controls()
                    self.pwl_progress.configure(value=len(results))
                    output_root = Path(
                        self.pwl_output_var.get()
                    ).expanduser().resolve()
                    self.pwl_log_text.insert(
                        "end",
                        f"\n완료: {len(results)}개\n"
                        f"manifest: {output_root / 'pwl_manifest.csv'}\n",
                    )
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set(
                        f"PWL {len(results)}개 변환 완료 · "
                        "DC=0 V / 10 mVpp / tail=0 V"
                    )
                    self.transient_folder_var.set(str(output_root))
                    self._refresh_transient_pwl_files()
                elif kind == "pwl_stopped":
                    self._finish_pwl_controls()
                    self.pwl_log_text.insert("end", f"\n중지됨: {payload}\n")
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set("PWL 변환 중지됨")
                elif kind == "pwl_error":
                    self._finish_pwl_controls()
                    self.pwl_log_text.insert("end", f"\nERROR: {payload}\n")
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set("PWL 변환 실패")
                    messagebox.showerror("PWL 변환 오류", str(payload))
                elif kind == "tran_log":
                    self.transient_log_text.insert(
                        "end", str(payload) + "\n"
                    )
                    self.transient_log_text.see("end")
                elif kind == "tran_done":
                    results = payload  # type: ignore[assignment]
                    self._finish_transient_controls()
                    self.last_transient_results = list(results)
                    if self.last_transient_results:
                        self._install_transient_results(
                            self.last_transient_results
                        )
                        self.transient_status_var.set(
                            f"Transient {len(results)} jobs 완료"
                        )
                    else:
                        self.transient_status_var.set("Transient 실행 중지됨")
                elif kind == "tran_error":
                    self._finish_transient_controls()
                    self.transient_log_text.insert(
                        "end", f"\nERROR: {payload}\n"
                    )
                    self.transient_log_text.see("end")
                    self.transient_status_var.set("Transient 실행 실패")
                    messagebox.showerror("Transient 실행 오류", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish_controls(self) -> None:
        """Restore controls after success, cancellation, or an error."""

        self.op_progress.stop()
        self.ac_progress.stop()
        self.worker = None
        self.op_stop_button.configure(state="disabled")
        self.ac_stop_button.configure(state="disabled")
        self._update_run_button()
        self._update_transient_run_button()

    def _show_empty_results(self) -> None:
        """Populate result tabs with instructions before the first simulation."""

        messages = (
            (self.mag_frame, "AC Sweep 실행 후 Magnitude 그래프가 표시됩니다."),
            (self.phase_frame, "AC Sweep 실행 후 Phase 그래프가 표시됩니다."),
            (
                self.op_frame,
                "DC 동작점 (.op) 실행 후 채널별 노드 전압 회로도가 표시됩니다.",
            ),
        )
        for frame, message in messages:
            self.ttk.Label(frame, text=message, justify="center").pack(expand=True)

    @staticmethod
    def _clear_frame(frame: object) -> None:
        """Destroy every child widget in a result-tab frame."""

        for widget in frame.winfo_children():
            widget.destroy()

    def _render_results(
        self,
        results: Sequence[ChannelResult],
        analyses: Sequence[str] | None = None,
    ) -> None:
        """Redraw only the completed analysis while preserving every visible tab."""

        ac_results = [result for result in results if result.ac_rows]
        op_results = [result for result in results if result.op_voltages]
        requested = (
            normalize_analyses(analyses)
            if analyses is not None
            else normalize_analyses(
                (
                    *(("ac",) if ac_results else ()),
                    *(("op",) if op_results else ()),
                )
            )
        )
        selected_ac_tab = self.ac_result_notebook.select()
        if "ac" in requested:
            for group in self.ac_plot_cursors.values():
                self._dispose_cursor_group(group)
                group.clear()
            self.metric_selection_overlay = None
            self.metric_table = None
            for frame in (self.mag_frame, self.phase_frame):
                self._clear_frame(frame)
            if ac_results:
                self._render_magnitude(ac_results)
                self._render_phase(ac_results)
            else:
                self.ttk.Label(
                    self.mag_frame, text="이번 실행에는 AC 결과가 없습니다."
                ).pack(expand=True)
                self.ttk.Label(
                    self.phase_frame, text="이번 실행에는 AC 결과가 없습니다."
                ).pack(expand=True)
            if selected_ac_tab:
                self.ac_result_notebook.select(selected_ac_tab)
        if "op" in requested:
            pending_view = getattr(self, "op_pending_view_state", None)
            if pending_view is None:
                self._capture_operating_point_view_state()
            else:
                (
                    self.op_detail_selected_tab,
                    self.op_pane_fraction,
                    self.op_canvas_view,
                ) = pending_view
            self._clear_frame(self.op_frame)
            if op_results:
                self._render_operating_point(op_results)
            else:
                self.ttk.Label(
                    self.op_frame, text="이번 실행에는 DC 동작점 결과가 없습니다."
                ).pack(expand=True)
            self.op_pending_view_state = None

    def _refresh_magnitude_tab(self) -> None:
        """Validate controls and redraw only the AC magnitude tab."""

        from tkinter import messagebox

        try:
            self._magnitude_axis_limits()
        except ValueError as exc:
            messagebox.showerror("AC Magnitude 축 범위 오류", str(exc))
            return
        self._dispose_cursor_group(self.ac_plot_cursors["magnitude"])
        self.ac_plot_cursors["magnitude"].clear()
        self._clear_frame(self.mag_frame)
        ac_results = list(self.last_ac_results)
        if ac_results:
            self._render_magnitude(ac_results)
        else:
            self.ttk.Label(
                self.mag_frame, text="이번 실행에는 AC 결과가 없습니다."
            ).pack(expand=True)

    def _refresh_phase_tab(self) -> None:
        """Redraw only the AC phase tab after an axis selection changes."""

        self._dispose_cursor_group(self.ac_plot_cursors["phase"])
        self.ac_plot_cursors["phase"].clear()
        self._clear_frame(self.phase_frame)
        ac_results = list(self.last_ac_results)
        if ac_results:
            self._render_phase(ac_results)
        else:
            self.ttk.Label(
                self.phase_frame, text="이번 실행에는 AC 결과가 없습니다."
            ).pack(expand=True)

    def _refresh_operating_point_tab(self) -> None:
        """Redraw only the OP schematic tab, preserving the outer tab selection."""

        self._capture_operating_point_view_state()
        self._clear_frame(self.op_frame)
        op_results = list(self.last_op_results)
        if op_results:
            self._render_operating_point(op_results)
        else:
            self.ttk.Label(
                self.op_frame, text="이번 실행에는 DC 동작점 결과가 없습니다."
            ).pack(expand=True)

    def _plot_controls(
        self,
        parent: object,
        x_variable: object,
        y_variable: object,
        y_values: Sequence[str],
        refresh_command: Callable[[], None],
        range_variables: tuple[object, object, object, object] | None = None,
    ) -> None:
        """Build axis-scale/range controls shared by AC plot tabs."""

        controls = self.ttk.Frame(parent, padding=(8, 6))
        controls.pack(fill="x")
        top = self.ttk.Frame(controls)
        top.pack(fill="x")
        self.ttk.Label(top, text="X축").pack(side="left")
        x_combo = self.ttk.Combobox(
            top,
            textvariable=x_variable,
            values=("Linear", "Decade"),
            state="readonly",
            width=10,
        )
        x_combo.pack(side="left", padx=(4, 14))
        self.ttk.Label(top, text="Y축").pack(side="left")
        y_combo = self.ttk.Combobox(
            top,
            textvariable=y_variable,
            values=tuple(y_values),
            state="readonly" if len(y_values) > 1 else "disabled",
            width=17,
        )
        y_combo.pack(side="left", padx=(4, 14))
        if range_variables is not None:
            x_min, x_max, y_min, y_max = range_variables
            self.ttk.Separator(top, orient="vertical").pack(
                side="left", fill="y", padx=(0, 12)
            )
            self.ttk.Label(top, text="X 범위").pack(side="left")
            self.ttk.Entry(top, textvariable=x_min, width=9).pack(
                side="left", padx=(4, 2)
            )
            self.ttk.Label(top, text="~").pack(side="left")
            self.ttk.Entry(top, textvariable=x_max, width=9).pack(
                side="left", padx=(2, 12)
            )
            self.ttk.Label(top, text="Y 범위").pack(side="left")
            self.ttk.Entry(top, textvariable=y_min, width=9).pack(
                side="left", padx=(4, 2)
            )
            self.ttk.Label(top, text="~").pack(side="left")
            self.ttk.Entry(top, textvariable=y_max, width=9).pack(
                side="left", padx=(2, 8)
            )
            self.ttk.Button(
                top, text="범위 적용", command=refresh_command
            ).pack(side="left")
            self.ttk.Button(
                top, text="자동 범위", command=self._reset_magnitude_axis_limits
            ).pack(side="left", padx=(5, 0))
        help_prefix = (
            "범위 입력은 현재 축 단위 기준, 빈칸은 자동 · "
            if range_variables is not None
            else ""
        )
        self.ttk.Label(
            controls,
            text=help_prefix
            + (
                "커서 추가 → 선을 따라 이동 → 클릭해 고정 · 반복하면 다중 커서 · "
                "아래 도구: 확대/이동/저장"
            ),
        ).pack(anchor="w", pady=(5, 0))
        for combo in (x_combo, y_combo):
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event: refresh_command(),
            )

    @staticmethod
    def _parse_axis_limits(
        minimum_text: str,
        maximum_text: str,
        axis_name: str,
        logarithmic: bool,
    ) -> tuple[float | None, float | None]:
        """Parse optional axis bounds and enforce ordering/log constraints."""

        def parse_one(text: str, field_name: str) -> float | None:
            """Parse one optional finite bound for the enclosing axis."""

            stripped = text.strip()
            if not stripped:
                return None
            try:
                value = float(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}에는 숫자 또는 빈칸을 입력하세요."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{field_name}에는 유한한 숫자를 입력하세요.")
            if logarithmic and value <= 0:
                raise ValueError(
                    f"{axis_name}이 Decade일 때 {field_name}은 0보다 커야 합니다."
                )
            return value

        minimum = parse_one(minimum_text, f"{axis_name} 최소")
        maximum = parse_one(maximum_text, f"{axis_name} 최대")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(
                f"{axis_name} 최소값은 최대값보다 작아야 합니다."
            )
        return minimum, maximum

    def _magnitude_axis_limits(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Read and validate the four manual magnitude-axis bounds."""

        x_min, x_max = self._parse_axis_limits(
            self.mag_x_min_var.get(),
            self.mag_x_max_var.get(),
            "X축",
            self.mag_x_scale_var.get() == "Decade",
        )
        y_min, y_max = self._parse_axis_limits(
            self.mag_y_min_var.get(),
            self.mag_y_max_var.get(),
            "Y축",
            self.mag_y_scale_var.get() == "Decade (|V|)",
        )
        return x_min, x_max, y_min, y_max

    def _reset_magnitude_axis_limits(self) -> None:
        """Clear all manual magnitude limits and redraw with autoscaling."""

        for variable in (
            self.mag_x_min_var,
            self.mag_x_max_var,
            self.mag_y_min_var,
            self.mag_y_max_var,
        ):
            variable.set("")
        self._refresh_magnitude_tab()

    def _embed_figure(
        self,
        parent: object,
        figure: object,
        axes: object,
        data_lines: Sequence[object],
        cursor_group: str,
    ) -> object:
        """Embed a figure and register its cursor under one AC plot group."""

        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
        except ImportError as exc:
            raise RuntimeError(
                "그래프를 표시하려면 matplotlib가 필요합니다."
            ) from exc
        host = self.ttk.Frame(parent)
        host.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        canvas = FigureCanvasTkAgg(figure, master=host)
        toolbar = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        cursor = InteractivePlotCursor(canvas, axes, data_lines)
        cursor_controls = self.ttk.Frame(host)
        cursor_controls.pack(side="bottom", fill="x", pady=(2, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 추가",
            command=cursor.arm,
        ).pack(side="left")
        self.ttk.Button(
            cursor_controls,
            text="커서 전체 삭제",
            command=cursor.clear,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Label(
            cursor_controls,
            text="추가 → 선을 따라 이동 → 세로 보조선 위치 클릭 · 반복 추가 가능",
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.ac_plot_cursors.setdefault(cursor_group, []).append(cursor)
        return canvas

    def _render_magnitude(self, results: Sequence[ChannelResult]) -> None:
        """Render the magnitude graph left and the 16-channel metrics list right."""

        from matplotlib.figure import Figure

        self._plot_controls(
            self.mag_frame,
            self.mag_x_scale_var,
            self.mag_y_scale_var,
            ("Linear (dB)", "Decade (|V|)"),
            self._refresh_magnitude_tab,
            (
                self.mag_x_min_var,
                self.mag_x_max_var,
                self.mag_y_min_var,
                self.mag_y_max_var,
            ),
        )
        paned = self.ttk.Panedwindow(self.mag_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        plot_host = self.ttk.Frame(paned)
        metrics_host = self.ttk.Frame(paned, width=390)
        paned.add(plot_host, weight=4)
        paned.add(metrics_host, weight=1)
        figure = Figure(figsize=(12.5, 7.2), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        use_decade_x = self.mag_x_scale_var.get() == "Decade"
        use_decade_y = self.mag_y_scale_var.get() == "Decade (|V|)"
        x_min, x_max, y_min, y_max = self._magnitude_axis_limits()
        data_lines: list[object] = []
        all_y_values: list[float] = []
        metric_entries: dict[int, tuple[object, AcMetrics, float]] = {}
        for result in results:
            label = f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)"
            frequencies = [row[0] for row in result.ac_rows]
            db_values = [row[1] for row in result.ac_rows]
            y_values = (
                [10.0 ** (value / 20.0) for value in db_values]
                if use_decade_y
                else db_values
            )
            all_y_values.extend(y_values)
            line = axes.plot(frequencies, y_values, label=label)[0]
            data_lines.append(line)
            metric = result.ac_metrics
            if metric is not None:
                marker_y = (
                    10.0 ** (metric.peak_db / 20.0)
                    if use_decade_y
                    else metric.peak_db
                )
                axes.plot(
                    metric.center_hz,
                    marker_y,
                    marker="o",
                    markersize=5,
                    color=line.get_color(),
                    label="_nolegend_",
                )
                metric_entries[result.channel.ch] = (line, metric, marker_y)
        if use_decade_x:
            axes.set_xscale("log")
        if use_decade_y:
            axes.set_yscale("log")
            axes.set_ylabel("|Vout| [V]")
        else:
            axes.set_ylabel("Magnitude [dB]")
        axes.set_xlim(left=x_min, right=x_max)
        auto_y_min, auto_y_max = data_occupancy_limits(
            all_y_values,
            logarithmic=use_decade_y,
        )
        resolved_y_min = auto_y_min if y_min is None else y_min
        resolved_y_max = auto_y_max if y_max is None else y_max
        if resolved_y_min >= resolved_y_max:
            raise ValueError(
                "수동 Y축 경계와 자동 Y축 경계를 함께 적용한 결과 "
                "최소값이 최대값 이상입니다."
            )
        axes.set_ylim(bottom=resolved_y_min, top=resolved_y_max)
        axes.set_title("AC Magnitude")
        axes.set_xlabel("Frequency [Hz]")
        axes.grid(True, which="both", alpha=0.32)
        axes.legend(fontsize=8, ncol=2)
        canvas = self._embed_figure(
            plot_host,
            figure,
            axes,
            data_lines,
            "magnitude",
        )
        self.metric_selection_overlay = MetricSelectionOverlay(
            canvas, axes, metric_entries
        )
        self._build_metrics_table(
            metrics_host, results, self.metric_selection_overlay
        )

    def _render_phase(self, results: Sequence[ChannelResult]) -> None:
        """Render all selected AC phase curves with interactive inspection."""

        from matplotlib.figure import Figure

        self._plot_controls(
            self.phase_frame,
            self.phase_x_scale_var,
            self.phase_y_scale_var,
            ("Linear (deg)",),
            self._refresh_phase_tab,
        )
        figure = Figure(figsize=(12.5, 7.8), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        data_lines: list[object] = []
        for result in results:
            label = f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)"
            line = axes.plot(
                [row[0] for row in result.ac_rows],
                [row[2] for row in result.ac_rows],
                label=label,
            )[0]
            data_lines.append(line)
        if self.phase_x_scale_var.get() == "Decade":
            axes.set_xscale("log")
        axes.set_title("AC Phase")
        axes.set_xlabel("Frequency [Hz]")
        axes.set_ylabel("Phase [deg]")
        axes.grid(True, which="both", alpha=0.32)
        axes.legend(fontsize=8, ncol=2)
        self._embed_figure(
            self.phase_frame,
            figure,
            axes,
            data_lines,
            "phase",
        )

    def _build_metrics_table(
        self,
        parent: object,
        results: Sequence[ChannelResult],
        overlay: MetricSelectionOverlay,
    ) -> None:
        """Build the copyable 16-row list that controls persistent peak cursors."""

        frame = self.ttk.LabelFrame(
            parent, text="16채널 f0 / Gain / Q", padding=6
        )
        frame.pack(fill="both", expand=True)
        self.ttk.Label(
            frame,
            text=(
                "행 클릭: 선택 추가/해제\n"
                "여러 행을 선택하면 그래프에 여러 커서가 표시됩니다."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 5))
        columns = ("channel", "center", "gain", "q")
        table = self.ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=16,
        )
        headings = {
            "channel": "채널",
            "center": "중심주파수 [Hz]",
            "gain": "Gain [dB]",
            "q": "Q",
        }
        widths = {
            "channel": 58,
            "center": 125,
            "gain": 92,
            "q": 72,
        }
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(
                column,
                width=widths[column],
                anchor="center",
                stretch=column == "center",
            )
        result_by_channel = {
            result.channel.ch: result for result in results
        }
        channels = self.channels or [
            result.channel for result in sorted(
                results, key=lambda item: item.channel.ch
            )
        ]
        self.metric_copy_rows = {}
        for channel in sorted(channels, key=lambda item: item.ch):
            result = result_by_channel.get(channel.ch)
            metric = result.ac_metrics if result else None
            item_id = f"ch{channel.ch:02d}"
            table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    item_id,
                    self._format_metric(metric.center_hz if metric else None),
                    self._format_metric(metric.peak_db if metric else None, digits=4),
                    self._format_metric(metric.q if metric else None, digits=4),
                ),
            )
            self.metric_copy_rows[item_id] = format_metric_copy_line(
                channel, metric
            )
        scroll = self.ttk.Scrollbar(
            frame, orient="vertical", command=table.yview
        )
        table.configure(yscrollcommand=scroll.set)
        table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        buttons = self.ttk.Frame(parent, padding=(0, 6, 0, 0))
        buttons.pack(fill="x")
        self.ttk.Button(
            buttons,
            text="선택 행 복사",
            command=lambda: self._copy_metric_rows(False),
        ).pack(side="left")
        self.ttk.Button(
            buttons,
            text="전체 16채널 복사",
            command=lambda: self._copy_metric_rows(True),
        ).pack(side="left", padx=(5, 0))
        table.bind(
            "<Button-1>",
            lambda event: self._toggle_metric_table_row(
                event, table, overlay
            ),
        )
        table.bind(
            "<<TreeviewSelect>>",
            lambda _event: self._sync_metric_selection(table, overlay),
        )
        table.bind(
            "<Control-c>",
            lambda _event: self._copy_metric_rows(False),
        )
        table.bind(
            "<Control-C>",
            lambda _event: self._copy_metric_rows(False),
        )
        self.metric_table = table
        for channel in sorted(self.selected_metric_channels):
            item_id = f"ch{channel:02d}"
            if table.exists(item_id):
                table.selection_add(item_id)
        self._sync_metric_selection(table, overlay)

    def _toggle_metric_table_row(
        self,
        event: object,
        table: object,
        overlay: MetricSelectionOverlay,
    ) -> str | None:
        """Toggle one metric row on a plain click so Ctrl is not required."""

        if table.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        item_id = table.identify_row(event.y)
        if not item_id:
            return None
        if item_id in table.selection():
            table.selection_remove(item_id)
        else:
            table.selection_add(item_id)
        table.focus(item_id)
        self._sync_metric_selection(table, overlay)
        return "break"

    def _sync_metric_selection(
        self,
        table: object,
        overlay: MetricSelectionOverlay,
    ) -> None:
        """Mirror selected metric rows into the persistent graph overlay."""

        channel_numbers = []
        for item_id in table.selection():
            match = re.fullmatch(r"ch(\d+)", item_id)
            if match:
                channel_numbers.append(int(match.group(1)))
        self.selected_metric_channels = set(channel_numbers)
        overlay.show_channels(sorted(channel_numbers))

    def _copy_metric_rows(self, copy_all: bool) -> str:
        """Copy selected or all metric lines as tab-separated plain text."""

        table = self.metric_table
        if table is None:
            return "break"
        item_ids = (
            tuple(table.get_children())
            if copy_all
            else tuple(table.selection())
        )
        if not item_ids:
            item_ids = tuple(table.get_children())
        text = "\n".join(
            self.metric_copy_rows[item_id]
            for item_id in item_ids
            if item_id in self.metric_copy_rows
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_var.set(f"AC 메트릭 {len(item_ids)}행 복사됨")
        return "break"

    def _capture_operating_point_view_state(self) -> None:
        """Save the OP detail tab, pane ratio, and schematic scroll position."""

        notebook = self.op_detail_notebook
        if notebook is not None:
            try:
                selected = notebook.select()
                for name, tab in self.op_detail_tabs.items():
                    if str(tab) == selected:
                        self.op_detail_selected_tab = name
                        break
            except Exception:
                pass

        paned = self.op_paned
        if paned is not None:
            try:
                paned.update_idletasks()
                width = int(paned.winfo_width())
                if width > 1:
                    sash = int(paned.sashpos(0))
                    self.op_pane_fraction = max(
                        0.05,
                        min(0.95, sash / width),
                    )
            except Exception:
                pass

        canvas = self.schematic_canvas
        if canvas is not None:
            try:
                self.op_canvas_view = (
                    float(canvas.xview()[0]),
                    float(canvas.yview()[0]),
                )
            except Exception:
                pass

    def _restore_operating_point_view_state(self, paned: object) -> None:
        """Restore saved OP geometry after Tk has laid out replacement widgets."""

        if paned is not self.op_paned:
            return
        notebook = self.op_detail_notebook
        selected_detail = self.op_detail_tabs.get(
            self.op_detail_selected_tab
        )
        if notebook is not None and selected_detail is not None:
            try:
                notebook.select(selected_detail)
            except Exception:
                pass
        try:
            paned.update_idletasks()
            width = int(paned.winfo_width())
            if self.op_pane_fraction is not None and width > 1:
                paned.sashpos(0, round(width * self.op_pane_fraction))
        except Exception:
            pass
        canvas = self.schematic_canvas
        if canvas is not None and self.op_canvas_view is not None:
            try:
                canvas.update_idletasks()
                canvas.xview_moveto(self.op_canvas_view[0])
                canvas.yview_moveto(self.op_canvas_view[1])
            except Exception:
                pass

    def _render_operating_point(
        self, results: Sequence[ChannelResult]
    ) -> None:
        """Render one selectable channel schematic plus OP/component detail panes."""

        labels = {
            f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)": result
            for result in results
        }
        if self.selected_op_channel_var.get() not in labels:
            self.selected_op_channel_var.set(next(iter(labels)))
        result = labels[self.selected_op_channel_var.get()]

        controls = self.ttk.Frame(self.op_frame, padding=(8, 6))
        controls.pack(fill="x")
        self.ttk.Label(controls, text="채널").pack(side="left")
        channel_combo = self.ttk.Combobox(
            controls,
            textvariable=self.selected_op_channel_var,
            values=tuple(labels),
            state="readonly",
            width=24,
        )
        channel_combo.pack(side="left", padx=(4, 16))
        channel_combo.bind(
            "<<ComboboxSelected>>",
            self._on_op_channel_changed,
        )
        self.ttk.Label(controls, text="회로도 확대").pack(side="left")
        self.ttk.Button(
            controls,
            text="−",
            width=3,
            command=lambda: self._change_schematic_zoom(1.0 / 1.2),
        ).pack(side="left", padx=(5, 2))
        zoom = self.ttk.Scale(
            controls,
            from_=0.5,
            to=2.5,
            variable=self.schematic_zoom_var,
            length=150,
        )
        zoom.pack(side="left", padx=2)
        zoom.bind(
            "<ButtonRelease-1>",
            lambda _event: self._redraw_current_schematic(),
        )
        self.ttk.Button(
            controls,
            text="+",
            width=3,
            command=lambda: self._change_schematic_zoom(1.2),
        ).pack(side="left", padx=(2, 4))
        self.ttk.Button(
            controls,
            text="100%",
            command=lambda: self._set_schematic_zoom(1.0),
        ).pack(side="left", padx=(0, 5))
        self.ttk.Label(
            controls,
            textvariable=self.schematic_zoom_var,
            width=5,
        ).pack(side="left")
        self.ttk.Label(
            controls,
            text=(
                "파랑: OP 노드 전압 [mV] · 초록: 적용 소자값 · "
                "휠: 확대/축소 · 좌클릭 드래그: 이동"
            ),
        ).pack(side="left", padx=(16, 0))

        paned = self.ttk.Panedwindow(self.op_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        drawing_host = self.ttk.Frame(paned)
        details_host = self.ttk.Frame(paned, width=330)
        paned.add(drawing_host, weight=4)
        paned.add(details_host, weight=1)
        self.op_paned = paned

        canvas = self.tk.Canvas(drawing_host, background="#ececec")
        x_scroll = self.ttk.Scrollbar(
            drawing_host, orient="horizontal", command=canvas.xview
        )
        y_scroll = self.ttk.Scrollbar(
            drawing_host, orient="vertical", command=canvas.yview
        )
        canvas.configure(
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        drawing_host.rowconfigure(0, weight=1)
        drawing_host.columnconfigure(0, weight=1)
        self.schematic_canvas = canvas
        self.schematic_result = result
        canvas.bind("<ButtonPress-1>", self._start_schematic_pan)
        canvas.bind("<B1-Motion>", self._drag_schematic)
        canvas.bind("<ButtonRelease-1>", self._stop_schematic_pan)
        canvas.bind("<MouseWheel>", self._zoom_schematic_with_wheel)
        canvas.bind("<Button-4>", self._zoom_schematic_with_wheel)
        canvas.bind("<Button-5>", self._zoom_schematic_with_wheel)
        self._draw_operating_point_schematic(canvas, result)
        self._build_op_detail_tables(details_host, result)
        self.root.after_idle(
            lambda current=paned: self._restore_operating_point_view_state(
                current
            )
        )

    @staticmethod
    def _start_schematic_pan(event: object) -> None:
        """Start Canvas scan-based panning at the pressed mouse position."""

        canvas = event.widget
        canvas.scan_mark(event.x, event.y)
        canvas.configure(cursor="fleur")

    @staticmethod
    def _drag_schematic(event: object) -> None:
        """Pan the enlarged schematic while the left mouse button is dragged."""

        event.widget.scan_dragto(event.x, event.y, gain=1)

    @staticmethod
    def _stop_schematic_pan(event: object) -> None:
        """Restore the normal pointer after schematic dragging ends."""

        event.widget.configure(cursor="")

    def _zoom_schematic_with_wheel(self, event: object) -> str:
        """Zoom around the mouse location for Windows, macOS, and Linux wheels."""

        button = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        zoom_in = button == 4 or delta > 0
        factor = 1.15 if zoom_in else 1.0 / 1.15
        self._set_schematic_zoom(
            float(self.schematic_zoom_var.get()) * factor,
            focus=(event.x, event.y),
        )
        return "break"

    def _change_schematic_zoom(self, factor: float) -> None:
        """Apply a multiplicative zoom step from the +/- buttons."""

        self._set_schematic_zoom(float(self.schematic_zoom_var.get()) * factor)

    def _redraw_current_schematic(self) -> None:
        """Apply the scale slider value to the existing schematic canvas."""

        self._set_schematic_zoom(float(self.schematic_zoom_var.get()))

    def _set_schematic_zoom(
        self,
        requested_scale: float,
        focus: tuple[float, float] | None = None,
    ) -> None:
        """Redraw at a bounded scale while preserving the focus point."""

        canvas = self.schematic_canvas
        result = self.schematic_result
        if canvas is None or result is None or not canvas.winfo_exists():
            return
        canvas.update_idletasks()
        focus_x, focus_y = focus or (
            max(1, canvas.winfo_width()) / 2.0,
            max(1, canvas.winfo_height()) / 2.0,
        )
        region = tuple(float(value) for value in canvas.cget("scrollregion").split())
        if len(region) == 4 and region[2] > region[0] and region[3] > region[1]:
            relative_x = (canvas.canvasx(focus_x) - region[0]) / (
                region[2] - region[0]
            )
            relative_y = (canvas.canvasy(focus_y) - region[1]) / (
                region[3] - region[1]
            )
        else:
            relative_x = relative_y = 0.5
        scale = max(0.5, min(2.5, requested_scale))
        self.schematic_zoom_var.set(round(scale, 3))
        self._draw_operating_point_schematic(canvas, result)
        canvas.update_idletasks()
        new_region = tuple(
            float(value) for value in canvas.cget("scrollregion").split()
        )
        if len(new_region) == 4:
            width = max(1.0, new_region[2] - new_region[0])
            height = max(1.0, new_region[3] - new_region[1])
            target_x = relative_x * width - focus_x
            target_y = relative_y * height - focus_y
            canvas.xview_moveto(max(0.0, min(1.0, target_x / width)))
            canvas.yview_moveto(max(0.0, min(1.0, target_y / height)))

    def _draw_operating_point_schematic(
        self, canvas: object, result: ChannelResult
    ) -> None:
        """Draw the cropped PDF image with simulated mV and component overlays."""

        try:
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "회로도를 표시하려면 Pillow가 필요합니다: py -m pip install Pillow"
            ) from exc
        if not DEFAULT_SCHEMATIC.is_file():
            raise FileNotFoundError(f"회로도 이미지를 찾을 수 없습니다: {DEFAULT_SCHEMATIC}")
        scale = max(0.5, min(2.5, float(self.schematic_zoom_var.get())))
        with Image.open(DEFAULT_SCHEMATIC) as source:
            width = max(1, round(source.width * scale))
            height = max(1, round(source.height * scale))
            image = source.convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
        photo = ImageTk.PhotoImage(image)
        self.schematic_photo = photo
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")

        node_positions = (
            ("/vin", "vin", 100, 195),
            ("Net-_V3-Pad1_", "V3+", 90, 304),
            ("Net-_U1-+_", "U1+", 342, 225),
            ("Net-_C2-Pad1_", "C2-L", 285, 346),
            ("Net-_U1--_", "U1-", 455, 345),
            ("Net-_U2-+_", "U2+", 675, 528),
            ("/v_filt_out", "VFILT", 675, 345),
            ("Net-_D1-A_", "D1-A", 855, 342),
            ("Net-_D1-K_", "D1-K", 1055, 365),
            ("Net-_U3-+_", "U3+", 840, 445),
            ("/v_detect_out", "VDET", 1120, 340),
            ("/v_ref", "VREF", 1248, 447),
            ("/v_comp_out", "VCOMP", 1460, 370),
            ("0.9V", "0.9V", 660, 658),
            ("1.8V", "1.8V", 565, 110),
            ("GND", "GND", 45, 754),
        )
        for node, display_name, x_pos, y_pos in node_positions:
            voltage = self._lookup_voltage(result.op_voltages, node)
            text = (
                f"{display_name} {format_voltage_mv(voltage, digits=7)}"
                if voltage is not None
                else f"{display_name} N/A"
            )
            self._canvas_boxed_text(
                canvas,
                x_pos * scale,
                y_pos * scale,
                text,
                "#0d47a1",
                "#e3f2fd",
                scale,
                font_size=8,
            )

        parameters = result.channel.spice_parameters()
        component_positions = (
            ("R1", 263, 181),
            ("RA", 371, 276),
            ("RA", 572, 401),
            ("R2", 786, 439),
            ("R2", 786, 577),
            ("R4", 835, 406),
            ("R5", 1027, 161),
            ("R6", 927, 488),
            ("R7", 1361, 300),
            ("R8", 1361, 509),
            ("C1", 386, 160),
            ("C1", 421, 320),
            ("C3", 1256, 481),
        )
        for parameter, x_pos, y_pos in component_positions:
            self._canvas_boxed_text(
                canvas,
                x_pos * scale,
                y_pos * scale,
                self._component_value(parameters[parameter]),
                "#1b5e20",
                "#e8f5e9",
                scale,
                anchor="center",
                font_size=11,
            )
        canvas.configure(scrollregion=(0, 0, width, height))

    @staticmethod
    def _canvas_boxed_text(
        canvas: object,
        x_pos: float,
        y_pos: float,
        text: str,
        foreground: str,
        background: str,
        scale: float,
        anchor: str = "sw",
        font_size: int = 10,
    ) -> None:
        """Draw legible foreground text on a color-matched opaque rectangle."""

        text_id = canvas.create_text(
            x_pos,
            y_pos,
            text=text,
            anchor=anchor,
            fill=foreground,
            font=("Segoe UI", max(8, round(font_size * scale)), "bold"),
        )
        bounds = canvas.bbox(text_id)
        if bounds:
            rectangle = canvas.create_rectangle(
                bounds[0] - 2,
                bounds[1] - 1,
                bounds[2] + 2,
                bounds[3] + 1,
                fill=background,
                outline=foreground,
                width=1,
            )
            canvas.tag_lower(rectangle, text_id)

    def _build_op_detail_tables(
        self, parent: object, result: ChannelResult
    ) -> None:
        """Build the all-node mV table and editable next-run component form."""

        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        node_tab = self.ttk.Frame(notebook)
        component_tab = self.ttk.Frame(notebook)
        notebook.add(node_tab, text=f"노드 전압 ({len(result.op_voltages)})")
        notebook.add(component_tab, text="적용 소자값 편집")
        self.op_detail_notebook = notebook
        self.op_detail_tabs = {
            "node": node_tab,
            "components": component_tab,
        }
        selected_detail = self.op_detail_tabs.get(
            self.op_detail_selected_tab,
            node_tab,
        )
        notebook.select(selected_detail)
        notebook.bind(
            "<<NotebookTabChanged>>",
            self._on_op_detail_tab_changed,
        )

        node_table = self.ttk.Treeview(
            node_tab,
            columns=("node", "voltage"),
            show="headings",
        )
        node_table.heading("node", text="노드")
        node_table.heading("voltage", text="전압 [mV]")
        node_table.column("node", width=210, stretch=True)
        node_table.column("voltage", width=115, anchor="e")
        node_scroll = self.ttk.Scrollbar(
            node_tab, orient="vertical", command=node_table.yview
        )
        node_table.configure(yscrollcommand=node_scroll.set)
        node_table.grid(row=0, column=0, sticky="nsew")
        node_scroll.grid(row=0, column=1, sticky="ns")
        node_tab.rowconfigure(0, weight=1)
        node_tab.columnconfigure(0, weight=1)
        for node, voltage in sorted(
            result.op_voltages.items(),
            key=lambda item: _canonical_vector_name(item[0]),
        ):
            node_table.insert(
                "", "end", values=(node, f"{voltage * 1000.0:.9g}")
            )

        applied = self._channel_by_number(result.channel.ch)
        self.component_editor_channel = applied.ch
        self.component_edit_vars = {}
        self.component_divider_edit_source = "resistors"
        self.component_divider_nominal_total_kohm = (
            applied.r7_kohm + applied.r8_kohm
        )
        self.component_vdet_v = self._lookup_voltage(
            result.op_voltages,
            "/v_detect_out",
        )
        editor = self.ttk.Frame(component_tab, padding=7)
        editor.pack(fill="both", expand=True)
        self.ttk.Label(
            editor,
            text=(
                "편집값은 메모리에만 적용되며 기본 CSV는 바뀌지 않습니다.\n"
                "메인 시뮬레이션 버튼을 누르면 현재 입력값을 자동 적용합니다."
            ),
            justify="left",
            foreground="#7a3e00",
            wraplength=310,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.ttk.Label(
            editor,
            text=(
                f"ch{applied.ch:02d} · 설계 fc={applied.f_c_hz:g} Hz · "
                f"설계 Q={applied.q:g}"
            ),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        for row, (key, label, attribute) in enumerate(
            COMPONENT_EDIT_FIELDS, start=2
        ):
            value = getattr(applied, attribute)
            variable = self.tk.StringVar(
                value=(
                    format_resistance_kohm(value)
                    if key in RESISTANCE_EDIT_KEYS
                    else f"{value:.12g}"
                )
            )
            self.component_edit_vars[key] = variable
            self.ttk.Label(editor, text=label).grid(
                row=row, column=0, sticky="e", padx=(0, 6), pady=2
            )
            entry = self.ttk.Entry(editor, textvariable=variable, width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if key in {"R7", "R8"}:
                entry.bind(
                    "<KeyRelease>",
                    lambda _event: self._preview_vref_from_resistors(),
                )
            if key in {"R4", "R5"}:
                variable.trace_add(
                    "write",
                    lambda *_args: self._preview_r6_from_r4_r5(),
                )
            entry.bind(
                "<Return>",
                lambda _event: self._apply_component_editor_from_gui(),
            )
        threshold_row = 2 + len(COMPONENT_EDIT_FIELDS)
        self.ttk.Label(editor, text="목표 V_margin [mV]").grid(
            row=threshold_row, column=0, sticky="e", padx=(0, 6), pady=(4, 2)
        )
        margin_entry = self.ttk.Entry(
            editor,
            textvariable=self.component_margin_var,
            width=16,
        )
        margin_entry.grid(
            row=threshold_row,
            column=1,
            sticky="ew",
            pady=(4, 2),
        )
        margin_entry.bind(
            "<KeyRelease>",
            lambda _event: self._preview_resistors_from_margin(),
        )
        margin_entry.bind(
            "<Return>",
            lambda _event: self._apply_component_editor_from_gui(),
        )
        simulated_vref = self._lookup_voltage(result.op_voltages, "/v_ref")
        initial_vref = (
            simulated_vref
            if simulated_vref is not None
            else applied.vthr_v
        )
        if self.component_vdet_v is not None:
            initial_margin_v = voltage_margin_v(
                self.component_vdet_v,
                initial_vref,
            )
            self.component_margin_var.set(
                f"{initial_margin_v * 1000.0:.12g}"
            )
            self.component_actual_margin_var.set(
                format_margin_mv(initial_margin_v)
            )
        else:
            self.component_margin_var.set("")
            self.component_actual_margin_var.set("—")

        self.ttk.Label(editor, text="계산 Vref [mV]").grid(
            row=threshold_row + 1,
            column=0,
            sticky="e",
            padx=(0, 6),
            pady=2,
        )
        vref_entry = self.ttk.Entry(
            editor,
            textvariable=self.component_vref_var,
            width=16,
            state="readonly",
        )
        vref_entry.grid(
            row=threshold_row + 1,
            column=1,
            sticky="ew",
            pady=2,
        )
        visible_vref_v = divider_vref_v(
            float(self.component_edit_vars["R7"].get()),
            float(self.component_edit_vars["R8"].get()),
        )
        self.component_vref_var.set(f"{visible_vref_v * 1000.0:.12g}")
        self.ttk.Label(
            editor,
            text=(
                "Vmargin = Vref − Vdet, Vref = 최근 DC Vdet + 목표 Vmargin · "
                "모든 저항은 0.01 kΩ로 반올림 · "
                "R7+R8 합계 고정 안 함 · "
                "R4/R5 변경 시 R6=R4||R5 자동 입력"
            ),
            foreground="#164a7b",
            justify="left",
            wraplength=310,
        ).grid(
            row=threshold_row + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 3),
        )

        self.ttk.Label(
            editor,
            text="최근 DC 실제 V_margin (Vref−Vdet)",
        ).grid(
            row=threshold_row + 3,
            column=0,
            sticky="e",
            padx=(0, 6),
            pady=2,
        )
        self.ttk.Label(
            editor,
            textvariable=self.component_actual_margin_var,
        ).grid(row=threshold_row + 3, column=1, sticky="w", pady=2)

        buttons = self.ttk.Frame(editor)
        buttons.grid(
            row=threshold_row + 4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(7, 4),
        )
        self.ttk.Button(
            buttons,
            text="변경값 적용 + DC 실행",
            command=self._apply_component_editor_from_gui,
        ).pack(side="left")
        self.ttk.Button(
            buttons,
            text="새 버전 저장",
            command=self._save_component_version_from_gui,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            buttons,
            text="버전 불러오기",
            command=self._browse_table,
        ).pack(side="left", padx=(5, 0))

        models = self.ttk.LabelFrame(editor, text="고정 소자 / 모델", padding=5)
        models.grid(
            row=threshold_row + 5,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        fixed_rows = (
            ("V1", "DC 0.9 V"),
            ("V2", "DC 0.9 V"),
            ("V3", "DC 0.9 V"),
            ("V4", "DC 0 V / AC 1 V"),
            ("U1, U2, U3", "OPA379"),
            ("U4", "LPV7215"),
            ("D1, D2", "BAT54W"),
        )
        for row, (reference, value) in enumerate(fixed_rows):
            self.ttk.Label(models, text=reference, width=12).grid(
                row=row, column=0, sticky="w"
            )
            self.ttk.Label(models, text=value).grid(
                row=row, column=1, sticky="w"
            )
        editor.columnconfigure(1, weight=1)

    def _channel_by_number(self, channel_number: int) -> Channel:
        """Return the current in-memory channel used by the next simulation."""

        for channel in self.channels:
            if channel.ch == channel_number:
                return channel
        raise ValueError(f"적용 소자값에 ch{channel_number:02d}가 없습니다.")

    def _on_op_detail_tab_changed(self, event: object = None) -> None:
        """Remember whether the user is viewing node voltages or component edits."""

        event_notebook = getattr(event, "widget", None)
        notebook = event_notebook or self.op_detail_notebook
        if notebook is not self.op_detail_notebook:
            return
        if notebook is None:
            return
        try:
            selected = notebook.select()
        except Exception:
            return
        for name, tab in self.op_detail_tabs.items():
            if str(tab) == selected:
                self.op_detail_selected_tab = name
                return

    def _preview_r6_from_r4_r5(self) -> None:
        """Auto-fill R6=R4||R5 only when R4 or R5 itself changes."""

        if self.component_auto_update_suspended:
            return
        try:
            r4_kohm = float(
                str(self.component_edit_vars["R4"].get()).strip()
            )
            r5_kohm = float(
                str(self.component_edit_vars["R5"].get()).strip()
            )
            r6_kohm = parallel_resistance_kohm(r4_kohm, r5_kohm)
        except (KeyError, TypeError, ValueError):
            return
        self.component_edit_vars["R6"].set(
            format_resistance_kohm(r6_kohm)
        )

    def _format_component_resistance_fields(self, channel: Channel) -> None:
        """Normalize all visible resistor fields without retriggering R6."""

        self.component_auto_update_suspended = True
        try:
            for key, _label, attribute in COMPONENT_EDIT_FIELDS:
                if key not in RESISTANCE_EDIT_KEYS:
                    continue
                variable = self.component_edit_vars.get(key)
                if variable is not None:
                    variable.set(
                        format_resistance_kohm(
                            getattr(channel, attribute)
                        )
                    )
        finally:
            self.component_auto_update_suspended = False

    def _preview_vref_from_resistors(self) -> None:
        """Update read-only Vref and target margin after R7/R8 is typed."""

        self.component_divider_edit_source = "resistors"
        try:
            r7 = float(str(self.component_edit_vars["R7"].get()).strip())
            r8 = float(str(self.component_edit_vars["R8"].get()).strip())
            vref_v = divider_vref_v(r7, r8)
        except (KeyError, TypeError, ValueError):
            return
        self.component_divider_nominal_total_kohm = r7 + r8
        self.component_vref_var.set(f"{vref_v * 1000.0:.12g}")
        if self.component_vdet_v is not None:
            margin_v = voltage_margin_v(self.component_vdet_v, vref_v)
            self.component_margin_var.set(f"{margin_v * 1000.0:.12g}")

    def _preview_resistors_from_margin(self) -> None:
        """Derive Vref and a 0.01 kΩ divider from the editable target margin."""

        self.component_divider_edit_source = "margin"
        try:
            if self.component_vdet_v is None:
                return
            margin_v = float(self.component_margin_var.get().strip()) / 1000.0
            vref_v = vref_from_margin_v(self.component_vdet_v, margin_v)
            r7_kohm, r8_kohm = divider_resistors_for_vref(
                vref_v,
                nominal_total_kohm=(
                    self.component_divider_nominal_total_kohm
                ),
            )
            self.component_edit_vars["R7"].set(
                format_resistance_kohm(r7_kohm)
            )
            self.component_edit_vars["R8"].set(
                format_resistance_kohm(r8_kohm)
            )
            rounded_vref_v = divider_vref_v(r7_kohm, r8_kohm)
            self.component_vref_var.set(
                f"{rounded_vref_v * 1000.0:.12g}"
            )
        except (KeyError, TypeError, ValueError):
            return

    def _component_editor_has_changes(self) -> bool:
        """Detect unsaved text edits without requiring the text to be valid."""

        if self.component_editor_channel is None or not self.component_edit_vars:
            return False
        channel = self._channel_by_number(self.component_editor_channel)
        for key, _label, attribute in COMPONENT_EDIT_FIELDS:
            variable = self.component_edit_vars.get(key)
            if variable is None:
                return True
            try:
                value = float(str(variable.get()).strip())
            except ValueError:
                return True
            expected = float(getattr(channel, attribute))
            if key in RESISTANCE_EDIT_KEYS:
                expected = round_resistance_kohm(expected)
            if not math.isclose(
                value,
                expected,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                return True
        if self.component_divider_edit_source == "margin":
            try:
                float(self.component_margin_var.get().strip())
            except ValueError:
                return True
        return False

    def _commit_component_editor(self) -> None:
        """Validate the visible component form and update its in-memory channel."""

        if self.component_editor_channel is None or not self.component_edit_vars:
            return
        original = self._channel_by_number(self.component_editor_channel)
        edit_source = self.component_divider_edit_source
        values = {
            key: str(variable.get())
            for key, variable in self.component_edit_vars.items()
        }
        requested_margin_mv = (
            self.component_margin_var.get()
            if edit_source == "margin"
            else None
        )
        updated = apply_component_updates(
            original,
            values,
            requested_margin_mv=requested_margin_mv,
            reference_vdet_v=self.component_vdet_v,
            divider_nominal_total_kohm=(
                self.component_divider_nominal_total_kohm
                if requested_margin_mv is not None
                else None
            ),
        )
        self._format_component_resistance_fields(updated)
        self.component_vref_var.set(f"{updated.vthr_v * 1000.0:.12g}")
        if edit_source == "resistors":
            self.component_divider_nominal_total_kohm = (
                updated.r7_kohm + updated.r8_kohm
            )
        self.component_divider_edit_source = "resistors"
        if updated == original:
            return
        self.channels = [
            updated if channel.ch == updated.ch else channel
            for channel in self.channels
        ]
        self.component_dirty = True
        base_label = (
            "기본값 (읽기 전용)"
            if self.component_source_path == DEFAULT_TABLE.resolve()
            else self.component_source_path.name
        )
        self.component_source_label_var.set(base_label + " / 메모리 수정")
        self._refresh_transient_component_state()

    def _apply_component_editor_from_gui(self) -> None:
        """Apply edited values and immediately rerun DC OP for that channel."""

        from tkinter import messagebox

        try:
            self._capture_operating_point_view_state()
            self.op_pending_view_state = (
                self.op_detail_selected_tab,
                self.op_pane_fraction,
                self.op_canvas_view,
            )
            channel_number = self.component_editor_channel
            target_margin_v = None
            if self.component_divider_edit_source == "margin":
                try:
                    target_margin_v = (
                        float(self.component_margin_var.get().strip())
                        / 1000.0
                    )
                except ValueError as exc:
                    raise ValueError(
                        "목표 Vmargin: mV 단위 숫자를 입력하세요."
                    ) from exc
            self._commit_component_editor()
            if channel_number is not None:
                updated = self._channel_by_number(channel_number)
                self.pending_margin_retune = (
                    (channel_number, target_margin_v)
                    if target_margin_v is not None
                    else None
                )
                self.status_var.set(
                    f"ch{channel_number:02d} 변경값 적용됨 / "
                    f"Vref={format_voltage_mv(updated.vthr_v)} / "
                    "DC 동작점 자동 실행"
                )
                self._start_run(
                    ("op",),
                    selected_channels=(updated,),
                )
        except Exception as exc:
            self.op_pending_view_state = None
            messagebox.showerror("소자값 입력 오류", str(exc))

    def _prepare_margin_retune(
        self,
        results: Sequence[ChannelResult],
    ) -> Channel | None:
        """Use the first new OP Vdet to refine a requested target Vmargin once."""

        pending = self.pending_margin_retune
        if pending is None:
            return None
        self.pending_margin_retune = None
        channel_number, target_margin_v = pending
        result = next(
            (
                item
                for item in results
                if item.channel.ch == channel_number and item.op_voltages
            ),
            None,
        )
        if result is None:
            return None
        vdet_v = self._lookup_voltage(
            result.op_voltages,
            "/v_detect_out",
        )
        if vdet_v is None:
            return None
        current = self._channel_by_number(channel_number)
        target_vref_v = vref_from_margin_v(vdet_v, target_margin_v)
        r7_kohm, r8_kohm = divider_resistors_for_vref(
            target_vref_v,
            nominal_total_kohm=(
                self.component_divider_nominal_total_kohm
                if self.component_editor_channel == channel_number
                else current.r7_kohm + current.r8_kohm
            ),
        )
        rounded_vref_v = divider_vref_v(r7_kohm, r8_kohm)
        updated = replace(
            current,
            r7_kohm=r7_kohm,
            r8_kohm=r8_kohm,
            vthr_v=rounded_vref_v,
        )
        if updated == current:
            return None
        self.channels = [
            updated if channel.ch == channel_number else channel
            for channel in self.channels
        ]
        self.component_dirty = True
        self._refresh_transient_component_state()
        if (
            self.component_editor_channel == channel_number
            and self.component_edit_vars
        ):
            self._format_component_resistance_fields(updated)
            self.component_vref_var.set(
                f"{rounded_vref_v * 1000.0:.12g}"
            )
            self.component_divider_edit_source = "resistors"
        return updated

    def _on_op_channel_changed(self, _event: object) -> None:
        """Commit the previous channel before switching the schematic/editor."""

        from tkinter import messagebox

        previous = self.component_editor_channel
        try:
            self._commit_component_editor()
        except Exception as exc:
            if previous is not None:
                for result in self.last_op_results:
                    if result.channel.ch == previous:
                        self.selected_op_channel_var.set(
                            f"ch{previous:02d} ({result.channel.f_c_hz:g} Hz)"
                        )
                        break
            messagebox.showerror("소자값 입력 오류", str(exc))
            return
        self._refresh_operating_point_tab()

    @staticmethod
    def _lookup_voltage(
        voltages: dict[str, float], requested_node: str
    ) -> float | None:
        """Find a voltage despite KiCad slash and ngspice ``v()`` spelling."""

        wanted = _canonical_vector_name(requested_node)
        if wanted in {"0", "gnd"}:
            return 0.0
        for node, voltage in voltages.items():
            if _canonical_vector_name(node) == wanted:
                return voltage
        return None

    @staticmethod
    def _component_value(spice_text: str) -> str:
        """Convert ngspice k/n suffixes into human-readable schematic units."""

        if spice_text.endswith("k"):
            return spice_text[:-1] + " kΩ"
        if spice_text.endswith("n"):
            return spice_text[:-1] + " nF"
        return spice_text

    def _component_rows(self, channel: Channel) -> tuple[tuple[str, str], ...]:
        """Return the full applied component/model summary for one channel."""

        p = channel.spice_parameters()
        return (
            ("R1", self._component_value(p["R1"])),
            ("RA1, RA2", self._component_value(p["RA"])),
            ("R2, R3", self._component_value(p["R2"])),
            ("R4", self._component_value(p["R4"])),
            ("R5", self._component_value(p["R5"])),
            ("R6", self._component_value(p["R6"])),
            ("R7", self._component_value(p["R7"])),
            ("R8", self._component_value(p["R8"])),
            ("C1, C2", self._component_value(p["C1"])),
            ("C3", self._component_value(p["C3"])),
            ("V1", "DC 0.9 V"),
            ("V2", "DC 0.9 V"),
            ("V3", "DC 0.9 V"),
            ("V4", "DC 0 V / AC 1 V"),
            ("U1, U2, U3", "OPA379"),
            ("U4", "LPV7215"),
            ("D1, D2", "BAT54W"),
        )

    @staticmethod
    def _format_metric(value: float | None, digits: int = 6) -> str:
        """Format an optional finite metric or return an em dash."""

        if value is None or not math.isfinite(value):
            return "—"
        return f"{value:.{digits}g}"

    def _open_results(self) -> None:
        """Open the most recent session folder in the platform file manager."""

        from tkinter import messagebox

        target = (
            self.last_results[0].run_dir.parent
            if self.last_results
            else Path(self.output_var.get()).expanduser()
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))


def build_parser() -> argparse.ArgumentParser:
    """Create CLI options for validation, headless execution, or the GUI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--netlist", type=Path, default=DEFAULT_NETLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ngspice", default="")
    parser.add_argument("--channels", default="all", help="예: 0,3,7 또는 all")
    parser.add_argument(
        "--analysis",
        choices=("ac", "op", "both"),
        default="both",
        help="실행할 해석 종류: ac, op 또는 both (기본: both)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested CLI mode and translate failures into exit code 1."""

    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            validate_generation(args.table, args.netlist, args.output_dir)
            return 0
        if args.headless:
            channels = load_channels(args.table)
            selected = parse_channel_selection(args.channels, channels)
            simulator = Simulator()
            results = simulator.run(
                selected,
                args.netlist,
                args.output_dir,
                find_ngspice(args.ngspice),
                SweepSettings(),
                args.analysis,
            )
            return 0 if len(results) == len(selected) else 2
        import tkinter as tk

        root = tk.Tk()
        SweeperGUI(root)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
