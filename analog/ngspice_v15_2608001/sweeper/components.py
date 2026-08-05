"""Component CSV protection, versioning, and divider calculations."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping, Sequence

from .constants import (
    CHANNEL_CSV_COLUMNS,
    COMPONENT_EDIT_FIELDS,
    DEFAULT_COMMON,
    DEFAULT_DETECTOR_OPAMP,
    DETECTOR_OPAMP_NAMES,
    DEFAULT_COMPONENT_VERSIONS,
    DEFAULT_TABLE,
    DIVIDER_NOMINAL_TOTAL_KOHM,
    DIVIDER_RESISTANCE_DECIMALS,
    DIVIDER_SUPPLY_V,
    REQUIRED_COLUMNS,
    RESISTANCE_EDIT_KEYS,
)
from .models import Channel
from .values import (
    format_resistance_kohm,
    round_resistance_kohm,
)


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
    """Calculate Vthr for the R7-top/R8-bottom divider in the template.

    The netlist connects R7 from 1.8 V to ``/v_thr`` and R8 from ``/v_thr`` to
    ground, so the unloaded divider equation is
    ``Vthr = supply * R8 / (R7 + R8)``.  The historic public function name is
    retained for source compatibility.
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
    """Return independently rounded R7/R8 values for a requested Vthr.

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
            "Vthr, 기준 총 저항, 분압 전원, 소수 자릿수는 유한해야 합니다."
        )
    if nominal_total_kohm <= 0.0 or supply_v <= 0.0:
        raise ValueError("기준 총 저항과 분압 전원은 0보다 커야 합니다.")
    if isinstance(decimals, bool) or int(decimals) != decimals or decimals < 0:
        raise ValueError("저항 소수 자릿수는 0 이상의 정수여야 합니다.")
    if not 0.0 < vref_v < supply_v:
        raise ValueError(
            f"계산 Vthr는 0보다 크고 {supply_v * 1000:g} mV보다 "
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
            "요청한 Vthr는 지정한 저항 분해능에서 양수 R7/R8로 만들 수 없습니다."
        )
    return r7_kohm, r8_kohm

def vref_from_margin_v(
    vdet_v: float,
    margin_v: float,
    supply_v: float = DIVIDER_SUPPLY_V,
) -> float:
    """Derive Vthr from ``Vmargin = Vthr - Venv_DC`` and validate rails.

    A positive margin is the detector's remaining headroom to the comparator
    threshold at the DC operating point.  The detector must rise by this
    amount before ``Venv > Vthr`` becomes true.  The historic public function
    name and argument names are retained for source compatibility.
    """

    values = (float(vdet_v), float(margin_v), float(supply_v))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Venv_DC, Vmargin, 분압 전원은 유한한 값이어야 합니다.")
    if supply_v <= 0.0:
        raise ValueError("분압 전원은 0보다 커야 합니다.")
    if margin_v <= 0.0:
        raise ValueError(
            "목표 Vmargin은 Vthr가 Venv_DC보다 높도록 0보다 커야 합니다."
        )
    vref_v = vdet_v + margin_v
    if not 0.0 < vref_v < supply_v:
        raise ValueError(
            "Vthr = Venv_DC + Vmargin 결과는 "
            f"0보다 크고 {supply_v * 1000:g} mV보다 작아야 합니다."
        )
    return vref_v

def parse_detector_opamp(row: dict[str, str]) -> str:
    """Read the optional detector op-amp column and validate the model name.

    The column is optional so the read-only factory CSV, which predates model
    selection, still loads unchanged.
    """

    raw = (row.get("detector_opamp") or "").strip()
    if not raw:
        return DEFAULT_DETECTOR_OPAMP
    for name in DETECTOR_OPAMP_NAMES:
        if raw.casefold() == name.casefold():
            return name
    raise ValueError(
        f"detector_opamp {raw!r}는 지원 목록에 없습니다: "
        + ", ".join(DETECTOR_OPAMP_NAMES)
    )

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
                    detector_opamp=parse_detector_opamp(row),
                )
                if abs(channel.vthr_v - divider) > 0.001:
                    raise ValueError(
                        f"ch{ch}: CSV Vthr={channel.vthr_v:.4f} V와 "
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
        "detector_opamp": channel.detector_opamp,
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

def divider_resistors_for_requested_vthr(
    channel: Channel,
    vthr_v: float,
    divider_nominal_total_kohm: float | None = None,
) -> tuple[float, float]:
    """Convert a requested Vthr into an independently rounded R7/R8 pair."""

    nominal_total_kohm = (
        channel.r7_kohm + channel.r8_kohm
        if divider_nominal_total_kohm is None
        else float(divider_nominal_total_kohm)
    )
    return divider_resistors_for_vref(
        vthr_v,
        nominal_total_kohm=nominal_total_kohm,
    )


def apply_component_updates(
    channel: Channel,
    values: Mapping[str, str],
    requested_margin_mv: str | None = None,
    reference_vdet_v: float | None = None,
    divider_nominal_total_kohm: float | None = None,
    requested_vthr_mv: str | None = None,
    detector_opamp: str | None = None,
) -> Channel:
    """Validate editor text and return a new immutable channel instance.

    Normally R7/R8 determine ``vthr_v``.  Vthr and Vmargin are two views of the
    same divider: setting either one derives the other through
    ``Vmargin = Vthr - Venv_DC``.  Whichever the user edited last is converted
    into an independently rounded 0.01 kΩ R7/R8 pair.  The explicitly supplied
    nominal divider total, or the channel's previous total when omitted, keeps
    the loading scale stable without forcing the rounded pair to an exact sum.
    """

    if requested_margin_mv is not None and requested_vthr_mv is not None:
        raise ValueError(
            "Vthr와 목표 Vmargin 중 하나만 기준으로 지정할 수 있습니다."
        )
    updates: dict[str, object] = {}
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
                "Vmargin으로 Vthr를 계산하려면 최근 DC Venv_DC 결과가 필요합니다."
            )
        try:
            margin_v = float(requested_margin_mv.strip()) / 1000.0
        except ValueError as exc:
            raise ValueError("목표 Vmargin: mV 단위 숫자를 입력하세요.") from exc
        vref_v = vref_from_margin_v(reference_vdet_v, margin_v)
        r7_kohm, r8_kohm = divider_resistors_for_requested_vthr(
            channel, vref_v, divider_nominal_total_kohm
        )
        updates["r7_kohm"] = r7_kohm
        updates["r8_kohm"] = r8_kohm
    if requested_vthr_mv is not None:
        try:
            vthr_v = float(requested_vthr_mv.strip()) / 1000.0
        except ValueError as exc:
            raise ValueError("Vthr: mV 단위 숫자를 입력하세요.") from exc
        if not math.isfinite(vthr_v):
            raise ValueError("Vthr는 유한한 값이어야 합니다.")
        r7_kohm, r8_kohm = divider_resistors_for_requested_vthr(
            channel, vthr_v, divider_nominal_total_kohm
        )
        updates["r7_kohm"] = r7_kohm
        updates["r8_kohm"] = r8_kohm
    if detector_opamp is not None:
        name = detector_opamp.strip()
        if name not in DETECTOR_OPAMP_NAMES:
            raise ValueError(
                "U3 opamp는 지원 목록의 모델이어야 합니다: "
                + ", ".join(DETECTOR_OPAMP_NAMES)
            )
        updates["detector_opamp"] = name
    updated = replace(channel, **updates)
    threshold = divider_vref_v(updated.r7_kohm, updated.r8_kohm)
    return replace(updated, vthr_v=threshold)
