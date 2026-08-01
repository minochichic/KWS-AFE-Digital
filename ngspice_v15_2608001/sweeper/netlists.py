"""AC, operating-point, and transient netlist generation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_DETECTOR_OPAMP,
    DETECTOR_OPAMP_LIBRARIES,
    DETECTOR_OPAMP_NAMES,
    DETECTOR_OPAMP_REFERENCE,
    MODEL_LIBRARY_DIR,
    PARAM_NAMES,
)
from .models import (
    ANALYSIS_JOB_SPECS,
    AnalysisJob,
    Channel,
    DetectorSettings,
    SweepSettings,
    TransientSettings,
    normalize_analyses,
)


def make_parameter_include(channel: Channel) -> str:
    """Render the per-channel ``.param`` include consumed by every analysis."""

    p = channel.spice_parameters()
    return "\n".join(
        [
            f"* Auto-generated parameters for channel {channel.ch}",
            f"* fc={channel.f_c_hz:.12g}Hz Q={channel.q:.12g} "
            f"gain={channel.gain_db:.12g}dB Vthr={channel.vthr_v:.6f}V",
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
    detector_opamp: str = DEFAULT_DETECTOR_OPAMP,
) -> str:
    """Build one standalone AC or OP netlist from the shared circuit template."""

    analysis = analysis.lower()
    if analysis == "dc":
        analysis = "op"
    if analysis not in {"ac", "op"}:
        raise ValueError(f"지원하지 않는 해석 종류: {analysis!r}")
    settings.validate((analysis,))
    lines = strip_generated_sections(template_text.splitlines())
    lines, extra_includes = apply_detector_opamp(lines, detector_opamp)
    insert_at = 1
    for index, line in enumerate(lines):
        if re.match(r"^\s*\.include\b", line, re.I):
            insert_at = index + 1
    for offset, statement in enumerate(
        [*extra_includes, f'.include "{spice_path(include_path)}"']
    ):
        lines.insert(insert_at + offset, statement)
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
        settings.vthr_node,
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
    detector_opamp: str = DEFAULT_DETECTOR_OPAMP,
) -> str:
    """Build a transient-only netlist that replaces the existing input source."""

    settings.validate()
    source_name, _positive_node, _negative_node = find_input_voltage_source(
        template_text, settings.vin_node
    )
    lines = strip_generated_sections(template_text.splitlines())
    lines = remove_input_voltage_source(lines, source_name)
    lines, extra_includes = apply_detector_opamp(lines, detector_opamp)
    insert_at = 1
    for index, line in enumerate(lines):
        if re.match(r"^\s*\.include\b", line, re.I):
            insert_at = index + 1
    for offset, statement in enumerate(
        [
            *extra_includes,
            f'.include "{spice_path(parameter_include_path)}"',
            f'.include "{spice_path(stimulus_include_path)}"',
        ]
    ):
        lines.insert(insert_at + offset, statement)
    lines.append("")
    lines.extend(transient_directives(settings, stop_time_s))
    lines.extend([".end", ""])
    return "\n".join(lines)


def detector_opamp_library_path(model: str) -> Path:
    """Return the .lib file that defines one selectable detector op-amp."""

    for name in DETECTOR_OPAMP_NAMES:
        if model.casefold() == name.casefold():
            return MODEL_LIBRARY_DIR / DETECTOR_OPAMP_LIBRARIES[name]
    raise ValueError(
        f"지원하지 않는 detector opamp: {model!r}. "
        "사용 가능: " + ", ".join(DETECTOR_OPAMP_NAMES)
    )


def replace_detector_opamp(lines: Sequence[str], model: str) -> list[str]:
    """Swap only the model name on the detector op-amp's subcircuit call.

    All three supported models declare the same terminal order
    (IN+, IN-, V+, V-, OUT), verified against their .lib files, so the node
    list is left exactly as the template defines it.
    """

    reference = DETECTOR_OPAMP_REFERENCE.casefold()
    updated: list[str] = []
    replaced = False
    for line in lines:
        parts = line.split()
        if parts and parts[0].casefold() == reference:
            if len(parts) < 3:
                raise ValueError(
                    f"{DETECTOR_OPAMP_REFERENCE} 문장이 불완전합니다: {line!r}"
                )
            parts[-1] = model
            updated.append(" ".join(parts))
            replaced = True
            continue
        updated.append(line)
    if not replaced:
        raise ValueError(
            f"네트리스트에서 {DETECTOR_OPAMP_REFERENCE} 문장을 찾지 못했습니다."
        )
    return updated


def apply_detector_opamp(
    lines: Sequence[str],
    detector_opamp: str,
) -> tuple[list[str], list[str]]:
    """Swap the detector op-amp and report any library include it needs.

    Every analysis shares one circuit, so AC, OP, Transient, and Detector must
    all honour the selected model or a comparison would silently keep OPA379.
    """

    library_path = detector_opamp_library_path(detector_opamp)
    is_default = detector_opamp.casefold() == DEFAULT_DETECTOR_OPAMP.casefold()
    if is_default and not any(
        line.split()[:1] == [DETECTOR_OPAMP_REFERENCE]
        or (
            line.split()
            and line.split()[0].casefold()
            == DETECTOR_OPAMP_REFERENCE.casefold()
        )
        for line in lines
    ):
        # A reduced template without a detector stage needs no substitution,
        # and the default model is what the template already declares.
        return list(lines), []
    updated = replace_detector_opamp(lines, detector_opamp)
    if is_default:
        # The template already includes the default model's library.
        return updated, []
    return updated, [f'.include "{spice_path(library_path)}"']


def make_gated_sine_pairs(
    settings: DetectorSettings,
) -> list[tuple[float, float]]:
    """Sample an exact-amplitude gated sine for a PWL voltage source.

    The interval count per period is rounded upward to a multiple of four.
    Therefore the PWL data contains both positive and negative sine peaks while
    no interval exceeds the requested maximum timestep.  This keeps the source
    Vpp equal to ``settings.input_vpp_v`` at the verified ``/vin`` terminals.
    """

    settings.validate()
    period_s = 1.0 / settings.frequency_hz
    intervals_per_cycle = max(
        4,
        math.ceil(period_s / settings.maximum_step_s),
    )
    intervals_per_cycle = 4 * math.ceil(intervals_per_cycle / 4)
    sample_step_s = period_s / intervals_per_cycle
    amplitude_v = settings.input_vpp_v / 2.0
    pairs: list[tuple[float, float]] = [(0.0, 0.0)]
    if settings.gate_on_s > 0.0:
        pairs.append((settings.gate_on_s, 0.0))
    sample_count = math.floor(
        settings.gate_duration_s / sample_step_s + 1e-12
    )
    for sample_index in range(1, sample_count + 1):
        relative_s = sample_index * sample_step_s
        absolute_s = settings.gate_on_s + relative_s
        if absolute_s >= settings.gate_off_s:
            break
        voltage_v = amplitude_v * math.sin(
            2.0 * math.pi * settings.frequency_hz * relative_s
        )
        pairs.append((absolute_s, voltage_v))
    if pairs[-1][0] == settings.gate_off_s:
        pairs[-1] = (settings.gate_off_s, 0.0)
    else:
        pairs.append((settings.gate_off_s, 0.0))
    pairs.append((settings.total_time_s, 0.0))
    return pairs


def make_gated_sine_source_include(
    source_name: str,
    positive_node: str,
    negative_node: str,
    settings: DetectorSettings,
) -> str:
    """Render the gated sine while preserving the original V-source terminals."""

    pairs = make_gated_sine_pairs(settings)
    lines = [
        "* Auto-generated Active Detector gated-sine stimulus",
        f"* frequency_hz={settings.frequency_hz:.12g}",
        f"* input_vpp_v={settings.input_vpp_v:.12g}",
        f"* gate_on_s={settings.gate_on_s:.12g}",
        f"* gate_off_s={settings.gate_off_s:.12g}",
        f"{source_name} {positive_node} {negative_node} ",
    ]
    first_time, first_voltage = pairs[0]
    lines[-1] += f"PWL({first_time:.12g} {first_voltage:.12g}"
    for time_s, voltage_v in pairs[1:-1]:
        lines.append(f"+ {time_s:.12g} {voltage_v:.12g}")
    last_time, last_voltage = pairs[-1]
    lines.append(f"+ {last_time:.12g} {last_voltage:.12g})")
    lines.append("")
    return "\n".join(lines)


def detector_directives(settings: DetectorSettings) -> list[str]:
    """Return saved-node and transient statements for a detector-only job."""

    settings.validate()
    nodes = (
        settings.vin_node,
        settings.vfilt_node,
        settings.venv_node,
        settings.vthr_node,
    )
    return [
        ".save " + " ".join(f"v({node})" for node in nodes),
        (
            f".tran {settings.maximum_step_s:.12g} "
            f"{settings.total_time_s:.12g} 0 "
            f"{settings.maximum_step_s:.12g}"
        ),
    ]


def make_detector_netlist(
    template_text: str,
    parameter_include_path: Path,
    stimulus_include_path: Path,
    settings: DetectorSettings,
    detector_opamp: str = DEFAULT_DETECTOR_OPAMP,
) -> str:
    """Build a full-channel detector netlist with gated sine applied at /vin.

    ``detector_opamp`` replaces only the XU3 model name and adds that model's
    library.  Selecting the default reproduces the template's original circuit.
    """

    settings.validate()
    source_name, _positive_node, _negative_node = find_input_voltage_source(
        template_text,
        settings.vin_node,
    )
    lines = strip_generated_sections(template_text.splitlines())
    lines = remove_input_voltage_source(lines, source_name)
    lines, extra_includes = apply_detector_opamp(lines, detector_opamp)
    insert_at = 1
    for index, line in enumerate(lines):
        if re.match(r"^\s*\.include\b", line, re.I):
            insert_at = index + 1
    for offset, statement in enumerate(
        [
            *extra_includes,
            f'.include "{spice_path(parameter_include_path)}"',
            f'.include "{spice_path(stimulus_include_path)}"',
        ]
    ):
        lines.insert(insert_at + offset, statement)
    lines.append("")
    lines.extend(detector_directives(settings))
    lines.extend([".end", ""])
    return "\n".join(lines)
