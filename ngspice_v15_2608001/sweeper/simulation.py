"""Process runners for AC/OP and PWL-driven transient simulations."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from ngspice_runner import resolve_ngspice, run_ngspice
from wav_pwl import read_pwl_data

from .components import load_channels, write_channels_csv
from .constants import DETECTOR_RAW_FILENAME, TRAN_RAW_FILENAME
from .models import (
    AcMetrics,
    Channel,
    ChannelResult,
    DetectorResult,
    DetectorSettings,
    SweepSettings,
    TransientResult,
    TransientSettings,
    normalize_analyses,
)
from .netlists import (
    build_analysis_jobs,
    find_input_voltage_source,
    make_detector_netlist,
    make_gated_sine_source_include,
    make_analysis_netlist,
    make_parameter_include,
    make_pwl_source_include,
    make_transient_netlist,
)
from .results import (
    _canonical_vector_name,
    calculate_ac_metrics,
    read_detector_raw,
    read_ac_raw,
    read_op_raw,
    read_transient_raw,
    write_ac_metrics_summary,
    write_normalized_csv,
)
from .values import db_to_linear, scale_pwl_pairs_to_vpp


def find_ngspice(configured: str = "") -> str:
    """Backward-compatible wrapper around the standalone launcher."""
    return resolve_ngspice(configured)


def _csv_node_prefix(node: str) -> str:
    """Convert a verified hierarchical node name into a stable CSV prefix."""

    prefix = re.sub(r"[^A-Za-z0-9]+", "_", node.strip("/"))
    return prefix.strip("_").lower() or "node"


# ngspice reports netlist and model problems near the START of its -o log, so a
# plain tail of the combined output usually shows only the banner.  These
# patterns pull the meaningful lines out regardless of where they appear.
_NGSPICE_ERROR_PATTERN = re.compile(
    r"error|fatal|abort|singular|no such|unknown|cannot|can't|could not"
    r"|unable to|failed|too small|not found|out of memory",
    re.I,
)
_NGSPICE_WARNING_PATTERN = re.compile(r"warning", re.I)
_MAX_DIAGNOSTIC_LINES = 15
_MAX_TAIL_LINES = 20

# Windows MAX_PATH is 260 including the terminating null, so 259 usable
# characters.  Python opens longer paths on this platform, but ngspice is a
# plain Win32 program and simply reports the file as missing.
_WINDOWS_MAX_PATH_CHARS = 259
# The per-stimulus directory only has to be unique and recognizable; the full
# source path stays in transient_summary.csv and in the stimulus include header.
_MAX_STIMULUS_LABEL_CHARS = 8
_RESULTS_PATH_REMEDY = (
    "GUI의 '결과 폴더'를 더 짧은 경로로 지정하면 해결됩니다. "
    "예: C:\\ngspice_results"
)


def _stimulus_directory_name(stimulus_index: int, stimulus: Path) -> str:
    """Return a short, unique directory name for one transient stimulus.

    The index alone guarantees uniqueness, so only a truncated word label is
    added.  Embedding the full PWL stem used to push the per-channel files past
    MAX_PATH, which made ngspice fail with a misleading missing-file error.  The
    full source path stays in ``transient_summary.csv`` and in the generated
    stimulus include header.
    """

    safe_word = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stimulus.parent.name,
    ).strip("._")[:_MAX_STIMULUS_LABEL_CHARS]
    name = f"stim_{stimulus_index:04d}"
    return f"{name}_{safe_word}" if safe_word else name


def _reject_paths_over_windows_limit(
    paths: Sequence[Path],
    remedy: str = _RESULTS_PATH_REMEDY,
) -> None:
    """Fail before launching when ngspice could not open a generated path.

    Without this check the run fails inside ngspice with a bare
    ``No such file or directory`` for a path that Python just wrote
    successfully, which is very hard to interpret.
    """

    if sys.platform != "win32":
        return
    offenders = [
        path for path in paths if len(str(path)) > _WINDOWS_MAX_PATH_CHARS
    ]
    if not offenders:
        return
    longest = max(offenders, key=lambda path: len(str(path)))
    length = len(str(longest))
    raise RuntimeError(
        f"생성 경로가 Windows 경로 한계({_WINDOWS_MAX_PATH_CHARS}자)를 "
        f"{length - _WINDOWS_MAX_PATH_CHARS}자 초과합니다 (총 {length}자).\n"
        "Python은 이 파일을 만들 수 있지만 ngspice는 열지 못합니다.\n\n"
        f"{longest}\n\n{remedy}"
    )


def _matching_log_lines(
    text: str,
    pattern: re.Pattern[str],
    limit: int = _MAX_DIAGNOSTIC_LINES,
) -> list[str]:
    """Return numbered log lines matching one severity pattern."""

    found: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or not pattern.search(stripped):
            continue
        if len(found) >= limit:
            found.append("  ... (같은 수준의 이후 메시지 생략)")
            break
        found.append(f"  {number:>6}행: {stripped}")
    return found


def _log_tail(text: str, limit: int = _MAX_TAIL_LINES) -> list[str]:
    """Return the last non-empty lines of one log for trailing context."""

    return [line for line in text.splitlines() if line.strip()][-limit:]


def _ngspice_failure_message(
    label: str,
    launch: object,
    raw_path: Path,
    ngspice_output: str,
    launcher_output: str,
) -> str:
    """Build the shared AC/OP, Transient, and Detector failure diagnosis.

    All three runners produce the same sections so a user never has to learn a
    different layout per analysis: what failed, which log lines explain it,
    trailing context, actionable hints, and both full log paths.
    """

    failure = (
        f"종료 코드 {launch.returncode}"
        if launch.returncode != 0
        else f"종료 코드는 0이지만 {raw_path.name} 없음"
    )
    sections = [f"{label}: ngspice {failure}"]

    # Startup failures never reach the -o log because ngspice cannot open it
    # yet; they land on the child's stdout, which is the launcher log.  Both
    # sources must be searched or those failures look like a silent exit.
    sources = (("ngspice", ngspice_output), ("launcher", launcher_output))
    highlighted: list[str] = []
    heading = ""
    for severity, pattern in (
        ("오류", _NGSPICE_ERROR_PATTERN),
        ("경고", _NGSPICE_WARNING_PATTERN),
    ):
        for source_label, text in sources:
            highlighted = _matching_log_lines(text, pattern)
            if highlighted:
                heading = f"{source_label} 로그의 {severity} 행"
                break
        if highlighted:
            break
    if highlighted:
        sections.append(f"{heading}:\n" + "\n".join(highlighted))
    else:
        sections.append(
            "ngspice 로그에 명시적인 오류 행이 없습니다. "
            "수렴 실패, 사용자/외부 종료, 디스크 공간 부족을 확인하세요."
        )

    for log_label, text in (
        ("ngspice", ngspice_output),
        ("launcher", launcher_output),
    ):
        tail = _log_tail(text)
        if tail:
            sections.append(f"{log_label} 로그 마지막 부분:\n" + "\n".join(tail))

    combined = f"{ngspice_output}\n{launcher_output}"
    hints: list[str] = []
    init_path = Path(getattr(launch, "cwd", raw_path.parent)) / ".spiceinit"
    if re.search(r"no such function\s+['\"]?if", combined, re.I):
        hints.append(
            "PSpice 호환 초기화가 적용되지 않았습니다. "
            f"{init_path}에 'set ngbehavior=pski'가 있는지 확인하세요."
        )
    elif "compatibility modes selected" not in combined.lower():
        hints.append(
            "로그에 호환 모드 선택 기록이 없습니다. "
            f"{init_path}를 ngspice가 읽었는지 확인하세요."
        )
    if re.search(
        r"could not find|no such file|(?:can ?not|can't|unable to|failed to)"
        r"\s+open",
        combined,
        re.I,
    ):
        hints.append(
            "모델 include 경로를 확인하세요. netlist_template.cir의 "
            "BAT54WT1, LPV7215, OPA379 경로가 이 PC와 달라도 같은 증상이 납니다."
        )
    if hints:
        sections.append(
            "확인할 점:\n" + "\n".join(f"- {hint}" for hint in hints)
        )

    sections.append(
        "전체 진단 로그:\n"
        f"  {getattr(launch, 'ngspice_log', '<없음>')}\n"
        f"  {getattr(launch, 'launcher_log', '<없음>')}"
    )
    return "\n\n".join(sections)

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
        # Microseconds match the transient and detector runners; without them
        # two runs started in the same second collide on mkdir(exist_ok=False).
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
            _reject_paths_over_windows_limit(
                (
                    inc_path,
                    *(job.netlist_path for job in jobs),
                    *(job.raw_path for job in jobs),
                    *(
                        run_dir / f"{job.spec.log_stem}_{name}.log"
                        for job in jobs
                        for name in ("ngspice", "launcher")
                    ),
                )
            )
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
                        template_text,
                        inc_path,
                        settings,
                        job.spec.key,
                        channel.detector_opamp,
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
                    raise RuntimeError(
                        _ngspice_failure_message(
                            f"ch{channel.ch:02d} {job.spec.label}",
                            launch,
                            job.raw_path,
                            ngspice_output,
                            launcher_output,
                        )
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
                    (
                        "frequency_hz",
                        f"{_csv_node_prefix(settings.output_node)}_magnitude_db",
                        f"{_csv_node_prefix(settings.output_node)}_magnitude_linear",
                        f"{_csv_node_prefix(settings.output_node)}_phase_deg",
                    ),
                    (
                        (frequency_hz, magnitude_db, db_to_linear(magnitude_db), phase_deg)
                        for frequency_hz, magnitude_db, phase_deg in ac_rows
                    ),
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
        on_result: Callable[[TransientResult], None] | None = None,
    ) -> list[TransientResult]:
        """Execute every selected stimulus/channel pair and normalize results.

        ``on_result`` is invoked as soon as each job is parsed, so a caller can
        display a finished channel while the remaining ones still run.  It is
        called on this worker thread, so a GUI must marshal it to its own loop.
        """

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
            stimulus_dir = session_dir / _stimulus_directory_name(
                stimulus_index,
                stimulus,
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
                _reject_paths_over_windows_limit(
                    (
                        parameter_include,
                        run_dir / f"channel_{channel.ch:02d}_tran.cir",
                        run_dir / TRAN_RAW_FILENAME,
                        run_dir / "tran_ngspice.log",
                        run_dir / "tran_launcher.log",
                        stimulus_include,
                    )
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
                        channel.detector_opamp,
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
                    raise RuntimeError(
                        _ngspice_failure_message(
                            f"{stimulus.name} ch{channel.ch:02d}",
                            launch,
                            raw_path,
                            ngspice_output,
                            launcher_output,
                        )
                    )
                rows = read_transient_raw(raw_path, settings)
                if settings.write_csv:
                    write_normalized_csv(
                        run_dir / "transient.csv",
                        (
                            "time_s",
                            "vin_v",
                            "v_filt_v",
                            "v_env_v",
                            "v_thr_v",
                        ),
                        rows,
                    )
                result = TransientResult(
                    channel=channel,
                    stimulus_path=stimulus,
                    run_dir=run_dir,
                    rows=rows,
                    log_text=log_text,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
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


class DetectorSimulator:
    """Run one full-channel gated-sine Active Detector job independently."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        """Create an idle detector runner with its own cancellation state."""

        self.log = log
        self.stop_event = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Request cancellation and terminate only the detector subprocess."""

        self.stop_event.set()
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            process.terminate()

    def run(
        self,
        channel: Channel,
        template_path: Path,
        output_root: Path,
        ngspice: str,
        settings: DetectorSettings,
        component_revision: int,
    ) -> DetectorResult:
        """Run one detector channel and return its single result.

        Retained as the original public entry point; ``run_channels()`` is the
        session-oriented form used by the GUI.
        """

        results = self.run_channels(
            (channel,),
            template_path,
            output_root,
            ngspice,
            settings,
            component_revision,
        )
        if not results:
            raise RuntimeError("Active Detector 실행이 중지되었습니다.")
        return results[0]

    def run_channels(
        self,
        channels: Sequence[Channel],
        template_path: Path,
        output_root: Path,
        ngspice: str,
        settings: DetectorSettings,
        component_revision: int,
        on_result: Callable[[DetectorResult], None] | None = None,
    ) -> list[DetectorResult]:
        """Run every selected channel sequentially inside one session folder.

        ``on_result`` fires as soon as each channel is parsed so a caller can
        show a finished channel while the rest still run.  It is called on this
        worker thread, so a GUI must marshal it to its own loop.
        """

        settings.validate()
        selected = list(channels)
        if not selected:
            raise ValueError("Active Detector 실행 채널을 하나 이상 선택하세요.")
        template = template_path.expanduser().resolve()
        if not template.is_file():
            raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {template}")
        template_text = template.read_text(encoding="utf-8-sig")
        session_dir = output_root.expanduser().resolve() / (
            "detector_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        write_channels_csv(
            session_dir / "applied_components.csv", selected
        )
        session_log = session_dir / "launcher.log"
        session_log.write_text(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"DETECTOR SESSION START channels="
            f"{','.join(str(item.ch) for item in selected)} "
            f"component_revision={component_revision} "
            f"frequency_hz={settings.frequency_hz:.12g} "
            f"input_vpp_v={settings.input_vpp_v:.12g}\n",
            encoding="utf-8",
        )
        results: list[DetectorResult] = []
        for position, channel in enumerate(selected, start=1):
            if self.stop_event.is_set():
                self.log("사용자가 Active Detector 실행을 중지했습니다.")
                break
            self.log(
                f"[{position}/{len(selected)}] ch{channel.ch:02d} "
                f"Active Detector 실행 ({channel.detector_opamp})"
            )
            result = self._run_channel(
                channel,
                session_dir,
                session_log,
                template_text,
                ngspice,
                settings,
                component_revision,
            )
            if result is None:
                break
            results.append(result)
            if on_result is not None:
                on_result(result)
        self.log(f"Active Detector 결과 폴더: {session_dir}")
        return results

    def _run_channel(
        self,
        channel: Channel,
        session_dir: Path,
        session_log: Path,
        template_text: str,
        ngspice: str,
        settings: DetectorSettings,
        component_revision: int,
    ) -> DetectorResult | None:
        """Generate, execute, parse, and normalize one detector simulation."""

        source_name, positive_node, negative_node = find_input_voltage_source(
            template_text,
            settings.vin_node,
        )
        run_dir = session_dir / f"ch{channel.ch:02d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        parameter_include = run_dir / f"channel_{channel.ch:02d}_params.inc"
        _reject_paths_over_windows_limit(
            (
                parameter_include,
                run_dir / f"channel_{channel.ch:02d}_detector.cir",
                run_dir / "detector_stimulus.inc",
                run_dir / DETECTOR_RAW_FILENAME,
                run_dir / "detector_ngspice.log",
                run_dir / "detector_launcher.log",
            )
        )
        parameter_include.write_text(
            make_parameter_include(channel),
            encoding="utf-8",
        )
        stimulus_include = run_dir / "detector_stimulus.inc"
        stimulus_include.write_text(
            make_gated_sine_source_include(
                source_name,
                positive_node,
                negative_node,
                settings,
            ),
            encoding="ascii",
        )
        netlist_path = run_dir / f"channel_{channel.ch:02d}_detector.cir"
        netlist_path.write_text(
            make_detector_netlist(
                template_text,
                parameter_include,
                stimulus_include,
                settings,
                channel.detector_opamp,
            ),
            encoding="utf-8",
        )
        init_path = run_dir / ".spiceinit"
        init_lines = [
            "* Auto-generated ngspice initialization for Active Detector",
            "set filetype=ascii",
        ]
        if settings.pspice_compat:
            init_lines.append("set ngbehavior=pski")
        init_path.write_text("\n".join(init_lines) + "\n", encoding="utf-8")
        raw_path = run_dir / DETECTOR_RAW_FILENAME

        def remember_process(process: subprocess.Popen[str] | None) -> None:
            """Expose only the current detector child for cancellation."""

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
            log_stem="detector",
        )
        ngspice_output = (
            launch.ngspice_log.read_text(encoding="utf-8", errors="replace")
            if launch.ngspice_log.is_file()
            else ""
        )
        launcher_output = (
            launch.launcher_log.read_text(encoding="utf-8", errors="replace")
            if launch.launcher_log.is_file()
            else ""
        )
        log_text = (
            f"[Detector ngspice]\n{ngspice_output.rstrip()}\n\n"
            f"[Detector launcher]\n{launcher_output.rstrip()}"
        ).strip()
        if self.stop_event.is_set():
            # A deliberate stop is not a failure, so the caller ends the loop
            # instead of raising and showing an error dialog.
            return None
        if launch.returncode != 0 or not raw_path.is_file():
            raise RuntimeError(
                _ngspice_failure_message(
                    f"ch{channel.ch:02d} Active Detector",
                    launch,
                    raw_path,
                    ngspice_output,
                    launcher_output,
                )
            )
        rows = read_detector_raw(raw_path, settings)
        if settings.write_csv:
            write_normalized_csv(
                run_dir / "detector.csv",
                ("time_s", "vin_v", "v_filt_v", "v_env_v", "v_thr_v"),
                rows,
            )
        write_normalized_csv(
            run_dir / "detector_settings.csv",
            ("name", "value"),
            (
                ("component_revision", component_revision),
                ("frequency_hz", settings.frequency_hz),
                ("input_vpp_v", settings.input_vpp_v),
                ("gate_on_s", settings.gate_on_s),
                ("gate_off_s", settings.gate_off_s),
                ("total_time_s", settings.total_time_s),
                ("maximum_step_s", settings.maximum_step_s),
                ("detector_opamp", channel.detector_opamp),
            ),
        )
        self.log(f"ch{channel.ch:02d} 완료: {len(rows)}점")
        return DetectorResult(
            channel=channel,
            settings=settings,
            component_revision=component_revision,
            run_dir=run_dir,
            rows=rows,
            log_text=log_text,
        )

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
