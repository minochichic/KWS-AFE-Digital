#!/usr/bin/env python3
"""Reliable ngspice process launcher for Windows and POSIX.

This module is deliberately independent of the Tk GUI.  It resolves ngspice
from an explicit value or PATH, writes launch diagnostics before starting the
child process, redirects child stdout/stderr directly to disk, and records the
exit status after the process finishes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class LaunchResult:
    """Immutable record of one ngspice child process and its diagnostic files."""

    executable: str
    command: tuple[str, ...]
    cwd: Path
    returncode: int
    launcher_log: Path
    ngspice_log: Path
    rawfile: Path | None


def _timestamp() -> str:
    """Return a timezone-aware timestamp suitable for durable launcher logs."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(path: Path, text: str) -> None:
    """Append diagnostics and force them to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace", newline="\n") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # Some virtual/network filesystems do not implement fsync.
            pass


def command_text(command: Sequence[str]) -> str:
    """Render a platform-correct command line for diagnostics only."""

    if sys.platform == "win32":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _prefer_console(path_text: str) -> str:
    """Prefer ngspice_con.exe beside the Windows GUI executable."""
    path = Path(path_text)
    if path.name.lower() == "ngspice.exe":
        console_path = path.with_name("ngspice_con.exe")
        if console_path.is_file():
            return str(console_path.resolve())
    return str(path.resolve())


def resolve_ngspice(configured: str = "") -> str:
    """Resolve an explicit path/command, environment override, or PATH entry."""
    requested = configured.strip().strip('"')
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
    else:
        env_override = os.environ.get("NGSPICE_EXE", "").strip().strip('"')
        if env_override:
            candidates.append(env_override)
        # The console binary is the reliable choice for Python automation.
        candidates.extend(("ngspice_con.exe", "ngspice_con", "ngspice", "ngspice.exe"))

    checked: list[str] = []
    for candidate in candidates:
        checked.append(candidate)
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            return _prefer_console(str(candidate_path))
        resolved = shutil.which(candidate)
        if resolved:
            return _prefer_console(resolved)

    if not requested:
        common_windows = (
            Path(r"C:\Spice64\bin\ngspice_con.exe"),
            Path(r"C:\Program Files\ngspice\bin\ngspice_con.exe"),
            Path(r"C:\Program Files (x86)\ngspice\bin\ngspice_con.exe"),
            Path(r"C:\Spice64\bin\ngspice.exe"),
            Path(r"C:\Program Files\ngspice\bin\ngspice.exe"),
            Path(r"C:\Program Files (x86)\ngspice\bin\ngspice.exe"),
        )
        for path in common_windows:
            checked.append(str(path))
            if path.is_file():
                return _prefer_console(str(path))

    detail = ", ".join(checked) if checked else "(없음)"
    raise FileNotFoundError(
        "ngspice 실행 파일을 찾지 못했습니다. "
        "PowerShell에서 'Get-Command ngspice'를 확인하거나 GUI에서 직접 선택하세요.\n"
        f"확인한 후보: {detail}"
    )


def run_ngspice(
    *,
    executable: str,
    netlist: Path,
    run_dir: Path,
    init_dir: Path,
    stop_event: threading.Event | None = None,
    on_process: Callable[[subprocess.Popen[str] | None], None] | None = None,
    session_log: Path | None = None,
    rawfile: Path | None = None,
    log_stem: str = "",
) -> LaunchResult:
    """Run ngspice while making launcher diagnostics durable."""
    resolved = resolve_ngspice(executable)
    run_dir = run_dir.resolve()
    netlist = netlist.resolve()
    init_dir = init_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if log_stem and not all(ch.isalnum() or ch in "_.-" for ch in log_stem):
        raise ValueError(f"잘못된 로그 이름: {log_stem!r}")
    prefix = f"{log_stem}_" if log_stem else ""
    launcher_log = run_dir / f"{prefix}launcher.log"
    ngspice_log = run_dir / f"{prefix}ngspice.log"
    resolved_rawfile = rawfile.resolve() if rawfile is not None else None
    if resolved_rawfile is not None and resolved_rawfile.parent != run_dir:
        raise ValueError("rawfile은 채널 실행 폴더 안에 있어야 합니다.")
    child_env = os.environ.copy()
    child_env["SPICE_USERINIT_DIR"] = str(init_dir)
    command_parts = [
        resolved,
        "-b",
        "-o",
        str(ngspice_log),
    ]
    if resolved_rawfile is not None:
        # ngspice is already launched with run_dir as cwd.  A basename avoids
        # every Windows quoting/path-conversion issue seen with control-language
        # file commands while still using the documented command-line -r path.
        command_parts.extend(("-r", resolved_rawfile.name))
    command_parts.append(str(netlist))
    command = tuple(command_parts)

    header = "\n".join(
        (
            f"[{_timestamp()}] START",
            f"requested_executable={executable or '<PATH auto-detect>'}",
            f"resolved_executable={resolved}",
            f"cwd={run_dir}",
            f"SPICE_USERINIT_DIR={init_dir}",
            f"rawfile={resolved_rawfile or '<none>'}",
            f"command={command_text(command)}",
            "[child stdout/stderr follows]",
            "",
        )
    )
    append_log(launcher_log, header)
    if session_log is not None:
        append_log(
            session_log,
            f"[{_timestamp()}] START cwd={run_dir} executable={resolved}\n"
            f"command={command_text(command)}",
        )

    creationflags = (
        subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW")
        else 0
    )
    process: subprocess.Popen[str] | None = None
    try:
        # Direct-to-file redirection avoids a full PIPE blocking the launcher.
        with launcher_log.open(
            "a", encoding="utf-8", errors="replace", newline="\n"
        ) as child_output:
            child_output.flush()
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                stdout=child_output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                env=child_env,
            )
            if on_process is not None:
                on_process(process)
            while process.poll() is None:
                if stop_event is not None and stop_event.wait(0.1):
                    process.terminate()
                    break
                if stop_event is None:
                    process.wait(timeout=None)
            returncode = process.wait()
            child_output.flush()
            try:
                os.fsync(child_output.fileno())
            except OSError:
                pass
    except Exception as exc:
        append_log(
            launcher_log,
            f"[{_timestamp()}] LAUNCH ERROR {type(exc).__name__}: {exc}",
        )
        if session_log is not None:
            append_log(
                session_log,
                f"[{_timestamp()}] LAUNCH ERROR cwd={run_dir} "
                f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        if on_process is not None:
            on_process(None)

    footer = f"[{_timestamp()}] END returncode={returncode}"
    append_log(launcher_log, footer)
    if session_log is not None:
        append_log(session_log, f"{footer} cwd={run_dir}")
    return LaunchResult(
        executable=resolved,
        command=command,
        cwd=run_dir,
        returncode=returncode,
        launcher_log=launcher_log,
        ngspice_log=ngspice_log,
        rawfile=resolved_rawfile,
    )


def probe_ngspice(configured: str, log_path: Path) -> int:
    """Resolve PATH and capture `ngspice -v` without requiring a netlist."""
    try:
        resolved = resolve_ngspice(configured)
        command = (resolved, "-v")
        append_log(
            log_path,
            "\n".join(
                (
                    f"[{_timestamp()}] PROBE START",
                    f"requested={configured or '<PATH auto-detect>'}",
                    f"resolved={resolved}",
                    f"command={command_text(command)}",
                    "",
                )
            ),
        )
        with log_path.open(
            "a", encoding="utf-8", errors="replace", newline="\n"
        ) as output:
            completed = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        append_log(log_path, f"[{_timestamp()}] PROBE END returncode={completed.returncode}")
        return completed.returncode
    except Exception as exc:
        append_log(
            log_path,
            f"[{_timestamp()}] PROBE ERROR {type(exc).__name__}: {exc}",
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone PATH/probe diagnostic command."""

    parser = argparse.ArgumentParser(description="ngspice PATH/logging probe")
    parser.add_argument("--ngspice", default="", help="경로 또는 PATH 명령; 생략 시 자동")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "runner_probe.log",
    )
    args = parser.parse_args(argv)
    status = probe_ngspice(args.ngspice, args.log.resolve())
    print(f"진단 로그: {args.log.resolve()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
