#!/usr/bin/env python3
"""Convert PCM WAV trees into source-independent ngspice PWL data files.

The generated ``.pwl`` files contain two whitespace-separated columns:
time in seconds and voltage in volts.  They are intentionally independent of
the circuit's voltage-source name and node names.  The transient netlist
builder reads this data and creates the actual ``V... PWL(...)`` source at the
existing input-source location.
"""

from __future__ import annotations

import csv
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


TARGET_VPP_DEFAULT = 0.010
MANIFEST_COLUMNS = (
    "word",
    "source_wav",
    "output_pwl",
    "sample_rate_hz",
    "channels",
    "sample_width_bytes",
    "frames",
    "audio_duration_s",
    "pwl_end_s",
    "input_min",
    "input_max",
    "input_mean",
    "output_min_v",
    "output_max_v",
    "output_mean_v",
    "output_vpp_v",
    "status",
)


@dataclass(frozen=True)
class PwlConversionResult:
    """Metadata and output path for one converted WAV file."""

    word: str
    source_wav: Path
    output_pwl: Path
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frames: int
    audio_duration_s: float
    pwl_end_s: float
    input_min: float
    input_max: float
    input_mean: float
    output_min_v: float
    output_max_v: float
    output_mean_v: float
    output_vpp_v: float
    status: str

    def manifest_row(self) -> dict[str, object]:
        """Return one stable CSV row for conversion traceability."""

        return {
            "word": self.word,
            "source_wav": str(self.source_wav),
            "output_pwl": str(self.output_pwl),
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "frames": self.frames,
            "audio_duration_s": f"{self.audio_duration_s:.12g}",
            "pwl_end_s": f"{self.pwl_end_s:.12g}",
            "input_min": f"{self.input_min:.12g}",
            "input_max": f"{self.input_max:.12g}",
            "input_mean": f"{self.input_mean:.12g}",
            "output_min_v": f"{self.output_min_v:.12g}",
            "output_max_v": f"{self.output_max_v:.12g}",
            "output_mean_v": f"{self.output_mean_v:.12g}",
            "output_vpp_v": f"{self.output_vpp_v:.12g}",
            "status": self.status,
        }


def _decode_pcm_sample(payload: bytes, width: int) -> int:
    """Decode one little-endian PCM sample of width 1, 2, 3, or 4 bytes."""

    if width == 1:
        return payload[0] - 128
    if width in {2, 4}:
        return int.from_bytes(payload, byteorder="little", signed=True)
    if width == 3:
        extended = payload + (b"\xff" if payload[2] & 0x80 else b"\x00")
        return int.from_bytes(extended, byteorder="little", signed=True)
    raise ValueError(
        f"지원하지 않는 PCM sample width: {width} bytes "
        "(지원: 8/16/24/32-bit integer PCM)"
    )


def read_pcm_wav(path: Path) -> tuple[int, int, int, list[float]]:
    """Read an integer-PCM WAV and return rate/channels/width/mono samples.

    Multichannel input is converted to mono by averaging all channels within
    each frame.  Google Speech Commands v2 files are normally mono 16-bit PCM,
    but the extra widths make the converter fail clearly or behave correctly
    for common user-added WAV files.
    """

    source = path.expanduser().resolve()
    try:
        with wave.open(str(source), "rb") as handle:
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            payload = handle.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"WAV 파일을 읽을 수 없습니다: {source}: {exc}") from exc
    if compression != "NONE":
        raise ValueError(f"압축 WAV는 지원하지 않습니다: {source} ({compression})")
    if channel_count < 1 or sample_rate < 1 or frame_count < 1:
        raise ValueError(
            f"WAV 헤더가 잘못되었습니다: channels={channel_count}, "
            f"rate={sample_rate}, frames={frame_count}"
        )
    bytes_per_frame = channel_count * sample_width
    expected_bytes = frame_count * bytes_per_frame
    if len(payload) != expected_bytes:
        raise ValueError(
            f"WAV PCM 길이 불일치: 예상 {expected_bytes} bytes, "
            f"실제 {len(payload)} bytes"
        )
    mono: list[float] = []
    for frame_index in range(frame_count):
        frame_offset = frame_index * bytes_per_frame
        channel_values = []
        for channel_index in range(channel_count):
            offset = frame_offset + channel_index * sample_width
            channel_values.append(
                _decode_pcm_sample(payload[offset : offset + sample_width], sample_width)
            )
        mono.append(sum(channel_values) / channel_count)
    return sample_rate, channel_count, sample_width, mono


def normalize_zero_dc_vpp(
    samples: Sequence[float],
    target_vpp: float = TARGET_VPP_DEFAULT,
) -> tuple[list[float], dict[str, float | str]]:
    """Remove the sample mean and scale peak-to-peak range to ``target_vpp``.

    Subtracting the arithmetic mean makes the discrete audio samples have
    zero DC.  Multiplication by ``target_vpp / (max-min)`` preserves waveform
    shape while making the output peak-to-peak span exactly the requested
    value.  A constant file cannot have non-zero Vpp, so it becomes 0 V and is
    explicitly marked ``constant/silence``.
    """

    if not samples:
        raise ValueError("정규화할 오디오 샘플이 없습니다.")
    if not math.isfinite(target_vpp) or target_vpp <= 0.0:
        raise ValueError("target Vpp는 0보다 큰 유한한 값이어야 합니다.")
    minimum = min(samples)
    maximum = max(samples)
    mean = math.fsum(samples) / len(samples)
    span = maximum - minimum
    if span <= 0.0:
        values = [0.0] * len(samples)
        status = "constant/silence: 0 V output"
    else:
        scale = target_vpp / span
        values = [(sample - mean) * scale for sample in samples]
        # Remove the remaining floating-point sum error without changing Vpp.
        residual = math.fsum(values) / len(values)
        values = [value - residual for value in values]
        status = "ok"
    output_minimum = min(values)
    output_maximum = max(values)
    output_mean = math.fsum(values) / len(values)
    return values, {
        "input_min": float(minimum),
        "input_max": float(maximum),
        "input_mean": float(mean),
        "output_min_v": float(output_minimum),
        "output_max_v": float(output_maximum),
        "output_mean_v": float(output_mean),
        "output_vpp_v": float(output_maximum - output_minimum),
        "status": status,
    }


def write_pwl_data(
    path: Path,
    voltages: Sequence[float],
    sample_rate_hz: int,
    source_wav: Path,
    *,
    overwrite: bool,
) -> float:
    """Write time/voltage pairs and append a one-sample-period 0 V tail.

    ngspice holds the final PWL value after the last declared point.  Appending
    ``time = frame_count / sample_rate`` at 0 V therefore guarantees silence
    for any transient stop time beyond the WAV duration.
    """

    if sample_rate_hz < 1 or not voltages:
        raise ValueError("PWL 출력에는 양의 sample rate와 샘플이 필요합니다.")
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"이미 존재하는 PWL 파일: {target}")
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("* ngspice PWL data v1: time_s voltage_v\n")
            handle.write(f"* source_wav={source_wav.resolve()}\n")
            handle.write(f"* sample_rate_hz={sample_rate_hz}\n")
            handle.write(f"* audio_frames={len(voltages)}\n")
            handle.write("* normalization=zero sample mean, 0.010 V peak-to-peak\n")
            for index, voltage in enumerate(voltages):
                handle.write(
                    f"{index / sample_rate_hz:.12g} {voltage:.12g}\n"
                )
            tail_time = len(voltages) / sample_rate_hz
            handle.write(f"{tail_time:.12g} 0\n")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(voltages) / sample_rate_hz


def read_pwl_data(path: Path) -> list[tuple[float, float]]:
    """Parse and validate one generated two-column PWL data file."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PWL 파일을 찾을 수 없습니다: {source}")
    pairs: list[tuple[float, float]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8-sig", errors="strict").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", "#", ";")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(
                f"{source.name} {line_number}행: time/value 두 열이 필요합니다."
            )
        try:
            time_s, voltage_v = map(float, parts)
        except ValueError as exc:
            raise ValueError(
                f"{source.name} {line_number}행: 숫자가 아닙니다."
            ) from exc
        if not math.isfinite(time_s) or not math.isfinite(voltage_v):
            raise ValueError(
                f"{source.name} {line_number}행: 유한한 숫자가 필요합니다."
            )
        if time_s < 0.0 or (pairs and time_s <= pairs[-1][0]):
            raise ValueError(
                f"{source.name} {line_number}행: 시간은 0 이상이며 엄격히 "
                "증가해야 합니다."
            )
        pairs.append((time_s, voltage_v))
    if len(pairs) < 2:
        raise ValueError(f"PWL 파일에는 최소 두 점이 필요합니다: {source}")
    return pairs


def discover_wav_files(input_root: Path, output_root: Path | None = None) -> list[Path]:
    """Find WAV files recursively while excluding an output tree inside input."""

    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"WAV 루트 폴더가 아닙니다: {root}")
    excluded = output_root.expanduser().resolve() if output_root else None
    found = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() != ".wav":
            continue
        resolved = candidate.resolve()
        if excluded is not None and (resolved == excluded or excluded in resolved.parents):
            continue
        found.append(resolved)
    return sorted(found, key=lambda item: str(item).casefold())


def discover_pwl_files(root: Path) -> list[Path]:
    """Find converted PWL data files recursively in deterministic order."""

    directory = root.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"PWL 폴더가 아닙니다: {directory}")
    return sorted(
        (
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pwl"
        ),
        key=lambda item: str(item).casefold(),
    )


def _output_relative_path(input_root: Path, source_wav: Path) -> Path:
    """Mirror word folders and create one when the selected root is a word."""

    relative = source_wav.relative_to(input_root)
    if relative.parent == Path("."):
        relative = Path(input_root.name) / relative
    return relative.with_suffix(".pwl")


def convert_one_wav(
    source_wav: Path,
    output_pwl: Path,
    *,
    target_vpp: float = TARGET_VPP_DEFAULT,
    overwrite: bool = True,
    word: str = "",
) -> PwlConversionResult:
    """Convert one WAV, write its PWL data, and return measured statistics."""

    rate, channels, width, samples = read_pcm_wav(source_wav)
    voltages, statistics = normalize_zero_dc_vpp(samples, target_vpp)
    pwl_end = write_pwl_data(
        output_pwl,
        voltages,
        rate,
        source_wav,
        overwrite=overwrite,
    )
    return PwlConversionResult(
        word=word or source_wav.parent.name,
        source_wav=source_wav.resolve(),
        output_pwl=output_pwl.resolve(),
        sample_rate_hz=rate,
        channels=channels,
        sample_width_bytes=width,
        frames=len(samples),
        audio_duration_s=len(samples) / rate,
        pwl_end_s=pwl_end,
        input_min=float(statistics["input_min"]),
        input_max=float(statistics["input_max"]),
        input_mean=float(statistics["input_mean"]),
        output_min_v=float(statistics["output_min_v"]),
        output_max_v=float(statistics["output_max_v"]),
        output_mean_v=float(statistics["output_mean_v"]),
        output_vpp_v=float(statistics["output_vpp_v"]),
        status=str(statistics["status"]),
    )


def write_conversion_manifest(
    path: Path,
    results: Iterable[PwlConversionResult],
) -> None:
    """Write a UTF-8 manifest that can be inspected in Excel."""

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.manifest_row())


def convert_wav_tree(
    input_root: Path,
    output_root: Path,
    *,
    target_vpp: float = TARGET_VPP_DEFAULT,
    overwrite: bool = True,
    progress: Callable[[int, int, Path, Path], None] | None = None,
) -> list[PwlConversionResult]:
    """Convert a dataset subtree and mirror its word-directory structure."""

    source_root = input_root.expanduser().resolve()
    target_root = output_root.expanduser().resolve()
    wav_files = discover_wav_files(source_root, target_root)
    if not wav_files:
        raise FileNotFoundError(f"WAV 파일이 없습니다: {source_root}")
    converted: list[PwlConversionResult] = []
    for index, source in enumerate(wav_files, start=1):
        relative = _output_relative_path(source_root, source)
        target = target_root / relative
        if progress is not None:
            progress(index, len(wav_files), source, target)
        converted.append(
            convert_one_wav(
                source,
                target,
                target_vpp=target_vpp,
                overwrite=overwrite,
                word=relative.parts[0],
            )
        )
    write_conversion_manifest(target_root / "pwl_manifest.csv", converted)
    return converted
