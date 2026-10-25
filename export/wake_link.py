"""The STM32 <-> FPGA wake link, as executable bytes (docs/wake_stm32_plan.md 3).

    STM32 -> FPGA   0xA5 | N | frame[0..N-1] (2 bytes, LE, bit k = ch k) | CRC-8
    FPGA  -> STM32  0x5A | class_idx | keyword_idx | margin (int32 LE) | CRC-8

CRC-8 is polynomial 0x07, init 0, no reflection (CRC-8/SMBUS), over every
byte before it, sync included. rtl/kws_crc8.v, rtl/tb/tb_wake_top.v and the
STM32's CRC peripheral all compute this; check value over b"123456789" is 0xF4.

This file is the reference the firmware and any PC-side tool are written
against -- the same role export/pack.py plays for the weight ROMs. It needs
neither torch nor a checkpoint.

    # the 100 native frames of clip 0 from a self-test export, as one packet
    python -m export.wake_link rtl/gen/bd_base/selftest/input.hex --clip 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, NamedTuple, Sequence

SYNC_IN, SYNC_OUT = 0xA5, 0x5A
T, NATIVE_T, PAD_LEFT = 128, 100, 14
RESULT_BYTES = 8


def crc8(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def frame_packet(frames: Sequence[int]) -> bytes:
    """N native 16-bit frames -> one STM32 -> FPGA packet."""
    if not 0 < len(frames) < 256:
        raise ValueError(f"N must fit one byte, got {len(frames)}")
    body = bytearray([SYNC_IN, len(frames)])
    for f in frames:
        if not 0 <= f < 1 << 16:
            raise ValueError(f"frame {f:#x} is not 16 bits")
        body += f.to_bytes(2, "little")
    return bytes(body + bytes([crc8(body)]))


class Result(NamedTuple):
    class_idx: int
    keyword_idx: int
    margin: int


def parse_result(pkt: bytes) -> Result:
    """One FPGA -> STM32 packet. Raises on bad length, sync or CRC."""
    if len(pkt) != RESULT_BYTES:
        raise ValueError(f"result packet is {RESULT_BYTES} bytes, got {len(pkt)}")
    if pkt[0] != SYNC_OUT:
        raise ValueError(f"bad sync {pkt[0]:#04x}")
    if crc8(pkt[:-1]) != pkt[-1]:
        raise ValueError("bad CRC")
    return Result(pkt[1], pkt[2], int.from_bytes(pkt[3:7], "little", signed=True))


def result_packet(r: Result) -> bytes:
    body = bytes([SYNC_OUT, r.class_idx, r.keyword_idx]) + \
        r.margin.to_bytes(4, "little", signed=True)
    return body + bytes([crc8(body)])


def native_frames(input_hex: Path, clip: int) -> List[int]:
    """The 100 native frames of one clip from an export's input.hex
    (T words per clip, padding included). Refuses a clip whose padding is not
    all -1 (zero words): the link adds that padding back, so dropping
    non-zero pad frames here would lose data silently."""
    words = [int(l, 16) for l in Path(input_hex).read_text().split()]
    if len(words) % T:
        raise ValueError(f"{input_hex}: {len(words)} words is not a multiple of {T}")
    w = words[clip * T:(clip + 1) * T]
    if len(w) != T:
        raise IndexError(f"clip {clip} is out of range")
    if any(w[:PAD_LEFT]) or any(w[PAD_LEFT + NATIVE_T:]):
        raise ValueError(f"clip {clip}: padding frames are not -1")
    return [x & 0xFFFF for x in w[PAD_LEFT:PAD_LEFT + NATIVE_T]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_hex", type=Path)
    ap.add_argument("--clip", type=int, default=0)
    args = ap.parse_args()
    pkt = frame_packet(native_frames(args.input_hex, args.clip))
    print(f"{len(pkt)} bytes, CRC {pkt[-1]:#04x}")
    print(pkt.hex(" "))


if __name__ == "__main__":
    main()
