from pathlib import Path

import pytest

from export.wake_link import (
    NATIVE_T,
    Result,
    crc8,
    frame_packet,
    native_frames,
    parse_result,
    result_packet,
)

ROOT = Path(__file__).resolve().parents[1]
SELFTEST = ROOT / "rtl" / "gen" / "bd_base" / "selftest" / "input.hex"


def test_crc8_is_crc8_smbus():
    # the published check value -- rtl/kws_crc8.v and tb_wake_top pin the same
    assert crc8(b"123456789") == 0xF4
    assert crc8(b"") == 0


def test_frame_packet_layout():
    frames = list(range(NATIVE_T))
    frames[0] = 0xBEEF
    pkt = frame_packet(frames)
    assert len(pkt) == 2 + 2 * NATIVE_T + 1
    assert pkt[0] == 0xA5 and pkt[1] == NATIVE_T
    assert pkt[2:4] == bytes([0xEF, 0xBE])            # little-endian, ch0 in bit 0
    assert crc8(pkt[:-1]) == pkt[-1]
    assert crc8(pkt) == 0                             # appending the CRC zeroes it


def test_frame_packet_rejects_bad_frames():
    with pytest.raises(ValueError):
        frame_packet([0x10000])
    with pytest.raises(ValueError):
        frame_packet([])


@pytest.mark.parametrize("margin", [0, 27918, -8612, -(1 << 21), (1 << 21) - 1])
def test_result_roundtrip(margin):
    r = Result(11, 5, margin)
    assert parse_result(result_packet(r)) == r


def test_parse_result_rejects_corruption():
    pkt = bytearray(result_packet(Result(5, 5, 100)))
    pkt[3] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        parse_result(bytes(pkt))
    with pytest.raises(ValueError, match="sync"):
        parse_result(b"\x00" + result_packet(Result(5, 5, 1))[1:])


@pytest.mark.skipif(not SELFTEST.is_file(), reason="self-test export not present")
def test_native_frames_from_the_selftest_export():
    f = native_frames(SELFTEST, 0)
    assert len(f) == NATIVE_T and all(0 <= x < 1 << 16 for x in f)
    assert any(f)                                     # a real clip, not padding
