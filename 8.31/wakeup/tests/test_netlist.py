"""넷리스트 — 배선 검사, 넷리스트 시뮬 대 gatelevel 대조, 그리고 대조가 실수를 잡는지.

  python -m pytest wakeup/tests/test_netlist.py -q -s      (8.31/ 에서)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.gatelevel import DigitalBoard, scan_segments
from wakeup.netlist import (NetSim, board_from_params, build_netlist,
                            compare_streams, e96_divider, make_streams)

FROZEN = Path(__file__).resolve().parents[2] / "hardware" / "f256_ok_s3.json"


@pytest.fixture(scope="module")
def frozen():
    board = board_from_params(json.loads(FROZEN.read_text(encoding="utf-8")))
    return board, build_netlist(board, vdd=1.8, name="t")


def test_frozen_netlist_passes_wiring_checks(frozen):
    _, nl = frozen
    assert nl.check() == []


def test_frozen_netlist_matches_gatelevel(frozen):
    board, nl = frozen
    r = compare_streams(nl, board, make_streams(board, n=4, T=700), 1.8)
    print(f"\n  {r['frames']}프레임, WAKE {r['wake_frames']}, 불일치 {r['mismatch']}")
    assert r["wake_frames"] > 50, "시험 스트림에서 WAKE 가 거의 안 나와 대조가 무의미하다"
    assert r["mismatch"] == 0


def _mutated(nl, fn):
    m = copy.deepcopy(nl)
    fn(m)
    return m


def test_comparison_catches_a_resistor_on_the_wrong_polarity(frozen):
    """ch6 저항을 CH6 대신 /CH6 에 물리면 반드시 드러나야 한다."""
    board, nl = frozen

    def swap(m):
        p = next(p for p in m.passives if p.b == "SC1" and p.a == "CH6")
        p.a = "/CH6"
    r = compare_streams(_mutated(nl, swap), board, make_streams(board, n=3, T=700), 1.8)
    assert r["mismatch"] > 0


def test_comparison_catches_a_decoder_output_on_the_wrong_pin(frozen):
    """WIN2 의 /D10 을 옆 핀 /D11 로 잘못 꽂은 경우."""
    board, nl = frozen

    def miswire(m):
        for ref, (pt, conn) in m.parts.items():
            if pt == "74LVC30A" and "WIN2" in conn.values():
                for pin, net in conn.items():
                    if net == "/D10":
                        conn[pin] = "/D11"
            if pt == "74LVC138A":
                for pin, net in conn.items():
                    if pin == "/Y3" and conn.get("E3") == "C3":
                        conn[pin] = "/D11"
    r = compare_streams(_mutated(nl, miswire), board,
                        make_streams(board, n=3, T=700), 1.8)
    assert r["mismatch"] > 0


def test_comparison_catches_a_wrong_divider(frozen):
    """k 를 하나 낮춘 것과 같은 분압(=허용이 느슨해짐)."""
    board, nl = frozen

    def loosen(m):
        top = next(p for p in m.passives if p.a == "VDD" and p.b == "VTH1")
        top.ohms *= 2.0
    r = compare_streams(_mutated(nl, loosen), board, make_streams(board, n=3, T=700), 1.8)
    assert r["mismatch"] > 0


def test_power_on_reset_clears_a_dirty_start(frozen):
    """전원 투입 때 latch 가 전부 1 이어도 POR 이 지워 WAKE 가 뜨지 않는다."""
    _, nl = frozen
    out = NetSim(nl, 1.8, init_ff=1, init_count=9).run(torch.zeros(8, 200))
    assert out["wake_seq"].sum() == 0 and out["run"].sum() == 0


def test_generator_handles_other_windows_and_timeouts():
    """그룹 1·2 디코드, 0 이 섞인 timeout(45 = 101101) -- 생성 경로가 다른 설정."""
    g = torch.Generator().manual_seed(1)
    tmpl = torch.randint(-1, 2, (2, 8), generator=g)
    tmpl[:, 0] = 1
    m = [int((r != 0).sum()) for r in tmpl]
    board = DigitalBoard(template=tmpl, k=[max(2, x - 1) for x in m], tau=[9, 20],
                         timeout=45, start_k=1, n_decoders=3, match_window=1)
    nl = build_netlist(board, vdd=1.8, name="alt")
    assert nl.check() == []
    r = compare_streams(nl, board, make_streams(board, n=3, T=600, seed=3), 1.8)
    print(f"\n  대안 설정: WAKE {r['wake_frames']}, 불일치 {r['mismatch']}")
    assert r["mismatch"] == 0


def test_e96_divider_is_close_to_the_ideal_ratio():
    for ratio in (5.5 / 8, 4.5 / 6, 0.5, 0.9):
        r7, r8 = e96_divider(ratio)
        assert abs(r8 / (r7 + r8) - ratio) < 2e-3


def test_rearm_restarts_in_the_same_frame():
    """채널이 계속 켜져 있으면 timeout 프레임에서 곧바로 다음 구간이 시작된다."""
    b = DigitalBoard(template=torch.tensor([[1]]), k=[1], tau=[1], timeout=5,
                     start_k=1, match_window=1)
    x = torch.ones(1, 1, 18)
    starts, _ = scan_segments(b, x)
    assert starts.tolist() == [0, 5, 10, 15]
