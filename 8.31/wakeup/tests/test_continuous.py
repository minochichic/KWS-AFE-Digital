"""연속 오디오 평가 — 사건 귀속 로직.

클립 단위 FPR 은 "단어 하나당 확률"이라 시간당 횟수가 아니다. 여기서 재는
것이 계획서 3.4·3.5 가 약속한 FA/h 다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.continuous import SR, build_stream, evaluate_stream


class _FakeBoard:
    """지정한 프레임에서만 WAKE 를 내는 가짜 보드."""

    def __init__(self, frames):
        self.frames = set(frames)

    def run(self, feats):
        T = feats.shape[2]
        w = torch.zeros(1, T)
        for f in self.frames:
            if 0 <= f < T:
                w[0, f] = 1.0
        return {"wake_seq": w}


def _feats(T):
    return torch.zeros(1, 4, T)


def test_wake_inside_the_window_counts_as_a_detection():
    onsets = [int(2.0 * SR)]                  # 2초 지점에 대상
    r = evaluate_stream(_FakeBoard([210]), _feats(6000), onsets)   # 2.10 s
    assert r["tpr"] == 1.0 and r["n_false"] == 0


def test_wake_outside_the_window_is_a_false_alarm():
    onsets = [int(2.0 * SR)]
    r = evaluate_stream(_FakeBoard([500]), _feats(6000), onsets)   # 5.00 s
    assert r["tpr"] == 0.0 and r["n_false"] == 1


def test_two_wakes_on_one_target_credit_once_and_the_rest_are_false():
    """한 발화가 WAKE 를 두 번 내면 하나만 검출로 치고 나머지는 오검출이다."""
    onsets = [int(2.0 * SR)]
    r = evaluate_stream(_FakeBoard([205, 240]), _feats(6000), onsets)
    assert r["tpr"] == 1.0
    assert r["n_wakes"] == 2 and r["n_false"] == 1


def test_fa_per_hour_scales_with_stream_length():
    """같은 오검출 수라도 스트림이 길면 시간당 값은 줄어야 한다."""
    short = evaluate_stream(_FakeBoard([100]), _feats(6000), [])    # 60 s
    long_ = evaluate_stream(_FakeBoard([100]), _feats(36000), [])   # 360 s
    assert short["fa_per_hour"] > long_["fa_per_hour"]
    assert abs(short["fa_per_hour"] - 60.0) < 1e-6      # 1회 / 60초 = 60/h


def test_build_stream_records_real_target_positions():
    waves = torch.randn(20, SR) * 0.05
    words = ["sheila" if i % 4 == 0 else "no" for i in range(20)]
    st, on = build_stream(waves, words, "sheila", minutes=0.5,
                          targets_per_min=10.0, seed=0)
    assert st.numel() > 0 and len(on) >= 2
    # 기록된 위치에 실제로 무언가 있어야 한다 (무음 구간이 아니라)
    for o in on[:3]:
        seg = st[o:o + SR]
        assert seg.abs().max() > 0, "대상 위치가 무음이다"


def test_grid_search_matches_a_full_state_machine_run():
    """구간 스캔 + k 격자 계산이 상태기계를 끝까지 돌린 것과 같아야 한다.

    CLR 은 timeout 만 보고 k 와 무관하므로 구간 분할이 k 에 의존하지 않는다는
    것이 이 계산의 전제다. 그 전제가 깨지면 동작점 선택이 통째로 틀린다.
    """
    from wakeup.config import Config
    from wakeup.model import WakeupModel
    from wakeup.gatelevel import board_from_model, scan_segments
    from wakeup.search import search_timing
    from wakeup.synthetic import make_batch
    from wakeup.continuous import stream_features, evaluate_stream, sweep_k

    torch.manual_seed(0)
    cfg = Config()
    cfg.frontend.n_channels = 8
    cfg.frontend.init_on_rate = 0.15
    cfg.head.n_states = 2
    cfg.head.tau = [8, 16]
    cfg.head.timeout = 28
    cfg.head.match_window = 2
    m = WakeupModel(cfg)
    x, y, _ = make_batch(256, [8, 16], seed=0)
    m.frontend.init_fixed_scale(x)
    m.frontend.init_thresholds(x, on_rate=0.15)
    search_timing(m, x, y)
    m.eval()

    words = ["TGT" if v > 0.5 else "oth" for v in y]
    st, on = build_stream(x, words, "TGT", minutes=1.5, targets_per_min=10.0,
                          seed=0)
    feats = stream_features(m, st)
    board = board_from_model(m)
    starts, mx = scan_segments(board, feats)
    assert len(starts) > 5, "계측 구간이 너무 적어 비교가 무의미하다"

    e = m.export()
    k = [int(v) for v in e["k"]]
    rows = sweep_k(starts, mx, on, [int(v) for v in e["m"]],
                   frames=feats.shape[2], tau_last=int(board.tau[-1]))
    grid = [r for r in rows if r["k"] == k][0]
    full = evaluate_stream(board, feats, on)
    assert grid["n_wakes"] == full["n_wakes"], (
        f"WAKE 수가 다르다: 격자 {grid['n_wakes']} vs 전체 {full['n_wakes']}")
    assert abs(grid["tpr"] - full["tpr"]) < 1e-9
    assert abs(grid["fa_per_hour"] - full["fa_per_hour"]) < 1e-6
