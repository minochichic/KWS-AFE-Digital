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
    r = evaluate_stream(_FakeBoard([210]), _feats(6000), onsets, frame_ms=10.0)   # 2.10 s
    assert r["tpr"] == 1.0 and r["n_false"] == 0


def test_wake_outside_the_window_is_a_false_alarm():
    onsets = [int(2.0 * SR)]
    r = evaluate_stream(_FakeBoard([500]), _feats(6000), onsets, frame_ms=10.0)   # 5.00 s
    assert r["tpr"] == 0.0 and r["n_false"] == 1


def test_two_wakes_on_one_target_credit_once_and_the_rest_are_false():
    """한 발화가 WAKE 를 두 번 내면 하나만 검출로 치고 나머지는 오검출이다."""
    onsets = [int(2.0 * SR)]
    r = evaluate_stream(_FakeBoard([205, 240]), _feats(6000), onsets, frame_ms=10.0)
    assert r["tpr"] == 1.0
    assert r["n_wakes"] == 2 and r["n_false"] == 1


def test_fa_per_hour_scales_with_stream_length():
    """같은 오검출 수라도 스트림이 길면 시간당 값은 줄어야 한다."""
    short = evaluate_stream(_FakeBoard([100]), _feats(6000), [], frame_ms=10.0)    # 60 s
    long_ = evaluate_stream(_FakeBoard([100]), _feats(36000), [], frame_ms=10.0)   # 360 s
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
    rows = sweep_k(starts, mx, on, [int(v) for v in e["m"]], frame_ms=10.0,
                   frames=feats.shape[2], tau_last=int(board.tau[-1]))
    grid = [r for r in rows if r["k"] == k][0]
    full = evaluate_stream(board, feats, on, frame_ms=10.0)
    assert grid["n_wakes"] == full["n_wakes"], (
        f"WAKE 수가 다르다: 격자 {grid['n_wakes']} vs 전체 {full['n_wakes']}")
    assert abs(grid["tpr"] - full["tpr"]) < 1e-9
    assert abs(grid["fa_per_hour"] - full["fa_per_hour"]) < 1e-6


def test_split_utterances_finds_one_clip_per_word():
    """발화 사이 간격이 1.3초인 녹음에서 발화당 하나로 떨어져야 한다.

    min_gap 이 짧으면 한 발화가 둘로 쪼개진다 -- "sheila" 는 /sh/ 와 뒤 모음
    사이에 에너지 골짜기가 있어 0.4초로 두면 10발화가 20조각이 됐다.
    """
    from wakeup.continuous import split_utterances
    torch.manual_seed(0)
    sr, gap = SR, 1.3
    w = torch.randn(int(sr * 14)) * 0.002
    onsets = []
    for i in range(10):
        a = int(sr * (1.0 + i * gap))
        onsets.append(a)
        # 앞쪽 큰 소리 + 짧은 골짜기 + 뒤쪽 작은 소리 (sheila 모양)
        w[a:a + int(sr * 0.15)] += torch.randn(int(sr * 0.15)) * 0.5
        b = a + int(sr * 0.45)
        w[b:b + int(sr * 0.2)] += torch.randn(int(sr * 0.2)) * 0.2

    clips = split_utterances(w)
    assert len(clips) == 10, f"발화당 하나가 아니다: {len(clips)}개"
    assert all(c.numel() == sr for c in clips)
    # 짧은 간격으로 두면 쪼개지는 것도 확인 (기본값이 왜 0.8 인지)
    assert len(split_utterances(w, min_gap_s=0.3)) > 10


def test_enrollment_clips_only_enter_the_train_split():
    """섞어 쓰는 모드에서 녹음이 val/test 에 새면 자기 목소리로 자기를 평가한다."""
    from wakeup.data import split_extra
    ex = [torch.zeros(4) + i for i in range(8)]
    g = split_extra(ex, enroll_only=False)
    assert len(g["train"]) == 8
    assert g["val"] is None and g["test"] is None


def test_speaker_dependent_mode_splits_the_recording_three_ways():
    """val 에 양성이 있어야 체크포인트를 고를 수 있고, test 는 그 val 과도
    달라야 한다. 한때 test 에 val 을 그대로 넣어 체크포인트를 고른 데이터로
    시험 점수를 매겼다 -- TPR 0.877 이 나왔지만 실제 녹음 검출은 21발화 중
    1개였다."""
    from wakeup.data import split_extra
    ex = [torch.zeros(4) + i for i in range(20)]
    g = split_extra(ex, enroll_only=True, verbose=False)
    sets = {k: {float(w[0]) for w in g[k]} for k in ("train", "val", "test")}
    assert all(len(v) > 0 for v in sets.values())
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (sets[a] & sets[b]), f"{a} 와 {b} 가 겹친다"
    assert set().union(*sets.values()) == {float(i) for i in range(20)}


def test_speaker_dependent_mode_drops_the_dataset_positives():
    from wakeup.data import TargetVsRest
    items = [(torch.randn(SR) * 0.05, "sheila" if i < 100 else "no")
             for i in range(600)]
    extra = [torch.randn(SR) * 0.05 for _ in range(10)]
    ds = TargetVsRest(items, [], "sheila", "train", extra_pos=extra,
                      enroll_only=True, silence_fraction=0.0)
    assert len(ds.pos) == 0, "데이터셋 양성이 남아 있다"
    assert ds.n_extra == 100, "양성 개수는 종전과 같아야 배치 구성이 안 바뀐다"
    w, y = ds[0]
    assert float(y) == 1.0 and w.numel() == SR


def test_enrollment_is_oversampled_to_the_requested_fraction():
    from wakeup.data import TargetVsRest
    items = [(torch.randn(SR) * 0.05, "sheila" if i < 100 else "no")
             for i in range(600)]
    extra = [torch.randn(SR) * 0.05 for _ in range(10)]
    ds = TargetVsRest(items, [], "sheila", "train", extra_pos=extra,
                      enroll_frac=0.3, silence_fraction=0.0)
    total_pos = len(ds.pos) + ds.n_extra
    frac = ds.n_extra / total_pos
    assert abs(frac - 0.3) < 0.02, f"녹음 비율이 {frac:.3f} (목표 0.30)"
    # 반복해서 꺼내도 10개 원본을 돌려 쓴다
    i = len(ds.pos)
    a, ya = ds[i]
    assert float(ya) == 1.0 and a.numel() == SR


def test_recorded_negatives_reach_the_requested_fraction():
    """양성만 시연자 녹음이고 음성이 전부 데이터셋이면, 단어가 아니라 녹음
    채널을 배우는 게 더 쉽다. 같은 마이크의 음성을 섞어 그 길을 막는다."""
    from wakeup.data import TargetVsRest
    items = [(torch.randn(SR) * 0.05, "sheila" if i < 100 else "no")
             for i in range(600)]
    pos = [torch.randn(SR) * 0.05 for _ in range(10)]
    neg = [torch.randn(SR) * 0.05 for _ in range(10)]
    ds = TargetVsRest(items, [], "sheila", "train", extra_pos=pos,
                      enroll_only=True, extra_neg=neg, neg_frac=0.5,
                      silence_fraction=0.0)
    assert ds.n_extra_neg == len(ds.neg), "녹음 음성이 절반이 아니다"
    # 마지막 구간이 녹음 음성이고 라벨이 0 이다
    w, y = ds[len(ds) - 1]
    assert float(y) == 0.0 and w.numel() == SR
    assert len(ds) == ds.n_extra + len(ds.neg) + ds.n_extra_neg


def test_frame_grid_must_be_an_integer_multiple_of_the_stft_hop():
    """stream_features 가 ratio = round(win/hop) 으로 max-pool 한다. 비정수면
    조용히 엉뚱한 격자가 된다 -- 7.8125/10 은 round 가 1 이라 10 ms 격자."""
    import pytest
    from wakeup.config import Config
    c = Config()
    c.frontend.envelope_win_ms = 7.8125      # hop 은 10.0 그대로
    with pytest.raises(ValueError, match="정수배"):
        c.validate()
    c.frontend.stft_hop_ms = 7.8125          # 같이 맞추면 통과
    c.validate()


def test_on_rate_default_is_the_measured_value_not_the_broken_one():
    """0.0(채널 평균)은 켜짐률 57% 로 '전부 통과' 를 일으킨 알려진 고장값이다.
    그게 기본값으로 남아 있어서, --on-rate 0.15 로 돌린 확정 구성을 플래그
    없이 재현했을 때 TPR 이 0.575 에서 0.394 로 떨어졌다."""
    from wakeup.config import Config
    assert Config().frontend.init_on_rate == 0.15


def test_continuous_scoring_requires_an_explicit_frame_ms():
    """frame_ms 에 기본값(10.0)이 있던 동안 gain_curve 가 그걸 안 넘겨서,
    모델이 7.8125 ms 로 도는데 채점만 10 ms 격자로 했다. 이벤트 시각이
    1.28배 늘어나 정답 창을 벗어났고 검출률이 55% 에서 4% 로 보였다."""
    import inspect
    from wakeup.continuous import evaluate_stream, sweep_k
    for fn in (evaluate_stream, sweep_k):
        p = inspect.signature(fn).parameters["frame_ms"]
        assert p.default is inspect.Parameter.empty, (
            f"{fn.__name__} 의 frame_ms 에 기본값이 생겼다 — "
            "호출자가 빠뜨려도 조용히 틀린 격자로 채점된다")


def test_frame_rate_error_grows_with_elapsed_time():
    """틀린 frame_ms 의 시각 오차는 **경과 시간에 비례**한다. 정답 창이
    1초 폭이라 스트림 앞부분에서는 티가 안 나고(1초 지점의 오차는 0.28초),
    20분 스트림 끝에서는 수십 초가 어긋나 전부 놓친다. 클립 지표는 멀쩡한데
    연속 검출률만 55% 에서 4% 로 보였던 이유가 이것이다."""
    from wakeup.continuous import sweep_k
    maxcount = torch.ones(1, 1) * 8

    # 1초 지점: 틀린 격자여도 1.28 s 라 창(0.9~1.9 s) 안 — 안 걸린다
    early = sweep_k(torch.tensor([128]), maxcount, [int(1.0 * SR)], [8],
                    frames=200000, frame_ms=10.0)
    assert early[-1]["tpr"] == 1.0, "앞부분에서는 틀린 격자도 통과해야 한다"

    # 100초 지점: 틀린 격자는 128 s 라 창(99.9~100.9 s) 밖
    late_ok = sweep_k(torch.tensor([12800]), maxcount, [int(100.0 * SR)], [8],
                      frames=200000, frame_ms=1000.0 / 128)
    late_bad = sweep_k(torch.tensor([12800]), maxcount, [int(100.0 * SR)], [8],
                       frames=200000, frame_ms=10.0)
    assert late_ok[-1]["tpr"] == 1.0, "맞는 격자인데 귀속되지 않았다"
    assert late_bad[-1]["tpr"] == 0.0, "틀린 격자인데 귀속됐다"
