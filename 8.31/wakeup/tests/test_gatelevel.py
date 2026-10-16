"""게이트 수준 회로가 학습 모델과 일치하는지 — 스키매틱을 그리기 전의 관문.

  python -m pytest wakeup/tests/test_gatelevel.py -q -s      (8.31/ 에서)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.config import Config
from wakeup.gatelevel import DigitalBoard, board_from_model
from wakeup.model import WakeupModel, detect_start
from wakeup.search import search_timing
from wakeup.synthetic import make_batch


# --------------------------------------------------------------- 최소 단위
def _board(template, k, tau, timeout, start_k=1, n_dec=3):
    return DigitalBoard(template=torch.tensor(template), k=list(k),
                        tau=list(tau), timeout=timeout, start_k=start_k,
                        n_decoders=n_dec)


def test_counter_is_zero_on_the_start_frame():
    """tau 는 START 프레임을 0 으로 세는 상대 시각이어야 한다."""
    b = _board([[1, 1]], k=(2,), tau=(2,), timeout=9)
    x = torch.zeros(1, 2, 12)
    x[0, :, 3] = 1.0          # START 는 프레임 3
    x[0, :, 5] = 1.0          # 3 + tau(2) = 5 에서 형판 일치
    assert b.run(x)["wake"].item() == 1.0


def test_pattern_one_frame_early_does_not_wake():
    b = _board([[1, 1]], k=(2,), tau=(2,), timeout=9)
    x = torch.zeros(1, 2, 12)
    x[0, :, 3] = 1.0
    x[0, :, 4] = 1.0          # 한 프레임 이르다
    assert b.run(x)["wake"].item() == 0.0


def test_timeout_clears_run_and_pass():
    """timeout 뒤에는 PASS 가 지워져 이전 부분 성공이 남지 않는다."""
    b = _board([[1, 1], [1, -1]], k=(2, 2), tau=(1, 3), timeout=5)
    x = torch.zeros(1, 2, 24)
    x[0, :, 2] = 1.0          # START
    x[0, :, 3] = 1.0          # tau1 통과
    #  tau2 는 안 맞음 -> timeout 에서 CLR
    x[0, :, 14] = 1.0         # 두 번째 발화 START
    x[0, :, 15] = 1.0         # tau1 만 통과, 역시 실패
    assert b.run(x)["wake"].item() == 0.0, "CLR 이 안 되어 PASS 가 남았다"


def test_two_decoders_alias_every_64_counts():
    """디코더 2개면 상위 2비트가 안 들어가 64 카운트마다 같은 신호가 또 뜬다."""
    b2 = DigitalBoard(template=torch.tensor([[1]]), k=[1], tau=[17],
                      timeout=100, start_k=1, n_decoders=2)
    assert b2.decode_mask == 0b111111
    assert b2.decode_hits(81, 17), "17 과 81 은 하위 6비트가 같다"
    assert b2.aliases(), "별칭이 보고되어야 한다"

    b3 = DigitalBoard(template=torch.tensor([[1]]), k=[1], tau=[17],
                      timeout=100, start_k=1, n_decoders=3)
    assert b3.decode_mask == 0xFF
    assert not b3.decode_hits(81, 17)
    assert not b3.aliases(), "디코더 3개면 별칭이 없어야 한다"


def test_two_decoders_make_the_timeout_itself_fire_early():
    """별칭의 실제 피해는 tau 가 아니라 timeout 에서 먼저 터진다.

    timeout 을 63 보다 크게 잡으면 DECODE_TO 자체가 별칭이라 훨씬 이른
    카운트에서 CLR 이 걸린다. 80 은 하위 6비트가 16 이므로 카운트 16 에서
    회로가 IDLE 로 돌아가 버리고, 그 뒤의 어떤 상태도 채점되지 못한다.
    """
    common = dict(template=torch.tensor([[1, 1]]), k=[2], tau=[20],
                  timeout=80, start_k=1)
    x = torch.zeros(1, 2, 60)
    x[0, :, 1] = 1.0                      # START (프레임 1, 카운터 0)
    x[0, :, 1 + 20] = 1.0                 # tau=20 에서 형판 일치

    b2 = DigitalBoard(**common, n_decoders=2)
    assert b2.decode_hits(16, 80), "80 의 하위 6비트는 16 이다"
    assert b2.run(x)["wake"].item() == 0.0, "카운트 16 에서 CLR 되어야 한다"

    b3 = DigitalBoard(**common, n_decoders=3)
    assert b3.run(x)["wake"].item() == 1.0, "디코더 3개면 정상 동작해야 한다"


def test_timeout_under_64_is_safe_with_two_decoders():
    """디코더 2개로 가려면 timeout <= 63 으로 묶으면 된다 — 창이 630 ms 로 제한."""
    common = dict(template=torch.tensor([[1, 1]]), k=[2], tau=[20],
                  timeout=48, start_k=1)
    x = torch.zeros(1, 2, 60)
    x[0, :, 1] = 1.0
    x[0, :, 21] = 1.0
    assert DigitalBoard(**common, n_decoders=2).run(x)["wake"].item() == 1.0
    assert not DigitalBoard(**common, n_decoders=2).aliases()


# ------------------------------------------------------- 학습 모델과의 일치
def _trained(seed=0, steps=400):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.head.temperature = cfg.head.and_temperature = 0.7
    cfg.head.l1_gate, cfg.head.min_channels = 0.05, 3
    m = WakeupModel(cfg)
    xtr, ytr, _ = make_batch(512, cfg.head.tau, seed=seed)
    m.frontend.init_fixed_scale(xtr[:256])
    m.frontend.init_thresholds(xtr[:256])
    search_timing(m, xtr, ytr)
    opt = torch.optim.Adam([{"params": m.head.parameters(), "lr": 3e-2},
                            {"params": m.frontend.parameters(), "lr": 3e-3}])
    m.train()
    for i in range(steps):
        j = torch.randint(0, 448, (1,)).item()
        o = m.loss(xtr[j:j + 64], ytr[j:j + 64])
        opt.zero_grad(); o["loss"].backward(); opt.step()
        if (i + 1) % 50 == 0:
            m.refit_k(xtr[:256], ytr[:256])
    m.eval()
    return m


def test_gate_level_matches_the_trained_model_bit_for_bit():
    """이게 통과해야 스키매틱을 그릴 수 있다."""
    m = _trained()
    board = board_from_model(m, start="delay", n_decoders=3)
    xte, yte, _ = make_batch(256, m.cfg.head.tau, seed=4242)

    with torch.no_grad():
        feat = m.features(xte)
        want = m.hard(xte)["wake"]
    got = board.run(feat)["wake"]

    agree = (got == want).float().mean().item()
    print(f"\n  게이트 수준 vs 학습 모델: {agree*100:.2f}% 일치 "
          f"({int((got != want).sum())}/{len(want)} 불일치)")
    print(f"  tau={list(board.tau)} timeout={board.timeout} "
          f"k={list(board.k)} start_k={board.start_k}")
    assert not board.aliases(), f"디코더 별칭이 있다: {board.aliases()}"
    assert agree == 1.0, "회로와 모델이 다른 함수를 계산한다"


def test_start_detection_agrees_with_the_model():
    m = _trained()
    board = board_from_model(m, start="delay")
    xte, _, _ = make_batch(128, m.cfg.head.tau, seed=77)
    with torch.no_grad():
        feat = m.features(xte)
        _, found = detect_start(feat, m.cfg.start.k)
    assert torch.equal(board.run(feat)["found"].bool(), found)


def test_wake_frame_is_the_last_sample_time():
    """WAKE 는 마지막 상태가 채점되는 프레임에 뜬다 -> 폭 계산의 근거."""
    m = _trained()
    board = board_from_model(m, start="delay")
    xte, yte, _ = make_batch(256, m.cfg.head.tau, seed=4242)
    with torch.no_grad():
        feat = m.features(xte)
        start, _ = detect_start(feat, m.cfg.start.k)
    r = board.run(feat)
    hit = r["wake"] > 0.5
    if hit.any():
        want = start[hit] + board.tau[-1]
        assert torch.equal(r["wake_frame"][hit], want)


def test_gate_level_matches_with_a_match_window():
    """채점 창을 켜도 회로와 모델이 같은 함수를 계산해야 한다.

    창을 켜면 PASS 가 에지 트리거 D 플립플롭이 아니라 셋 우세 래치가 된다.
    회로가 바뀌는 변경이므로 골든 벡터를 다시 확인한다.
    """
    torch.manual_seed(3)
    cfg = Config()
    cfg.head.temperature = cfg.head.and_temperature = 0.7
    cfg.head.match_window = 2
    cfg.head.l1_gate, cfg.head.min_channels = 0.05, 3
    m = WakeupModel(cfg)
    xtr, ytr, _ = make_batch(384, cfg.head.tau, seed=3)
    m.frontend.init_fixed_scale(xtr[:256])
    m.frontend.init_thresholds(xtr[:256])
    search_timing(m, xtr, ytr)
    m.eval()

    board = board_from_model(m, start="delay", n_decoders=3)
    assert board.match_window == 2
    xte, _, _ = make_batch(192, cfg.head.tau, seed=3003)
    with torch.no_grad():
        feat = m.features(xte)
        want = m.hard(xte)["wake"]
    got = board.run(feat)["wake"]
    agree = (got == want).float().mean().item()
    print(f"\n  창 2프레임: {agree*100:.2f}% 일치")
    assert agree == 1.0


def test_window_admits_a_shifted_pattern_that_the_point_sample_misses():
    """창이 정렬 오차를 흡수하는지 — 이 변경의 존재 이유."""
    tmpl = torch.tensor([[1, 1]])
    common = dict(template=tmpl, k=[2], tau=[10], timeout=30, start_k=1)
    x = torch.zeros(1, 2, 40)
    x[0, :, 2] = 1.0            # START (프레임 2, 카운터 0)
    x[0, :, 2 + 10 + 2] = 1.0   # 패턴이 2프레임 늦다
    assert DigitalBoard(**common, match_window=0).run(x)["wake"].item() == 0.0
    assert DigitalBoard(**common, match_window=2).run(x)["wake"].item() == 1.0


# ------------------------------------------- START 시프트 레지스터와 시각 원점
def test_ripple_transients_are_the_trailing_ones_cleared():
    from wakeup.gatelevel import ripple_transients
    assert ripple_transients(11) == [10, 8]       # 1011 -> 1010 -> 1000 -> 1100
    assert ripple_transients(7) == [6, 4, 0]
    assert ripple_transients(4) == []


def test_start_needs_n_consecutive_frames():
    """x3 이면 2프레임 켜짐은 START 가 아니고, 3번째 프레임이 카운트 0 이다."""
    b = DigitalBoard(template=torch.tensor([[1]]), k=[1], tau=[4], timeout=10,
                     start_k=1, start_frames=3, start_mode="causal")
    x = torch.zeros(1, 1, 20)
    x[0, 0, 2:4] = 1.0                    # 2프레임만 -> START 아님
    assert b.run(x)["found"].item() == 0.0
    x[0, 0, 6:9] = 1.0                    # 6,7,8 -> 프레임 8 에서 RUN
    assert b.run(x)["found"].item() == 1.0
    # causal: 디코드는 tau-(N-1) = 2 -> 프레임 8+2 = 10, 학습 시각으로는 6+4 = 10
    x[0, 0, 10] = 1.0
    r = b.run(x)
    assert r["wake"].item() == 1.0 and r["wake_frame"].item() == 10


def test_causal_pulls_the_decode_and_truncates_the_first_window():
    """확정 설정(tau 2/8, 창 ±2, timeout 60, x3)을 causal 로 만들면 창1 앞이 잘린다."""
    b = DigitalBoard(template=torch.tensor([[1, 0], [0, 1]]), k=[1, 1],
                     tau=[2, 8], timeout=60, start_k=1, match_window=2,
                     start_frames=3, start_mode="causal")
    assert b.window_counts(0) == [0, 1, 2]       # 학습 시각 0..4 중 2..4 만 남는다
    assert b.window_counts(1) == [4, 5, 6, 7, 8]
    assert b.timeout_count == 58
    d = DigitalBoard(template=torch.tensor([[1, 0], [0, 1]]), k=[1, 1],
                     tau=[2, 8], timeout=60, start_k=1, match_window=2,
                     start_frames=3, start_mode="delay")
    assert d.window_counts(0) == [0, 1, 2, 3, 4] and d.data_delay == 2


def test_causal_refuses_a_window_entirely_before_confirmation():
    with pytest.raises(ValueError):
        DigitalBoard(template=torch.tensor([[1]]), k=[1], tau=[1], timeout=10,
                     start_k=1, start_frames=3, start_mode="causal")


def test_delay_board_matches_the_model_with_min_frames():
    """delay 구성은 x3 학습 모델과 같은 함수여야 한다 (원점 = 구간 첫 프레임)."""
    torch.manual_seed(5)
    cfg = Config()
    cfg.head.temperature = cfg.head.and_temperature = 0.7
    cfg.head.match_window = 2
    cfg.head.l1_gate, cfg.head.min_channels = 0.05, 3
    m = WakeupModel(cfg)
    xtr, ytr, _ = make_batch(384, cfg.head.tau, seed=5)
    m.frontend.init_fixed_scale(xtr[:256])
    m.frontend.init_thresholds(xtr[:256])
    search_timing(m, xtr, ytr)
    cfg.start.min_frames = 3              # 탐색이 START 조건을 다시 고르므로 뒤에 고정
    m.eval()

    board = board_from_model(m, start="delay", n_decoders=3)
    assert board.start_frames == 3
    xte, _, _ = make_batch(192, cfg.head.tau, seed=5005)
    with torch.no_grad():
        feat = m.features(xte)
        want = m.hard(xte)["wake"]
        _, found = detect_start(feat, cfg.start.k, 3)
    r = board.run(feat)
    assert torch.equal(r["found"].bool(), found)
    agree = (r["wake"] == want).float().mean().item()
    print(f"\n  x3 delay vs 모델: {agree*100:.2f}% 일치, START {found.float().mean():.2f}")
    assert agree == 1.0


def test_single_board_ignores_min_frames():
    """종전 스트림 수치를 재현하는 회로 -- min_frames 를 보지 않는다."""
    cfg = Config()
    cfg.start.min_frames = 3
    m = WakeupModel(cfg)
    assert board_from_model(m, start="single").start_frames == 1
    assert board_from_model(m, start="causal").start_frames == 3
    with pytest.raises(TypeError):
        board_from_model(m)                       # 어느 회로인지 반드시 밝힌다


# ------------------------------------------------------- S 게이팅이 빠지면
def _hazard_x(T, on):
    x = torch.zeros(1, 2, T)
    for f in on:
        x[0, :, f] = 1.0
    return x


def test_idle_counter_zero_sets_pass_without_run_gate():
    """S 에 RUN 이 없으면 IDLE 의 카운트 0 이 창1 로 읽힌다.

    s1 이 '전부 꺼짐' 을 요구하면 조용할 때마다 MATCH1 이 선다 -- 확정 형판
    s1 = 0000 0010 (k=6) 도 무음에서 7개가 맞아 그렇다.
    """
    common = dict(template=torch.tensor([[-1, -1], [1, 1]]), k=[2, 2],
                  tau=[1, 6], timeout=12, start_k=1, match_window=1)
    # 0..9 무음(IDLE), 10 에서 START, 10..12 켜짐(s1 창 실패), 16 에서 s2 일치
    x = _hazard_x(24, on=[10, 11, 12, 16])
    assert DigitalBoard(**common).run(x)["wake"].item() == 0.0
    assert DigitalBoard(**common, gate_run=False).run(x)["wake"].item() == 1.0


def test_ripple_transient_sets_pass_without_clk_gate():
    """S 를 CLK 로 막지 않으면 11->12 에서 지나가는 10, 8 이 창(8..10)으로 읽힌다."""
    common = dict(template=torch.tensor([[-1, -1]]), k=[2], tau=[9],
                  timeout=20, start_k=1, match_window=1)
    # 프레임 0 START. 창(카운트 8..10)에서는 켜져 있어 실패, 카운트 12 에서 꺼짐
    x = _hazard_x(30, on=[0, 8, 9, 10])
    assert DigitalBoard(**common).run(x)["wake"].item() == 0.0
    assert DigitalBoard(**common, gate_clk=False).run(x)["wake"].item() == 1.0
