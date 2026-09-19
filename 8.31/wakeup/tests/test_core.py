"""코어 단위 테스트. 학습 전에 회로 등가성부터 확인한다.

  python -m pytest wakeup/tests/test_core.py -q       (8.31/ 에서)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.config import Config, HeadConfig, FrontendConfig
from wakeup.head import TemplateHead
from wakeup.model import WakeupModel, detect_start, gather_states
from wakeup.ste import step_ste, sign_ste, gate_ste


# --------------------------------------------------------------------- STE
def test_step_ste_forward_is_a_real_step():
    x = torch.tensor([-1.0, -1e-9, 0.0, 1e-9, 1.0])
    assert torch.equal(step_ste(x), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_ste_gradient_flows_inside_clip_and_stops_outside():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    sign_ste(x, clip=1.0).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_step_ste_gradient_is_half_of_sign():
    # step = (sign+1)/2 이므로 통과 기울기도 절반이다. 문턱의 실효 학습률이
    # 형판 가중치의 절반이라는 뜻 -- lr_threshold 를 따로 두는 이유.
    x = torch.tensor([-0.5, 0.5], requires_grad=True)
    step_ste(x, clip=1.0).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------- START 검출
def test_detect_start_finds_first_frame_over_k():
    x = torch.zeros(1, 4, 10)
    x[0, :2, 3] = 1.0          # 2채널만 켜짐 -> k=3 에서는 미달
    x[0, :3, 6] = 1.0          # 3채널 -> 여기가 START
    start, found = detect_start(x, k=3)
    assert found.item() and start.item() == 6


def test_detect_start_reports_not_found_on_silence():
    start, found = detect_start(torch.zeros(2, 4, 10), k=1)
    assert not found.any()


# ------------------------------------------------------------------- gather
def test_gather_states_picks_the_right_columns():
    B, C, T = 2, 3, 20
    x = torch.arange(B * C * T, dtype=torch.float32).reshape(B, C, T)
    start = torch.tensor([0, 5])
    tau = torch.tensor([1, 4])
    got = gather_states(x, start, tau)                 # [B, S, W, C], W=1
    assert got.shape == (2, 2, 1, 3)
    assert torch.equal(got[0, 0, 0], x[0, :, 1])
    assert torch.equal(got[1, 1, 0], x[1, :, 9])       # start 5 + tau 4


def test_gather_clamps_past_the_end():
    x = torch.zeros(1, 2, 10)
    got = gather_states(x, torch.tensor([9]), torch.tensor([5]))
    assert got.shape == (1, 1, 1, 2)       # 예외 없이 마지막 프레임으로 물린다


def test_gather_window_spans_both_sides():
    B, C, T = 1, 2, 20
    x = torch.arange(B * C * T, dtype=torch.float32).reshape(B, C, T)
    got = gather_states(x, torch.tensor([5]), torch.tensor([4]), window=1)
    assert got.shape == (1, 1, 3, 2)
    for j, t in enumerate((8, 9, 10)):                 # start 5 + tau 4 ± 1
        assert torch.equal(got[0, 0, j], x[0, :, t])


def test_match_window_is_an_or_over_time():
    """창 안에서 한 번이라도 맞으면 통과. 정렬 오차를 흡수하는 장치다."""
    M = torch.tensor([[1, 1]], dtype=torch.float32)
    h = _head_with(M, k=[2.0])
    # 창의 한가운데는 틀리고, 한쪽 끝에서만 맞는다
    xs = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]])   # [1,1,3,2]
    assert h.hard(xs)["wake"].item() == 1.0
    # 창 어디에서도 안 맞으면 탈락
    ng = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]])
    assert h.hard(ng)["wake"].item() == 0.0


def test_window_of_one_matches_the_no_window_path():
    M = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)
    h = _head_with(M, k=[3.0])
    flat = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])          # [1,1,4]
    win = flat.unsqueeze(2)                                 # [1,1,1,4]
    assert h(flat)["count"].item() == h(win)["count"].item()


# ------------------------------------------------------------------- 형판
def _head_with(M, k, temperature=0.5, and_temperature=0.5):
    """형판을 직접 심어 둔 head. M: [S, C] in {-1,0,+1}."""
    S, C = M.shape
    cfg = HeadConfig(n_states=S, tau=list(range(1, S + 1)), timeout=S + 5,
                     temperature=temperature, and_temperature=and_temperature,
                     l1_gate=0.0, min_channels=0)
    h = TemplateHead(cfg, C)
    with torch.no_grad():
        h.gate_logit.copy_(torch.where(M != 0, 1.0, -1.0))
        h.weight.copy_(torch.where(M >= 0, 1.0, -1.0))
        h.k.copy_(torch.tensor(k, dtype=torch.float32))
    return h


def test_ternary_roundtrip():
    M = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)
    h = _head_with(M, k=[3.0])
    g, s = h.ternary()
    assert torch.equal((g * s).squeeze(0), M.squeeze(0))
    assert h.used_channels().item() == 3          # X 하나는 안 센다


def test_match_count_ignores_dont_care():
    #      c0 요구1, c1 요구0, c2 = X, c3 요구1
    M = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)
    h = _head_with(M, k=[3.0])
    #        c0=1(맞음) c1=0(맞음) c2=1(무시) c3=1(맞음)  -> count 3
    x = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])
    assert h(x)["count"].item() == pytest.approx(3.0)
    #        c3 을 틀리게 -> count 2
    x2 = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    assert h(x2)["count"].item() == pytest.approx(2.0)


def test_hard_decision_matches_count_ge_k():
    M = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)
    h = _head_with(M, k=[3.0])
    ok = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])       # count 3 >= 3
    ng = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])       # count 2 <  3
    assert h.hard(ok)["wake"].item() == 1.0
    assert h.hard(ng)["wake"].item() == 0.0


def test_and_over_states_needs_every_state():
    M = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    h = _head_with(M, k=[2.0, 2.0])
    both = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    one = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    assert h.hard(both)["wake"].item() == 1.0
    assert h.hard(one)["wake"].item() == 0.0


def test_soft_and_agrees_with_hard_at_low_temperature():
    M = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    h = _head_with(M, k=[2.0, 2.0], temperature=0.02, and_temperature=0.02)
    both = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    one = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    assert torch.sigmoid(h(both)["wake_logit"]).item() > 0.99
    assert torch.sigmoid(h(one)["wake_logit"]).item() < 0.01


def test_export_gives_circuit_constants():
    M = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)
    h = _head_with(M, k=[3.0])
    e = h.export(vdd=1.8)
    assert torch.equal(e["M"], M.to(torch.int8))
    assert e["m"].tolist() == [3] and e["k"].tolist() == [3]
    # V_TH = 1.8 * (3 - 0.5) / 3
    assert e["V_TH"].item() == pytest.approx(1.8 * 2.5 / 3)
    # 여유 = 한 채널 몫의 절반
    assert e["margin_V"].item() == pytest.approx(1.8 / 6)
    assert e["n_resistors"] == 3


def test_l1_pushes_gates_off():
    cfg = HeadConfig(n_states=1, tau=[1], timeout=6, l1_gate=1.0, min_channels=0)
    h = TemplateHead(cfg, 4)
    before = h.used_channels().sum().item()
    opt = torch.optim.SGD(h.parameters(), lr=0.5)
    for _ in range(30):
        opt.zero_grad()
        h.penalties()["l1"].backward()
        opt.step()
    assert h.used_channels().sum().item() < before


# ------------------------------------------------------------------- 설정
def test_config_rejects_timeout_before_last_state():
    with pytest.raises(ValueError, match="timeout"):
        HeadConfig(n_states=2, tau=[4, 10], timeout=8).validate(FrontendConfig())


def test_config_rejects_tau_zero():
    with pytest.raises(ValueError, match=r"tau\[0\]"):
        HeadConfig(n_states=2, tau=[0, 10], timeout=20).validate(FrontendConfig())


def test_config_rejects_timeout_past_the_six_bit_decode():
    with pytest.raises(ValueError, match="63"):
        HeadConfig(n_states=1, tau=[10], timeout=80).validate(FrontendConfig())


def test_config_rejects_per_clip_normalization():
    c = Config()
    c.frontend.normalize = "minmax"
    with pytest.raises(ValueError, match="fixed"):
        c.validate()


# ------------------------------------------------------------------- 통합
def test_model_runs_end_to_end_and_gradients_reach_everything():
    cfg = Config()
    cfg.frontend.n_channels = 8
    m = WakeupModel(cfg)
    wave = torch.randn(4, 16000) * 0.1
    m.frontend.init_fixed_scale(wave)
    m.frontend.init_thresholds(wave)

    m.train()
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = m.loss(wave, y)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    for name in ("frontend.threshold", "head.weight", "head.gate_logit", "head.k"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_hard_path_produces_binary_wake():
    cfg = Config()
    m = WakeupModel(cfg)
    wave = torch.randn(4, 16000) * 0.1
    m.frontend.init_fixed_scale(wave)
    m.frontend.init_thresholds(wave)
    w = m.hard(wave)["wake"]
    assert set(w.unique().tolist()) <= {0.0, 1.0}


def test_frontend_output_is_binary_and_right_shape():
    cfg = Config()
    m = WakeupModel(cfg)
    wave = torch.randn(3, 16000) * 0.1
    m.frontend.init_fixed_scale(wave)
    x = m.features(wave)
    assert x.shape == (3, cfg.frontend.n_channels, cfg.frontend.native_T)
    assert set(x.unique().tolist()) <= {0.0, 1.0}


def test_frontend_refuses_before_scale_init():
    m = WakeupModel(Config())
    with pytest.raises(RuntimeError, match="init_fixed_scale"):
        m.features(torch.randn(2, 16000))


def test_checkpoint_round_trip():
    """체크포인트 저장/로딩. 30에폭을 돌린 뒤 여기서 터진 적이 있다 --
    Frontend.state_dict() 를 재정의해 키를 접두사 없이 끼워 넣었더니 부모의
    공유 dict 에 박혀 "Unexpected key(s): _scale_ready" 가 났다."""
    cfg = Config()
    a = WakeupModel(cfg)
    wave = torch.randn(4, 16000) * 0.1
    a.frontend.init_fixed_scale(wave)
    a.frontend.init_thresholds(wave)

    b = WakeupModel(Config())
    b.load_state_dict(a.state_dict())          # strict=True
    assert bool(b.frontend.scale_ready), "스케일 준비 상태가 안 넘어왔다"
    assert torch.equal(a.hard(wave)["wake"], b.hard(wave)["wake"])
