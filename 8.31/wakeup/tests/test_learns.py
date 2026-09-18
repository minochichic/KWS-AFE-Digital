"""구조가 학습되는지 확인한다. 합성 과제라 빠르고 결정적이다.

  python -m pytest wakeup/tests/test_learns.py -q -s      (8.31/ 에서)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.config import Config
from wakeup.model import WakeupModel
from wakeup.search import search_timing
from wakeup.synthetic import make_batch


def _fit(steps: int = 800, n_train: int = 512, seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.frontend.n_channels = 8
    cfg.head.l1_gate = 0.0          # 먼저 풀 수 있는지만 본다
    cfg.head.min_channels = 0
    cfg.head.temperature = 0.7
    cfg.start.k = 3

    m = WakeupModel(cfg)
    xtr, ytr, _ = make_batch(n_train, cfg.head.tau, seed=seed)
    xte, yte, _ = make_batch(256, cfg.head.tau, seed=seed + 999)

    m.frontend.init_fixed_scale(xtr[:256])
    m.frontend.init_thresholds(xtr[:256])
    # tau 를 추측값으로 두면 무너진다(acc 0.76±0.18). 탐색이 필수다.
    search_timing(m, xtr, ytr)

    opt = torch.optim.Adam([
        {"params": m.head.parameters(), "lr": 3e-2},
        {"params": m.frontend.parameters(), "lr": 3e-3},
    ])
    m.train()
    bs = 64
    for i in range(steps):
        j = torch.randint(0, n_train - bs, (1,)).item()
        out = m.loss(xtr[j:j + bs], ytr[j:j + bs])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if cfg.head.refit_k_every and (i + 1) % cfg.head.refit_k_every == 0:
            m.refit_k(xtr[:256], ytr[:256])

    m.eval()
    wake = m.hard(xte)["wake"]
    acc = (wake == yte).float().mean().item()
    tpr = wake[yte == 1].mean().item()
    fpr = wake[yte == 0].mean().item()
    return m, acc, tpr, fpr


def test_model_learns_the_synthetic_pattern():
    m, acc, tpr, fpr = _fit()
    print(f"\n  경성 판정  acc {acc:.3f}  TPR {tpr:.3f}  FPR {fpr:.3f}")
    print(f"  m(s) = {m.head.used_channels().tolist()}  k = {m.head.k.round().tolist()}")
    assert acc > 0.94, f"합성 과제도 못 푼다 (acc {acc:.3f})"
    assert tpr > 0.95, f"검출률이 낮다 (TPR {tpr:.3f})"
    # k 탐색이 FPR 을 상한(k_max_fpr) 아래로 묶어 주어야 한다
    assert fpr < 0.10, f"오검출이 많다 (FPR {fpr:.3f})"


def test_start_is_found_on_every_positive():
    cfg = Config()
    m = WakeupModel(cfg)
    x, y, _ = make_batch(128, cfg.head.tau, seed=7)
    m.frontend.init_fixed_scale(x)
    m.frontend.init_thresholds(x)
    found = m(x)["found"]
    assert found.float().mean().item() > 0.95, "START 를 자주 놓친다"


def test_l1_trades_resistors_for_accuracy():
    """X 를 늘리는 압력이 실제로 저항을 줄이는지."""
    torch.manual_seed(0)
    cfg = Config()
    cfg.frontend.n_channels = 8
    cfg.head.min_channels = 2
    cfg.head.temperature = 0.7
    cfg.start.k = 3

    counts = {}
    for lam in (0.0, 0.25):
        torch.manual_seed(0)
        cfg.head.l1_gate = lam
        m = WakeupModel(cfg)
        x, y, _ = make_batch(512, cfg.head.tau, seed=0)
        m.frontend.init_fixed_scale(x[:256])
        m.frontend.init_thresholds(x[:256])
        opt = torch.optim.Adam(m.parameters(), lr=3e-2)
        m.train()
        for i in range(200):
            j = torch.randint(0, 448, (1,)).item()
            out = m.loss(x[j:j + 64], y[j:j + 64])
            opt.zero_grad(); out["loss"].backward(); opt.step()
            if (i + 1) % 50 == 0:
                m.refit_k(x[:256], y[:256])
        counts[lam] = m.export()["n_resistors"]

    print(f"\n  저항 수: L1 0 -> {counts[0.0]}개,  L1 0.25 -> {counts[0.25]}개")
    assert counts[0.25] < counts[0.0], "L1 이 저항을 줄이지 못한다"


def test_tau_search_beats_a_guessed_tau():
    """tau 를 고정해 두면 정렬이 어긋나 검출률이 무너진다는 것을 못 박아 둔다."""
    def fit(search: bool, seed: int = 1, steps: int = 400):
        torch.manual_seed(seed)
        cfg = Config()
        cfg.head.l1_gate = 0.0
        cfg.head.min_channels = 0
        cfg.head.temperature = 0.7
        cfg.head.and_temperature = 0.7
        m = WakeupModel(cfg)
        xtr, ytr, _ = make_batch(512, cfg.head.tau, seed=seed)
        xte, yte, _ = make_batch(256, cfg.head.tau, seed=seed + 999)
        m.frontend.init_fixed_scale(xtr[:256])
        m.frontend.init_thresholds(xtr[:256])
        if search:
            search_timing(m, xtr, ytr)
        opt = torch.optim.Adam(m.parameters(), lr=3e-2)
        m.train()
        for i in range(steps):
            j = torch.randint(0, 448, (1,)).item()
            o = m.loss(xtr[j:j + 64], ytr[j:j + 64])
            opt.zero_grad(); o["loss"].backward(); opt.step()
            if (i + 1) % 50 == 0:
                m.refit_k(xtr[:256], ytr[:256])
        m.eval()
        wk = m.hard(xte)["wake"]
        return wk[yte == 1].mean().item()

    guessed, searched = fit(False), fit(True)
    print(f"\n  TPR: 고정 tau {guessed:.3f} -> 탐색 {searched:.3f}")
    assert searched > guessed + 0.15
