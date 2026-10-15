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
    # 종전엔 searched > guessed + 0.15 를 요구했다. 그 격차는 문턱 초기화
    # 기본값이 0.0(채널 평균, 켜짐률 57%)이던 시절의 것이다. 기본값을 측정값
    # 0.15 로 고치자 합성 과제에서는 고정 tau 도 1.000 이 나온다 -- "tau 를
    # 찍어도 된다" 가 아니라, 당시 붕괴의 상당 부분이 tau 가 아니라 문턱
    # 탓이었다는 뜻이다(README 의 ste_clip 항목과 같은 결론). 실제 음성에서는
    # tau 탐색이 여전히 결정적이므로 탐색을 지운 것은 아니다.
    assert searched >= guessed - 1e-6, "탐색이 고정 tau 보다 나빠졌다"
    assert searched > 0.9, f"탐색 TPR 이 {searched:.3f} 로 너무 낮다"


def test_tau_upper_bound_comes_from_the_decoder_not_the_current_timeout():
    """tau 상한은 디코더 제약(63 - min_wake_frames)에서만 와야 한다.
    탐색 전의 잠정 timeout(CLI 기본 28)을 함께 쓰면 상한이 23 으로 묶여,
    7.8125 ms 에서 180 ms -- 단어의 뒷부분을 볼 수 없게 된다."""
    import inspect
    from wakeup import search
    code = [l.split("#")[0] for l in
            inspect.getsource(search.search_tau).splitlines()]
    line = [l for l in code if "max_off = min(" in l][0]
    assert "timeout" not in line, f"잠정 timeout 이 tau 상한을 묶고 있다: {line}"
    assert "63 - mw" in line


def test_pick_offsets_says_so_when_it_cannot_keep_the_gap(capsys):
    from wakeup.search import pick_offsets
    auc = torch.rand(24)
    sel = pick_offsets(auc, 2, min_gap=20, min_tau=1, max_tau=23)
    assert len(sel) == 2
    out = capsys.readouterr().out
    if abs(sel[0] - sel[1]) < 20:
        assert "경고" in out, "간격을 못 지켰는데 조용히 넘어갔다"


def test_tau_search_rejects_offsets_that_kill_a_state():
    """tau 탐색이 한 상태를 죽이고(k<=1) 나머지 하나로 점수를 내면 안 된다.
    tau 상한을 푼 뒤 5시드 중 3회가 tau2=445 ms 로 가서 s2 가 음성의 92% 를
    통과시켰다 -- 비교기 하나가 통째로 낭비다."""
    import inspect
    from wakeup import search
    code = [l.split("#")[0] for l in
            inspect.getsource(search.search_tau).splitlines()]
    body = "\n".join(code)
    assert "(k <= 1).any()" in body, "죽은 상태를 벌하는 규칙이 없다"
