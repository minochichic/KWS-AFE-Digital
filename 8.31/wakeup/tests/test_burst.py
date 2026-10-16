"""짧은 소리 + 무음 -- 학습 음성과 동작점 검사.

  python -m pytest wakeup/tests/test_burst.py -q      (8.31/ 에서)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakeup.data import TargetVsRest
from wakeup.netlist import board_from_params
from wakeup.train import _click_count

FROZEN = Path(__file__).resolve().parents[2] / "hardware" / "f256_ok_s3.json"


def test_click_count_flags_the_old_operating_point():
    """k=[6,5] 는 무음에서 이미 통과 -> 짧은 소리마다 WAKE. [7,6] 은 0."""
    b = board_from_params(json.loads(FROZEN.read_text(encoding="utf-8")))
    assert _click_count(b, [6, 5]) == 48
    assert _click_count(b, [7, 6]) == 0


def _toy(burst: float):
    g = torch.Generator().manual_seed(0)
    items = [(torch.randn(16000, generator=g) * 0.05, w)
             for w in ("yes", "sheila", "go") * 12]
    return TargetVsRest(items, [torch.randn(48000, generator=g) * 0.01], "sheila",
                        "train", burst_fraction=burst, neg_per_pos=2.0)


def test_burst_negatives_are_short_negatives_and_off_by_default():
    assert _toy(0.0).n_burst == 0, "기본값은 꺼져 있어야 종전 학습이 재현된다"
    ds = _toy(1.0)
    assert ds.n_burst == 12
    first = len(ds) - ds.n_silence - ds.n_burst
    for j in range(first, first + ds.n_burst):
        w, y = ds[j]
        assert float(y) == 0.0
        loud = (w.abs() > 0.02).nonzero().flatten()
        assert len(loud) > 0, "조각이 무음이면 START 도 안 서서 아무것도 못 가르친다"
        span_ms = (loud.max() - loud.min() + 1).item() / 16.0
        assert span_ms <= 100.5, f"조각이 {span_ms:.0f} ms -- 100 ms 를 넘는다"
