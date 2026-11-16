"""experiments/score_live.py: a run's answer on logged board bits, grouped by V_thr."""
from __future__ import annotations

import json

import numpy as np
import torch

from data.afe import AFEFrontend
from experiments.afe_clip import classify, load, words_of
from experiments.score_live import load_scored, run_levels, score
from models.binary_matchboxnet import BinaryMatchboxNet
from train.config import load_config


def test_score_matches_classify_and_skips_unsettled(tmp_path):
    from data.speech_commands import KEYWORDS
    names = KEYWORDS + ["silence", "unknown"]
    torch.manual_seed(0)
    cfg = load_config("configs/base.yaml")
    run = tmp_path / "run"
    run.mkdir()
    cfg.save(run / "config.yaml")
    torch.save({"model": BinaryMatchboxNet(cfg.model).state_dict(),
                "afe": AFEFrontend(cfg.afe).state_dict()}, run / "best.pt")
    (run / "vthr_levels.csv").write_text("ch,vthr_mV\n" + "".join(f"{c},{900 + 16 * (c % 7):g}\n"
                                                                  for c in range(16)))
    rng = np.random.default_rng(0)
    bits = (rng.random((12, 100, 16)) < 0.3).astype(np.uint8)
    model = load(run)[0]
    pred, _ = classify(model, bits)
    log = tmp_path / "live.jsonl"
    with log.open("w") as f:
        for i, b in enumerate(bits):
            # half labelled with the model's own answer, half with another word
            lab = names[int(pred[i])] if i % 2 == 0 else names[(int(pred[i]) + 1) % 12]
            f.write(json.dumps({"ref": f"a{i}", "label": lab, "clip": words_of(b),
                                "vthr_mv": [900.0] * 16, "fpga": lab}) + "\n")
        f.write(json.dumps({"ref": None, "label": "yes", "clip": words_of(bits[0])}) + "\n")
    recs = load_scored([log])
    assert len(recs) == 12                                   # the unsettled one is dropped
    s = score(recs, model, names)
    assert s.tolist() == [1.0, 0.0] * 6
    assert run_levels(run) == tuple(900.0 + 16 * (c % 7) for c in range(16))
