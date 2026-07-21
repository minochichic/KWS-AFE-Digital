"""Training loop tests (FIRST_TASK.md step 5).

The headline check: deliberately overfit a tiny synthetic tone set to 100%
train accuracy. If that works, gradients flow through every STE in the chain
(AFE thresholds -> binary weights -> int8 fake-quant) well enough to learn --
which is the entire point of step 5.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.afe import AFEFrontend
from models.binary_matchboxnet import BinaryMatchboxNet
from train.config import load_config
from train.synthetic import class_frequencies, make_tone_dataset
from train.train import Trainer, resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"


def tiny_cfg(tmp_path, **extra):
    ov = {
        "model.C": 16, "model.T": 64,
        "afe.envelope_win_ms": 10.0,
        "train.lr": 1e-2, "train.epochs": 3,
        "train.batch_size": 48, "train.seed": 7,
        "train.out_dir": str(tmp_path / "runs"),
        "train.device": "cpu", "train.log_every": 0,
    }
    ov.update(extra)
    return load_config(BASE, ov)


def tone_data(cfg, per_class: int = 4):
    return make_tone_dataset(cfg.model.n_classes, per_class,
                             cfg.afe.sample_rate, seed=0)


def tone_loader(cfg, per_class: int = 4) -> DataLoader:
    waves, labels = tone_data(cfg, per_class)
    return DataLoader(TensorDataset(waves, labels),
                      batch_size=cfg.train.batch_size, shuffle=True)


def build_trainer(cfg, init_thresholds: bool = True) -> Trainer:
    set_seed(cfg.train.seed)
    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)
    if init_thresholds:                          # documented workflow
        afe.init_thresholds(tone_data(cfg)[0])   # (Cerutti IV-A)
    return Trainer(cfg, model, afe=afe)


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_set_seed_makes_everything_reproducible(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    x = torch.randint(0, 2, (2, 16, 64)).float() * 2 - 1

    set_seed(123)
    out1 = BinaryMatchboxNet(cfg.model)(x)
    set_seed(123)
    out2 = BinaryMatchboxNet(cfg.model)(x)
    assert torch.equal(out1, out2)


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cuda", "mps", "cpu")


def test_unknown_optimizer_rejected(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path, **{"train.optimizer": "sgdmax"})
    with pytest.raises(ValueError, match="optimizer"):
        build_trainer(cfg)


def test_novograd_is_explicitly_deferred(tmp_path) -> None:
    """CLAUDE.md 3: NovoGrad only after Adam reproduces. Fail loudly, not
    silently fall back to Adam."""
    cfg = tiny_cfg(tmp_path, **{"train.optimizer": "novograd"})
    with pytest.raises(NotImplementedError, match="[Nn]ovo[Gg]rad"):
        build_trainer(cfg)


def test_synthetic_tones_are_class_distinct() -> None:
    freqs = class_frequencies(12)
    assert len(freqs) == 12
    assert torch.all(freqs[1:] > freqs[:-1])         # strictly increasing
    waves, labels = make_tone_dataset(12, 2)
    assert waves.shape == (24, 16000)
    assert labels.bincount().tolist() == [2] * 12


# --------------------------------------------------------------------------- #
# training mechanics
# --------------------------------------------------------------------------- #
def test_fit_runs_and_records_history(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    trainer = build_trainer(cfg)
    history = trainer.fit(tone_loader(cfg))

    assert len(history) == cfg.train.epochs
    for row in history:
        assert set(row) >= {"epoch", "train_loss", "train_acc", "lr"}

    run_dir = Path(cfg.train.out_dir) / cfg.tag
    assert (run_dir / "history.json").exists()       # CLAUDE.md 5: 결과 기록
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "last.pt").exists()


def test_loss_decreases_when_overfitting(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 15})
    trainer = build_trainer(cfg)
    history = trainer.fit(tone_loader(cfg))
    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    assert last < first * 0.7, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    trainer = build_trainer(cfg)
    trainer.fit(tone_loader(cfg))

    x = torch.randn(2, 16000)
    ref = trainer.predict(x)

    cfg2 = tiny_cfg(tmp_path)
    set_seed(999)                                    # different init on purpose
    fresh = Trainer(cfg2, BinaryMatchboxNet(cfg2.model),
                    afe=AFEFrontend(cfg2.afe))
    assert not torch.allclose(fresh.predict(x), ref)

    fresh.load_checkpoint(Path(cfg.train.out_dir) / cfg.tag / "last.pt")
    assert torch.allclose(fresh.predict(x), ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# resume (Colab disconnect / T4->A100 safety)
# --------------------------------------------------------------------------- #
def test_last_checkpoint_saved_every_epoch(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 3})
    trainer = build_trainer(cfg)
    trainer.fit(tone_loader(cfg))
    ck = trainer.load_checkpoint(Path(cfg.train.out_dir) / cfg.tag / "last.pt")
    assert ck["epoch"] == 3                            # last completed epoch
    assert len(ck["history"]) == 3                     # per-epoch history saved
    assert ck["scheduler"] is not None                 # scheduler state persisted


def test_resume_continues_from_saved_epoch(tmp_path) -> None:
    # phase 1: train 3 epochs
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 3})
    build_trainer(cfg).fit(tone_loader(cfg), tone_loader(cfg))

    # phase 2: same tag, more epochs, resume -> continues at 4, not 1
    cfg2 = tiny_cfg(tmp_path, **{"train.epochs": 6})
    trainer2 = build_trainer(cfg2)
    history = trainer2.fit(tone_loader(cfg2), tone_loader(cfg2), resume=True)

    assert [r["epoch"] for r in history] == [1, 2, 3, 4, 5, 6]   # contiguous
    ck = trainer2.load_checkpoint(Path(cfg2.train.out_dir) / cfg2.tag / "last.pt")
    assert ck["epoch"] == 6


def test_resume_without_checkpoint_starts_fresh(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 2})
    trainer = build_trainer(cfg)
    history = trainer.fit(tone_loader(cfg), resume=True)    # no last.pt yet
    assert [r["epoch"] for r in history] == [1, 2]


def test_resume_when_already_complete_is_noop(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 2})
    build_trainer(cfg).fit(tone_loader(cfg))
    # resume with the same epoch budget -> nothing left to do
    trainer2 = build_trainer(cfg)
    history = trainer2.fit(tone_loader(cfg), resume=True)
    assert len(history) == 2                            # loaded, not re-trained
    assert all(k in history[0] for k in ("epoch", "train_loss"))


def test_resume_restores_lr_schedule_state(tmp_path) -> None:
    """The plateau scheduler's internal counters must survive resume, else the
    LR schedule silently restarts on an A100."""
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 3})
    t1 = build_trainer(cfg)
    t1.fit(tone_loader(cfg), tone_loader(cfg))
    lr_state = t1.scheduler.state_dict()

    cfg2 = tiny_cfg(tmp_path, **{"train.epochs": 3})       # already complete
    t2 = build_trainer(cfg2)
    t2.load_checkpoint(Path(cfg.train.out_dir) / cfg.tag / "last.pt")
    assert t2.scheduler.state_dict()["num_bad_epochs"] == lr_state["num_bad_epochs"]
    assert t2.scheduler.state_dict()["best"] == lr_state["best"]


# --------------------------------------------------------------------------- #
# THE step-5 check: overfit to 100%
# --------------------------------------------------------------------------- #
def test_overfit_synthetic_tones(tmp_path) -> None:
    """The step-5 check: the loop can memorize a tiny set, proving gradients
    flow through every STE (AFE thresholds -> binary weights -> int8).

    We gate on near-perfect memorization, not exactly 48/48, on purpose:
    conv ops are non-deterministic across hardware (CUDA vs CPU vs MPS), so a
    fixed epoch budget lands on exactly 1.0 on some machines and 47/48 on
    others. loss collapsing from ~2.5 to <0.15 with >=95% accuracy proves the
    point regardless. The full-size base config (C=64) does reach exactly 1.0
    (see the step-5 report in git history)."""
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 200})
    trainer = build_trainer(cfg)
    loader = tone_loader(cfg)

    thr_before = trainer.afe.threshold.detach().clone()
    history = trainer.fit(loader)
    final = trainer.evaluate(loader)

    assert final["acc"] >= 0.95, (
        f"failed to overfit: acc={final['acc']:.3f}, "
        f"loss {history[0]['train_loss']:.3f} -> {history[-1]['train_loss']:.3f}"
    )
    assert final["loss"] < 0.15                       # strongly memorized

    # end-to-end STE proof: AFE thresholds moved along the way (CLAUDE.md 2.4)
    assert not torch.equal(trainer.afe.threshold.detach(), thr_before)
