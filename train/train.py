"""Training loop for BinaryMatchboxNet (+ jointly trained AFE thresholds).

Scope (FIRST_TASK.md step 5): a config-driven, reproducible Trainer proven on
synthetic data. The real Speech Commands pipeline is step 6 -- the CLI below
therefore only exposes the synthetic overfit smoke test for now.

Reproducibility (CLAUDE.md 5): seed everything, write history.json +
config.yaml + checkpoints under <out_dir>/<tag>/ for every run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.config import Config, load_config

# imported lazily in main() to keep `python -m train.train --help` fast


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds CUDA too, when present


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.train.lr,
                                betas=tuple(cfg.train.betas),
                                weight_decay=cfg.train.weight_decay)
    if name == "novograd":
        # Deliberately loud: silently falling back to Adam would produce runs
        # whose config lies about what trained them.
        raise NotImplementedError(
            "NovoGrad + Warmup-Hold-Decay is phase 2 (CLAUDE.md 3: start with "
            "Adam until reproduction is stable)")
    raise ValueError(f"unknown optimizer {cfg.train.optimizer!r}")


class Trainer:
    """Joint training of BinaryMatchboxNet and (optionally) the AFE front end.

    If `afe` is given, batches are (waveform, label) and the AFE runs inside
    the training graph so its thresholds learn end-to-end (CLAUDE.md 2.4).
    Otherwise batches are already-binarized images (image, label).
    """

    def __init__(self, cfg: Config, model: nn.Module,
                 afe: Optional[nn.Module] = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.afe = afe.to(self.device) if afe is not None else None

        params = list(self.model.parameters())
        if self.afe is not None:
            params += list(self.afe.parameters())
        self.optimizer = _make_optimizer(cfg, params)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.train.label_smoothing)

        sch = cfg.train.scheduler
        if sch == "plateau":
            # Cerutti VI-A: lr /10 when stuck for 10 epochs. Both numbers are
            # config now -- see train/config.py lr_factor for why the default
            # is harsher on a binary run than on a continuous one.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor=float(getattr(cfg.train, "lr_factor", 0.1)),
                patience=int(getattr(cfg.train, "lr_patience", 10)),
                min_lr=cfg.train.min_lr)
        elif sch == "none":
            self.scheduler = None
        else:
            raise NotImplementedError(
                f"scheduler {sch!r} is phase 2 (supported now: plateau, none)")

        # AMP only where it actually works; on CPU/MPS it would silently no-op
        # or error, so gate on CUDA.
        self.amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.run_dir = Path(cfg.train.out_dir) / cfg.tag
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ #
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.afe is not None:
            x = self.afe(x, target_T=self.cfg.model.T)
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if self.afe is not None:
            self.afe.eval()
        with torch.no_grad():
            return self._forward(x.to(self.device)).cpu()

    # ------------------------------------------------------------------ #
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.afe is not None:
            self.afe.train()

        total, correct, loss_sum = 0, 0, 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=self.amp):
                logits = self._forward(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for g in self.optimizer.param_groups for p in g["params"]),
                    self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = y.numel()
            total += bs
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()

            le = self.cfg.train.log_every
            if le and (step + 1) % le == 0:
                print(f"  epoch {epoch} step {step + 1}: "
                      f"loss {loss_sum / total:.4f} acc {correct / total:.3f}")

        return {"loss": loss_sum / total, "acc": correct / total}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        if self.afe is not None:
            self.afe.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self._forward(x)
            loss_sum += self.criterion(logits, y).item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        return {"loss": loss_sum / total, "acc": correct / total}

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            resume: bool = False) -> List[Dict]:
        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.yaml")

        # Resume from the per-epoch last.pt if asked and present. Colab drops
        # long runs; re-running the same command with --resume continues from
        # the last completed epoch instead of restarting. Checkpoints are
        # device-agnostic, so a T4 run resumes on an A100 unchanged.
        start_epoch, best_acc, prev_wall = 1, -1.0, 0.0
        last_path = self.run_dir / "last.pt"
        if resume and last_path.exists():
            ck = self.load_checkpoint(last_path)
            start_epoch = ck["epoch"] + 1
            best_acc = ck.get("best_acc", -1.0)
            prev_wall = ck.get("wall_time_s", 0.0)
            self.history = ck.get("history", [])
            print(f"resuming '{cfg.tag}' from epoch {ck['epoch']} "
                  f"-> {start_epoch}/{cfg.train.epochs} on {self.device}")

        if start_epoch > cfg.train.epochs:
            print(f"'{cfg.tag}' already trained {cfg.train.epochs} epochs; "
                  f"nothing to do (change --tag or delete last.pt to retrain)")
            return self.history

        t0 = time.time()
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            tr = self.train_epoch(train_loader, epoch)
            row: Dict = {"epoch": epoch, "train_loss": tr["loss"],
                         "train_acc": tr["acc"],
                         "lr": self.optimizer.param_groups[0]["lr"]}

            monitor = tr["loss"]
            if val_loader is not None:
                va = self.evaluate(val_loader)
                row.update(val_loss=va["loss"], val_acc=va["acc"])
                monitor = va["loss"]

            if self.scheduler is not None:
                self.scheduler.step(monitor)

            self.history.append(row)
            wall = prev_wall + (time.time() - t0)

            # best.pt when val improves; last.pt EVERY epoch so a crash between
            # epochs loses at most one epoch of work.
            if val_loader is not None and row.get("val_acc", -1.0) > best_acc:
                best_acc = row["val_acc"]
                self.save_checkpoint(self.run_dir / "best.pt", epoch,
                                     best_acc=best_acc, wall_time_s=wall)
            self.save_checkpoint(last_path, epoch, best_acc=best_acc,
                                 wall_time_s=wall)
            self._write_history(wall)

            if cfg.train.log_every:
                # lr is in `row` and has always been written to history.json,
                # but it was not on screen -- and it is the one number that
                # separates "converged" from "ReduceLROnPlateau collapsed the
                # rate early and the run has been frozen since". A flat loss
                # curve looks identical either way without it.
                msg = (f"epoch {epoch:3d}/{cfg.train.epochs}  "
                       f"loss {row['train_loss']:.4f}  acc {row['train_acc']:.3f}")
                if "val_acc" in row:
                    msg += f"  val_acc {row['val_acc']:.3f}"
                msg += f"  lr {row['lr']:.2e}"
                print(msg)

        return self.history

    def _write_history(self, wall: float) -> None:
        with (self.run_dir / "history.json").open("w") as f:
            json.dump({"history": self.history,
                       "wall_time_s": round(wall, 1),
                       "device": str(self.device)}, f, indent=2)

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Path, epoch: int, best_acc: float = -1.0,
                        wall_time_s: float = 0.0) -> None:
        state = {"epoch": epoch,
                 "best_acc": best_acc,
                 "wall_time_s": wall_time_s,
                 "history": self.history,
                 "model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict(),
                 "scheduler": (self.scheduler.state_dict()
                               if self.scheduler is not None else None),
                 "scaler": self.scaler.state_dict(),
                 "config": self.cfg.to_dict()}
        if self.afe is not None:
            state["afe"] = self.afe.state_dict()
        # atomic write: a disconnect mid-save must not corrupt last.pt
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path) -> Dict:
        """Restore full training state; returns the saved metadata dict.

        Tolerant of older checkpoints missing scheduler/scaler/history keys.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        if self.afe is not None and "afe" in state:
            self.afe.load_state_dict(state["afe"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        return state


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overfit-smoke", action="store_true",
                    help="train on synthetic tones to verify the loop (no "
                         "dataset needed)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<tag>/last.pt if it exists "
                         "(safe on a fresh run: just starts from epoch 1)")
    ap.add_argument("overrides", nargs="*",
                    help="dotted config overrides, e.g. data.root=/content/ds "
                         "model.C=32")
    args = ap.parse_args()

    if args.overfit_smoke:
        _run_overfit_smoke(args)
    else:
        _run_speech_commands(args)


def _base_overrides(args, default_tag: str) -> dict:
    from experiments.inspect_config import _parse_overrides
    overrides = _parse_overrides(args.overrides)
    overrides["tag"] = args.tag or default_tag
    if args.epochs:
        overrides["train.epochs"] = args.epochs
    return overrides


def _run_overfit_smoke(args) -> None:
    from data.afe import AFEFrontend
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.synthetic import make_tone_dataset
    from torch.utils.data import DataLoader, TensorDataset

    cfg = load_config(args.config, _base_overrides(args, "overfit_smoke"))

    set_seed(cfg.train.seed)
    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)

    waves, labels = make_tone_dataset(cfg.model.n_classes, per_class=4,
                                      sample_rate=cfg.afe.sample_rate, seed=0)
    afe.init_thresholds(waves)
    loader = DataLoader(TensorDataset(waves, labels),
                        batch_size=cfg.train.batch_size, shuffle=True)

    trainer = Trainer(cfg, model, afe=afe)
    trainer.fit(loader)
    final = trainer.evaluate(loader)
    print(f"\nfinal: loss {final['loss']:.4f}  acc {final['acc']:.3f}")


def _run_speech_commands(args) -> None:
    """Real Speech Commands v2 training (Colab: downloads on first call)."""
    from data.afe import AFEFrontend, _NEEDS_SCALE, collect_init_batch
    from data.speech_commands import build_dataloaders
    from models.binary_matchboxnet import BinaryMatchboxNet

    cfg = load_config(args.config, _base_overrides(args, "sc_v2"))

    set_seed(cfg.train.seed)
    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data, cfg.train.batch_size, cfg.afe.sample_rate, seed=cfg.train.seed)

    # The same two-step init the notebook does, because that is the path every
    # recorded run came from and a CLI run has to be comparable to them.
    #
    # It used to be one batch and init_thresholds alone, which is wrong twice.
    # Every normalize mode carrying a dataset-level constant needs
    # init_fixed_scale FIRST -- "fixed", "agc", "xmax", "xmix", "xlse" -- and
    # without it this entry point simply raises, so it could only ever train
    # "minmax". And one batch is too few: delta is a 2% quantile and frames
    # inside a clip are correlated, so its effective sample size tracks CLIPS,
    # not frames. Measured across two seeds it moved 2.4x on one batch
    # (data/afe.py collect_init_batch). That number is the V_ref offset handed
    # to the analog side.
    waves = collect_init_batch(train_loader)         # 2048 clips
    if cfg.afe.normalize in _NEEDS_SCALE:
        afe.init_fixed_scale(waves)                  # delta first
    afe.init_thresholds(waves)                       # then the thresholds

    trainer = Trainer(cfg, model, afe=afe)
    trainer.fit(train_loader, val_loader, resume=args.resume)

    # Score best.pt, not whatever the last epoch happened to leave behind.
    # fit() does not restore it, and every downstream script -- fixed_accuracy,
    # threshold_placement, export.emit -- loads best.pt. Reporting the final
    # epoch here meant the headline number and the number everything else
    # works from were two different models, agreeing only when the last epoch
    # happened to be the best one.
    last = trainer.evaluate(test_loader)
    best_path = trainer.run_dir / "best.pt"
    if best_path.is_file():
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        if afe is not None and "afe" in state:
            afe.load_state_dict(state["afe"])
        best = trainer.evaluate(test_loader)
        print(f"\ntest (best.pt, epoch {state.get('epoch', '?')}): "
              f"loss {best['loss']:.4f}  acc {best['acc']:.3f}  "
              f"({'MEETS' if best['acc'] >= 0.85 else 'below'} 85% target)")
        if abs(best["acc"] - last["acc"]) > 0.002:
            print(f"  (the final epoch scored {last['acc']:.3f}; best.pt is "
                  f"what every other script reads)")
    else:
        print(f"\ntest (final epoch, no best.pt): loss {last['loss']:.4f}  "
              f"acc {last['acc']:.3f}  "
              f"({'MEETS' if last['acc'] >= 0.85 else 'below'} 85% target)")


if __name__ == "__main__":
    main()
