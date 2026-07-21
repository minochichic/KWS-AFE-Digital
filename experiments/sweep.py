"""(C, T) sweep over Speech Commands v2 -- the current milestone (CLAUDE.md 1).

Finds the smallest (C, T) that clears the 85% 12-class target, which then
fixes the hardware size. Runs on Colab (needs the dataset); each point trains
a model, evaluates on the test split, and appends a row to a results JSON that
IS version-controlled (unlike checkpoints).

    python -m experiments.sweep --config configs/base.yaml \
        --C 16 32 48 64 --T 40 64 96 128 --epochs 100

Envelope window follows T so native_T covers it (25 ms for T<=40 else 10 ms),
unless --envelope-ms is given explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from data.afe import AFEFrontend
from data.speech_commands import build_dataloaders, class_names
from models.binary_matchboxnet import BinaryMatchboxNet
from train.config import load_config
from train.train import Trainer, set_seed


def envelope_ms_for_T(T: int, override: float) -> float:
    if override > 0:
        return override
    return 25.0 if T <= 40 else 10.0


def run_point(config_path: str, C: int, T: int, epochs: int,
              envelope_ms: float, seed: int, extra: Dict = None) -> Dict:
    overrides = {
        "model.C": C, "model.T": T,
        "afe.envelope_win_ms": envelope_ms,
        "train.epochs": epochs, "train.seed": seed,
        "tag": f"sweep_C{C}_T{T}",
    }
    overrides.update(extra or {})                    # e.g. data.root
    cfg = load_config(config_path, overrides)

    set_seed(seed)
    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data, cfg.train.batch_size, cfg.afe.sample_rate, seed=seed)

    # init AFE thresholds from one training batch (Cerutti IV-A)
    waves, _ = next(iter(train_loader))
    afe.init_thresholds(waves)

    trainer = Trainer(cfg, model, afe=afe)
    t0 = time.time()
    trainer.fit(train_loader, val_loader)
    test = trainer.evaluate(test_loader)

    return {
        "C": C, "T": T, "envelope_ms": envelope_ms, "epochs": epochs,
        "seed": seed, "params": n_params,
        "val_acc": max(r.get("val_acc", 0.0) for r in trainer.history),
        "test_acc": round(test["acc"], 4),
        "test_loss": round(test["loss"], 4),
        "wall_s": round(time.time() - t0, 1),
        "meets_target": test["acc"] >= 0.85,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--C", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--T", type=int, nargs="+", default=[40, 64, 96, 128])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--envelope-ms", type=float, default=0.0,
                    help="override; default 25 ms for T<=40 else 10 ms")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="experiments/results/sweep.json")
    ap.add_argument("overrides", nargs="*",
                    help="dotted config overrides, e.g. data.root=/content/ds")
    args = ap.parse_args()

    from experiments.inspect_config import _parse_overrides
    extra = _parse_overrides(args.overrides)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    if out.exists():
        results = json.loads(out.read_text()).get("results", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"sweep on {device}: C={args.C} T={args.T} epochs={args.epochs}\n")

    # point-level resume: a (C,T) already in the results file is skipped, so
    # re-running the sweep after a Colab disconnect continues where it stopped.
    done = {(r["C"], r["T"]) for r in results}

    for C in args.C:
        for T in args.T:
            if (C, T) in done:
                print(f"=== C={C} T={T}: already in {out.name}, skipping ===\n")
                continue
            env = envelope_ms_for_T(T, args.envelope_ms)
            print(f"=== C={C} T={T} (envelope {env:.0f} ms) ===")
            row = run_point(args.config, C, T, args.epochs, env, args.seed,
                            extra=extra)
            results.append(row)
            print(f"  -> test_acc {row['test_acc']:.3f}  params {row['params']:,}"
                  f"  {'✓ TARGET' if row['meets_target'] else ''}  ({row['wall_s']:.0f}s)\n")
            out.write_text(json.dumps(
                {"classes": class_names(), "results": results}, indent=2))

    print_summary(results)


def print_summary(results: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print(f"{'C':>4}{'T':>5}{'params':>10}{'test_acc':>10}  target")
    print("-" * 60)
    for r in sorted(results, key=lambda x: (x["C"], x["T"])):
        mark = "  ✓" if r["meets_target"] else ""
        print(f"{r['C']:>4}{r['T']:>5}{r['params']:>10,}"
              f"{r['test_acc']:>10.3f}{mark}")

    passing = [r for r in results if r["meets_target"]]
    if passing:
        best = min(passing, key=lambda r: (r["params"], r["C"], r["T"]))
        print("-" * 60)
        print(f"smallest config >= 85%: C={best['C']} T={best['T']} "
              f"({best['params']:,} params, {best['test_acc']:.3f})")
    else:
        print("-" * 60)
        print("no config reached 85% -- see CLAUDE.md note on the AFE ceiling; "
              "first ablation: conv2 separable:false")


if __name__ == "__main__":
    main()
