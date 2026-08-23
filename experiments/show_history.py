"""The training curve of a finished run, including the learning rate.

Reads runs/<tag>/history.json, which needs neither torch nor the dataset, so it
answers "did this converge or did the schedule collapse?" in a second.

THE LEARNING RATE IS THE POINT. A loss curve flat for fifty epochs looks
identical whether the optimiser converged or ReduceLROnPlateau (factor 0.1,
patience 10) dropped the rate to min_lr early and froze the run. Both fx_d0 and
fx_q50 were read without it and the question could not be settled either way.

Run:
    python -m experiments.show_history --tag fx_mean_ctl
    python -m experiments.show_history --tag fx_nobin --every 1
    python -m experiments.show_history --tag fx_k2 --tag fx_d0     # side by side
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load(runs: str, tag: str) -> List[Dict]:
    p = Path(runs) / tag / "history.json"
    if not p.is_file():
        raise SystemExit(f"{p} 가 없다. 태그를 확인하거나 학습이 끝났는지 보라.")
    doc = json.loads(p.read_text())
    # the file is {"history": [...], "wall_time_s": ..., "device": ...}, not a
    # bare list -- iterating the top level gives strings and a confusing crash
    rows = doc["history"] if isinstance(doc, dict) else doc
    if not rows:
        raise SystemExit(f"{p} 의 history 가 비어 있다.")
    return rows


def show(tag: str, rows: List[Dict], every: int) -> None:
    wall = ""
    print(f"=== {tag}  ({len(rows)} epochs){wall}")
    cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    have = [c for c in cols if c in rows[0]]
    print("  " + "".join(f"{c:>12}" for c in have))
    for r in rows:
        if r["epoch"] % every and r is not rows[-1]:
            continue
        line = ""
        for c in have:
            v = r.get(c)
            if v is None:
                line += f"{'—':>12}"
            elif c == "epoch":
                line += f"{v:>12d}"
            elif c == "lr":
                line += f"{v:>12.2e}"
            else:
                line += f"{v:>12.4f}"
        print("  " + line)

    # The two readings that need saying out loud rather than eyeballing.
    lrs = [r["lr"] for r in rows if "lr" in r]
    if lrs:
        floor = min(lrs)
        first_at_floor = next((r["epoch"] for r in rows
                               if r.get("lr", 1.0) <= floor * 1.001), None)
        drops = sum(1 for a, b in zip(lrs, lrs[1:]) if b < a * 0.99)
        print(f"\n  lr {lrs[0]:.1e} -> {lrs[-1]:.1e}, {drops} drops")
        if first_at_floor is not None and first_at_floor < len(rows) * 0.6:
            print(f"  ⚠ hit its floor at epoch {first_at_floor} of {len(rows)}: "
                  f"the last {len(rows) - first_at_floor} epochs ran at the "
                  f"minimum rate.\n    A flat loss after that is the SCHEDULE "
                  f"stopping, not the model converging.")
        else:
            print("  The rate was still moving late, so a flat loss is the "
                  "model converging\n    rather than the schedule giving up.")

    best = max(rows, key=lambda r: r.get("val_acc", -1.0))
    if "val_acc" in best:
        print(f"  best val_acc {best['val_acc']:.4f} at epoch {best['epoch']} "
              f"(best.pt is this one)")
    gap = best.get("train_acc", 0.0) - best.get("val_acc", 0.0)
    print(f"  train - val at that epoch: {gap:+.4f}"
          + ("   underfitting: no gap to close with more data or regularisation"
             if gap < 0.02 else
             "   a real gap: regularisation or more data can help"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True,
                    help="repeat to compare runs side by side")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--every", type=int, default=10,
                    help="print every Nth epoch (the last is always shown)")
    args = ap.parse_args()
    for i, tag in enumerate(args.tag):
        if i:
            print()
        show(tag, load(args.runs, tag), args.every)


if __name__ == "__main__":
    main()
